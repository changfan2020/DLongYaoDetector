# -*- coding: utf-8 -*-
"""
================================================================================
 龙妖首板基因识别器 —— 因子打分与失效判定核心 (v2)
================================================================================
 四大基因模块 -> 12 因子 -> 分位打分(0~100) -> 模块加权 -> 总分(0~100)

 优先级链(高于加权打分):
   1) 一票否决失效信号   : 触发即剔除候选池
   2) 情绪周期 4 档折扣  : 主升×1.0 / 震荡×0.8 / 退潮×0.5(门槛提至80) / 冰点暂停
   3) 入选规则           : >=70 观察池 | 55~69 备选 | <55 剔除

 ★ 严禁未来函数: 实时打分只使用当日及之前已产生的数据。
 ★ akshare 为网页爬虫数据源, 盘中存在延迟、精细封单数据有限;
   本系统仅用于策略原型验证, 不直接等同于实盘交易依据。
================================================================================
"""
import os
import json
import time
import datetime as dt

import numpy as np
import pandas as pd

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
import warnings
warnings.filterwarnings("ignore")

import akshare as ak

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG = os.path.join(BASE_DIR, "config.json")
DS = os.path.join(DATA_DIR, "dataset_v2.csv")
IND_MAP_CSV = os.path.join(DATA_DIR, "industry_map.csv")

# 与 build_dataset.py 完全一致的因子方向定义
FACTORS = {
    "plate_strength": ("plate", +1), "plate_has_leader": ("plate", +1),
    "plate_limit_cnt": ("plate", +1),
    "rel_turn": ("price", +1), "volume_ratio": ("price", +1),
    "bomb_times": ("price", -1), "price_pos": ("price", -1),
    "chip_pressure": ("chip", -1), "big_money_net": ("chip", +1),
    "market_emotion": ("emo", +1), "high_limit_premium": ("emo", +1),
    "bomb_rate": ("emo", -1),
}
MODULES = ["plate", "price", "chip", "emo"]
MOD_CN = {"plate": "板块基因", "price": "量价基因",
          "chip": "筹码资金基因", "emo": "市场情绪基因"}
N_BINS = 20

# 板块强度阈值: 板块涨停家数低于该值 -> 该板块首板总分打折(需求文档规则)
PLATE_WEAK_THRESHOLD = 2
PLATE_WEAK_DISCOUNT = 0.7


# ================================================================ 工具
def board_of(code):
    c = str(code)
    return 0 if c[:2] in ("60", "00") else (1 if c[:2] == "30" else -1)


def _to_float(v):
    """东财封单额形如 '1.35亿' / '3500万', 统一转元"""
    if v is None:
        return None
    try:
        s = str(v).strip()
        mul = 1.0
        if s.endswith("亿"):
            mul, s = 1e8, s[:-1]
        elif s.endswith("万"):
            mul, s = 1e4, s[:-1]
        return float(s) * mul
    except Exception:
        return None


def lim_pct_of(code):
    return 0.20 if str(code)[:2] == "30" else 0.10


def emotion_cycle(zt_count, max_lb, bomb_rate):
    """情绪周期 4 档: 主升/震荡/退潮/冰点 -> (档位, 折扣系数)"""
    try:
        if zt_count is None or (isinstance(zt_count, float) and np.isnan(zt_count)):
            return "震荡", 0.8
        zt = float(zt_count)
        lb = 0.0 if max_lb is None or np.isnan(float(max_lb)) else float(max_lb)
        br = 35.0 if bomb_rate is None or np.isnan(float(bomb_rate)) else float(bomb_rate)
    except Exception:
        return "震荡", 0.8
    if zt < 20 or (lb <= 2 and br > 50):
        return "冰点", 0.0
    if br > 45 or lb <= 3:
        return "退潮", 0.5
    if lb >= 6 and zt >= 60 and br < 30:
        return "主升", 1.0
    return "震荡", 0.8


def pct_score(val, edges, direction):
    if edges is None or val is None or (isinstance(val, float) and np.isnan(val)):
        return 50.0
    p = np.searchsorted(np.asarray(edges, dtype=float), float(val), side="right") - 1
    p = float(np.clip(p, 0, N_BINS - 1)) / (N_BINS - 1.0) * 100.0
    return p if direction > 0 else 100.0 - p


# ================================================================ 分析器
class DragonAnalyzer:
    def __init__(self, config_path=CONFIG):
        self.cfg = {}
        self.bins = {}
        self.mw = {"plate": 25.0, "price": 25.0, "chip": 15.0, "emo": 10.0}
        self.sw = {f: 1.0 for f in FACTORS}
        self.thr_hi, self.thr_lo, self.thr_tui = 70.0, 55.0, 80.0
        self.center, self.scale = None, 1.0      # 分数仿射拉伸(与训练集一致)
        self.ind_map = {}
        self._ds = None
        self._ind_loaded = False
        self.load_config(config_path)

    # ---------------------------------------------------------- 配置
    def load_config(self, path=CONFIG):
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                c = json.load(f)
            self.cfg = c
            self.bins = c.get("quantile_bins", {})
            mw = c.get("module_weights", {})
            rev = {v: k for k, v in MOD_CN.items()}
            for cn, w in mw.items():
                if cn in rev:
                    self.mw[rev[cn]] = float(w)
            for f, w in c.get("sub_weights", {}).items():
                if f in FACTORS:
                    self.sw[f] = float(w)
            t = c.get("thresholds", {})
            self.thr_hi = float(t.get("select", 70))
            self.thr_lo = float(t.get("备选", 55))
            self.thr_tui = float(t.get("退潮门槛", 80))
            self.center = c.get("score_center")
            self.scale = float(c.get("score_scale", 1.0) or 1.0)
            return True
        except Exception as e:
            print("[config] 加载失败: %r, 使用默认权重" % e)
            return False

    def load_industry(self):
        if self._ind_loaded:
            return
        self._ind_loaded = True
        if os.path.exists(IND_MAP_CSV):
            try:
                m = pd.read_csv(IND_MAP_CSV, dtype={"code": str})
                self.ind_map = dict(zip(m["code"].str.zfill(6), m["industry"]))
            except Exception:
                pass

    @property
    def dataset(self):
        if self._ds is None and os.path.exists(DS):
            try:
                self._ds = pd.read_csv(DS, dtype={"code": str, "date": str})
            except Exception:
                self._ds = pd.DataFrame()
        return self._ds

    # ---------------------------------------------------------- 打分
    def score_features(self, feats):
        """feats: dict(因子->原始值) -> (总分, 各模块分, 各因子分)"""
        fs = {}
        for f, (m, d) in FACTORS.items():
            fs[f] = pct_score(feats.get(f), self.bins.get(f), d)
        mods, mraw = {}, {}
        for m in MODULES:
            xs = [f for f in FACTORS if FACTORS[f][0] == m]
            num = sum(self.sw.get(f, 1.0) * fs[f] for f in xs)
            den = sum(self.sw.get(f, 1.0) for f in xs)
            mods[m] = num / den if den > 0 else 50.0
            mraw[m] = mods[m]
        tot = sum(self.mw[m] * mods[m] for m in MODULES) / max(sum(self.mw.values()), 1e-9)
        # 分数仿射拉伸: 12个分位因子平均后向50收敛, 需按训练集尺度重新展开,
        # 否则 70 分门槛几乎不可达。参数仅由训练集拟合, 不引入未来信息。
        if self.center is not None:
            tot = float(np.clip(50.0 + (float(tot) - float(self.center)) * self.scale, 0.0, 100.0))
        return tot, mraw, fs

    def apply_rules(self, tot, mraw, feats, alerts):
        """情绪周期折扣 + 板块弱势折扣 + 入选判定"""
        zt = feats.get("mkt_zt_count")
        lb = feats.get("mkt_max_lb")
        br = feats.get("bomb_rate")
        cycle, disc = emotion_cycle(zt, lb, br)
        final = tot * disc
        # 板块强度低于阈值 -> 该板块首板总分统一打折
        plate_cnt = feats.get("plate_limit_cnt")
        if plate_cnt is not None and not pd.isna(plate_cnt) and \
                float(plate_cnt) < PLATE_WEAK_THRESHOLD:
            final *= PLATE_WEAK_DISCOUNT
        if cycle == "冰点":
            status = "暂停"
        elif any(a.get("veto") for a in alerts):
            status = "剔除"
        else:
            thr = self.thr_tui if cycle == "退潮" else self.thr_hi
            if final >= thr:
                status = "观察"
            elif final >= self.thr_lo:
                status = "备选"
            else:
                status = "剔除"
        return dict(total=round(float(tot), 1), final=round(float(final), 1),
                    cycle=cycle, discount=disc, status=status)

    # ---------------------------------------------------------- 历史日打分(离线样本)
    def score_dataset_date(self, date):
        df = self.dataset
        if df is None or len(df) == 0:
            return None
        d = df[df["date"].astype(str) == str(date)]
        if len(d) == 0:
            return None
        out = []
        for _, r in d.iterrows():
            feats = {f: r.get(f) for f in FACTORS}
            feats["mkt_zt_count"] = r.get("mkt_zt_count")
            feats["mkt_max_lb"] = r.get("mkt_max_lb")
            feats["bomb_rate"] = r.get("bomb_rate")
            tot, mraw, fs = self.score_features(feats)
            alerts = self.check_veto(r, feats)
            rl = self.apply_rules(tot, mraw, feats, alerts)
            out.append(dict(
                代码=str(r["code"]), 名称=str(r.get("name", "")),
                行业=str(r.get("industry", "")),
                收盘=float(r["close"]) if not pd.isna(r.get("close")) else None,
                换手=round(float(r["turnover"]), 2) if not pd.isna(r.get("turnover")) else None,
                流通市值亿=round(float(r["float_mkt"]) / 1e8, 1) if not pd.isna(r.get("float_mkt")) else None,
                modules={MOD_CN[m]: round(float(mraw[m]), 1) for m in MODULES},
                factors={f: None if pd.isna(r.get(f)) else round(float(r[f]), 3) for f in FACTORS},
                fscores={f: round(float(fs[f]), 1) for f in FACTORS},
                score=rl["total"], final=rl["final"], cycle=rl["cycle"],
                discount=rl["discount"], status=rl["status"],
                prob=self.prob_of(rl["final"]),
                label=int(r["label_Y"]), ret_next=round(float(r["ret_next_oc"]), 2)
                if not pd.isna(r.get("ret_next_oc")) else None,
                alerts=alerts))
        out.sort(key=lambda x: -x["final"])
        m = d.iloc[0]
        cyc, disc = emotion_cycle(m.get("mkt_zt_count"), m.get("mkt_max_lb"), m.get("bomb_rate"))
        return dict(date=str(date), source="dataset", stocks=out,
                    market=dict(涨停家数=int(m["mkt_zt_count"]), 最高连板=int(m["mkt_max_lb"]),
                                炸板率=round(float(m["bomb_rate"]), 1),
                                涨停溢价=round(float(m["high_limit_premium"]), 2),
                                cycle=cyc, discount=disc),
                    base_rate=round(float(d["label_Y"].mean()) * 100, 2))

    # ---------------------------------------------------------- 实时/当日分析
    def analyze_live(self, date=None):
        """拉取当日涨停池 -> 首板筛选 -> 因子计算 -> 打分 -> 失效判定"""
        date = date or dt.datetime.now().strftime("%Y%m%d")
        self.load_industry()
        try:
            pool = ak.stock_zt_pool_em(date=date)
        except Exception as e:
            return dict(ok=False, date=date, msg="涨停池拉取失败: %r" % e, stocks=[])
        if pool is None or len(pool) == 0:
            return dict(ok=False, date=date, msg="当日无涨停数据(非交易日或数据源未更新)", stocks=[])
        pool["代码"] = pool["代码"].astype(str).str.zfill(6)
        # 炸板池(计算全市场炸板率)
        zb = 0
        try:
            z = ak.stock_zt_pool_zbgc_em(date=date)
            zb = 0 if z is None else len(z)
        except Exception:
            pass
        # 首板: 连板数 == 1
        if "连板数" in pool.columns:
            fb = pool[pd.to_numeric(pool["连板数"], errors="coerce") == 1].copy()
        else:
            fb = pool.copy()
        zt_n = len(pool)
        try:
            max_lb = int(pd.to_numeric(pool["连板数"], errors="coerce").max())
        except Exception:
            max_lb = 1
        bomb_rate = zb / max(zt_n + zb, 1) * 100

        # 行业涨停家数(板块强度)
        ind_cnt = {}
        for c in pool["代码"]:
            ind = self.ind_map.get(c)
            if ind:
                ind_cnt[ind] = ind_cnt.get(ind, 0) + 1
        # 板块最高连板(板块龙头)
        ind_lb = {}
        if "连板数" in pool.columns:
            for c, n in zip(pool["代码"], pd.to_numeric(pool["连板数"], errors="coerce")):
                ind = self.ind_map.get(c)
                if ind and not pd.isna(n):
                    ind_lb[ind] = max(ind_lb.get(ind, 0), int(n))

        # 市场情绪 / 涨停溢价(取数据集中最近一日的溢价作为近似, 无则用0)
        prem = self._recent_premium(date)
        market = dict(涨停家数=zt_n, 炸板家数=zb, 最高连板=max_lb,
                      炸板率=round(bomb_rate, 1), 涨停溢价=round(prem, 2))
        cyc, disc = emotion_cycle(zt_n, max_lb, bomb_rate)
        market["cycle"] = cyc
        market["discount"] = disc

        out = []
        for _, r in fb.iterrows():
            code = r["代码"]
            try:
                feats = self._live_features(code, date)
            except Exception as e:
                feats = None
            if feats is None:
                continue
            if feats.get("yizi"):
                continue                       # 一字首板剔除
            ind = self.ind_map.get(code, "")
            feats.update(dict(
                plate_strength=ind_cnt.get(ind, 0) / max(zt_n, 1) * 100 if ind else np.nan,
                plate_has_leader=1.0 if ind_lb.get(ind, 0) >= 2 else 0.0,
                plate_limit_cnt=float(ind_cnt.get(ind, 0)) if ind else 0.0,
                market_emotion=zt_n / 60.0 + max_lb - bomb_rate / 100 * 4.0,
                high_limit_premium=prem, bomb_rate=bomb_rate,
                mkt_zt_count=zt_n, mkt_max_lb=max_lb,
            ))
            tot, mraw, fs = self.score_features(feats)
            alerts = self.check_veto(r, feats)
            rl = self.apply_rules(tot, mraw, feats, alerts)
            out.append(dict(
                代码=code, 名称=str(r.get("名称", "")), 行业=ind,
                收盘=feats.get("close"), 换手=round(feats.get("turnover", 0) or 0, 2),
                流通市值亿=round((feats.get("float_mkt") or 0) / 1e8, 1),
                封板时间=str(r.get("首次封板时间", "")),
                炸板次数=int(r["炸板次数"]) if "炸板次数" in r and not pd.isna(r["炸板次数"]) else 0,
                modules={MOD_CN[m]: round(float(mraw[m]), 1) for m in MODULES},
                fscores={f: round(float(fs[f]), 1) for f in FACTORS},
                score=rl["total"], final=rl["final"], cycle=rl["cycle"],
                discount=rl["discount"], status=rl["status"],
                prob=self.prob_of(rl["final"]), alerts=alerts))
        out.sort(key=lambda x: -x["final"])
        return dict(ok=True, date=date, source="live", stocks=out, market=market,
                    base_rate=self._recent_base_rate())

    # ---------------------------------------------------------- 实时因子
    def _sina_realtime(self, sym):
        """新浪实时行情(盘中日K不含当日行, 用它合成), 失败返回 None"""
        try:
            import requests
            url = "https://hq.sinajs.cn/list=" + sym
            r = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"},
                             timeout=8)
            f = r.text.split('"')[1].split(",")
            if len(f) < 32 or not f[0]:
                return None
            return dict(open=float(f[1]), preclose=float(f[2]),
                        current=float(f[3]), high=float(f[4]), low=float(f[5]),
                        volume=float(f[8]), amount=float(f[9]))
        except Exception:
            return None

    def _live_features(self, code, date):
        """下载个股日K(约120日)并按与训练集完全一致的公式计算窗口因子"""
        sym = ("sh" if code[0] == "6" else "sz") + code
        d0 = dt.datetime.strptime(str(date), "%Y%m%d")
        start = (d0 - dt.timedelta(days=200)).strftime("%Y%m%d")
        k = None
        for att in range(3):
            try:
                k = ak.stock_zh_a_daily(symbol=sym, start_date=start,
                                        end_date=str(date), adjust="")
                break
            except Exception:
                time.sleep(0.8 * (att + 1))
        if k is None or len(k) < 45:
            return None
        k = k.reset_index(drop=True)
        for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share"):
            k[c] = pd.to_numeric(k[c], errors="coerce")
        k["date"] = k["date"].astype(str).str.replace("-", "", regex=False)
        k = k[k["date"].astype(str) <= str(date)]
        if len(k) < 45:
            return None
        # 盘中/盘后初期, 新浪日K尚未包含当日行 -> 用实时行情合成当日行
        if str(k.iloc[-1]["date"]) < str(date):
            q = self._sina_realtime(sym)
            if q is None or q["current"] <= 0:
                return None
            k = pd.concat([k, pd.DataFrame([dict(
                date=str(date), open=q["open"], high=q["high"], low=q["low"],
                close=q["current"], volume=q["volume"], amount=q["amount"],
                outstanding_share=float(k.iloc[-1]["outstanding_share"]))])],
                ignore_index=True)
        pos = len(k) - 1
        row = k.iloc[pos]
        preclose = float(k.iloc[pos - 1]["close"])
        lp = lim_pct_of(code)
        lim = round(preclose * (1 + lp), 2)
        close, high, low = float(row["close"]), float(row["high"]), float(row["low"])
        if not (close >= lim - 0.011 and close >= high - 1e-6):
            return None                       # 非封板收盘(可能已炸板)
        os_share = float(row["outstanding_share"])
        turnover = float(row["volume"]) / os_share * 100 if os_share > 0 else np.nan

        s = max(0, pos - 60)
        close_h = k["close"].values[s:pos].astype(float)
        vol_h = k["volume"].values[s:pos].astype(float)
        amt_h = k["amount"].values[s:pos].astype(float)
        turn_h = (k["volume"].values[s:pos] / os_share * 100) if os_share > 0 else \
            np.full(pos - s, np.nan)
        tp_h = (k["high"].values[s:pos].astype(float) +
                k["low"].values[s:pos].astype(float) +
                k["close"].values[s:pos].astype(float)) / 3.0

        t20 = np.nanmean(turn_h[-20:]) if len(turn_h) >= 20 else np.nan
        rel_turn = turnover / t20 if (t20 and t20 > 0) else np.nan
        v5 = np.nanmean(vol_h[-5:]) if len(vol_h) >= 5 else np.nan
        vol_ratio = float(row["volume"]) / v5 if (v5 and v5 > 0) else np.nan
        p30 = close_h[-30] if len(close_h) >= 30 else np.nan
        price_pos = (close / p30 - 1) * 100 if (p30 and p30 > 0) else np.nan

        w = np.exp(-0.03 * np.arange(len(close_h) - 1, -1, -1)) * np.nan_to_num(amt_h)
        tot_w = w.sum()
        if tot_w > 0:
            above = w[tp_h > close].sum() / tot_w
            near = w[(tp_h > close * 0.97) & (tp_h < close * 1.10)].sum() / tot_w
            chip_pressure = float(above * 100 + near * 50)
        else:
            chip_pressure = np.nan
        rng = high - low
        mf = ((close - low) - (high - close)) / rng if rng > 0 else 0.0
        a20 = np.nanmean(amt_h[-20:]) if len(amt_h) >= 20 else np.nan
        big_money = float(mf * float(row["amount"]) / a20 * 100) if (a20 and a20 > 0) else np.nan

        return dict(
            close=close, turnover=turnover, float_mkt=close * os_share,
            rel_turn=rel_turn, volume_ratio=vol_ratio, price_pos=price_pos,
            chip_pressure=chip_pressure, big_money_net=big_money,
            yizi=bool(low >= lim - 0.011),
        )

    # ---------------------------------------------------------- 一票否决
    def check_veto(self, row, feats):
        """失效信号(优先级高于打分): 返回告警列表, veto=True 表示直接剔除"""
        alerts = []

        def g(k, default=None):
            try:
                v = row[k]
                return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            except Exception:
                return default

        # 1) 尾盘炸板 / 炸板次数过多
        bt = g("炸板次数", feats.get("bomb_times"))
        try:
            bt = float(bt)
        except Exception:
            bt = np.nan
        if not np.isnan(bt) and bt >= 1:
            lz = str(g("最后炸板时间", "") or "")
            late = False
            if lz and ":" in lz:
                try:
                    late = int(lz.split(":")[0]) * 60 + int(lz.split(":")[1]) >= 14 * 60 + 30
                except Exception:
                    late = False
            if bt >= 3:
                alerts.append(dict(veto=True, type="反复炸板",
                                   msg="当日炸板 %d 次, 承接极差" % int(bt)))
            elif late:
                alerts.append(dict(veto=True, type="尾盘炸板",
                                   msg="尾盘炸板(%s)无法回封" % lz))
            else:
                alerts.append(dict(veto=False, type="盘中止跌炸板",
                                   msg="早盘短暂炸板 %d 次(轻扣)" % int(bt)))
        # 2) 封单崩塌 / 封单过弱
        seal_raw = g("封板资金", None)
        seal = _to_float(seal_raw) if seal_raw is not None else feats.get("seal_amt")
        fm = feats.get("float_mkt")
        if seal is not None and fm:
            try:
                ratio = float(seal) / float(fm)
                if ratio < 0.003:
                    alerts.append(dict(veto=True, type="封单崩塌",
                                       msg="封单仅占流通市值 %.2f%%, 封板力度不足" % (ratio * 100)))
                elif ratio < 0.01:
                    alerts.append(dict(veto=False, type="封单偏弱",
                                       msg="封单占流通市值 %.2f%%" % (ratio * 100)))
            except Exception:
                pass
        decay = feats.get("seal_decay")
        if decay is not None and not pd.isna(decay) and float(decay) < -0.35:
            alerts.append(dict(veto=True, type="封单持续崩塌",
                               msg="封单较上一轮扫描衰减 %.0f%%" % (abs(float(decay)) * 100)))
        # 3) 板块脱离主线(板块内无涨停效应)
        pc = feats.get("plate_limit_cnt")
        if pc is not None and not pd.isna(pc) and float(pc) < PLATE_WEAK_THRESHOLD:
            alerts.append(dict(veto=False, type="脱离主线",
                               msg="所属板块当日涨停仅 %d 家, 无板块效应(总分已打%.0f折)"
                                   % (int(float(pc)), PLATE_WEAK_DISCOUNT * 10)))
        # 4) 高位首板
        pp = feats.get("price_pos")
        if pp is not None and not pd.isna(pp) and float(pp) > 60:
            alerts.append(dict(veto=False, type="高位首板",
                               msg="近30日累计涨幅 %.0f%%, 位置偏高" % float(pp)))
        # 5) 冰点周期
        if feats.get("cycle") == "冰点":
            alerts.append(dict(veto=True, type="冰点周期", msg="市场冰点, 暂停新标的筛选"))
        return alerts

    # ---------------------------------------------------------- 竞价二次校验(盘前)
    def precheck(self, date, codes, open_drop=-3.0):
        """次日 9:25 竞价: 大幅低开/核按钮 -> 移出观察池"""
        res = []
        for c in codes:
            try:
                sym = ("sh" if c[0] == "6" else "sz") + c
                d0 = dt.datetime.strptime(str(date), "%Y%m%d")
                k = ak.stock_zh_a_daily(
                    symbol=sym,
                    start_date=(d0 - dt.timedelta(days=20)).strftime("%Y%m%d"),
                    end_date=str(date), adjust="")
                if k is None or len(k) == 0:
                    res.append(dict(代码=c, 状态="无数据", 竞价涨幅=None))
                    continue
                row = k.iloc[-1]
                pre = float(k.iloc[-2]["close"]) if len(k) >= 2 else float(row["close"])
                op = float(row["open"])
                pct = (op / pre - 1) * 100 if pre > 0 else np.nan
                st = "移出观察池" if (not np.isnan(pct) and pct < open_drop) else "保留"
                res.append(dict(代码=c, 竞价涨幅=round(float(pct), 2), 状态=st,
                                reason="竞价大幅低开 %.2f%%(核按钮)" % pct
                                if st == "移出观察池" else ""))
            except Exception as e:
                res.append(dict(代码=c, 状态="查询失败", reason=repr(e)[:60]))
            time.sleep(0.1)
        return res

    # ---------------------------------------------------------- 概率/校准
    def prob_of(self, final_score):
        for b in self.cfg.get("calibration", []):
            if b["lo"] <= final_score < b["hi"]:
                return round(b["p"] * 100, 1)
        return None

    def _recent_premium(self, date):
        df = self.dataset
        if df is not None and len(df):
            d = df[df["date"].astype(str) <= str(date)]
            if len(d):
                return float(d.iloc[-1]["high_limit_premium"])
        return 0.0

    def _recent_base_rate(self):
        df = self.dataset
        if df is not None and len(df):
            return round(float(df["label_Y"].mean()) * 100, 2)
        return None


# ================================================================ CLI 自检
if __name__ == "__main__":
    import sys
    a = DragonAnalyzer()
    print("=" * 74)
    print("龙妖首板基因识别器 · 打分核心自检")
    print("=" * 74)
    print("模块权重:", {MOD_CN[m]: a.mw[m] for m in MODULES})
    print("入选门槛: >=%.0f 观察 | %.0f~%.0f 备选 | 退潮期门槛 %.0f"
          % (a.thr_hi, a.thr_lo, a.thr_hi, a.thr_tui))
    ok = a.dataset is not None and len(a.dataset) > 0
    print("数据集:", "已加载 %d 条" % len(a.dataset) if ok else "未找到(先运行 build_dataset.py)")
    if ok:
        dates = sorted(a.dataset["date"].astype(str).unique())
        last = dates[-1]
        r = a.score_dataset_date(last)
        print("-" * 74)
        print("回看 %s: 涨停 %s 家 | 最高连板 %s | 炸板率 %s%% | 情绪周期 %s(×%.1f)"
              % (last, r["market"]["涨停家数"], r["market"]["最高连板"],
                 r["market"]["炸板率"], r["market"]["cycle"], r["market"]["discount"]))
        print("候选首板 %d 只, 全体晋级率 %.2f%%" % (len(r["stocks"]), r["base_rate"]))
        print("-" * 74)
        print("%-8s %-6s %6s %6s %6s %6s %6s %6s  %s" %
              ("代码", "名称", "板块", "量价", "筹码", "情绪", "总分", "晋级率", "状态"))
        for s in r["stocks"][:12]:
            m = s["modules"]
            print("%-8s %-6s %6.1f %6.1f %6.1f %6.1f %6.1f %6s%%  %s"
                  % (s["代码"], (s["名称"] or "")[:5], m["板块基因"], m["量价基因"],
                     m["筹码资金基因"], m["市场情绪基因"], s["final"],
                     s["prob"] if s["prob"] is not None else "-", s["status"]))
        n = {"观察": 0, "备选": 0, "剔除": 0, "暂停": 0}
        for s in r["stocks"]:
            n[s["status"]] = n.get(s["status"], 0) + 1
        print("-" * 74)
        print("入选分布:", n)
        obs = [s for s in r["stocks"] if s["status"] == "观察"]
        if obs:
            real = [s["label"] for s in obs if s["label"] is not None]
            if real:
                print("观察池实际晋级率: %.1f%% (基线 %.2f%%)"
                      % (sum(real) / len(real) * 100, r["base_rate"]))
    print("=" * 74)
    print("自检完成 ✔")
