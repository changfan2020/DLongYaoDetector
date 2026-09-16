# -*- coding: utf-8 -*-
"""
================================================================================
 龙妖首板基因识别器 —— 数据集构建脚本 (v2)
================================================================================
 底层策略: 龙妖战法  目标任务: 首板识别 + 次日开盘介入
 预测目标: 首板次日 1 进 2 晋级概率 (Y=1 晋级 / Y=0 断板)

 四大基因模块 (12 个因子):
   模块1 板块基因   : plate_strength / plate_has_leader / plate_limit_cnt
   模块2 量价基因   : rel_turn / volume_ratio / bomb_times / price_pos
   模块3 筹码资金   : chip_pressure / big_money_net
   模块4 市场情绪   : market_emotion / high_limit_premium / bomb_rate

 ★ 严禁未来函数: 所有因子只使用 T 日及之前已产生的数据; 标签 Y 使用 T+1 日数据。
 ★ 时序切分: train 2023.01~2025.12 / valid 2026.01~2026.06 / test 2026.07~今, 不随机打乱。
 ★ 样本清洗: 剔除 ST/退市/上市不足20日/一字首板/大盘权重股/暴跌日/暴雷跌停/数据残缺。

 数据源说明(akshare 为网页爬虫源, 存在延迟与限流, 仅用于策略原型验证):
   A. 新浪日K stock_zh_a_daily       —— 深度历史全量 OHLCV + 流通股本 (主数据源)
   B. 新浪行业 stock_sector_detail    —— 84 个行业板块成分股映射
   C. 东财涨停池 stock_zt_pool_em     —— 近 15 日富因子(封板时间/炸板次数/封单额)
   D. 交易所股票清单 stock_info_a_code_name —— 股票名称(ST 识别)
================================================================================
"""
import argparse
import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd

# 环境内代理不稳定, 一律直连(东财 push2* 域名在本机被限流, push2ex/新浪/交易所正常)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import warnings
warnings.filterwarnings("ignore")

import akshare as ak

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------- 路径常量
BARS_SHARDS = [os.path.join(DATA_DIR, "bars_shard%d.csv" % i) for i in range(3)]
DONE_SHARDS = [p + ".done" for p in BARS_SHARDS]
IND_MAP_CSV = os.path.join(DATA_DIR, "industry_map.csv")
NAME_CSV = os.path.join(DATA_DIR, "code_name.csv")
ZT_POOL_CSV = os.path.join(DATA_DIR, "zt_pool_raw.csv")
OUT_CSV = os.path.join(DATA_DIR, "dataset_v2.csv")
REPORT_CSV = os.path.join(DATA_DIR, "build_report.csv")

SHARDS = 3                      # 下载并行进程数
DOWNLOAD_START = "20221001"     # 需覆盖 2023.01 起样本, 预留 60 日因子回看窗口
MIN_LISTED_DAYS = 20            # 上市不足 20 日剔除
MAX_FLOAT_MKT = 500e8           # 流通市值上限(剔除大盘权重股), 单位: 元
CRASH_DROP = -0.05              # 大盘单日暴跌阈值(中位数跌幅 < -5% 当日样本全剔)
SPLIT_TRAIN_END = "20251231"
SPLIT_VALID_END = "20260630"


def log(msg):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, "build_dataset.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def board_of(code):
    """0=主板10cm, 1=创业板20cm, -1=剔除(北交所/科创板/其他)"""
    c = str(code)
    if c[:2] in ("60", "00"):
        return 0
    if c[:2] == "30":
        return 1
    return -1


def lim_pct_of(code):
    return 0.20 if str(code)[:2] == "30" else 0.10


# ================================================================ Stage 0: 行业映射
def stage0_industry():
    """新浪 84 行业板块成分股 -> code->行业 映射(缓存)"""
    if os.path.exists(IND_MAP_CSV) and os.path.getsize(IND_MAP_CSV) > 1000:
        m = pd.read_csv(IND_MAP_CSV, dtype={"code": str})
        log("[行业] 使用缓存映射: %d 只股票" % len(m))
        return dict(zip(m["code"].str.zfill(6), m["industry"]))
    log("[行业] 拉取新浪行业板块成分(84个板块, 约1-2分钟)...")
    try:
        sectors = ak.stock_sector_spot(indicator="行业")
    except Exception as e:
        log("[行业] 板块列表拉取失败: %r (板块因子将置为空)" % e)
        return {}
    labels = sectors["label"].tolist()
    names = sectors["板块"].tolist()
    rows = []
    for i, (lb, nm) in enumerate(zip(labels, names)):
        for att in range(3):
            try:
                c = ak.stock_sector_detail(sector=lb)
                if c is not None and len(c):
                    for code in c["code"].astype(str):
                        rows.append((code.zfill(6)[-6:], nm))
                break
            except Exception:
                time.sleep(1.0 * (att + 1))
        if (i + 1) % 20 == 0:
            log("  行业进度 %d/%d" % (i + 1, len(labels)))
    m = pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code")
    m.to_csv(IND_MAP_CSV, index=False, encoding="utf-8")
    log("[行业] 完成: %d 只股票, %d 个行业" % (len(m), m["industry"].nunique()))
    return dict(zip(m["code"], m["industry"]))


# ================================================================ Stage 1: 股票名称(ST识别)
def stage0_name():
    if os.path.exists(NAME_CSV) and os.path.getsize(NAME_CSV) > 1000:
        n = pd.read_csv(NAME_CSV, dtype={"code": str})
        return dict(zip(n["code"].str.zfill(6), n["name"].astype(str)))
    log("[清单] 拉取交易所股票清单(ST识别)...")
    for att in range(3):
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={df.columns[0]: "code", df.columns[1]: "name"})
            df["code"] = df["code"].astype(str).str.zfill(6)
            df[["code", "name"]].to_csv(NAME_CSV, index=False, encoding="utf-8")
            log("[清单] 完成: %d 只" % len(df))
            return dict(zip(df["code"], df["name"].astype(str)))
        except Exception as e:
            log("[清单] 第%d次失败: %r" % (att + 1, e))
            time.sleep(2)
    return {}


# ================================================================ Stage 2: 东财涨停池(近15日富因子)
def stage1_em_pool(days=18):
    """东财涨停池: 封板时间/炸板次数/封单额(服务端仅保留最近约15个交易日)"""
    old = pd.DataFrame()
    have = set()
    if os.path.exists(ZT_POOL_CSV) and os.path.getsize(ZT_POOL_CSV) > 100:
        old = pd.read_csv(ZT_POOL_CSV, dtype={"代码": str, "日期": str})
        have = set(old["日期"].unique())
    cal = ak.tool_trade_date_hist_sina()
    cal = [d.strftime("%Y%m%d") for d in cal["trade_date"]]
    now = dt.datetime.now().strftime("%Y%m%d")
    cal = [d for d in cal if d <= now][-days:]
    todo = [d for d in cal if d not in have]
    log("[数据源C] 东财涨停池: 需拉取 %d 天 (缓存 %d 天)" % (len(todo), len(have)))
    got = []
    for d in todo:
        try:
            df = ak.stock_zt_pool_em(date=d)
            if df is not None and len(df):
                df["日期"] = d
                got.append(df)
                log("  涨停池 %s: %d 只" % (d, len(df)))
            time.sleep(0.3)
        except Exception as e:
            log("  涨停池 %s 失败: %r" % (d, e))
    if got:
        new = pd.concat(got, ignore_index=True)
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
        old.to_csv(ZT_POOL_CSV, index=False, encoding="utf-8")
    log("[数据源C] 涨停池累计 %d 条" % (0 if not len(old) else len(old)))
    return old if len(old) else pd.DataFrame()


# ================================================================ Stage 3: 新浪日K全量下载
def _dl_worker(sid, codes, start_d, end_d, out_path, done_path):
    import akshare as _ak
    import warnings as _w
    _w.filterwarnings("ignore")
    n_ok, t0 = 0, time.time()
    first_write = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    f = open(out_path, "a", encoding="utf-8", newline="")
    d = open(done_path, "a", encoding="utf-8")
    buf = []
    for i, code in enumerate(codes):
        sym = ("sh" if code[0] == "6" else "sz") + code
        k = None
        for att in range(3):
            try:
                k = _ak.stock_zh_a_daily(symbol=sym, start_date=start_d,
                                         end_date=end_d, adjust="")
                break
            except Exception:
                time.sleep(1.2 * (att + 1))
        if k is None or len(k) == 0:
            continue
        k = k.copy()
        k["code"] = code
        keep = ["date", "code", "open", "high", "low", "close",
                "volume", "amount", "outstanding_share"]
        for c in keep:
            if c not in k.columns:
                k[c] = np.nan
        buf.append(k[keep])
        n_ok += 1
        d.write(code + "\n")
        if (i + 1) % 100 == 0:
            pd.concat(buf).to_csv(f, header=first_write, index=False)
            first_write = False
            buf = []
            d.flush()
            print("[shard%d] %d/%d (%.0fs)" % (sid, i + 1, len(codes), time.time() - t0),
                  flush=True)
    if buf:
        pd.concat(buf).to_csv(f, header=False, index=False)
    f.close()
    d.close()
    print("[shard%d] 完成 %d 只 (%.0fs)" % (sid, n_ok, time.time() - t0), flush=True)


def stage2_bars(end_date):
    """下载全市场日K(断点续传, 多进程)"""
    log("[数据源A] 新浪日K: %s ~ %s" % (DOWNLOAD_START, end_date))
    name_map = stage0_name()
    pool = [c for c in name_map
            if board_of(c) >= 0 and "ST" not in name_map[c] and "退" not in name_map[c]]
    pool.sort()
    log("[数据源A] 目标股票 %d 只" % len(pool))

    done = set()
    for p in DONE_SHARDS:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                done |= set(x.strip() for x in f if x.strip())
    todo = [c for c in pool if c not in done]
    log("[数据源A] 断点续传: 已完成 %d, 待下载 %d" % (len(done), len(todo)))

    if todo:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        ps = []
        for i in range(SHARDS):
            codes = todo[i::SHARDS]
            if not codes:
                continue
            p = ctx.Process(target=_dl_worker,
                            args=(i, codes, DOWNLOAD_START, end_date,
                                  BARS_SHARDS[i], DONE_SHARDS[i]))
            p.start()
            ps.append(p)
            time.sleep(0.5)
        for p in ps:
            p.join()
    tot = sum(os.path.getsize(p) for p in BARS_SHARDS if os.path.exists(p))
    log("[数据源A] 日K文件共 %.1f MB" % (tot / 1024 / 1024))

    dfs = [pd.read_csv(p, dtype={"code": str}) for p in BARS_SHARDS if os.path.exists(p)]
    bars = pd.concat(dfs, ignore_index=True)
    bars["date"] = bars["date"].astype(str).str.replace("-", "", regex=False)
    for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share"):
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars = bars.dropna(subset=["close", "volume"])
    bars = bars[bars["close"] > 0]
    log("[数据源A] 载入 %d 行日K, %d 只股票" % (len(bars), bars["code"].nunique()))
    return bars


# ================================================================ Stage 4: 因子与样本
def _consec_lb(zt, code):
    """分组内连续涨停计数(向量化)"""
    arr = zt.astype(np.int64)
    newgrp = np.empty(len(arr), dtype=bool)
    newgrp[0] = True
    newgrp[1:] = code[1:] != code[:-1]
    reset = np.where(arr == 0, 1, 0)
    reset[newgrp] = 1
    seg = np.cumsum(reset)
    cum = np.cumsum(arr)
    start_cum = np.zeros(len(arr), dtype=np.int64)
    idx = np.nonzero(reset)[0]
    start_cum[idx] = cum[idx] - arr[idx]
    start_cum = np.maximum.accumulate(start_cum)
    lb = (cum - start_cum).astype(np.int32)
    lb[arr == 0] = 0
    return lb


def build_features(bars, ind_map, log_=log):
    log_("[因子] 排序与基础字段计算...")
    bars = bars.sort_values(["code", "date"]).reset_index(drop=True)
    code_arr = bars["code"].values
    g = bars.groupby("code", sort=False)
    bars["preclose"] = g["close"].shift(1)
    bars["pos"] = g.cumcount()
    bars = bars.dropna(subset=["preclose"])
    bars = bars[(bars["preclose"] > 0) & (bars["close"] > 0)]

    lp = np.where(bars["code"].str[:2].values == "30", 0.20, 0.10)
    bars["lim"] = np.round(bars["preclose"].values * (1 + lp), 2)
    bars["pct"] = bars["close"].values / bars["preclose"].values - 1
    # 涨停(封板收盘): 收盘价触及涨停价且收盘=最高
    bars["zt"] = (bars["close"].values >= bars["lim"].values - 0.011) & \
                 (bars["close"].values >= bars["high"].values - 1e-6)
    bars["touch"] = bars["high"].values >= bars["lim"].values - 0.011
    bars["yizi"] = bars["low"].values >= bars["lim"].values - 0.011     # 一字板
    bars["bomb"] = bars["touch"].values & (~bars["zt"].values)          # 触板未封=炸板
    bars["dn"] = (bars["close"].values <= bars["preclose"].values * 0.9)  # 跌停(暴雷代理)

    log_("[因子] 连板数计算...")
    bars["lb"] = _consec_lb(bars["zt"].values.astype(np.int8),
                            bars["code"].values)
    bars["turnover"] = bars["volume"] / bars["outstanding_share"].replace(0, np.nan) * 100
    # 除权等异常收益置空, 避免污染
    bars.loc[bars["pct"].abs() > 0.35, "pct"] = np.nan

    # ---------- 市场级情绪(按交易日) ----------
    log_("[因子] 市场情绪(涨停家数/炸板率/连板高度/涨停溢价)...")
    byd = bars.groupby("date")
    mkt = pd.DataFrame({
        "zt_count": byd["zt"].sum(),
        "touch_cnt": byd["touch"].sum(),
        "bomb_cnt": byd["bomb"].sum(),
        "max_lb": byd["lb"].max(),
        "mkt_ret": byd["pct"].median(),
    })
    mkt["bomb_rate"] = mkt["bomb_cnt"] / (mkt["zt_count"] + mkt["bomb_cnt"]).replace(0, np.nan)
    mkt["bomb_rate"] = mkt["bomb_rate"].fillna(0.5)

    dates = mkt.index.tolist()
    # 昨日涨停股在今日的平均涨幅 = 涨停溢价
    zt_prev = {}
    zt_by_date = {d: set(g["code"][g["zt"]].values)
                  for d, g in bars.groupby("date")}
    ret_by_date = {d: dict(zip(g["code"], g["pct"])) for d, g in bars.groupby("date")}
    prem = {}
    for i, d in enumerate(dates):
        if i == 0:
            prem[d] = np.nan
            continue
        prev = zt_by_date.get(dates[i - 1], set())
        r = ret_by_date.get(d, {})
        vals = [r[c] for c in prev if c in r and not np.isnan(r[c])]
        prem[d] = float(np.mean(vals)) * 100 if vals else np.nan
    mkt["prev_premium"] = pd.Series(prem)
    mkt["prev_premium"] = mkt["prev_premium"].fillna(0.0)

    # ---------- 板块级(行业×日期) ----------
    log_("[因子] 板块强度(行业涨停家数/板块龙头)...")
    bars["ind"] = bars["code"].map(ind_map)
    gi = bars.groupby(["date", "ind"])
    plate = pd.DataFrame({
        "p_zt": gi["zt"].sum(),
        "p_maxlb": gi["lb"].max(),
    }).reset_index()
    plate_dict = {(r["date"], r["ind"]): (r["p_zt"], r["p_maxlb"])
                  for _, r in plate.iterrows()}

    # ---------- 逐股数组(供窗口因子) ----------
    log_("[因子] 构建逐股数组...")
    stock_arr = {}
    for c, gg in bars.groupby("code", sort=False):
        stock_arr[c] = dict(
            date=gg["date"].values,
            close=gg["close"].values.astype(float),
            high=gg["high"].values.astype(float),
            low=gg["low"].values.astype(float),
            vol=gg["volume"].values.astype(float),
            amt=gg["amount"].values.astype(float),
            turn=gg["turnover"].values.astype(float),
            os=float(gg["outstanding_share"].iloc[-1]) if len(gg) else np.nan,
        )

    # ---------- 候选样本: 非一字首板 ----------
    log_("[样本] 筛选非一字首板候选...")
    cand = bars[(bars["zt"]) & (~bars["yizi"]) & (bars["lb"] == 1) &
                (bars["pos"] >= MIN_LISTED_DAYS)].copy()
    crash_days = set(mkt.index[mkt["mkt_ret"] < CRASH_DROP])
    cand = cand[~cand["date"].isin(crash_days)]
    # 流通市值(剔除大盘权重股): 收盘 × 流通股本
    cand["float_mkt"] = cand["close"].values * cand["outstanding_share"].values
    cand = cand[cand["float_mkt"] <= MAX_FLOAT_MKT]
    log_("[样本] 候选首板 %d 条 (剔除一字/次新/暴跌日/权重股后)" % len(cand))

    cand = cand.sort_values(["code", "date"]).reset_index(drop=True)

    # ---------- 逐样本窗口因子 ----------
    log_("[因子] 计算窗口因子(相对换手/量比/价格位置/筹码压力/资金流)...")
    em_extra = _load_em_extra()
    rows = []
    cur_code, cur = None, None
    for i in range(len(cand)):
        r = cand.iloc[i]
        code = r["code"]
        if code != cur_code:
            cur_code = code
            cur = stock_arr.get(code)
        if cur is None:
            continue
        pos = np.searchsorted(cur["date"], r["date"])
        if pos >= len(cur["date"]) or cur["date"][pos] != r["date"]:
            continue
        if pos < 30:      # 需要 60 日回看, 不足则跳过(数据残缺)
            continue
        s = max(0, pos - 60)
        close_h = cur["close"][s:pos]
        vol_h = cur["vol"][s:pos]
        amt_h = cur["amt"][s:pos]
        turn_h = cur["turn"][s:pos]
        tp_h = (cur["high"][s:pos] + cur["low"][s:pos] + cur["close"][s:pos]) / 3.0

        # 相对换手: 当日换手 / 近20日均换手
        t20 = np.nanmean(turn_h[-20:]) if len(turn_h) >= 20 else np.nan
        rel_turn = (r["turnover"] / t20) if (t20 and t20 > 0) else np.nan
        # 量比: 当日量 / 近5日均量
        v5 = np.nanmean(vol_h[-5:]) if len(vol_h) >= 5 else np.nan
        vol_ratio = (r["volume"] / v5) if (v5 and v5 > 0) else np.nan
        # 价格位置: 近30日累计涨幅
        p30 = close_h[-30] if len(close_h) >= 30 else np.nan
        price_pos = (r["close"] / p30 - 1) * 100 if (p30 and p30 > 0) else np.nan
        # 上方筹码压力: 近60日成交额加权, 价格高于当日收盘的筹码占比(近端衰减加权)
        w = np.exp(-0.03 * np.arange(len(close_h) - 1, -1, -1))   # 越近权重越大
        w = w * np.nan_to_num(amt_h, nan=0.0)
        tot = w.sum()
        if tot > 0:
            above = w[(tp_h > r["close"])].sum() / tot
            near = w[(tp_h > r["close"] * 0.97) & (tp_h < r["close"] * 1.10)].sum() / tot
            chip_pressure = float(above * 100 + near * 50)
        else:
            chip_pressure = np.nan
        # 主力资金净流入(估计): 日内资金流 × 成交额 / 近20日均成交额
        rng = cur["high"][pos] - cur["low"][pos]
        if rng > 0:
            mf = ((r["close"] - r["low"]) - (r["high"] - r["close"])) / rng
        else:
            mf = 0.0
        a20 = np.nanmean(amt_h[-20:]) if len(amt_h) >= 20 else np.nan
        big_money = float(mf * r["amount"] / a20 * 100) if (a20 and a20 > 0) else np.nan

        d = r["date"]
        m = mkt.loc[d]
        ind = r["ind"]
        p_zt, p_maxlb = plate_dict.get((d, ind), (0, 0)) if isinstance(ind, str) else (0, 0)
        zt_n = max(int(m["zt_count"]), 1)
        ex = em_extra.get((d, code), {})

        rows.append(dict(
            date=d, code=code, name=_nm(code),
            industry=ind if isinstance(ind, str) else "",
            board=board_of(code),
            # --- 模块1 板块基因 ---
            plate_strength=float(p_zt) / zt_n * 100 if isinstance(ind, str) else np.nan,
            plate_has_leader=1.0 if (p_maxlb >= 2) else 0.0,
            plate_limit_cnt=float(p_zt),
            # --- 模块2 量价基因 ---
            rel_turn=rel_turn, volume_ratio=vol_ratio,
            bomb_times=ex.get("bomb_times", np.nan),
            price_pos=price_pos,
            # --- 模块3 筹码资金 ---
            chip_pressure=chip_pressure, big_money_net=big_money,
            # --- 模块4 市场情绪 ---
            market_emotion=float(m["zt_count"]) / 60.0 + float(m["max_lb"]) - \
                           float(m["bomb_rate"]) * 4.0,
            high_limit_premium=float(m["prev_premium"]),
            bomb_rate=float(m["bomb_rate"]) * 100,
            # --- 辅助 ---
            turnover=float(r["turnover"]), float_mkt=float(r["float_mkt"]),
            close=float(r["close"]), amount=float(r["amount"]),
            mkt_zt_count=int(m["zt_count"]), mkt_max_lb=int(m["max_lb"]),
            mkt_ret=float(m["mkt_ret"]) * 100 if not np.isnan(m["mkt_ret"]) else np.nan,
            seal_time_min=ex.get("seal_time_min", np.nan),
            seal_amt=ex.get("seal_amt", np.nan),
        ))
    df = pd.DataFrame(rows)
    log_("[因子] 因子计算完成: %d 条" % len(df))
    return df, mkt, bars


def _nm(code):
    return _NAME_MAP.get(code, "")


_NAME_MAP = {}
_EM_EXTRA = {}


def _load_em_extra():
    """东财涨停池富因子: (date,code) -> 炸板次数/封板时间/封单额"""
    global _EM_EXTRA
    if _EM_EXTRA:
        return _EM_EXTRA
    if not os.path.exists(ZT_POOL_CSV) or os.path.getsize(ZT_POOL_CSV) < 100:
        return {}
    try:
        p = pd.read_csv(ZT_POOL_CSV, dtype={"代码": str, "日期": str})
    except Exception:
        return {}
    out = {}
    col_bomb = "炸板次数" if "炸板次数" in p.columns else None
    col_time = "首次封板时间" if "首次封板时间" in p.columns else None
    col_amt = "封板资金" if "封板资金" in p.columns else None
    for _, r in p.iterrows():
        key = (str(r["日期"]), str(r["代码"]).zfill(6))
        rec = {}
        if col_bomb:
            try:
                rec["bomb_times"] = float(r[col_bomb])
            except Exception:
                pass
        if col_time:
            try:
                t = str(r[col_time])
                hh, mm = t.split(":")[0], t.split(":")[1]
                rec["seal_time_min"] = int(hh) * 60 + int(mm) - 570  # 相对9:30分钟
            except Exception:
                pass
        if col_amt:
            try:
                rec["seal_amt"] = float(str(r[col_amt]).replace("亿", "")) * 1e8
            except Exception:
                pass
        out[key] = rec
    _EM_EXTRA = out
    return out


# ================================================================ Stage 5: 标签+切分
def attach_label(df, bars):
    """标签 Y: 次日是否再次涨停(1进2); 同时计算次日开盘介入收益"""
    zt_set = {d: set(g["code"][g["zt"]].values) for d, g in bars.groupby("date")}
    oc_by_date = {d: dict(zip(g["code"], zip(g["open"], g["close"], g["high"], g["low"])))
                  for d, g in bars.groupby("date")}
    dates = sorted(zt_set.keys())
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    y, nxt_d, ret_oc, ret_cc, y_high = [], [], [], [], []
    for _, r in df.iterrows():
        nd = nxt.get(r["date"])
        if nd is None:
            y.append(np.nan); nxt_d.append(""); ret_oc.append(np.nan)
            ret_cc.append(np.nan); y_high.append(np.nan)
            continue
        nxt_d.append(nd)
        y.append(1 if r["code"] in zt_set.get(nd, set()) else 0)
        o = oc_by_date.get(nd, {}).get(r["code"])
        if o:
            op, cl, hi, lo = o
            ret_oc.append((cl / op - 1) * 100 if op > 0 else np.nan)   # 次日开盘买→收盘卖
            ret_cc.append((cl / r["close"] - 1) * 100)
            y_high.append((hi / op - 1) * 100 if op > 0 else np.nan)
        else:
            ret_oc.append(np.nan); ret_cc.append(np.nan); y_high.append(np.nan)
    df["next_date"] = nxt_d
    df["label_Y"] = y
    df["ret_next_oc"] = ret_oc
    df["ret_next_cc"] = ret_cc
    df["ret_next_high"] = y_high
    df = df[df["next_date"] != ""]
    df = df.dropna(subset=["label_Y"])
    df["label_Y"] = df["label_Y"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true", help="跳过下载, 仅用本地缓存重建")
    ap.add_argument("--end", default=dt.datetime.now().strftime("%Y%m%d"))
    args = ap.parse_args()

    log("=" * 78)
    log("龙妖首板基因识别器 · 数据集构建 v2 开始  (end=%s)" % args.end)
    log("=" * 78)

    global _NAME_MAP
    _NAME_MAP = stage0_name()
    ind_map = stage0_industry()
    stage1_em_pool(18)

    if args.skip_download:
        dfs = [pd.read_csv(p, dtype={"code": str}) for p in BARS_SHARDS if os.path.exists(p)]
        bars = pd.concat(dfs, ignore_index=True)
        bars["date"] = bars["date"].astype(str).str.replace("-", "", regex=False)
        for c in ("open", "high", "low", "close", "volume", "amount", "outstanding_share"):
            bars[c] = pd.to_numeric(bars[c], errors="coerce")
        bars = bars.dropna(subset=["close"])
        log("[数据源A] 本地缓存载入 %d 行" % len(bars))
    else:
        bars = stage2_bars(args.end)

    df, mkt, bars2 = build_features(bars, ind_map)
    bars = bars2
    df = attach_label(df, bars)

    # 剔除近5日出现过跌停的暴雷标的
    dn_recent = {}
    bd = bars.groupby("date")
    dn_series = {d: set(g["code"][g["dn"]].values) for d, g in bd}
    dts = sorted(dn_series.keys())
    for i, d in enumerate(dts):
        s = set()
        for j in range(max(0, i - 5), i):
            s |= dn_series[dts[j]]
        dn_recent[d] = s
    n1 = len(df)
    df = df[[c not in dn_recent.get(d, set())
             for c, d in zip(df["code"].values, df["date"].values)]]
    log("[清洗] 剔除近5日跌停(暴雷代理): %d -> %d" % (n1, len(df)))

    # 数据残缺剔除(任一核心因子为空)
    core = ["rel_turn", "volume_ratio", "price_pos", "chip_pressure",
            "big_money_net", "market_emotion", "bomb_rate"]
    n0 = len(df)
    df = df.dropna(subset=core)
    log("[清洗] 剔除因子残缺: %d -> %d" % (n0, len(df)))

    # 时序切分
    def split_of(d):
        if d <= SPLIT_TRAIN_END:
            return "train"
        if d <= SPLIT_VALID_END:
            return "valid"
        return "test"
    df["split"] = df["date"].map(split_of)
    df = df.sort_values(["date", "code"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log("=" * 78)
    log("[输出] %s" % OUT_CSV)
    log("样本总数 %d | 交易日 %d (%s ~ %s)" %
        (len(df), df["date"].nunique(), df["date"].min(), df["date"].max()))
    for sp in ("train", "valid", "test"):
        s = df[df["split"] == sp]
        if len(s):
            log("  %-5s %6d 条  %s ~ %s  晋级率 %.2f%%" %
                (sp, len(s), s["date"].min(), s["date"].max(),
                 s["label_Y"].mean() * 100))
    log("全体 1进2 晋级率: %.2f%%" % (df["label_Y"].mean() * 100))
    log("次日开盘介入平均收益: %.3f%%" % (df["ret_next_oc"].mean()))
    log("=" * 78)
    log("数据集构建完成 ✔")


if __name__ == "__main__":
    main()
