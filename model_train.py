# -*- coding: utf-8 -*-
"""
================================================================================
 龙妖首板基因识别器 —— 模型训练脚本 (v2)
================================================================================
 模型: 多因子加权求和打分模型
     Score = w1*S_板块 + w2*S_量价 + w3*S_筹码 + w4*S_情绪      (各模块 0~100)
     S_模块 = Σ(子权重 * 子因子分) / Σ子权重

 权重范围(需求文档规定): w1 板块 10~40 | w2 量价 10~40 | w3 筹码 5~25 | w4 情绪 5~20
 搜索方式: 四模块权重全网格遍历 -> 子因子权重坐标下降细化
 选择标准: 验证集效果最优(不挑训练集最高的), 且 训练/验证 精确率差值 <= 15% (过拟合校验)

 输出指标: Precision / Recall / AUC / 次日盈亏 / 最大回撤 / 校准分箱 / 过拟合差值
 可选进阶: LightGBM 二分类对比测试(若环境可用)

 ★ 严禁未来函数: 因子分位箱仅用训练集拟合; 时序切分 train/valid/test 不随机打乱。
 ★ 正负样本极度不平衡(晋级率约15%), 核心看 Precision 与盈亏, 不看准确率。
================================================================================
"""
import os
import sys
import json
import time
import itertools
import datetime as dt

import numpy as np
import pandas as pd

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DS = os.path.join(DATA_DIR, "dataset_v2.csv")
CONFIG = os.path.join(BASE_DIR, "config.json")

LOG_PATH = os.path.join(LOG_DIR, "model_train.log")


def log(msg):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- 因子定义
FACTORS = {
    # 模块1 板块基因
    "plate_strength":   ("plate", +1),
    "plate_has_leader": ("plate", +1),
    "plate_limit_cnt":  ("plate", +1),
    # 模块2 量价基因
    "rel_turn":         ("price", +1),
    "volume_ratio":     ("price", +1),
    "bomb_times":       ("price", -1),
    "price_pos":        ("price", -1),
    # 模块3 筹码与资金基因
    "chip_pressure":    ("chip",  -1),
    "big_money_net":    ("chip",  +1),
    # 模块4 全局市场情绪基因
    "market_emotion":   ("emo",   +1),
    "high_limit_premium": ("emo", +1),
    "bomb_rate":        ("emo",   -1),
}
MODULES = ["plate", "price", "chip", "emo"]
MOD_NAME = {"plate": "板块基因", "price": "量价基因", "chip": "筹码资金基因", "emo": "市场情绪基因"}

SELECT_HI = 70.0     # 总分 >=70 优质首板, 纳入次日开盘观察池
SELECT_LO = 55.0     # 55~69 备选观察标的; <55 剔除
N_BINS = 20          # 因子分位箱数
TARGET_SD = 15.0     # 分数拉伸目标标准差
# 说明: 12 个分位因子加权平均后分数会向 50 收敛(标准差仅约 8),
#       直接套用 70 分门槛几乎不可达, 故做仿射拉伸使分布重新展开。
TOPK = 3             # 每日 Top-K 评估(实盘最关心的口径)


# ---------------------------------------------------------------- 情绪周期(4档)
def emotion_cycle(row):
    """依据 T 日全市场数据判定情绪周期: 返回 (档位, 折扣系数)
    主升 ×1.0 | 震荡 ×0.8 | 退潮 ×0.5(门槛提至80) | 冰点 暂停筛选
    """
    zt = row.get("mkt_zt_count", np.nan)
    lb = row.get("mkt_max_lb", np.nan)
    br = row.get("bomb_rate", np.nan)
    if np.isnan(zt):
        return "震荡", 0.8
    zt = float(zt); lb = float(lb) if not np.isnan(lb) else 0
    br = float(br) if not np.isnan(br) else 35.0
    if zt < 20 or (lb <= 2 and br > 50):
        return "冰点", 0.0
    if br > 45 or lb <= 3:
        return "退潮", 0.5
    if lb >= 6 and zt >= 60 and br < 30:
        return "主升", 1.0
    return "震荡", 0.8


def cycle_of_df(df):
    cs = [emotion_cycle(r) for _, r in df.iterrows()] if len(df) < 20000 else \
         [_cyc(z, l, b) for z, l, b in zip(df["mkt_zt_count"].values,
                                           df["mkt_max_lb"].values,
                                           df["bomb_rate"].values)]
    df["_cycle"] = [c[0] for c in cs]
    df["_disc"] = [c[1] for c in cs]
    return df


def _cyc(zt, lb, br):
    try:
        if np.isnan(zt):
            return "震荡", 0.8
    except Exception:
        return "震荡", 0.8
    zt = float(zt); lb = 0 if np.isnan(lb) else float(lb)
    br = 35.0 if np.isnan(br) else float(br)
    if zt < 20 or (lb <= 2 and br > 50):
        return "冰点", 0.0
    if br > 45 or lb <= 3:
        return "退潮", 0.5
    if lb >= 6 and zt >= 60 and br < 30:
        return "主升", 1.0
    return "震荡", 0.8


# ---------------------------------------------------------------- 分位箱(仅用训练集)
def fit_bins(train):
    bins = {}
    for f in FACTORS:
        v = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = v.dropna()
        if len(v) < 50:
            bins[f] = None
            continue
        qs = np.unique(np.nanpercentile(v, np.linspace(0, 100, N_BINS + 1)))
        bins[f] = [float(x) for x in qs]
    return bins


def score_factor(series, edges, direction):
    """0~100 分位打分, direction=-1 表示越小越好"""
    v = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if edges is None:
        s = pd.Series(np.full(len(v), 50.0), index=v.index)
    else:
        arr = v.values.astype(float)
        pct = np.searchsorted(np.array(edges), arr, side="right") - 1
        pct = np.clip(pct, 0, N_BINS - 1) / (N_BINS - 1.0)
        pct = np.where(np.isnan(arr), np.nan, pct)
        s = pd.Series(pct * 100.0, index=v.index)
    if direction < 0:
        s = 100.0 - s
    return s.fillna(50.0)     # 缺失因子给中性分 50


def compute_scores(df, bins, mw, sw):
    """mw: {module: weight}, sw: {factor: sub-weight} -> 返回 (总分, 各模块分DataFrame)"""
    S = {}
    for m in MODULES:
        fs = [f for f, (mm, _) in FACTORS.items() if mm == m]
        num = pd.Series(0.0, index=df.index)
        den = 0.0
        for f in fs:
            w = float(sw.get(f, 1.0))
            if w <= 0:
                continue
            num = num + w * score_factor(df[f], bins.get(f), FACTORS[f][1])
            den += w
        S[m] = num / den if den > 0 else pd.Series(50.0, index=df.index)
    tot = sum(float(mw[m]) * S[m] for m in MODULES)
    tot = tot / max(sum(float(mw[m]) for m in MODULES), 1e-9)
    return tot, pd.DataFrame(S)


# ---------------------------------------------------------------- 评估
def fit_stretch(tot):
    """在训练集上拟合仿射拉伸参数, 使总分分布重新展开"""
    t = np.asarray(tot, dtype=float)
    t = t[~np.isnan(t)]
    mu = float(t.mean())
    sd = float(t.std())
    k = float(TARGET_SD / sd) if sd > 1e-9 else 1.0
    return mu, float(min(k, 6.0))


def apply_stretch(tot, mu, k):
    return np.clip(50.0 + (np.asarray(tot, dtype=float) - mu) * k, 0.0, 100.0)


def daily_topk(df, score, K=TOPK):
    """每日按分数取 Top-K(冰点周期不出手), 返回 (精度, 样本数, 均收益, 基线)"""
    d = df.copy()
    d["_s"] = np.asarray(score, dtype=float)
    d = d[d["_disc"] > 0]                       # 冰点周期暂停筛选
    if len(d) == 0:
        return None
    rk = d.groupby("date")["_s"].rank(ascending=False, method="first")
    s = d[rk <= K]
    if len(s) == 0:
        return None
    return dict(n=int(len(s)), prec=float(s["label_Y"].mean()),
                ret=float(np.nanmean(s["ret_next_oc"])) if "ret_next_oc" in s else np.nan,
                base=float(d["label_Y"].mean()),
                mdd=max_drawdown(s))


def evaluate(df, score, disc=None, thr=SELECT_HI, need_n=100):
    d = df.copy()
    d["_s"] = score.values if hasattr(score, "values") else np.asarray(score)
    if disc is not None:
        d["_sf"] = d["_s"] * disc
        d = d[d["_disc"] > 0]                       # 冰点周期暂停筛选
        if len(d) == 0:
            return None
        thr_eff = np.where(d["_cycle"].values == "退潮", 80.0, thr)
        sel = d["_sf"].values >= thr_eff
    else:
        d["_sf"] = d["_s"]
        sel = d["_sf"].values >= thr
    n = int(sel.sum())
    if n < need_n:
        # 入选过少则退化为"每日 Top15%"口径, 保证统计有效
        d2 = d.copy()
        rk = d2.groupby("date")["_sf"].rank(pct=True, ascending=False)
        sel = (rk <= 0.15).values
        n = int(sel.sum())
        if n < 20:
            return None
    s = d[sel]
    y = s["label_Y"].values
    prec = float(y.mean())
    recall = float(y.sum()) / max(float(d["label_Y"].sum()), 1)
    ret = float(np.nanmean(s["ret_next_oc"].values)) if "ret_next_oc" in s else np.nan
    # AUC
    try:
        yy = d["label_Y"].values
        ss = d["_sf"].values
        order = np.argsort(ss)
        ranks = np.empty(len(ss), float)
        ranks[order] = np.arange(1, len(ss) + 1)
        # 处理并列
        srt = pd.Series(ss).rank(method="average").values
        n1 = yy.sum(); n0 = len(yy) - n1
        auc = (srt[yy == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    except Exception:
        auc = np.nan
    # 最大回撤(按日期累计收益)
    try:
        t = s[["date", "ret_next_oc"]].dropna().groupby("date")["ret_next_oc"].mean()
        eq = (1 + t.values / 100).cumprod()
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
        cum = float((eq[-1] - 1) * 100)
    except Exception:
        mdd, cum = np.nan, np.nan
    return dict(n=n, prec=prec, recall=recall, auc=float(auc),
                ret=ret, mdd=mdd, cum=cum, base=float(d["label_Y"].mean()))


def max_drawdown(d):
    try:
        t = d[["date", "ret_next_oc"]].dropna().groupby("date")["ret_next_oc"].mean()
        eq = (1 + t.values / 100).cumprod()
        return float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    except Exception:
        return np.nan


# ---------------------------------------------------------------- 主流程
def main():
    log("=" * 78)
    log("龙妖首板基因识别器 · 模型训练 v2 开始")
    log("=" * 78)
    if not os.path.exists(DS):
        log("数据集不存在: %s —— 请先运行 build_dataset.py" % DS)
        sys.exit(1)
    df = pd.read_csv(DS, dtype={"code": str, "date": str})
    df["date"] = df["date"].astype(str)
    log("载入样本 %d 条, 交易日 %d (%s ~ %s)" %
        (len(df), df["date"].nunique(), df["date"].min(), df["date"].max()))
    df = cycle_of_df(df)
    log("情绪周期分布: " + str(df["_cycle"].value_counts().to_dict()))

    tr = df[(df["split"] == "train") & (df["date"] >= "20230101")].copy()
    va = df[df["split"] == "valid"].copy()
    te = df[df["split"] == "test"].copy()
    log("切分: train=%d valid=%d test=%d | 晋级率 %.2f%% / %.2f%% / %.2f%%" %
        (len(tr), len(va), len(te), tr["label_Y"].mean() * 100,
         va["label_Y"].mean() * 100,
         te["label_Y"].mean() * 100 if len(te) else float("nan")))

    bins = fit_bins(tr)
    log("因子分位箱拟合完成(仅用训练集)")

    sw0 = {f: 1.0 for f in FACTORS}

    def eval_combo(mw, sw):
        """评估一组权重: (obj, train指标, valid指标, mu, k); 触发过拟合校验返回 None"""
        s_tr, _ = compute_scores(tr, bins, mw, sw)
        s_va, _ = compute_scores(va, bins, mw, sw)
        mu, k = fit_stretch(s_tr)                    # 分数尺度仅在训练集拟合
        st_tr = apply_stretch(s_tr, mu, k)
        st_va = apply_stretch(s_va, mu, k)
        k_tr = daily_topk(tr, st_tr)
        k_va = daily_topk(va, st_va)
        if k_tr is None or k_va is None:
            return None
        if abs(k_tr["prec"] - k_va["prec"]) > 0.15:  # 过拟合校验: 差值>15% 舍弃
            return None
        a_tr = _auc(tr["label_Y"].values, st_tr)
        a_va = _auc(va["label_Y"].values, st_va)
        rv = 0.0 if np.isnan(k_va["ret"]) else k_va["ret"]
        obj = k_va["prec"] * 0.5 + a_va * 0.3 + (rv / 100.0) * 0.2
        return (obj,
                dict(prec=k_tr["prec"], n=k_tr["n"], ret=k_tr["ret"], auc=a_tr),
                dict(prec=k_va["prec"], n=k_va["n"], ret=k_va["ret"], auc=a_va),
                mu, k)

    # ---------------- 阶段1: 四模块权重网格 ----------------
    grid_w1 = list(range(10, 41, 5))
    grid_w2 = list(range(10, 41, 5))
    grid_w3 = list(range(5, 26, 5))
    grid_w4 = list(range(5, 21, 5))
    combos = list(itertools.product(grid_w1, grid_w2, grid_w3, grid_w4))
    log("阶段1 四模块网格搜索: %d 组权重 (目标: 验证集每日 Top%d 晋级率)" % (len(combos), TOPK))

    best, best_obj = None, -1e9
    t0 = time.time()
    for i, (a, b, c, d) in enumerate(combos):
        r = eval_combo({"plate": a, "price": b, "chip": c, "emo": d}, sw0)
        if r is None:
            continue
        obj, rt, rv, mu, k = r
        if obj > best_obj:
            best_obj, best = obj, ({"plate": a, "price": b, "chip": c, "emo": d}, rt, rv)
        if (i + 1) % 200 == 0:
            log("  %d/%d (%.0fs) 当前最优 valid Top%d 晋级率=%.4f" %
                (i + 1, len(combos), time.time() - t0, TOPK,
                 best[2]["prec"] if best else float("nan")))
    if best is None:
        log("警告: 全部组合均触发过拟合校验, 放宽为仅看验证集")
        for a, b, c, d in combos:
            mw = {"plate": a, "price": b, "chip": c, "emo": d}
            s_tr, _ = compute_scores(tr, bins, mw, sw0)
            s_va, _ = compute_scores(va, bins, mw, sw0)
            mu, k = fit_stretch(s_tr)
            kv = daily_topk(va, apply_stretch(s_va, mu, k))
            if kv is None:
                continue
            if kv["prec"] > best_obj:
                kt = daily_topk(tr, apply_stretch(s_tr, mu, k))
                best_obj, best = kv["prec"], (mw, kt or kv, kv)
    mw, r_tr, r_va = best
    log("阶段1 最优模块权重: 板块%.0f 量价%.0f 筹码%.0f 情绪%.0f" %
        (mw["plate"], mw["price"], mw["chip"], mw["emo"]))
    log("   Top%d 晋级率 train=%.4f (n=%d) | valid=%.4f (n=%d) | 差值=%.4f" %
        (TOPK, r_tr["prec"], r_tr["n"], r_va["prec"], r_va["n"],
         abs(r_tr["prec"] - r_va["prec"])))

    # ---------------- 阶段2: 子因子权重坐标下降 ----------------
    log("阶段2 子因子权重坐标下降细化...")
    sw = dict(sw0)
    grid_sw = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_obj2 = best_obj
    for rd in range(3):
        improved = False
        for f in FACTORS:
            cur = sw[f]
            for g in grid_sw:
                if g == cur:
                    continue
                sw[f] = g
                r = eval_combo(mw, sw)
                if r is None:
                    continue
                if r[0] > best_obj2 + 1e-6:
                    best_obj2, cur, improved = r[0], g, True
            sw[f] = cur
        log("  round%d 完成 obj=%.5f" % (rd + 1, best_obj2))
        if not improved:
            break

    # ---------------- 最终评估 ----------------
    log("-" * 78)
    s_tr_raw, _ = compute_scores(tr, bins, mw, sw)
    mu, k = fit_stretch(s_tr_raw)
    log("分数尺度: 中心 %.2f, 拉伸 %.2f 倍 (目标标准差 %.0f)" % (mu, k, TARGET_SD))
    res = {}
    for name, part in (("train", tr), ("valid", va), ("test", te)):
        if len(part) == 0:
            continue
        s, _ = compute_scores(part, bins, mw, sw)
        st = apply_stretch(s, mu, k)
        r = evaluate(part, st, disc=part["_disc"].values)
        kt = daily_topk(part, st)
        if r is None and kt is None:
            continue
        if r is not None:
            r["mdd"] = max_drawdown(part.loc[assign_sel(part, st, part["_disc"].values)])
        auc = _auc(part["label_Y"].values, st)
        rec = dict(auc=float(auc),
                   n=(r["n"] if r else 0), prec=(r["prec"] if r else None),
                   base=(r["base"] if r else (kt["base"] if kt else None)),
                   recall=(r["recall"] if r else None),
                   ret=(r["ret"] if r else None), mdd=(r["mdd"] if r else None),
                   topk_n=(kt["n"] if kt else 0), topk_prec=(kt["prec"] if kt else None),
                   topk_ret=(kt["ret"] if kt else None),
                   topk_mdd=(kt["mdd"] if kt else None))
        res[name] = rec
        if kt:
            log("%-5s 每日Top%d: n=%4d 晋级率 %5.2f%% (基线 %5.2f%%) | 次日均收益 %+.3f%% | 最大回撤 %.2f%%"
                % (name, TOPK, kt["n"], kt["prec"] * 100, kt["base"] * 100,
                   kt["ret"], kt["mdd"]))
        if r:
            log("%-5s 门槛>=%.0f: n=%4d 晋级率 %5.2f%% (基线 %5.2f%%, 提升 %+.1f%%) | 召回 %.2f%% | AUC %.4f | 次日均收益 %+.3f%% | 最大回撤 %.2f%%"
                % (name, SELECT_HI, r["n"], r["prec"] * 100, r["base"] * 100,
                   (r["prec"] / max(r["base"], 1e-9) - 1) * 100, r["recall"] * 100,
                   auc, r["ret"], r["mdd"]))
    gap = abs(res["train"]["topk_prec"] - res["valid"]["topk_prec"]) \
        if ("valid" in res and res["train"].get("topk_prec") and res["valid"].get("topk_prec")) \
        else np.nan
    log("过拟合校验 |训练-验证| Top%d 晋级率差值 = %.4f (%s)" %
        (TOPK, gap, "通过" if gap <= 0.15 else "过拟合"))

    # 校准分箱(用 train+valid)
    cal_df = pd.concat([tr, va], ignore_index=True)
    s_cal, _ = compute_scores(cal_df, bins, mw, sw)
    cal_df["_s"] = apply_stretch(s_cal.values, mu, k)
    cal_df["_sf"] = cal_df["_s"] * cal_df["_disc"].values
    cal_df = cal_df[cal_df["_disc"] > 0]
    bins_cal = []
    edges = [0, 40, 50, 55, 60, 65, 70, 75, 80, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cal_df["_sf"] >= lo) & (cal_df["_sf"] < hi)
        if m.sum() >= 30:
            bins_cal.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                                 p=float(cal_df.loc[m, "label_Y"].mean()),
                                 ret=float(np.nanmean(cal_df.loc[m, "ret_next_oc"]))))
    log("校准分箱(分档 -> 实际晋级率):")
    for b in bins_cal:
        log("  [%3d,%3d) n=%5d  晋级率 %5.2f%%  次日均收益 %+.3f%%"
            % (b["lo"], b["hi"], b["n"], b["p"] * 100, b["ret"]))

    # ---------------- 可选: LightGBM 对比 (需 pip install lightgbm) ----------------
    lg = None
    if os.environ.get("DRAGON_LGBM") == "1":
        try:
            import lightgbm as lgb
            lg = True
        except Exception:
            try:
                os.system("\"%s\" -m pip install --quiet --disable-pip-version-check lightgbm"
                          % sys.executable)
                import lightgbm as lgb
                lg = True
            except Exception as e:
                log("[可选] LightGBM 不可用, 跳过对比测试: %r" % e)
    else:
        log("[可选] LightGBM 对比测试已跳过(设置环境变量 DRAGON_LGBM=1 可启用)")
    lgbm_metrics = None
    if lg:
        try:
            import lightgbm as lgb
            feats = list(FACTORS.keys())
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, verbose=-1)
            m.fit(tr[feats].astype(float), tr["label_Y"])
            pv = m.predict_proba(va[feats].astype(float))[:, 1]
            rk = pd.Series(pv).rank(pct=True)
            top = rk >= 0.85
            lgbm_metrics = dict(
                auc=float(_auc(va["label_Y"].values, pv)),
                prec_top15=float(va["label_Y"].values[top.values].mean()),
                base=float(va["label_Y"].mean()))
            log("[可选] LightGBM 对比: AUC=%.4f, Top15%%晋级率=%.2f%% (基线 %.2f%%)"
                % (lgbm_metrics["auc"], lgbm_metrics["prec_top15"] * 100,
                   lgbm_metrics["base"] * 100))
        except Exception as e:
            log("[可选] LightGBM 训练失败: %r" % e)

    # ---------------- 输出 config.json ----------------
    cfg = dict(
        version="2.0",
        updated=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        strategy=dict(name="龙妖战法", task="首板识别+次日开盘介入",
                      target="首板次日1进2晋级概率"),
        module_weights={"板块基因": float(mw["plate"]), "量价基因": float(mw["price"]),
                        "筹码资金基因": float(mw["chip"]), "市场情绪基因": float(mw["emo"])},
        sub_weights={f: float(sw[f]) for f in FACTORS},
        factor_dirs={f: FACTORS[f][1] for f in FACTORS},
        factor_module={f: FACTORS[f][0] for f in FACTORS},
        score_center=float(mu), score_scale=float(k), target_sd=float(TARGET_SD),
        topk=int(TOPK),
        quantile_bins=bins,
        thresholds=dict(select=SELECT_HI,备选=SELECT_LO, 退潮门槛=80.0),
        emotion_cycle=dict(主升=1.0, 震荡=0.8, 退潮=0.5, 冰点=0.0),
        calibration=bins_cal,
        metrics={k: {kk: _f(vv) for kk, vv in v.items()} for k, v in res.items()},
        overfit_gap=None if (isinstance(gap, float) and np.isnan(gap)) else float(gap),
        lightgbm=lgbm_metrics,
        data_ranges=dict(
            train=[tr["date"].min(), tr["date"].max()] if len(tr) else None,
            valid=[va["date"].min(), va["date"].max()] if len(va) else None,
            test=[te["date"].min(), te["date"].max()] if len(te) else None),
        n_samples=dict(train=int(len(tr)), valid=int(len(va)), test=int(len(te))),
        notes=("akshare 为网页爬虫数据源, 盘中存在延迟、精细封单数据有限; "
               "本系统仅用于策略原型验证, 不构成实盘交易依据。"),
    )
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log("最优权重已写入 %s" % CONFIG)
    log("=" * 78)
    log("模型训练完成 ✔")


def assign_sel(df, score, disc, thr=SELECT_HI):
    d = df.copy()
    d["_sf"] = np.asarray(score) * np.asarray(disc)
    d = d[np.asarray(disc) > 0]
    thr_eff = np.where(d["_cycle"].values == "退潮", 80.0, thr)
    sel = d["_sf"].values >= thr_eff
    if sel.sum() < 100:
        rk = d.groupby("date")["_sf"].rank(pct=True, ascending=False)
        d = d[(rk <= 0.15).values]
    else:
        d = d[sel]
    return d.index


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return x


def _auc(y, s):
    srt = pd.Series(s).rank(method="average").values
    n1 = y.sum(); n0 = len(y) - n1
    return (srt[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


if __name__ == "__main__":
    main()
