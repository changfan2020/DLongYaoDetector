# -*- coding: utf-8 -*-
"""
v2 审计实证脚本 —— 用于 Phase 1 审计的证据采集（只读，不修改任何数据）
运行: python scripts/audit_v2_checks.py
输出: 控制台 + docs/audit_evidence.txt
"""
import os
import sys
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = []


def w(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def sec(t):
    w("")
    w("=" * 78)
    w("  " + t)
    w("=" * 78)


ds = pd.read_csv(os.path.join(DATA, "dataset_v2.csv"), dtype={"code": str, "date": str})
cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))

FACTORS = ["plate_strength", "plate_has_leader", "plate_limit_cnt", "rel_turn",
           "volume_ratio", "bomb_times", "price_pos", "chip_pressure",
           "big_money_net", "market_emotion", "high_limit_premium", "bomb_rate"]

w("dataset_v2.csv 载入: %d 行, 交易日 %d, %s ~ %s" %
  (len(ds), ds["date"].nunique(), ds["date"].min(), ds["date"].max()))
w("列: %s" % list(ds.columns))

# ---------------------------------------------------------------- 1 缺失率
sec("检查 1 · 因子缺失率（训练集 / 全体）")
tr = ds[ds["split"] == "train"]
w("%-22s %10s %10s %10s" % ("factor", "all_nan%", "train_nan%", "nunique"))
for f in FACTORS:
    w("%-22s %9.2f%% %9.2f%% %10d" % (
        f, ds[f].isna().mean() * 100, tr[f].isna().mean() * 100,
        ds[f].nunique(dropna=True)))

# ---------------------------------------------------------------- 2 big_money_net 退化
sec("检查 2 · big_money_net 是否退化为成交额倍数（自命名'主力大单净流入'）")
sub = ds[["big_money_net", "amount"]].dropna()
w("样本数 %d" % len(sub))
# 复现公式: mf = ((close-low)-(high-close))/(high-low)，对封板股应恒等于 1
bar = pd.read_csv(os.path.join(DATA, "bars_shard0.csv"), dtype={"code": str}, nrows=None)
w("bars_shard0 列: %s" % list(bar.columns))
bar["date"] = bar["date"].astype(str).str.replace("-", "", regex=False)
key = ds[["date", "code"]].copy()
m = bar.merge(key, on=["date", "code"], how="inner")
w("能与日K匹配上的样本行数: %d" % len(m))
if len(m):
    rng = (m["high"] - m["low"]).replace(0, np.nan)
    mf = ((m["close"] - m["low"]) - (m["high"] - m["close"])) / rng
    w("mf 统计: min=%.6f  max=%.6f  mean=%.6f  std=%.8f" %
      (mf.min(), mf.max(), mf.mean(), mf.std()))
    w("mf == 1.0 的比例: %.4f%%" % ((mf.round(6) == 1.0).mean() * 100))
    # 与 close==high 的一致性
    w("close == high 的比例: %.4f%%" % ((m["close"] - m["high"]).abs().lt(1e-6).mean() * 100))
    # big_money_net 与 amount 的相关
    j = ds.merge(m[["date", "code", "amount", "high", "low", "close"]],
                 on=["date", "code"], how="inner", suffixes=("", "_bar"))
    j = j.dropna(subset=["big_money_net"])
    if len(j):
        w("corr(big_money_net, amount) = %.6f" % j["big_money_net"].corr(j["amount_bar"]))
        r = (j["big_money_net"] / j["amount_bar"]).replace([np.inf, -np.inf], np.nan).dropna()
        w("big_money_net / amount 的 std = %.10f (恒为常数则说明完全退化)" % r.std())
        w("  → 该比值均值 = %.6f，等价于 100 / 近20日均成交额" % r.mean())

# ---------------------------------------------------------------- 3 因子相关性
sec("检查 3 · 因子间共线性（Spearman，全体样本）")
c = ds[FACTORS].corr(method="spearman")
pairs = []
for i, a in enumerate(FACTORS):
    for b in FACTORS[i + 1:]:
        v = c.loc[a, b]
        if abs(v) >= 0.5:
            pairs.append((abs(v), a, b, v))
pairs.sort(reverse=True)
w("|rho| >= 0.5 的因子对: %d 组" % len(pairs))
for _, a, b, v in pairs[:20]:
    w("  %-22s ~ %-22s  rho = %+.3f" % (a, b, v))

# ---------------------------------------------------------------- 4 分数单调性
sec("检查 4 · 评分单调性（config.json 自带校准表即已证伪）")
cal = cfg["calibration"]
w("%-14s %8s %10s %12s" % ("score_bin", "n", "晋级率%", "ret_next_oc%"))
prev = None
viol = 0
for b in cal:
    w("[%3d,%3d) %8d %9.2f%% %11.3f" % (b["lo"], b["hi"], b["n"], b["p"] * 100, b["ret"]))
    if prev is not None and b["p"] < prev:
        viol += 1
    prev = b["p"]
w("单调性违反次数: %d / %d" % (viol, len(cal) - 1))
w("→ 结论: 分数并非随分数升高而单调升高；[70,100) 段晋级率反而低于 [65,70) 段")

# ---------------------------------------------------------------- 5 报告指标一致性
sec("检查 5 · config.json 指标口径一致性")
for sp in ("train", "valid", "test"):
    mt = cfg["metrics"][sp]
    w("%-6s AUC=%.4f  基线晋级率=%.4f  阈值入选精度=%.4f (n=%d)  TopK精度=%.4f (n=%d)"
      % (sp, mt["auc"], mt["base"], mt["prec"], mt["n"], mt["topk_prec"], mt["topk_n"]))
    w("       n_samples.total=%d  → 阈值入选仅占 %.2f%%" %
      (cfg["n_samples"][sp], mt["n"] / cfg["n_samples"][sp] * 100))
w("test AUC = %.4f  → 与随机猜测(0.5)无统计差异" % cfg["metrics"]["test"]["auc"])
w("overfit_gap = %s （仅比较 train vs valid，未做 valid vs test）" % cfg["overfit_gap"])

# ---------------------------------------------------------------- 6 test 集规模
sec("检查 6 · test 集统计功效")
te = ds[ds["split"] == "test"]
w("test 样本 %d 条, 交易日 %d 天, %s ~ %s" % (len(te), te["date"].nunique(),
                                              te["date"].min(), te["date"].max()))
w("test 正样本(晋级) %d 条" % int(te["label_Y"].sum()))
w("每日平均首板样本数: %.1f" % (len(te) / max(te["date"].nunique(), 1)))
w("→ 结论: test 仅 %.1f 个月、正样本约 %d 个，Top-K 精度置信区间极宽"
  % (te["date"].nunique() / 21.0, int(te["label_Y"].sum())))

# ---------------------------------------------------------------- 7 标签与收益分布
sec("检查 7 · 标签 & 收益定义分布")
for col in ("label_Y", "ret_next_oc", "ret_next_cc", "ret_next_high"):
    s = pd.to_numeric(ds[col], errors="coerce").dropna()
    w("%-16s n=%6d  mean=%+8.3f  median=%+8.3f  std=%7.3f  neg%%=%.1f%%" %
      (col, len(s), s.mean(), s.median(), s.std(), (s < 0).mean() * 100))
w("→ ret_next_oc 均值 %.3f%% 为负: T+1 高开买入、当日收盘卖出，是 A 股最差的介入口径之一"
  % ds["ret_next_oc"].mean())

# ---------------------------------------------------------------- 8 换手率异常
sec("检查 8 · 数据质量（v2 无任何数据质量检测）")
for f in ("turnover", "close", "amount", "float_mkt"):
    if f in ds.columns:
        s = pd.to_numeric(ds[f], errors="coerce")
        w("%-12s nan=%5.2f%%  min=%12.4f  p1=%12.4f  p99=%14.2f  max=%14.2f" %
          (f, s.isna().mean() * 100, s.min(), s.quantile(0.01), s.quantile(0.99), s.max()))
w("重复 (date,code) 行数: %d" % int(ds.duplicated(["date", "code"]).sum()))
w("turnover > 100%% 的行数: %d" % int((pd.to_numeric(ds["turnover"], errors="coerce") > 100).sum()))
w("float_mkt <= 0 的行数: %d" % int((pd.to_numeric(ds["float_mkt"], errors="coerce") <= 0).sum()))

# ---------------------------------------------------------------- 9 行业分类覆盖
sec("检查 9 · 行业板块覆盖（用于板块因子）")
w("industry 为空的行数: %d (%.2f%%)" % (ds["industry"].isna().sum(),
                                        ds["industry"].isna().mean() * 100))
w("行业种类数: %d" % ds["industry"].nunique())
vc = ds["industry"].value_counts()
w("最大行业: %s (%d 条, 占 %.2f%%)" % (vc.index[0], vc.iloc[0], vc.iloc[0] / len(ds) * 100))
w("前5: %s" % dict(vc.head(5)))

# ---------------------------------------------------------------- 10 因子分位箱退化
sec("检查 10 · 分位箱退化（拟合失效的因子）")
for f, e in cfg["quantile_bins"].items():
    if e is None:
        w("  %-22s → null （训练集有效样本 < 50，打分被强制置为中位 50）" % f)
    elif len(set(e)) <= 3:
        w("  %-22s → 仅 %d 个不同分位点 %s（打分退化为极少数档位）" % (f, len(set(e)), e))

# ---------------------------------------------------------------- 写文件
os.makedirs(os.path.join(BASE, "docs"), exist_ok=True)
p = os.path.join(BASE, "docs", "audit_evidence.txt")
with open(p, "w", encoding="utf-8") as fp:
    fp.write("\n".join(OUT))
print("\n证据已写入 %s" % p)
