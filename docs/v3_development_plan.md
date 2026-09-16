# DLongYaoDetector v3 开发计划

> 版本：v3.0-plan（2026-09-17）
> 定位变更：**从「预测首板次日晋级」→「A 股首板研究、次日竞价决策与策略验证工作台」**
> 上游依据：`DLongYaoDetector v3 开发计划提示词.md`
> 配套证据：`docs/v2_audit.md`、`docs/audit_evidence.txt`、`scripts/audit_v2_checks.py`

---

## 0. 一句话结论

v2 的问题不是「模型不够强」，而是**四层地基同时是坏的**：交易口径不可实现（`ret_next_oc` + 涨停开盘仍假设可成交）、
因子层有 2 个因子在数学上完全退化（`big_money_net`）、数据层无复权与质量校验、
评分层的「70 分」是人为拉伸出来的虚数。**在这些修好之前，任何模型迭代都是在噪声上做优化。**

本计划的排序原则：**先让回测可信 → 再谈因子与模型 → 最后才是竞价与 UI。**

---

## 1. 当前架构分析（v2 现状）

### 1.1 物理结构

```
DLongYaoDetector/
├── build_dataset.py     27KB  674 行  数据下载 + 12 因子计算 + 清洗 + 切分 + 打标签（单体）
├── model_train.py       24KB  553 行  分位箱 + 980 组网格 + 坐标下降 + 评估 + 校准 + 写 config
├── dragon_analyzer.py   28KB  614 行  实时/离线打分 + 情绪周期 + 一票否决 + 竞价 precheck + Flask 依赖
├── flask_server.py      15KB  373 行  8 个 API + 内置调度线程 + 收盘入库
├── index.html           18KB          纯前端看板（SVG 自绘，无 CDN）
├── config.json          10KB          权重 / 分位箱 / 拉伸参数 / 校准表 / 指标
├── autostart_server.py / run_server.bat
├── data/  (307MB)       bars_shard0-2.csv + 映射表 + dataset_v2.csv
└── logs/
```

### 1.2 数据流

```
新浪日K(不复权, 20221001起, 4414只) ─┐
新浪行业映射(84板块, 静态快照)      ├→ build_features() ─→ 首板候选
东财涨停池(仅近15交易日)           ─┘        │
                                             ├→ attach_label()  → label_Y / ret_next_oc / ret_next_cc / ret_next_high
                                             ├→ 清洗(ST/次新/一字/暴跌日/权重股/暴雷/残缺)
                                             └→ dataset_v2.csv (34508 条 / 917 交易日)
                                                        │
                     model_train.py ──→ config.json ────┤
                                                        └→ dragon_analyzer.py ─→ flask_server.py ─→ index.html
```

### 1.3 关键量化事实（实测，非推测）

| 项 | 数值 | 备注 |
|---|---|---|
| 样本量 | 34,508 条 / 917 交易日 / 20221122–20260910 | 首板且非一字 |
| 训练集 | 25,538（20230101–20251231） | |
| 验证集 | 5,647（20260105–20260630） | |
| 测试集 | 2,520（20260701–20260910，**仅 52 个交易日**） | 正样本 352 个 |
| 基线晋级率 | 训练 17.17% / 验证 13.99% / 测试 14.08% | |
| **测试集 AUC** | **0.4978** | 与随机猜（0.5）无差异 |
| 分数单调性 | **违反 2 次 / 8 箱** | [65,70)=25.14% → [70,75)=16.32% |

**这是整个 v3 立项的根本原因：系统目前没有可证明的预测能力，且「70 分以上更优」被自己的校准表证伪。**

---

## 2. 已发现的问题（摘要，详见 `docs/v2_audit.md`）

按严重度分三级：

### A 级（致命 · 结论级错误）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| A1 | `big_money_net` 因子**数学上完全退化** | `mf ≡ 1.000000`，std = 0，100% 样本 | 「筹码资金基因」模块实为成交额倍数，与 `rel_turn` ρ=**0.988**。四大基因有两个是同一个东西 |
| A2 | `bomb_times` 因子训练集 **100% 缺失** | 分位箱 = `null`，打分恒为中性 50 | 占权重但零信息量；实盘有值 → 训练/推理分布不一致 |
| A3 | `ret_next_oc` 不是可实现收益 | 均值 **−0.296%**，胜率 42.6% | 且假设 T+1 开盘必能成交——**7.33% 的样本次日以涨停价开盘，物理上买不到** |
| A4 | 分数「70 分」是**人为拉伸**产物 | `score_center=34.24 / score_scale=1.75`，把 sd 强行拉到 15 | 阈值不是统计结果，是仿射变换的副产品；且入选率在 train/valid/test 间从 4.29% 漂到 13.77%，**阈值不可移植** |
| A5 | 校准表用 **train+valid** 拟合，`prob_of()` 把 in-sample 概率当预测概率输出给前端 | [65,70) 显示 25.1%，测试集实际 14.1% | 对外展示的「晋级率」系统性乐观 |
| A6 | 验证集**双重使用**：既用来选模型（`model_train.py:346-358`），又用来做最终报告与校准 | 代码可见 | 报告指标不可信 |
| A7 | 盘中扫描首板是**口径错配** | `flask_server.py:184` 交易窗口 09:15–15:00 | T 日是否「收盘封板」只能盘后知道；盘中扫到的是「当前封板中」，训练样本是「收盘封板」。模型被喂了分布外特征（半日换手、半日涨停家数、半日炸板率） |
| A8 | 涨停池富因子（炸板次数/封板时间/封单额）**只有近 15 交易日**，却被写进 3 年样本与风险规则 | `build_dataset.py:148` | 依赖它们的规则在历史回测中**永远不生效**（死规则） |

### B 级（重要 · 影响结论可靠性）

| # | 问题 | 证据 |
|---|---|---|
| B1 | 日 K **不复权**（`adjust=""`），除权日涨停价判定错误 | `build_dataset.py:194` |
| B2 | 股票池用**当前** ST 名单过滤 3 年历史 → 时点无效(universe look-ahead) | `build_dataset.py:229-231` |
| B3 | 股票名称用当前名单回填历史 | `_nm()` |
| B4 | 无任何交易成本（佣金/最低佣金/印花税/过户费/滑点全缺） | 全项目无 cost 模块 |
| B5 | 无 Walk-Forward，单次 train/valid/test | `model_train.py` |
| B6 | 无 baseline 对比（随机 / 每日随机 3 只 / 简单规则） | 全项目无 |
| B7 | 「最大回撤」= 按日等权收益的累计，非资金曲线 | `model_train.py:266-272` |
| B8 | 剔除规则依赖当日全市场中位数跌幅（`mkt_ret < -5%` 全剔），口径粗暴且未验证 | `build_dataset.py:381-382` |
| B9 | 因子**方向全部人工硬编码**（`FACTORS` 里 +1/−1），未从数据验证 | `model_train.py:55-72` |
| B10 | 过拟合校验用 `|train−valid| Top3 精度 > 0.15` 直接丢弃组合 —— 而 valid Top3 仅 345 样本，噪声主导 | `model_train.py:314` |

### C 级（工程 · 可维护性）

| # | 问题 |
|---|---|
| C1 | **死代码**：`dragon_analyzer.py:517` `feats.get("cycle")` 该键从未写入 → 冰点 veto 永不触发 |
| C2 | **死代码**：`dragon_analyzer.py:501` `feats.get("seal_decay")` 全项目无任何赋值 → 封单崩塌 veto 永不触发 |
| C3 | **死代码**：`flask_server.py:125` 读 `s["封单额"]`，但 `analyze_live()` 从不产出该字段 → 封单衰减追踪永不生效 |
| C4 | **死代码**：`flask_server.py:292` 一行无意义列表推导 |
| C5 | 双路径不一致：`--skip-download` 少了 `close>0` 过滤 → 同一份缓存产出两个不同数据集，**不可复现** |
| C6 | 收盘入库（`close_and_store`）只写 score/final/status，**不写因子值、不回填标签** → README 宣称的「自动迭代（收盘入库→次日回填→定期重训）」链路**实际未实现** |
| C7 | 无 `tests/` 目录，零测试 |
| C8 | 无数据质量检测 |
| C9 | 无决策日志/可追溯链路，无法复现任一历史候选池 |
| C10 | `dataset_v2.csv` 把真实标签 `label_Y`、未来收益一并输出到 API，前端展示「晋级率」实为**回看时的真实标签**，极易被误读为预测结果 |

---

## 3. 哪些旧模块保留

保留 = 逻辑有价值，但**必须改接口/换位置**。

| v2 资产 | v3 处置 | 理由 |
|---|---|---|
| 新浪日K下载 + `.done` 断点续传（`_dl_worker`） | **保留**，迁入 `src/data/providers/akshare_provider.py`，新增节流与复权双轨 | 已验证可用；断点续传设计正确 |
| 涨停 / 一字 / 炸板 / 触板 判定 | **保留公式**，迁入 `src/features/price_features.py`，修正涨跌幅上限与复权 | 公式本身合理 |
| 连板计数 `_consec_lb` | **保留并重写**，加停牌断链处理 + 单测 | 向量化思路好，但未处理停牌 |
| 东财涨停池拉取 | **保留**，作为**增强数据源**，明确标注历史不可得 | 实时价值高 |
| V8 预热 + `SCAN_LOCK` 串行化 | **保留**（迁入 `src/services/scheduler.py`） | 踩过的坑，py_mini_racer 并发首 init 硬崩 |
| 「非交易时段完全静默 / 手动启动不自启」 | **保留**（用户明确要求） | 用户纪律 |
| `index.html` 视觉骨架与 SVG 自绘方案 | **保留骨架**，页面重组为 5 页 | 不依赖 CDN，离线可用 |
| `industry_map.csv` / `code_name.csv` | 保留，迁入 `data/raw/` | 迁移成本低 |

---

## 4. 哪些模块重构

| v2 | v3 目标模块 | 变更要点 |
|---|---|---|
| `build_dataset.py::build_features`（700 行单体） | `src/features/{price,market,plate,chip,auction}_features.py` + `registry.py` | 按因子族拆分；因子元数据（名/模块/方向/所需数据/是否实时可得）集中注册 |
| `build_dataset.py::attach_label` | `src/labels/labeler.py` | 多目标标签 + 真实 T+1 口径 + 可成交性标记 |
| `model_train.py` 打分 | `src/models/score_model.py` | 删除仿射拉伸；改为单调映射 + 分位阈值 |
| `model_train.py` 校准 | `src/models/calibration.py` | 校准只用 valid/test，输出单调性检验 |
| `model_train.py` 网格搜索 | `src/models/*` + `analysis/score_analysis.py` | 权重拟合下沉为可复用组件；评价体系扩展 |
| `dragon_analyzer.py::emotion_cycle` | `src/analysis/regime_analysis.py` | 四档保留，**权重由分 Regime 数据估计**，不人为指定 ×0.8 |
| `dragon_analyzer.py::check_veto` | `src/strategy/risk_filter.py` | 规则化 + 数据缺失输出 `unknown`，不默认「无风险」 |
| `dragon_analyzer.py::precheck` | `src/strategy/auction_selector.py` | 从「低开就剔除」升级为「竞价多因子重排序」 |
| 无 | `src/backtest/*` | **全新**：engine / execution / cost / metrics / walk_forward |
| 无 | `src/analysis/factor_analysis.py` | **全新**：IC / RankIC / 分组 / 稳定性标签 |
| 无 | `src/alerts|quality.py` | 数据质量检测 |
| `flask_server.py`（路由+调度+扫描混在 373 行） | `src/api/routes.py` + `src/services/scheduler.py` | 彻底分离 |
| 板块统计（仅行业涨停家数） | `src/strategy/theme_analyzer.py` | 主线/次主线/轮动/退潮 |

---

## 5. 哪些模块删除

**「不要为了保留旧逻辑而保留错误设计」——以下为明确删除项：**

| 删除对象 | 位置 | 替换为 |
|---|---|---|
| `ret_next_oc` 作为**策略收益** | `build_dataset.py:550` | `intraday_mark_return`（仅标记浮盈浮亏，字段名即警告） |
| 固定阈值 `select=70 / 备选=55` | `model_train.py:76-77`、`config.json` | 训练集内分位数阈值（如 Top 5%）+ valid 验证 |
| 仿射拉伸 `score_center` / `score_scale` / `TARGET_SD` | `model_train.py:183-194`、`dragon_analyzer.py:190-192` | 删除。分数保持原始量纲，阈值用分位数表达 |
| `bomb_times` 因子 | `FACTORS` 全部三处 | 删除（历史不可得）。炸板信息仅在**实时**路径作为风险提示，且标注 `unknown@history` |
| `big_money_net` 因子 | `build_dataset.py:434-441` | 删除。若需资金流，改用可验证的日K代理或明确标注为 `volume_proxy` |
| 盘中扫描首板（09:15–15:00 每 15 分钟） | `flask_server.py:184, 202-211` | 盘后候选生成（15:30 后）+ 次日竞价重排序 |
| 情绪周期折扣系数 ×1.0/×0.8/×0.5 | `emotion_cycle` 三处副本 | 分 Regime 数据驱动权重 |
| `--skip-download` 与全量下载的**双路径** | `build_dataset.py:581-588` | 单一 `loader.load_bars()` 出口，两条路径共用后处理 |
| 收盘入库假「自动迭代」 | `flask_server.py:161-179` | `decision_log.jsonl` 全字段落盘 + 标签回填任务 |
| C1–C4 死代码 | 见上表 | 删除，或用单测强制其真实生效 |
| `ret_next_cc` 作为可交易收益 | `attach_label` | 保留为研究字段（T 日收盘买入不可实现），重命名 `research_cl_to_next_close` |

---

## 6. 新目录结构

```
DLongYaoDetector/
├── config/
│   ├── config.json           # 运行参数：trading_cost / label_defs / backtest / thresholds(分位) / regime / paths
│   └── providers.json        # 数据源开关与限流参数
├── data/
│   ├── raw/                  # 原始层（只增不改）：bars_shard*.csv、industry_map、code_name、zt_pool_raw
│   ├── processed/            # dataset_v3.parquet（唯一训练/回测入口）
│   └── cache/                # 中间缓存、竞价快照、decision_log
├── docs/
│   ├── v2_audit.md           ✔ Phase 1
│   ├── v3_development_plan.md ✔ 本文档
│   ├── audit_evidence.txt    ✔ 实证输出
│   ├── plan_deviations.md    # 与上游计划的偏差记录（强制）
│   └── data_dictionary.md    # 字段口径字典（含每个字段的「可知时点」）
├── src/
│   ├── __init__.py
│   ├── config.py             # 配置加载 + 校验 + hash（用于复现）
│   ├── data/
│   │   ├── providers/
│   │   │   ├── base.py            # DataProvider 抽象
│   │   │   ├── akshare_provider.py
│   │   │   ├── tushare_provider.py   # 预留桩
│   │   │   └── local_provider.py     # 读 data/raw 缓存
│   │   ├── loader.py         # 唯一数据出口：load_bars / load_limit_pool / load_names
│   │   ├── cache.py
│   │   ├── calendar.py       # 交易日历、T+n 解析
│   │   └── quality.py        # 数据质量检测 -> data_quality_report
│   ├── features/
│   │   ├── registry.py       # 因子注册表（含 available_at: close/intraday/auction）
│   │   ├── price_features.py
│   │   ├── market_features.py
│   │   ├── plate_features.py
│   │   ├── chip_features.py
│   │   └── auction_features.py
│   ├── labels/
│   │   └── labeler.py        # 多目标 + tradability
│   ├── models/
│   │   ├── score_model.py
│   │   ├── classifier.py     # 可选：逻辑回归/GBDT 对比
│   │   └── calibration.py
│   ├── strategy/
│   │   ├── candidate_selector.py   # 盘后候选池 S/A/B
│   │   ├── auction_selector.py     # T+1 竞价重排序
│   │   ├── risk_filter.py
│   │   └── theme_analyzer.py
│   ├── backtest/
│   │   ├── engine.py         # 资金曲线 / 持仓 / 组合
│   │   ├── execution.py      # 可成交性判定 + 滑点
│   │   ├── cost.py           # 时变成本模型
│   │   ├── metrics.py        # 收益/风险/概率 全套指标
│   │   └── walk_forward.py
│   ├── analysis/
│   │   ├── factor_analysis.py
│   │   ├── regime_analysis.py
│   │   └── score_analysis.py
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   └── scheduler.py      # 盘后调度 + 竞价调度 + 静默策略
│   └── legacy/               # v2 单体移入，仅作对照，禁止被 src 其他模块 import（CI 检查）
├── web/                      # 新前端（5 页）
├── tests/
├── scripts/
│   ├── audit_v2_checks.py    ✔
│   ├── run_pipeline.py       # 一键：build → label → backtest → report
│   └── migrate_data.py       # v2 → v3 数据迁移
├── reports/
├── logs/
├── requirements.txt
├── readme.md
└── run_server.bat
```

---

## 7. 数据结构

### 7.1 时间锚点定义（**全系统唯一口径，写入 `docs/data_dictionary.md`**）

| 锚点 | 含义 | 数据可知时点 |
|---|---|---|
| `T` | 首板日 | 全部数据 **T 日 15:00 后**可知 |
| `T+1_auction` | 次日 09:15–09:25 竞价 | 09:25 后可知 |
| `T+1_open` | 次日 09:30 开盘价 | 09:30 后可知 |
| `T+1_close` | 次日收盘 | T+1 15:00 后可知 |
| `T+2_open / T+2_close` | 卖出窗口 | — |

### 7.2 `data/processed/dataset_v3.parquet` 主表

| 字段组 | 字段 | 说明 |
|---|---|---|
| 主键 | `date, code` | date = T 日（YYYYMMDD str） |
| 静态 | `name, board, industry, list_days, float_mkt, is_st_at_T` | `is_st_at_T` 为**时点**ST 状态（来自历史更名表，非当前名单） |
| 行情 | `open/high/low/close/preclose/volume/amount/turnover`（**前复权 + 原始双份**） | 复权序列用于因子，原始序列用于成交价与涨停价 |
| 因子 | 由 `features/registry.py` 注册，前缀区分：`f_*` | 每个因子附 `available_at` 标签 |
| 板块 | `plate_zt_cnt, plate_first_cnt, plate_max_lb, plate_amt_chg, plate_diffusion, theme_name, theme_role` | |
| Regime | `regime`（main_up/oscillation/decline/ice）, `regime_zt_cnt, regime_max_lb, regime_bomb_rate` | |
| 标签 | `label_1to2, label_next_positive, label_t2_positive, label_t2_gt_3, label_t2_gt_5, label_max_gain_{1,2,3,5}d, label_max_dd_{1,2,3,5}d` | |
| 收益 | `ret_buy_t1_open_sell_t2_open, ret_buy_t1_open_sell_t2_close, ret_hold{N}d, intraday_mark_return, research_cl_to_next_close` | |
| 可成交性 | `tradable_t1_open`(bool), `tradable_reason` | 涨停开盘/一字/停牌 → False |
| 元数据 | `split, data_version, config_hash` | 复现用 |

### 7.3 收益定义（**核心**）

```
intraday_mark_return        = T+1_close / T+1_open − 1     ← 仅浮盈浮亏标记，禁止作为策略收益
ret_buy_t1_open_sell_t2_open  = T+2_open  / T+1_open − 1   ← 模式 A
ret_buy_t1_open_sell_t2_close = T+2_close / T+1_open − 1   ← 模式 B
ret_hold{N}d                = T+N_close / T+1_open − 1     ← 模式 C，N∈{1,2,3,5}
ret_cond_exit               = 按 ExitPolicy 逐日模拟          ← 模式 D
最终策略收益                 = 上述任一口径 − trading_cost（含滑点/佣金/最低佣金/印花税/过户费）
```

**可成交性硬约束（新增，v2 缺失）**：
`T+1_open >= round(T_close × (1+限幅), 2) − 0.01` → `tradable_t1_open = False`。
实测该情形占 **7.33%**，且集中在最强样本上——不修正会让回测系统性乐观。

### 7.4 模式 D 退出策略（模块化）

```python
class ExitPolicy(Protocol):
    def should_exit(self, ctx: BarContext) -> tuple[bool, str]: ...

# 内置实现（可组合）：
BreakPrevLowPolicy        # 跌破前一日最低价
BreakMA5Policy            # 跌破 5 日均线
OpenHighCloseLowPolicy    # 高开低走
StopLossPolicy(-pct)      # 止损
TakeProfitPolicy(+pct)    # 止盈
MaxHoldDaysPolicy(n)      # 最大持有天数
```
`config/backtest.json` 以策略名 + 参数组合，禁止写死在引擎里。

### 7.5 交易成本（`config/config.json`）

```json
{
  "trading_cost": {
    "commission_rate": 0.00025,
    "min_commission": 5,
    "stamp_tax": 0.0005,
    "stamp_tax_schedule": [{"from": "20230828", "rate": 0.0005},
                           {"to":   "20230827", "rate": 0.001}],
    "transfer_fee": 0.00001,
    "slippage_buy": 0.001,
    "slippage_sell": 0.001
  }
}
```
> 说明：印花税 2023-08-28 由 1‰ 降至 0.5‰。**成本模型必须时变**，否则跨 2023 的回测会被高估 0.05%/笔。这是对上游计划的**加强**，记入 `plan_deviations.md`。

---

## 8. 回测架构（第一优先级）

### 8.1 分层

```
BacktestEngine
  ├── SignalSource        每日给出候选（策略 / 模型 / 简单规则 / 随机）
  ├── Executor            应用 tradability + slippage + 涨跌停不可成交
  ├── CostModel           时变费率
  ├── Portfolio           资金分配、最大同时持仓、等权/固定金额
  └── MetricsCalculator   全套指标 + 资金曲线
```

### 8.2 硬约束（全部可单测）

1. **禁止未来函数**：`Executor` 只接受 `available_at <= 当前决策时点` 的字段，违反即抛异常。
2. **可成交性**：涨停开盘、停牌、成交量为 0 → 不可成交，信号作废（不是"以开盘价成交"）。
3. **T+1 制度**：买入当日不可卖出，只能是 T+1 买 T+2 卖。
4. **资金约束**：每日最多持仓 M 只、单票上限、最低佣金 5 元（小额单成本占比自然上升）。
5. **份额取整**：按 100 股整手，剩余现金保留（v2 完全忽略）。

### 8.3 Walk-Forward

```
窗口：训练 24 个月 → 测试 3 个月，步进 3 个月，滚动至数据末端
例：train 2023-01~2024-12 → test 2025Q1
    train 2023-04~2025-03 → test 2025Q2   ... 持续滚动
```
输出：每个窗口的 AUC / Precision / Top-K / 收益 / 胜率 / 回撤，**并汇总为分布（均值 ± 标准差）**，
重点回答「因子效果是否随市场环境变化」。

### 8.4 基线对比（强制四路）

| 基线 | 定义 |
|---|---|
| B0 | 全部首板等权、随机参与 |
| B1 | 每日随机选 3 只（重复 100 次取分布） |
| B2 | 简单规则：板块涨停数 ≥3 且 无炸板 且 非高位 |
| B3 | v2 模型（迁移后重跑） |
| B4 | v3 模型 |

**必须证明复杂模型相对 B2 简单规则是否真的带来增量**——这是 v3 的核心问题之一。

---

## 9. 模型架构

### 9.1 目标

模型同时回答两问：
- **A**：这只股票次日是否具有较强延续性？ → `label_1to2`
- **B**：按真实交易规则参与，风险收益是否值得？ → `label_t2_positive` / `label_t2_gt_3`

### 9.2 评分流程（删除仿射拉伸）

```
原始因子值
  → 因子单调映射（分位箱，仅在训练窗口内拟合）
  → 模块加权（权重由训练窗口内优化得出，方向由数据验证而非硬编码）
  → 原始分（保留自然量纲）
  → 阈值 = 训练窗口内分位数（Top q%），在 valid/test 上验证
```

**关键删除**：不再用 `score_center/score_scale` 把分数拉伸到 sd=15。阈值以「分位数」表达，天然可移植。
若仍要输出 0–100 分，则用 `percentile_rank × 100`，并在文档中说明它是**相对分位**而非概率。

### 9.3 Score Calibration（强制输出）

| Score区间 | 样本数 | 晋级率 | 真实平均收益 | 中位收益 | 胜率 | 最大回撤 |
|---|---|---|---|---|---|---|

+ **单调性检验**：
  - Spearman(score, 晋级率) 与 Spearman(score, 真实收益)
  - 逐箱违反计数 + 违反幅度
  - 输出 `score_monotonicity_failed: true/false`
+ **若为 true：UI 禁止出现「分数越高越好」类表述，必须显式标注「当前评分排序能力未通过检验」**

### 9.4 模型评价指标（禁止只看 Accuracy）

概率类：AUC、PR-AUC、Precision、Recall、Lift、Top-K Precision
收益类：平均真实收益、中位收益、胜率、Profit Factor、最大回撤、Sharpe、Calmar
稳健类：Walk-Forward 各窗口指标分布、train-valid-test 三段一致性

---

## 10. 竞价模块架构

### 10.1 定位

`auction_analyzer` 是 v3 的核心增量：T 日盘后模型只能给出**候选**，T+1 09:15–09:25 的竞价才是**决策时刻**。

### 10.2 输入（`auction_features.py`）

竞价涨幅、竞价成交额、竞价换手、竞价量比、竞价未匹配量、高开幅度、
相对 T 日封单变化、板块竞价强弱、同板块核心股表现、市场整体竞价环境。

### 10.3 输出

```
final_score = f(base_score_T, auction_score)
```
**权重不写死**，由历史竞价数据拟合；`auction_data_available=False` 时降级为仅用开盘价与前 5 分钟表现，并在输出中显式标注。

### 10.4 ⚠️ 数据可得性风险（必须先行验证 —— Phase 9 前置任务）

akshare 对**分钟级/竞价级**历史数据的支持有限。**若无法获取历史竞价数据，则竞价模型无法回测**，
此时只能：
- (a) 降级为「T+1 开盘价 + 开盘涨幅」的代理变量（可得），并明确标注为近似；
- (b) 仅作为实时辅助提示，不纳入历史绩效声明。

**验证任务**：Phase 9 第一步先写 `scripts/probe_auction_data.py`，探测可得性与回溯深度，结果写入 `docs/plan_deviations.md`。

---

## 11. API 改造

| v2 | v3 | 变更 |
|---|---|---|
| `GET /` | `GET /` | 5 页前端 |
| `GET /api/health` | `GET /api/health` | 加入 `data_version / config_hash / data_quality_warning` |
| `GET /api/analyze?date=` | `GET /api/candidates?date=` | **改为盘后候选池**；返回 `score_breakdown` + `reject_reason` |
| `GET /api/history` | `GET /api/pool/history?date=` | |
| `GET /api/emotion` | `GET /api/market/history?days=` | 删死代码 |
| `GET /api/config` | `GET /api/config` | 返回脱敏后的有效参数 + `config_hash` |
| `GET /api/watchlist` | `GET /api/watchlist` | S/A/B 三级 |
| `GET /api/precheck` | `GET /api/auction?date=` | 升级为竞价重排序 |
| `GET /api/logs` | `GET /api/decision_log?date=` | JSONL 结构化 |
| — | `GET /api/backtest?strategy=&window=` | 新增 |
| — | `GET /api/research/factors` | 新增：IC / 分组 / 稳定性 |
| — | `GET /api/research/calibration` | 新增 |
| — | `POST /api/run/{build|label|backtest}` | 新增：手动触发（默认不自启） |

**强制**：所有响应带 `available_at` 语义说明；**任何情况下不得把 `label_*` 字段返回给前端**（v2 的 C10 问题）。

---

## 12. UI 改造（5 页）

| 页面 | 内容 |
|---|---|
| **Dashboard** | 市场周期 / 涨停家数 / 炸板率 / 连板高度 / 主线板块 / 今日候选池（S/A/B 计数） |
| **Candidates** | 基础评分 / 竞价评分 / 最终评分 / 板块 / 风险 / **因子贡献拆解** / 状态（含免责声明） |
| **Market** | 市场情绪历史（涨停家数、炸板率、连板高度、晋级率曲线 + Regime 着色） |
| **Backtest** | 累计收益 / 胜率 / 最大回撤 / **四路基线对比** / Walk-Forward 窗口表 |
| **Research** | 因子 IC / 因子分组收益 / 因子稳定性标签 / 模型 AUC / Score Calibration 表 + 单调性告警 |

**UI 硬规则**：
1. 等级（S/A/B）**必须**标注为「研究优先级，非买入建议」。
2. 若 `score_monotonicity_failed=true`，Research 页顶部显示红色告警条。
3. 若 `data_quality_warning`，全站顶部横幅提示。

---

## 13. 测试计划

`tests/` 目录，pytest。**每个 Phase 结束必须全绿才允许进入下一 Phase。**

| 测试文件 | 覆盖 |
|---|---|
| `test_limit_detection.py` | 涨停识别（含边界：±0.01 容差、收盘=最高、触板未封） |
| `test_first_board.py` | 首板识别、一字板剔除 |
| `test_consec_limit.py` | 连板计算（含**停牌断链**、跨股票隔离） |
| `test_board_20cm.py` | 创业板/科创板 20cm 限幅 |
| `test_board_10cm.py` | 主板 10cm 限幅 |
| `test_st.py` | ST 5% 限幅 + 时点 ST 判定 |
| `test_t1_return.py` | T+1 收益口径（模式 A/B/C）、可成交性判定 |
| `test_cost.py` | 佣金/最低佣金/印花税（**含 2023-08-28 费率切换**）/过户费/滑点 |
| `test_suspension.py` | 停牌处理（无成交、跨停牌持有） |
| `test_missing_data.py` | 缺失数据 → `unknown` 而非「无风险」 |
| **`test_no_future_leakage.py`** | **核心**：T 日信号构造只依赖 `available_at <= T_close` 的字段；用「数据截断重算」法验证——把数据截到 T 日重算因子，结果须与全量重算的 T 日值**逐位相等** |
| `test_split_integrity.py` | train/valid/test 无重叠、时间单调、无未来穿越 |
| `test_calibration.py` | 校准单调性检验自身的正确性（造单调/非单调数据） |
| `test_reproducibility.py` | 同 config_hash + data_version → 逐位可复现 |

**`test_no_future_leakage.py` 设计要点**（这是 v2 完全缺失、也最容易做错的一环）：
```python
def test_truncation_invariance():
    """把 bars 截断到 T 日，重算 T 日因子，必须与全量数据下算出的 T 日因子完全一致"""
    full   = build_features(bars_all)[date == T]
    trunc  = build_features(bars_all[bars_all.date <= T])[date == T]
    assert_frame_equal(full, trunc)
```
该测试可一次性捕获所有「窗口越界」类泄漏。

---

## 14. 数据迁移方案

| v2 资产 | v3 目标 | 动作 |
|---|---|---|
| `data/bars_shard*.csv`（307MB，不复权） | `data/raw/bars_shard*.csv` | **移动**（零拷贝，改路径常量）。**新增** `data/raw/bars_qfq_shard*.csv` 前复权序列 |
| `data/industry_map.csv` | `data/raw/` | 移动 |
| `data/code_name.csv` | `data/raw/` | 移动；**新增** `code_name_history.csv`（历史更名，供时点 ST 判定） |
| `data/zt_pool_raw.csv` | `data/raw/` | 移动 |
| `data/dataset_v2.csv` | `data/raw/dataset_v2_frozen.csv` | **冻结只读**，仅供 v2 对照复现，v3 不再读写 |
| `data/cache/pool_*.csv`、`daily_samples_*.csv` | `data/cache/` | 移动（历史价值低） |
| `config.json` | `config/config.json` | **重写**：权重/分位箱/拉伸参数**不迁移**（因子集已变，迁移等于把错误固化）；保留 `notes` 与数据源限制说明 |
| `logs/` | `logs/` | 保留 |

**迁移脚本**：`scripts/migrate_data.py`（幂等、可重复执行、移动前校验文件完整性）。
**回归保护**：迁移后必须能用 `src/legacy/` 的 v2 代码 + `dataset_v2_frozen.csv` 复现出 v2 的 evaluate 结果，作为「未污染」的证据。

**新增数据需求（Phase 2 前置验证）**：
1. **复权因子** — 用于修正除权日涨停价误判。探测 akshare `adjust="qfq"/"hfq"` 或独立复权因子接口。
2. **历史 ST 状态** — 修正 universe 时点偏差。
3. **历史行业/概念分类变更** — 降低行业映射的前视偏差（优先级低）。
4. **竞价/分钟级历史** — 见 §10.4。

---

## 15. 分阶段实施顺序与验收标准

> 每完成一个 Phase：**先跑测试 → 再检查结果 → 再提交改动**。禁止假设测试通过。

### Phase 1 · 审计旧代码 ✅ 已完成
**产出**：`docs/v2_audit.md`、`docs/audit_evidence.txt`、`scripts/audit_v2_checks.py`
**验收**：所有 A 级问题有可复现实证证据（非推测）→ 已满足

### Phase 2 · 重构数据层
建立 `DataProvider` 抽象、`loader`、`cache`、`quality`；完成数据迁移；补复权序列。
**验收**：
- `pytest tests/data tests/test_suspension.py tests/test_missing_data.py` 全绿
- `data/processed/dataset_v3.parquet` 可生成；数据质量报告无 ERROR 级问题
- 幂等：连续两次运行 `run_pipeline.py --stage=build` 产出文件 hash 一致
- 复权修正生效：除权日涨停误判数从 N 降到 0（有记录）

### Phase 3 · 重构标签系统
`labeler.py` 产出全部标签 + 可成交性标记。
**验收**：
- `pytest tests/test_t1_return.py tests/test_no_future_leakage.py test_split_integrity.py` 全绿
- 输出标签分布报告：各标签正例率、`tradable_t1_open=False` 占比（预期 ≈7.3%）、收益分布
- **手工核对**：随机抽 20 条样本，人工按日K核对 T+1/T+2 收益，偏差为 0

### Phase 4 · 建立真实 BacktestEngine
**验收**：
- `pytest tests/test_cost.py tests/test_backtest_engine.py` 全绿
- 给定**固定信号集**（人工构造 10 笔交易），引擎输出的每笔成本、收益、资金曲线可**手工核对**
- 支持：交易成本、滑点、T+1、持有周期、资金曲线、最大回撤
- 印花税费率切换正确（2023-08-28 前后各一笔对比）

### Phase 5 · 重跑 v2 策略，得到真实结果
**产出**：`reports/v2_real_backtest.md`
**必须回答**：**v2 是否真的存在 alpha？**
**验收**：给出含成本的资金曲线、四路基线对比、以及「修正可成交性前后」的对照。
若不显著优于 B1/B2 基线，必须**明确写出结论为「未发现 alpha」**，不得粉饰。

### Phase 6 · 因子分析
**产出**：`reports/factor_analysis.md`
**验收**：每个因子输出 IC / RankIC / 分组收益 / 1进2 成功率 / 分 Regime 表现 + 稳定性标签
（`factor_useless` / `factor_decay` / `factor_regime_sensitive`）
删除明显无效因子，**已预告删除**：`bomb_times`（历史 100% 缺失）、`big_money_net`（完全退化）
**验收判据**：删除后模型指标不劣化（若劣化则说明该因子其实有效，需重新审查）

### Phase 7 · 重新构建评分模型
**验收**：
- 不再出现任何固定 70 分与仿射拉伸参数
- Score Calibration 表完整（n / 晋级率 / 真实均收益 / 中位 / 胜率 / 最大回撤）
- 单调性检验输出 `score_monotonicity_failed`
- Top-K / Lift 指标齐全；AUC 在 **test** 上报告（不是 train）
- 相对 B2 简单规则基线的增量被明确量化

### Phase 8 · Regime Model
**验收**：分 Regime 的因子有效性表；权重由数据估计；输出「冰点是否应完全暂停」的统计证据

### Phase 9 · 竞价模型
**前置**：`scripts/probe_auction_data.py` 探测数据可得性
**验收**：`auction_score` 可回测（或按 §10.4 明确降级并标注）；`final_score = base + auction` 相对仅用 base 的增量被量化

### Phase 10 · 重构 Web UI
**验收**：5 页齐备；免责声明与单调性/质量告警强制展示；不返回任何 `label_*` 字段

---

## 16. 每阶段验收标准（汇总判据）

**通用门禁（每个 Phase 必须同时满足）**：
1. 该 Phase 相关 pytest **全绿**，且测试不是「为了通过而写」（断言必须能失败）
2. **无未来函数**：`test_no_future_leakage.py` 全绿
3. **可复现**：`config_hash + data_version` 相同 → 输出逐位一致
4. **有报告**：产出对应的 `reports/*.md` 或 `docs/*.md`，且**包含不利结果**
5. **无粉饰**：禁止只报告最优区间、禁止只报胜率不报亏损、禁止把浮盈当真实收益

**反模式检查清单（每次提交前自问）**：
- [ ] 是否为了保留旧逻辑而保留了错误设计？
- [ ] 是否凭感觉增加了因子？
- [ ] 是否人为修改权重直到回测变漂亮？
- [ ] 是否让 test 集参与了调参？
- [ ] 是否引入了未来函数？
- [ ] 是否只报告了最优时间区间？
- [ ] 是否只显示胜率而忽略了亏损？
- [ ] 是否把浮盈当成了真实收益？

---

## 17. 与上游计划的偏差记录（同步写入 `docs/plan_deviations.md`）

技术实现中对上游计划的**主动调整**，逐条记录「原方案 / 修改方案 / 修改原因」：

| # | 原方案 | 修改方案 | 原因 |
|---|---|---|---|
| D1 | 产品定位含「T 日盘中扫描首板」（v2 遗留） | 改为「**盘后生成候选池** + T+1 竞价重排序」 | T 日是否收盘封板只能盘后确定；盘中扫描的样本分布与训练集不一致（A7） |
| D2 | 模式 C「T+1 竞价/开盘确认信号后买入」未定义确认标准 | 明确为「竞价涨幅 ∈ [x,y] 且竞价量比 ≥ q」，x/y/q 由训练集网格确定 | 无定义则无法回测，属逻辑闭环缺失 |
| D3 | 未提及复权 | **新增**复权数据轨道，双序列（复权算因子 / 原始算成交价） | 不复权导致除权日涨停价判定错误（B1），属数据正确性优先项 |
| D4 | 假设竞价历史数据可得 | **新增前置探测任务**，并设计降级路径 | akshare 分钟/竞价历史数据可得性未验证，不能假设 |
| D5 | 要求用「炸板次数/封单额」做风险过滤 | 降级为「仅实时可用」，历史路径输出 `unknown` | 东财涨停池仅留近 15 交易日（A8），历史不可得，禁止伪造 |
| D6 | 交易成本参数为静态 | **改为时变**（印花税 2023-08-28 调整） | 静态成本使跨 2023 回测高估 0.05%/笔 |
| D7 | 「分数 0–100」的语义未界定 | 明确为**分位相对分**，禁止当作概率解读 | v2 的 70 分是仿射拉伸产物（A4），必须切断该误读 |

---

## 18. 最终目标产物形态

```
今日市场：
  情绪周期：震荡 | 涨停：47 | 炸板：18 | 最高板：5
  主线：机器人 / 算力 / 消费电子
  首板：32 → 初筛后：11 → 重点候选：3

候选（600XXX · 机器人）：
  基础评分 67 | 板块评分 82 | 量价 73 | 市场 65 | 风险 低
  T+1 竞价：高开 +2.8% | 竞价成交 正常 | 板块竞价 强 | 竞价评分 81
  最终评分 75 | 状态：重点观察（研究优先级，非买入建议）
  归因：板块 +12 / 量价 +8 / 情绪 +5 / 风险 −6 / 竞价 +9
```

**系统最重要的四件事**：可验证 · 可解释 · 可复现 · 无未来函数且符合真实 A 股交易规则。

---

*本计划为活文档。任何实现中的调整必须先写入 `docs/plan_deviations.md`，再改代码。*
