# -*- coding: utf-8 -*-
"""
================================================================================
 龙妖首板基因识别器 —— Flask 后端服务 (v2)
================================================================================
 启动: python flask_server.py    访问: http://127.0.0.1:5000

 接口:
   GET /                      网页看板
   GET /api/health            健康检查
   GET /api/analyze?date=     当日/指定日 首板基因打分(实时优先, 无实时则回看数据集)
   GET /api/history?date=     历史日打分(离线样本)
   GET /api/emotion?days=     全局情绪周期折线数据
   GET /api/config            模型参数(最优权重/校准分箱)
   GET /api/watchlist         当前观察池(次日开盘介入标的)
   GET /api/precheck?date=    盘前竞价二次校验(剔除核按钮/大幅低开)
   GET /api/logs?date=        扫描日志

 内置调度(后台线程, 无需外部定时任务):
   9:15~15:00 每 5 分钟盘中扫描首板, 结果写入内存 + 日志
   每轮记录: 扫描时间/标的分数/失效告警/封单衰减
   15:10 收盘入库: 当日首板样本落盘 data/daily_samples_YYYYMMDD.csv

 ★ akshare 为网页爬虫数据源, 盘中存在延迟, 精细封单数据有限, 仅用于策略原型验证。
 ★ 所有因子只使用当日及之前已产生的数据, 严禁未来函数。
================================================================================
"""
import os
import io
import json
import time
import datetime as dt
import threading

import numpy as np
import pandas as pd

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory, Response

from dragon_analyzer import DragonAnalyzer, MOD_CN, MODULES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
DS = os.path.join(DATA_DIR, "dataset_v2.csv")

app = Flask(__name__, static_folder=None)
A = DragonAnalyzer()

STATE = {
    "last_scan": None,
    "result": None,
    "history": [],          # 每轮扫描摘要
    "seal_snapshot": {},    # code -> 上轮封单额(用于封单衰减判定)
    "watchlist": {},        # date -> [codes]
}

# py_mini_racer(V8) 并发首次初始化会硬崩进程(Check failed:
# !IsConfigurablePoolInitialized), 所有 akshare 扫描调用必须串行,
# 且须在主线程预热一次 V8 后再启动调度线程。
SCAN_LOCK = threading.Lock()
_V8_KEEP = None


def warmup_v8():
    """主线程预热 V8 引擎, 进程内只初始化一次, 消除多线程竞态"""
    global _V8_KEEP
    try:
        import py_mini_racer
        _V8_KEEP = py_mini_racer.MiniRacer()   # 持有引用防止 GC 回收
        _V8_KEEP.eval("1+1")
        print(" V8 引擎预热完成 (py_mini_racer", py_mini_racer.__version__ if hasattr(py_mini_racer, "__version__") else "", ")")
    except Exception as e:
        print(" V8 预热跳过: %r" % e)

# ---------------------------------------------------------------- 日志
def scan_log(msg, date=None):
    date = date or dt.datetime.now().strftime("%Y%m%d")
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    try:
        with open(os.path.join(LOG_DIR, "scan_%s.log" % date), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- 扫描
def do_scan(date=None, force_live=True):
    # 全程串行: 多线程并发调用 akshare(新浪日K的 JS 解密用 py_mini_racer)
    # 会触发 V8 双重初始化硬崩, 必须整段持锁
    with SCAN_LOCK:
        return _do_scan_locked(date, force_live)


def _do_scan_locked(date=None, force_live=True):
    date = date or dt.datetime.now().strftime("%Y%m%d")
    t0 = time.time()
    res = None
    if force_live:
        try:
            r = A.analyze_live(date)
            if r.get("ok") and r.get("stocks"):
                res = r
        except Exception as e:
            scan_log("实时扫描异常: %r" % e, date)
    if res is None:
        r = A.score_dataset_date(date)
        if r:
            r["ok"] = True
            res = r
    if res is None:
        return dict(ok=False, date=date, msg="无可用数据", stocks=[])

    # 封单衰减追踪(连续两轮扫描对比)
    snap = STATE["seal_snapshot"]
    for s in res["stocks"]:
        code = s["代码"]
        cur = s.get("封单额")
        if cur and snap.get(code):
            try:
                decay = (float(cur) - float(snap[code])) / max(float(snap[code]), 1e-9)
                if decay < -0.35:
                    s["alerts"].append(dict(veto=True, type="封单持续崩塌",
                                            msg="封单较上轮衰减 %.0f%%" % (abs(decay) * 100)))
                    if s["status"] != "暂停":
                        s["status"] = "剔除"
            except Exception:
                pass
        if cur:
            snap[code] = cur

    res["scan_time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res["cost_sec"] = round(time.time() - t0, 1)
    STATE["last_scan"] = res["scan_time"]
    STATE["result"] = res
    obs = [s for s in res["stocks"] if s["status"] == "观察"]
    STATE["watchlist"][date] = [s["代码"] for s in obs]
    STATE["history"].append(dict(
        t=res["scan_time"], date=date, n=len(res["stocks"]),
        obs=len(obs), cycle=(res.get("market") or {}).get("cycle"),
        cost=res["cost_sec"]))
    STATE["history"] = STATE["history"][-200:]
    scan_log("扫描完成: %s | 首板 %d 只 | 观察池 %d 只 | 周期 %s | 耗时 %.1fs"
             % (date, len(res["stocks"]), len(obs),
                (res.get("market") or {}).get("cycle"), res["cost_sec"]), date)
    for s in res["stocks"]:
        if s.get("alerts"):
            for a in s["alerts"]:
                scan_log("  %s %s [%s] %s" % (s["代码"], s.get("名称", ""),
                                              a["type"], a["msg"]), date)
    return res


def close_and_store(date=None):
    """收盘入库: 直接落盘当日最后一轮盘中扫描缓存(★零额外网络请求, 静默模式)"""
    date = date or dt.datetime.now().strftime("%Y%m%d")
    try:
        r = STATE.get("result")
        if not r or not r.get("stocks") or str(r.get("date")) != str(date):
            scan_log("收盘入库跳过: 当日无盘中扫描缓存(静默模式不额外联网)", date)
            return
        rows = []
        for s in r["stocks"]:
            rows.append(dict(date=date, code=s["代码"], name=s.get("名称", ""),
                             industry=s.get("行业", ""), score=s["score"],
                             final=s["final"], status=s["status"],
                             cycle=s["cycle"], prob=s.get("prob")))
        p = os.path.join(DATA_DIR, "daily_samples_%s.csv" % date)
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
        scan_log("收盘入库 %d 条 -> %s" % (len(rows), os.path.basename(p)), date)
    except Exception as e:
        scan_log("收盘入库失败: %r" % e, date)


# ---------------------------------------------------------------- 调度线程
SCAN_INTERVAL_SEC = 900      # ★ 盘中 15 分钟一轮(降频, 减少网络占用)
TRADE_WINDOW = (9 * 60 + 15, 15 * 60)   # 仅交易时段联网; 其余时间完全静默


def in_trade_window(now=None):
    now = now or dt.datetime.now()
    if now.weekday() >= 5:                # 周六周日静默
        return False
    hm = now.hour * 60 + now.minute
    return TRADE_WINDOW[0] <= hm <= TRADE_WINDOW[1]


def scheduler():
    last_close_day = None
    while True:
        try:
            now = dt.datetime.now()
            hm = now.hour * 60 + now.minute
            day = now.strftime("%Y%m%d")
            if in_trade_window(now):               # 9:15~15:00 盘中扫描(15分钟一轮)
                gap = None
                if STATE["last_scan"]:
                    gap = (now - dt.datetime.strptime(
                        STATE["last_scan"], "%Y-%m-%d %H:%M:%S")).total_seconds()
                if gap is None or gap >= SCAN_INTERVAL_SEC:
                    try:
                        do_scan(day)
                    except Exception as e:
                        scan_log("调度扫描异常: %r" % e, day)
            elif (hm >= 15 * 60 + 10 and last_close_day != day
                    and (time.time() - _START_TS) > 180):
                close_and_store(day)               # 15:10 收盘入库(用缓存, 不联网)
                last_close_day = day
            # 非交易时段: 不发任何网络请求, 仅空转等待
            time.sleep(30)
        except Exception:
            time.sleep(60)


# ---------------------------------------------------------------- 路由
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify(dict(ok=True, time=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        last_scan=STATE["last_scan"],
                        config_loaded=bool(A.cfg),
                        dataset_rows=(0 if A.dataset is None else int(len(A.dataset)))))


@app.route("/api/analyze")
def api_analyze():
    date = request.args.get("date") or dt.datetime.now().strftime("%Y%m%d")
    force = request.args.get("force")
    cached = STATE["result"]
    # 节流: 距上一轮扫描不足 15 分钟直接返回缓存(防止看板60秒自动刷新变成分钟级全量扫描)
    if cached and cached.get("date") == date and not force:
        gap = None
        if STATE["last_scan"]:
            gap = (dt.datetime.now() - dt.datetime.strptime(
                STATE["last_scan"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        if gap is not None and gap < SCAN_INTERVAL_SEC:
            return jsonify(cached)
    # 静默: 非交易时段不联网, 回落到离线数据集(force=1 才强制实时)
    if not force and not in_trade_window():
        r = A.score_dataset_date(date)
        if r:
            r["ok"] = True
            r["source"] = "offline"
            return jsonify(r)
        return jsonify(dict(ok=False, date=date, stocks=[],
                            msg="非交易时段(静默): 无该日离线样本, 需实时请加 ?force=1"))
    r = do_scan(date)
    return jsonify(r)


@app.route("/api/history")
def api_history():
    date = request.args.get("date")
    if not date:
        df = A.dataset
        date = sorted(df["date"].astype(str).unique())[-1] if df is not None and len(df) else ""
    r = A.score_dataset_date(date)
    return jsonify(r if r else dict(ok=False, msg="无该日数据"))


@app.route("/api/emotion")
def api_emotion():
    days = int(request.args.get("days", 90))
    df = A.dataset
    if df is None or len(df) == 0:
        return jsonify(dict(ok=False, data=[]))
    d = df.copy()
    d["date"] = d["date"].astype(str)
    g = d.groupby("date").agg(
        涨停家数=("mkt_zt_count", "max"),
        最高连板=("mkt_max_lb", "max"),
        炸板率=("bomb_rate", "max"),
        涨停溢价=("high_limit_premium", "max"),
        晋级率=("label_Y", "mean"),
        样本数=("label_Y", "size"),
    ).reset_index()
    g = g.tail(days)
    g["晋级率"] = (g["晋级率"] * 100).round(2)
    g["炸板率"] = g["炸板率"].round(2)
    g["涨停溢价"] = g["涨停溢价"].round(2)
    cyc = [DragonAnalyzer.emotion_cycle if False else None for _ in range(len(g))]
    cycles = []
    for _, r in g.iterrows():
        c, disc = A.__class__.__dict__ and _cycle_helper(r["涨停家数"], r["最高连板"], r["炸板率"])
        cycles.append(dict(cycle=c, discount=disc))
    g["cycle"] = [c["cycle"] for c in cycles]
    g["discount"] = [c["discount"] for c in cycles]
    return jsonify(dict(ok=True, data=json.loads(g.to_json(orient="records"))))


def _cycle_helper(zt, lb, br):
    from dragon_analyzer import emotion_cycle
    return emotion_cycle(zt, lb, br)


@app.route("/api/config")
def api_config():
    return jsonify(A.cfg if A.cfg else dict(msg="config.json 未生成, 请先运行 model_train.py"))


@app.route("/api/watchlist")
def api_watchlist():
    return jsonify(dict(ok=True, watchlist=STATE["watchlist"],
                        last_scan=STATE["last_scan"]))


@app.route("/api/precheck")
def api_precheck():
    date = request.args.get("date") or dt.datetime.now().strftime("%Y%m%d")
    codes = request.args.get("codes", "")
    if not codes:
        wl = STATE["watchlist"]
        keys = sorted(wl.keys())
        codes = ",".join(wl[keys[-1]]) if keys else ""
    codes = [c.strip() for c in codes.split(",") if c.strip()]
    if not codes:
        return jsonify(dict(ok=False, msg="无待校验标的(观察池为空)"))
    r = A.precheck(date, codes)
    return jsonify(dict(ok=True, date=date, items=r))


@app.route("/api/logs")
def api_logs():
    date = request.args.get("date") or dt.datetime.now().strftime("%Y%m%d")
    p = os.path.join(LOG_DIR, "scan_%s.log" % date)
    if not os.path.exists(p):
        return jsonify(dict(ok=False, msg="无日志", lines=[]))
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()[-300:]
    return jsonify(dict(ok=True, date=date, lines=lines,
                        scans=STATE["history"][-50:]))


# ---------------------------------------------------------------- 启动
if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 5000))
    print("=" * 74)
    print(" 龙妖首板基因识别器 · Flask 服务 v2")
    print(" 访问地址: http://127.0.0.1:%d" % port)
    print("=" * 74)
    a = A
    if a.dataset is not None and len(a.dataset):
        print(" 数据集: %d 条样本" % len(a.dataset))
    print(" 配置: %s" % ("已加载" if a.cfg else "未生成(先运行 model_train.py)"))
    warmup_v8()                      # ★ 必须先于调度线程: V8 只能初始化一次
    t = threading.Thread(target=scheduler, daemon=True)
    t.start()
    print(" 调度线程已启动 (9:15~15:00 每5分钟扫描 / 15:10 收盘入库)")
    # 首轮扫描: 仅交易时段执行; 非交易时段完全静默(不联网)
    if in_trade_window():
        try:
            r = do_scan(force_live=True)
            print(" 首轮扫描: %s 首板 %d 只" % (r.get("date"), len(r.get("stocks", []))), flush=True)
        except Exception as e:
            print(" 首轮扫描失败: %r" % e, flush=True)
    else:
        print(" 非交易时段(静默): 跳过实时扫描, 看板显示离线数据集结果", flush=True)
    print(" 盘中扫描间隔: %d 分钟 | 交易窗口 09:15~15:00 | 周末静默" % (SCAN_INTERVAL_SEC // 60), flush=True)
    print("=" * 74, flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
