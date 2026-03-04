# -*- coding: utf-8 -*-
"""
A 股持仓管理脚本

对已持仓股票做均线趋势 + RSI + MACD 综合分析，输出操作建议：
  加仓 / 继续持有 / 减仓1/3 / 减仓1/2 / 清仓

包含止损（亏损 >= 8% 强制清仓）和止盈修正（盈利 >= 20% 阈值上移）。
大盘环境参与个股评分修正。

数据源: BaoStock + 本地缓存
运行：
  python analyze_stock.py                                    # 分析默认持仓
  python analyze_stock.py 600036.XSHG:40.00 002475.XSHE:50  # 指定 代码:成本价
"""
import pathlib
import sys
import time
import datetime
import numpy as np
import pandas as pd

from bs_utils import fetch_bars, rq_to_bs, fetch_stock_name

# =====================================================================
#  本地缓存（与 screener_macd.py 共用 .cn_cache）
# =====================================================================

_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cn_cache"


def _cache_path(rq_id, date_str):
    return _CACHE_DIR / date_str / f"{rq_id.replace('.', '_')}.csv"


def _read_cache(rq_id, date_str):
    p = _cache_path(rq_id, date_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df if len(df) >= 5 else None
    except Exception:
        return None


def _write_cache(rq_id, date_str, df):
    p = _cache_path(rq_id, date_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


# =====================================================================
#  默认持仓  —  格式: { "代码": 成本价 }
#  请在此处填写你的实际持仓成本价
# =====================================================================

DEFAULT_HOLDINGS = {
    "562510.XSHG": 0.751,   # 旅游ETF        — 填写你的成本价
    "562360.XSHG": 1.046,   # 机器人ETF       — 填写你的成本价
    "600895.XSHG": 38.1,   # 张江高科        — 填写你的成本价
    "600887.XSHG": 26.639,   # 伊利股份        — 填写你的成本价
    "002475.XSHE": 50.565,   # 立讯精密        — 填写你的成本价
    "512480.XSHG": 1.24,   # 芯片ETF         — 填写你的成本价
    "601138.XSHG": 59.368,   # 工业富联        — 填写你的成本价
    "605111.XSHG": 47.471,   # 新洁能          — 填写你的成本价
    "515230.XSHG": 0.916,   # 软件ETF         — 填写你的成本价
    "588093.XSHG": 1.732,   # 科创半导ETF     — 填写你的成本价
    "600036.XSHG": 38.383,   # 招商银行        — 填写你的成本价
}

PROFIT_THRESHOLD = 20.0  # 止盈修正线：盈利超过此比例时评分阈值上移


# =====================================================================
#  技术指标计算
# =====================================================================

def _ma(data, period):
    """简单移动平均。"""
    out = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        out[i] = data[i - period + 1: i + 1].mean()
    return out


def _ema(data, period):
    """指数移动平均。"""
    out = np.full_like(data, np.nan, dtype=float)
    k = 2.0 / (period + 1)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(close, period=14):
    """RSI 指标，返回与 close 等长的数组。"""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    return out


def _macd(close, fast=12, slow=26, signal=9):
    """MACD 指标，返回 (dif, dea, histogram) 三个等长数组。"""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


# =====================================================================
#  数据获取（带缓存）
# =====================================================================

def _detect_cache_key(today_str):
    """尝试从已有缓存目录推断最新交易日（避免无谓的网络请求）。"""
    cache_dirs = sorted(_CACHE_DIR.iterdir()) if _CACHE_DIR.exists() else []
    for d in reversed(cache_dirs):
        if d.is_dir() and d.name <= today_str:
            idx = d / "_IDX_000300.csv"
            if idx.exists():
                return d.name
    return None


def fetch_data(rq_ids, days=120):
    """获取多只股票行情，返回 {rq_id: DataFrame} 和 cache_key。"""
    today = datetime.date.today()
    start_dt = today - datetime.timedelta(days=days)
    start = start_dt.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    # 优先从本地缓存推断 cache_key，避免 BaoStock 偶发卡死
    cache_key = _detect_cache_key(today_str)
    idx_cached = _read_cache("_IDX_000300", cache_key) if cache_key else None

    if idx_cached is None:
        cache_key = today_str
        for _retry in range(3):
            try:
                idx_df = fetch_bars("sh.000300", start, today_str)
                if idx_df is not None and len(idx_df) >= 5:
                    cache_key = str(idx_df["date"].iloc[-1])
                    _write_cache("_IDX_000300", cache_key, idx_df)
                    break
            except Exception:
                time.sleep(2 * (_retry + 1))
        print(f"  沪深300数据: 在线获取, cache_key={cache_key}")
    else:
        print(f"  沪深300数据: 读取缓存, cache_key={cache_key}")

    results = {}
    miss = 0
    for rq_id in rq_ids:
        df = _read_cache(rq_id, cache_key)
        if df is None:
            bs_code = rq_to_bs(rq_id)
            time.sleep(0.3)
            df = fetch_bars(bs_code, start, today_str)
            if df is not None and len(df) >= 5:
                _write_cache(rq_id, cache_key, df)
                miss += 1
        if df is not None:
            results[rq_id] = df

    if miss > 0:
        print(f"  新下载: {miss} 只")

    idx_df = _read_cache("_IDX_000300", cache_key)
    return results, idx_df, cache_key


# =====================================================================
#  单只股票分析
# =====================================================================

def analyze_one(df, cost=None):
    """对单只股票做持仓分析，返回结果字典。"""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    if n < 20:
        return None

    # ----- 均线 -----
    ma5 = _ma(close, 5)
    ma10 = _ma(close, 10)
    ma20 = _ma(close, 20)
    ma60 = _ma(close, 60) if n >= 60 else np.full(n, np.nan)

    price = close[-1]
    prev_price = close[-2] if n >= 2 else price
    chg_pct = (price / prev_price - 1) * 100

    cur_ma5, cur_ma10, cur_ma20 = ma5[-1], ma10[-1], ma20[-1]
    cur_ma60 = ma60[-1] if not np.isnan(ma60[-1]) else None

    above_ma5 = price >= cur_ma5
    above_ma10 = price >= cur_ma10
    above_ma20 = price >= cur_ma20

    if cur_ma60 is not None:
        bullish = cur_ma5 > cur_ma10 > cur_ma20 > cur_ma60
        bearish = cur_ma5 < cur_ma10 < cur_ma20 < cur_ma60
    else:
        bullish = cur_ma5 > cur_ma10 > cur_ma20
        bearish = cur_ma5 < cur_ma10 < cur_ma20

    if bullish and above_ma5:
        arrangement = "多头排列"
    elif bearish and not above_ma20:
        arrangement = "空头排列"
    elif abs(cur_ma5 - cur_ma10) / price < 0.005 and abs(cur_ma10 - cur_ma20) / price < 0.005:
        arrangement = "均线粘合"
    else:
        arrangement = "震荡整理"

    # ----- 金叉/死叉（近 5 日）-----
    crosses = []
    lookback = min(5, n - 1)
    for i in range(-lookback, 0):
        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma5[i - 1]) or np.isnan(ma10[i - 1]):
            continue
        if ma5[i - 1] <= ma10[i - 1] and ma5[i] > ma10[i]:
            crosses.append({"type": "金叉", "pair": "MA5/MA10", "days_ago": -i})
        elif ma5[i - 1] >= ma10[i - 1] and ma5[i] < ma10[i]:
            crosses.append({"type": "死叉", "pair": "MA5/MA10", "days_ago": -i})
        if np.isnan(ma20[i]) or np.isnan(ma20[i - 1]):
            continue
        if ma10[i - 1] <= ma20[i - 1] and ma10[i] > ma20[i]:
            crosses.append({"type": "金叉", "pair": "MA10/MA20", "days_ago": -i})
        elif ma10[i - 1] >= ma20[i - 1] and ma10[i] < ma20[i]:
            crosses.append({"type": "死叉", "pair": "MA10/MA20", "days_ago": -i})

    # ----- 支撑位/压力位 -----
    recent_high = high[-20:].max()
    recent_low = low[-20:].min()

    support_levels = [("近20日低点", recent_low)]
    if not np.isnan(cur_ma20) and cur_ma20 < price:
        support_levels.append(("MA20", cur_ma20))
    if cur_ma60 is not None and not np.isnan(cur_ma60) and cur_ma60 < price:
        support_levels.append(("MA60", cur_ma60))
    if not np.isnan(cur_ma10) and cur_ma10 < price:
        support_levels.append(("MA10", cur_ma10))
    support_levels.sort(key=lambda x: x[1], reverse=True)
    support = support_levels[0]

    resist_levels = [("近20日高点", recent_high)]
    if not np.isnan(cur_ma20) and cur_ma20 > price:
        resist_levels.append(("MA20", cur_ma20))
    if cur_ma60 is not None and not np.isnan(cur_ma60) and cur_ma60 > price:
        resist_levels.append(("MA60", cur_ma60))
    if not np.isnan(cur_ma10) and cur_ma10 > price:
        resist_levels.append(("MA10", cur_ma10))
    resist_levels.sort(key=lambda x: x[1])
    resist = resist_levels[0]

    # ----- 量价配合 -----
    vol_avg5 = volume[-6:-1].mean() if n >= 6 else volume[:-1].mean()
    vol_ratio = volume[-1] / vol_avg5 if vol_avg5 > 0 else 1.0

    if vol_ratio > 1.5 and chg_pct > 0:
        vol_price = "放量上涨"
    elif vol_ratio > 1.5 and chg_pct < 0:
        vol_price = "放量下跌"
    elif vol_ratio < 0.8 and chg_pct < 0:
        vol_price = "缩量下跌"
    elif vol_ratio < 0.8 and chg_pct > 0:
        vol_price = "缩量上涨"
    else:
        vol_price = "量价平稳"

    ret5 = (close[-1] / close[-5] - 1) * 100 if n >= 5 else 0.0

    # ----- RSI -----
    rsi_arr = _rsi(close, 14)
    rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else None
    if rsi_val is not None:
        if rsi_val > 70:
            rsi_status = "超买"
        elif rsi_val < 30:
            rsi_status = "超卖"
        else:
            rsi_status = "中性"
    else:
        rsi_status = "N/A"

    # ----- MACD -----
    dif, dea, hist = _macd(close)
    cur_dif = dif[-1] if not np.isnan(dif[-1]) else None
    cur_dea = dea[-1] if not np.isnan(dea[-1]) else None
    cur_hist = hist[-1] if not np.isnan(hist[-1]) else None
    prev_hist = hist[-2] if n >= 2 and not np.isnan(hist[-2]) else None

    macd_cross = None
    if cur_dif is not None and cur_dea is not None and n >= 2:
        prev_dif = dif[-2] if not np.isnan(dif[-2]) else None
        prev_dea = dea[-2] if not np.isnan(dea[-2]) else None
        if prev_dif is not None and prev_dea is not None:
            if prev_dif <= prev_dea and cur_dif > cur_dea:
                macd_cross = "金叉"
            elif prev_dif >= prev_dea and cur_dif < cur_dea:
                macd_cross = "死叉"

    hist_turn = None
    if cur_hist is not None and prev_hist is not None:
        if prev_hist <= 0 and cur_hist > 0:
            hist_turn = "转正"
        elif prev_hist >= 0 and cur_hist < 0:
            hist_turn = "转负"

    # ----- 盈亏 -----
    pnl_pct = None
    if cost is not None and cost > 0:
        pnl_pct = (price - cost) / cost * 100

    # ----- 综合评分 -----
    score = 0.0
    reasons = []

    # 1) 均线排列
    if arrangement == "多头排列":
        score += 3
        reasons.append("均线多头排列，趋势向上")
    elif arrangement == "空头排列":
        score -= 3
        reasons.append("均线空头排列，趋势向下")
    elif arrangement == "均线粘合":
        reasons.append("均线粘合，方向待定")

    # 2) 均线金叉/死叉
    for c in crosses:
        if c["type"] == "金叉":
            bonus = 2 if c["pair"] == "MA10/MA20" else 1
            score += bonus
        else:
            penalty = -2 if c["pair"] == "MA10/MA20" else -1
            score += penalty
        reasons.append(f"{c['days_ago']}天前 {c['pair']} {c['type']}")

    # 3) 价格位置
    if above_ma5 and above_ma10 and above_ma20:
        score += 1
        reasons.append("价格在所有短中期均线之上")
    elif not above_ma5 and not above_ma10 and not above_ma20:
        score -= 1
        reasons.append("价格在所有短中期均线之下")

    # 4) 量价配合
    if vol_price == "放量上涨":
        score += 1
        reasons.append("放量上涨，资金积极")
    elif vol_price == "放量下跌":
        score -= 2
        reasons.append("放量下跌，抛压沉重")
    elif vol_price == "缩量下跌":
        score += 0.5
        reasons.append("缩量下跌，抛压减弱")
    elif vol_price == "缩量上涨":
        score -= 0.5
        reasons.append("缩量上涨，上攻动力不足")

    # 5) RSI
    if rsi_val is not None:
        if rsi_val > 70:
            score -= 1
            reasons.append(f"RSI={rsi_val:.1f} 超买，注意回调风险")
        elif rsi_val < 30:
            score += 1
            reasons.append(f"RSI={rsi_val:.1f} 超卖，抛压或已见底")

    # 6) MACD 柱状图方向
    if hist_turn == "转正":
        score += 1
        reasons.append("MACD柱转正，动能恢复")
    elif hist_turn == "转负":
        score -= 1
        reasons.append("MACD柱转负，动能衰减")

    # 7) MACD DIF/DEA 交叉
    if macd_cross == "金叉":
        score += 1
        reasons.append("MACD金叉，趋势转强")
    elif macd_cross == "死叉":
        score -= 1
        reasons.append("MACD死叉，趋势转弱")

    # ----- 止盈修正 -----
    profit_shift = 0.0
    if pnl_pct is not None and pnl_pct >= PROFIT_THRESHOLD:
        profit_shift = 2.0
        reasons.append(f"盈利 {pnl_pct:.1f}% 超过{PROFIT_THRESHOLD:.0f}%，阈值上移保护利润")

    # ----- 建议映射（5 档）-----
    if score >= (3 + profit_shift):
        advice = "加仓"
    elif score >= (0 + profit_shift):
        advice = "继续持有"
    elif score >= (-2 + profit_shift):
        advice = "减仓1/3"
    elif score >= (-4 + profit_shift):
        advice = "减仓1/2"
    else:
        advice = "清仓"

    return {
        "price": price,
        "cost": cost,
        "pnl_pct": pnl_pct,
        "chg_pct": chg_pct,
        "ret5": ret5,
        "ma5": cur_ma5,
        "ma10": cur_ma10,
        "ma20": cur_ma20,
        "ma60": cur_ma60,
        "arrangement": arrangement,
        "above_ma5": above_ma5,
        "above_ma10": above_ma10,
        "above_ma20": above_ma20,
        "crosses": crosses,
        "support": support,
        "resist": resist,
        "vol_ratio": vol_ratio,
        "vol_price": vol_price,
        "rsi": rsi_val,
        "rsi_status": rsi_status,
        "macd_dif": cur_dif,
        "macd_dea": cur_dea,
        "macd_hist": cur_hist,
        "macd_cross": macd_cross,
        "hist_turn": hist_turn,
        "score": score,
        "profit_shift": profit_shift,
        "force_clear": False,
        "advice": advice,
        "reasons": reasons,
        "date": str(df["date"].iloc[-1]),
    }


# =====================================================================
#  大盘环境分析
# =====================================================================

def analyze_market(idx_df):
    """分析沪深 300 均线状态，返回分析结果和大盘修正分。"""
    if idx_df is None or len(idx_df) < 20:
        return None, 0
    result = analyze_one(idx_df)
    if result is None:
        return None, 0
    if result["arrangement"] == "多头排列":
        market_adj = 1
    elif result["arrangement"] == "空头排列":
        market_adj = -1
    else:
        market_adj = 0
    return result, market_adj


# =====================================================================
#  输出
# =====================================================================

_ADVICE_ICON = {
    "加仓":    "🟢 加仓",
    "继续持有": "🔵 继续持有",
    "减仓1/3": "🟡 减仓1/3",
    "减仓1/2": "🟠 减仓1/2",
    "清仓":    "🔴 清仓",
}


def print_market(mkt, market_adj):
    """输出大盘环境。"""
    print()
    print("=" * 64)
    print("  大盘环境（沪深 300）")
    print("=" * 64)
    if mkt is None:
        print("  数据获取失败")
        return

    print(f"  收盘: {mkt['price']:.2f}  涨跌: {mkt['chg_pct']:+.2f}%  近5日: {mkt['ret5']:+.2f}%")
    ma_str = f"  MA5={mkt['ma5']:.2f}  MA10={mkt['ma10']:.2f}  MA20={mkt['ma20']:.2f}"
    if mkt["ma60"] is not None:
        ma_str += f"  MA60={mkt['ma60']:.2f}"
    print(ma_str)
    print(f"  均线排列: {mkt['arrangement']}")
    if mkt["rsi"] is not None:
        print(f"  RSI(14): {mkt['rsi']:.1f}  {mkt['rsi_status']}")

    adj_label = {1: "+1（偏乐观）", -1: "-1（偏保守）", 0: "0（中性）"}
    print(f"  大盘修正分: {adj_label.get(market_adj, str(market_adj))}")

    if mkt["arrangement"] == "多头排列":
        print("  → 大盘趋势向上，适合操作")
    elif mkt["arrangement"] == "空头排列":
        print("  → 大盘趋势向下，谨慎操作")
    else:
        print("  → 大盘方向不明，控制仓位")
    print()


def print_stock(rq_id, name, r):
    """输出单只股票分析。"""
    label = f"{name} ({rq_id})" if name else rq_id

    print("─" * 64)
    print(f"  {label}")
    print("─" * 64)

    # 价格与盈亏
    line1 = f"  现价: {r['price']:.2f}  涨跌: {r['chg_pct']:+.2f}%  近5日: {r['ret5']:+.2f}%"
    if r["cost"] is not None and r["cost"] > 0:
        line1 += f"  成本: {r['cost']:.2f}"
    print(line1)

    if r["pnl_pct"] is not None:
        pnl_icon = "盈" if r["pnl_pct"] >= 0 else "亏"
        print(f"  持仓盈亏: {r['pnl_pct']:+.2f}% ({pnl_icon})")

    # 均线
    ma_str = f"  MA5={r['ma5']:.2f}  MA10={r['ma10']:.2f}  MA20={r['ma20']:.2f}"
    if r["ma60"] is not None:
        ma_str += f"  MA60={r['ma60']:.2f}"
    print(ma_str)

    pos_parts = []
    for tag, above in [("MA5", r["above_ma5"]), ("MA10", r["above_ma10"]), ("MA20", r["above_ma20"])]:
        pos_parts.append(f"{tag}{'上' if above else '下'}")
    print(f"  价格位置: {', '.join(pos_parts)}")
    print(f"  均线排列: {r['arrangement']}")

    # 均线交叉
    if r["crosses"]:
        for c in r["crosses"]:
            print(f"  均线信号: {c['days_ago']}天前 {c['pair']} {c['type']}")
    else:
        print("  均线信号: 近5日无金叉/死叉")

    # 量价
    print(f"  量比: {r['vol_ratio']:.2f}  {r['vol_price']}")

    # RSI
    if r["rsi"] is not None:
        print(f"  RSI(14): {r['rsi']:.1f}  {r['rsi_status']}")

    # MACD
    if r["macd_dif"] is not None:
        macd_info = f"  MACD: DIF={r['macd_dif']:.3f}  DEA={r['macd_dea']:.3f}  柱={r['macd_hist']:.3f}"
        extras = []
        if r["macd_cross"]:
            extras.append(r["macd_cross"])
        if r["hist_turn"]:
            extras.append(f"柱{r['hist_turn']}")
        if extras:
            macd_info += f"  ({', '.join(extras)})"
        print(macd_info)

    # 支撑/压力
    sup_name, sup_val = r["support"]
    res_name, res_val = r["resist"]
    sup_pct = (r["price"] - sup_val) / r["price"] * 100 if sup_val < r["price"] else 0
    res_pct = (res_val - r["price"]) / r["price"] * 100 if res_val > r["price"] else 0
    print(f"  支撑: {sup_val:.2f} ({sup_name}, -{sup_pct:.1f}%)  "
          f"压力: {res_val:.2f} ({res_name}, +{res_pct:.1f}%)")

    # 建议
    icon = _ADVICE_ICON.get(r["advice"], r["advice"])
    print(f"  ────────")
    print(f"  评分: {r['score']:.1f}", end="")
    if r["profit_shift"] > 0:
        print(f"  (止盈阈值上移 +{r['profit_shift']:.0f})", end="")
    print()
    print(f"  建议: {icon}")
    for reason in r["reasons"]:
        print(f"    · {reason}")
    print()


# =====================================================================
#  命令行解析
# =====================================================================

def parse_holdings(args):
    """解析命令行参数，返回 {rq_id: cost}。"""
    if not args:
        return dict(DEFAULT_HOLDINGS)
    holdings = {}
    for arg in args:
        if ":" in arg:
            code, cost_str = arg.split(":", 1)
            holdings[code] = float(cost_str)
        else:
            holdings[arg] = 0.0
    return holdings


# =====================================================================
#  主流程
# =====================================================================

def main():
    holdings = parse_holdings(sys.argv[1:])

    t0 = time.time()
    print("=" * 64)
    print("  A 股持仓管理 — 均线趋势 + RSI + MACD")
    print("=" * 64)
    print(f"  持仓标的: {len(holdings)} 只")

    print("  获取行情数据...")
    data, idx_df, cache_key = fetch_data(list(holdings.keys()), days=120)
    print(f"  获取成功: {len(data)} 只  最新交易日: {cache_key}")

    mkt, market_adj = analyze_market(idx_df)
    print_market(mkt, market_adj)

    print("=" * 64)
    print("  个股持仓分析")
    print("=" * 64)

    results = {}
    names = {}
    for rq_id, cost in holdings.items():
        df = data.get(rq_id)
        if df is None:
            print(f"\n  {rq_id}: 数据获取失败，跳过")
            continue

        result = analyze_one(df, cost=cost if cost > 0 else None)
        if result is None:
            print(f"\n  {rq_id}: 数据不足，跳过")
            continue

        # 大盘修正
        result["score"] += market_adj
        if not result["force_clear"]:
            ps = result["profit_shift"]
            s = result["score"]
            if s >= (3 + ps):
                result["advice"] = "加仓"
            elif s >= (0 + ps):
                result["advice"] = "继续持有"
            elif s >= (-2 + ps):
                result["advice"] = "减仓1/3"
            elif s >= (-4 + ps):
                result["advice"] = "减仓1/2"
            else:
                result["advice"] = "清仓"
            if market_adj != 0:
                adj_txt = "大盘偏强+1" if market_adj > 0 else "大盘偏弱-1"
                result["reasons"].append(adj_txt)

        bs_code = rq_to_bs(rq_id)
        time.sleep(0.3)
        name = fetch_stock_name(bs_code)
        names[rq_id] = name
        results[rq_id] = result
        print_stock(rq_id, name, result)

    # 汇总表
    print("=" * 64)
    print("  汇总")
    print("=" * 64)
    header = (f"  {'代码':<16s} {'名称':<10s} {'现价':>7s} {'盈亏':>7s} "
              f"{'RSI':>5s} {'均线':>6s} {'评分':>5s} {'建议':>6s}")
    print(header)
    print("  " + "-" * 62)

    for rq_id in holdings:
        if rq_id not in results:
            continue
        r = results[rq_id]
        name = names.get(rq_id, "")
        pnl_str = f"{r['pnl_pct']:+.1f}%" if r["pnl_pct"] is not None else "  N/A"
        rsi_str = f"{r['rsi']:.0f}" if r["rsi"] is not None else "N/A"
        arr_short = r["arrangement"][:4]
        print(f"  {rq_id:<16s} {name:<10s} {r['price']:>7.2f} {pnl_str:>7s} "
              f"{rsi_str:>5s} {arr_short:>6s} {r['score']:>+5.1f} {r['advice']:>6s}")

    print()
    print(f"  总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
