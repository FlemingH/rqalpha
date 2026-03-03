# -*- coding: utf-8 -*-
"""
美股大盘信号检查脚本

检查 S&P 500 的 MA5 vs MA10，判断明天是否可以操作。
美股收盘后运行（北京时间早上 5:00 以后任何时间都行）。

数据源: Stooq（免费、无限流）
运行：python check_market_us.py
"""
import datetime
import urllib.request
import io
import numpy as np
import pandas as pd


def _stooq_fetch(symbol, start_str, end_str, timeout=10):
    """从 Stooq 下载日线 CSV，返回 DataFrame 或 None。"""
    d1 = start_str.replace("-", "")
    d2 = end_str.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read().decode()
        if "No data" in content or len(content.strip()) < 30:
            return None
        df = pd.read_csv(io.StringIO(content))
        df.columns = [c.strip().lower() for c in df.columns]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df if len(df) >= 5 else None
    except Exception:
        return None


def check():
    today = datetime.date.today()
    start = today - datetime.timedelta(days=30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    # S&P 500 指数
    df = _stooq_fetch("^spx", start_str, end_str)
    if df is None or len(df) < 10:
        # 重试一次
        df = _stooq_fetch("^spx", start_str, end_str)

    if df is None or len(df) < 10:
        n = len(df) if df is not None else 0
        print(f"  数据不足（只有 {n} 天），无法判断")
        return

    closes = df["close"].values.tolist()
    dates = df["date"].values.tolist()

    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    latest_date = dates[-1]
    latest_close = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else latest_close
    today_chg = (latest_close / prev_close - 1) * 100

    print()
    print("=" * 60)
    print("  美股大盘信号检查（S&P 500）")
    print("=" * 60)
    print(f"  数据日期: {latest_date}")
    print(f"  收盘点位: {latest_close:.2f}  今日涨跌: {today_chg:+.2f}%")
    print()
    print(f"  MA5  = {ma5:.2f}  （最近 5 天均价）")
    print(f"  MA10 = {ma10:.2f}  （最近 10 天均价）")
    print()

    if ma5 > ma10:
        diff = (ma5 / ma10 - 1) * 100
        print(f"  ✅ MA5 > MA10（高出 {diff:.2f}%）")
        print(f"  → 大盘趋势向上，明天可以买入")
    else:
        diff = (ma10 / ma5 - 1) * 100
        print(f"  ❌ MA5 < MA10（低了 {diff:.2f}%）")
        print(f"  → 大盘趋势向下，明天空仓不操作")
        if any(p > 0 for p in closes[-3:]):
            print(f"  → 如果有持仓，明天开盘全部卖出")

    print()
    print("  最近 10 天走势:")
    for i in range(-min(10, len(closes)), 0):
        d = dates[i]
        c = closes[i]
        chg = (c / closes[i - 1] - 1) * 100 if i > -len(closes) else 0
        bar = "▲" if chg > 0 else "▼" if chg < 0 else "─"
        print(f"    {d}  {c:>9.2f}  {chg:>+5.2f}%  {bar}")

    print()


if __name__ == "__main__":
    check()
