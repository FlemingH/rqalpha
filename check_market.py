# -*- coding: utf-8 -*-
"""
大盘信号检查脚本

检查沪深 300 的 MA5 vs MA10，判断明天是否可以操作。
每天收盘后运行（17:00 以后任何时间都行）。

运行：python check_market.py
"""
import signal
import sys
import datetime
import numpy as np


def check():
    import baostock as bs

    def _alarm(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _alarm)

    try:
        signal.alarm(10)
        bs.login()

        today = datetime.date.today().strftime("%Y-%m-%d")
        start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

        rs = bs.query_history_k_data_plus(
            "sh.000300", "date,close",
            start_date=start, end_date=today,
            frequency="d", adjustflag="3",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        signal.alarm(0)
        bs.logout()
    except Exception as e:
        signal.alarm(0)
        try:
            bs.logout()
        except:
            pass
        print(f"  获取数据失败: {e}")
        return
    finally:
        signal.signal(signal.SIGALRM, old)

    if len(rows) < 10:
        print(f"  数据不足（只有 {len(rows)} 天），无法判断")
        return

    closes = [float(r[1]) for r in rows]
    dates = [r[0] for r in rows]

    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    latest_date = dates[-1]
    latest_close = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else latest_close
    today_chg = (latest_close / prev_close - 1) * 100

    print()
    print("=" * 60)
    print("  大盘信号检查（沪深 300）")
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
            print(f"  → 如果有持仓，明天 9:30 开盘全部卖出")

    print()
    print("  最近 10 天走势:")
    for i in range(-min(10, len(closes)), 0):
        d = dates[i]
        c = closes[i]
        chg = (c / closes[i-1] - 1) * 100 if i > -len(closes) else 0
        bar = "▲" if chg > 0 else "▼" if chg < 0 else "─"
        print(f"    {d}  {c:>8.2f}  {chg:>+5.2f}%  {bar}")

    print()


if __name__ == "__main__":
    check()
