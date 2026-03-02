# -*- coding: utf-8 -*-
"""
BaoStock 工具模块
解决 BaoStock 在某些环境下连续查询挂起的问题：每次查询独立 login/logout。
两个选股脚本共用此模块。
"""
import os
import sys
import time
import signal
import datetime
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


# =====================================================================
#  BaoStock 查询
# =====================================================================

def bs_query_one(bs_code, fields, start, end, timeout=5):
    """单次查询（独立 login/logout + 超时保护），失败返回空列表。"""
    import baostock as bs

    def _alarm(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(timeout)
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start, end_date=end,
            frequency="d", adjustflag="3",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        signal.alarm(0)
        return rows
    except Exception:
        signal.alarm(0)
        # 超时或异常后不再尝试 logout（服务端已异常，logout 只会卡死）
        return []
    finally:
        signal.signal(signal.SIGALRM, old)


def bs_to_rq(bs_code):
    """sh.600000 -> 600000.XSHG"""
    prefix, num = bs_code.split(".")
    exch = "XSHG" if prefix == "sh" else "XSHE"
    return f"{num}.{exch}"


def rq_to_bs(rq_id):
    """600000.XSHG -> sh.600000"""
    code, exch = rq_id.split(".")
    prefix = "sh" if exch == "XSHG" else "sz"
    return f"{prefix}.{code}"


def fetch_bars(bs_code, start, end):
    """获取单只股票行情 DataFrame，失败返回 None。"""
    rows = bs_query_one(bs_code, "date,open,high,low,close,volume,amount", start, end)
    if len(rows) < 5:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df if len(df) >= 5 else None


def fetch_stock_name(bs_code):
    """从 BaoStock 获取股票名称，失败返回代码数字部分。"""
    import baostock as bs

    def _alarm(signum, frame):
        raise TimeoutError()

    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(5)
        bs.login()
        rs_info = bs.query_stock_basic(code=bs_code)
        name = bs_code.split(".")[1]
        while rs_info.next():
            row = rs_info.get_row_data()
            if len(row) > 1 and row[1]:
                name = row[1]
            break
        bs.logout()
        signal.alarm(0)
        return name
    except Exception:
        signal.alarm(0)
        return bs_code.split(".")[1]
    finally:
        signal.signal(signal.SIGALRM, old)


# =====================================================================
#  本地预筛
# =====================================================================

def local_prefilter_momentum(top_n=50):
    """
    用 bundle 本地数据快速预筛动量股 Top N。
    条件：股价 3~100、成交额 >1 亿、月涨幅 -15%~+30%、均线多头优先。
    """
    print("=" * 70)
    print("  第 1 步：bundle 本地快速预筛（动量）")
    print("=" * 70)

    h5_path = os.path.expanduser("~/.rqalpha/bundle/stocks.h5")
    candidates = []

    with h5py.File(h5_path, "r") as f:
        stock_ids = list(f.keys())
        print(f"  A 股总数: {len(stock_ids)}")

        for sid in tqdm(stock_ids, desc="  扫描中", ncols=70, file=sys.stdout):
            d = f[sid]
            if len(d) < 30:
                continue
            r = d[-30:]
            c, v, t = r["close"], r["volume"], r["total_turnover"]
            if c[-1] == 0 or v[-1] == 0:
                continue
            if c[-1] < 3.0 or c[-1] > 100.0:
                continue
            if t[-10:].mean() < 1e8:
                continue
            ret = c[-1] / c[-20] - 1 if c[-20] > 0 else 0
            if ret > 0.30 or ret < -0.15:
                continue
            ma5, ma10, ma20 = c[-5:].mean(), c[-10:].mean(), c[-20:].mean()
            ma = sum([c[-1] > ma5, ma5 > ma10, ma10 > ma20, c[-1] > ma20])
            score = ret * 0.5 + ma * 0.12
            candidates.append({"id": sid, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:top_n]
    print(f"  预筛通过: {len(candidates)} 只 → 取 Top {top_n}")
    return top


def local_prefilter_reversal(top_n=50):
    """
    用 bundle 本地数据快速预筛超跌反转股 Top N。
    条件：5 日跌 2%~15%、处于 20 日低位、有企稳信号。
    """
    print("=" * 70)
    print("  第 1 步：bundle 本地快速预筛（超跌反转）")
    print("=" * 70)

    h5_path = os.path.expanduser("~/.rqalpha/bundle/stocks.h5")
    candidates = []

    with h5py.File(h5_path, "r") as f:
        stock_ids = list(f.keys())
        print(f"  A 股总数: {len(stock_ids)}")

        for sid in tqdm(stock_ids, desc="  扫描中", ncols=70, file=sys.stdout):
            d = f[sid]
            if len(d) < 60:
                continue
            r = d[-60:]
            c, v = r["close"], r["volume"]
            h, l, t = r["high"], r["low"], r["total_turnover"]
            if c[-1] == 0 or v[-1] == 0:
                continue
            if c[-1] < 3.0 or c[-1] > 80.0:
                continue
            if t[-10:].mean() < 50000000:
                continue

            ret_5d = c[-1] / c[-5] - 1 if c[-5] > 0 else 0
            if ret_5d > -0.02 or ret_5d < -0.15:
                continue

            h20, l20 = h[-20:].max(), l[-20:].min()
            if h20 <= l20:
                continue
            pos = (c[-1] - l20) / (h20 - l20)
            if pos > 0.40:
                continue

            ret_60d = c[-1] / c[-60] - 1 if c[-60] > 0 else 0
            if ret_60d < -0.30:
                continue

            drop_f = c[-5] - c[-2]
            drop_l = c[-2] - c[-1]
            slowdown = abs(drop_l) < abs(drop_f) * 0.6 if abs(drop_f) > 0 else False
            vol_shrink = v[-2:].mean() < v[-5:-2].mean() if v[-5:-2].mean() > 0 else False
            if not (slowdown or vol_shrink):
                continue

            score = (-ret_5d) * 0.4 + (1 - pos) * 0.3
            if slowdown:
                score += 0.15
            if vol_shrink:
                score += 0.15
            candidates.append({"id": sid, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:top_n]
    print(f"  预筛通过: {len(candidates)} 只 → 取 Top {top_n}")
    return top


# =====================================================================
#  在线获取行情（带进度条）
# =====================================================================

def fetch_batch(rq_ids, label="获取中"):
    """
    批量获取股票行情，每只独立 login/logout。
    日期范围：上月 10 号 ~ 今天。
    需要多取一些历史数据，因为因子计算需要至少 15 天（均线、RSI 等）。
    返回 {rq_id: DataFrame} 字典。
    """
    today = datetime.date.today()
    # 上月 10 号，确保有足够数据做因子计算
    month_start = today.replace(day=1)
    prev_month = month_start - datetime.timedelta(days=1)  # 上月最后一天
    start = prev_month.replace(day=10)                     # 上月 10 号
    start_str = start.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    print(f"  数据范围: {start_str} ~ {today_str}（含上月历史用于因子计算）")

    results = {}   # {rq_id: DataFrame}
    names = {}     # {rq_id: 股票名称}
    fail = 0

    for rq_id in tqdm(rq_ids, desc=f"  {label}", ncols=70, file=sys.stdout):
        time.sleep(0.3)
        bs_code = rq_to_bs(rq_id)
        df = fetch_bars(bs_code, start_str, today_str)
        if df is not None:
            results[rq_id] = df
            # 获取名称
            time.sleep(0.2)
            names[rq_id] = fetch_stock_name(bs_code)
        else:
            fail += 1

    print(f"  获取成功: {len(results)} 只, 失败: {fail} 只")
    return results, names
