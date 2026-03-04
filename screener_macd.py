# -*- coding: utf-8 -*-
"""
MACD 反转选股脚本（与回测 v3 同步）

用 MACD 指标选出处于底部、即将反转的 A 股：
  1. bundle 本地预筛：基本面过滤 + MACD 底部反转信号（~2 秒）
  2. BaoStock 在线获取最新数据 + MACD 反转精选 Top 10（~30 秒）
     带本地缓存：每天第一次运行从 BaoStock 下载并缓存到 .cn_cache/
     后续运行直接读缓存，结果稳定且速度快

MACD 反转选股逻辑（与 backtest_macd.py v3 一致）：
  - 零轴下方金叉（DIF 从下穿上 DEA，两者均 < 0）
  - 底背离（股价创新低，但 MACD 不创新低 = 空方力竭）
  - 绿柱缩脚（MACD 柱从大负向零收敛 = 卖压减弱）
  - 放量反弹（底部缩量 + 反弹放量 = 资金介入）
  - 短期动量确认（close ≥ EMA5 × 0.99，防止假反转）
  - 相对强度因子（近 5 日跑赢沪深 300 越多越好）

运行：conda activate rqalpha && python screener_macd.py
"""
import os
import pathlib
import sys
import time
import datetime
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from bs_utils import fetch_bars, rq_to_bs, fetch_stock_name

# =====================================================================
#  本地缓存
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
#  EMA / MACD 计算
# =====================================================================

def _ema(data, period):
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def calc_macd(close, fast=12, slow=26, signal=9):
    close = np.array(close, dtype=float)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


# =====================================================================
#  第 1 步：bundle 本地预筛
# =====================================================================

def local_prefilter(top_n=200):
    print("=" * 70)
    print("  第 1 步：bundle 本地预筛（纯基本面，不计算 MACD）")
    print("=" * 70)

    h5_path = os.path.expanduser("~/.rqalpha/bundle/stocks.h5")
    candidates = []

    with h5py.File(h5_path, "r") as f:
        stock_ids = list(f.keys())
        print(f"  A 股总数: {len(stock_ids)}")

        for sid in tqdm(stock_ids, desc="  扫描中", ncols=70, file=sys.stdout):
            d = f[sid]
            if len(d) < 20:
                continue
            r = d[-20:]
            c = r["close"]
            v = r["volume"]
            t = r["total_turnover"]

            if c[-1] == 0 or v[-1] == 0:
                continue
            if c[-1] < 3.0 or c[-1] > 100.0:
                continue
            avg_turnover = t[-10:].mean()
            if avg_turnover < 5e7:
                continue
            if any(v[-3:] == 0):
                continue

            candidates.append({"id": sid, "turnover": avg_turnover})

    candidates.sort(key=lambda x: x["turnover"], reverse=True)
    top = candidates[:top_n]
    print(f"  基本面通过: {len(candidates)} 只 → 按成交额取 Top {min(top_n, len(candidates))}")
    return top


# =====================================================================
#  第 2 步：BaoStock 在线精选
# =====================================================================

def online_score(rq_ids):
    print()
    print("=" * 70)
    print("  第 2 步：BaoStock 在线获取 + MACD 反转精选")
    print("=" * 70)

    today = datetime.date.today()
    start_dt = today - datetime.timedelta(days=50)
    start = start_dt.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    print(f"  数据范围: {start} ~ {today_str}")

    # 沪深 300 指数（同时探测最新交易日作为 cache key）
    idx_ret5 = 0.0
    idx_df = None
    for _retry in range(3):
        try:
            idx_df = fetch_bars("sh.000300", start, today_str)
            if idx_df is not None and len(idx_df) >= 5:
                break
            else:
                print(f"  沪深300 数据不足，重试 {_retry+1}/3...")
                time.sleep(1)
        except Exception:
            print(f"  沪深300 获取失败，重试 {_retry+1}/3...")
            time.sleep(1)
    else:
        print("  沪深300 获取失败，相对强度因子设为 0")

    if idx_df is not None and len(idx_df) >= 5:
        cache_key = str(idx_df["date"].iloc[-1])
        idx_close = idx_df["close"].values
        idx_ret5 = idx_close[-1] / idx_close[-5] - 1
        _write_cache("_IDX_000300", cache_key, idx_df)
        print(f"  沪深300 近5日涨幅: {idx_ret5:+.2%}")
        print(f"  最新交易日: {cache_key}")
    else:
        cache_key = today.strftime("%Y-%m-%d")

    # 检查缓存情况
    cache_dir = _CACHE_DIR / cache_key
    cached_ids = set()
    if cache_dir.exists():
        cached_ids = {p.stem.replace("_", ".") for p in cache_dir.glob("*.csv")
                      if not p.stem.startswith("_")}
    need_download = [rid for rid in rq_ids if rid not in cached_ids]
    have_cache = [rid for rid in rq_ids if rid in cached_ids]
    if have_cache:
        print(f"  本地缓存: {len(have_cache)} 只, 需下载: {len(need_download)} 只")

    # 先从网络下载缺失的数据
    fail = 0
    consecutive_fail = 0
    base_sleep = 0.3
    for rq_id in tqdm(need_download, desc="  下载中", ncols=70, file=sys.stdout,
                      disable=not need_download):
        if consecutive_fail >= 10:
            tqdm.write("  ⚠ 连续失败 10 次，BaoStock 可能限流，等待 30 秒...")
            time.sleep(30)
            consecutive_fail = 0
        elif consecutive_fail >= 5:
            time.sleep(5)
        elif consecutive_fail >= 3:
            time.sleep(2)
        else:
            time.sleep(base_sleep)

        bs_code = rq_to_bs(rq_id)
        df = fetch_bars(bs_code, start, today_str)
        if df is not None and len(df) >= 26:
            _write_cache(rq_id, cache_key, df)
            consecutive_fail = 0
        else:
            fail += 1
            consecutive_fail += 1

    if need_download:
        print(f"  新下载: {len(need_download) - fail} 只, 失败: {fail}")

    # 统一从缓存读取并分析
    candidates = []
    analyze_fail = 0

    for rq_id in tqdm(rq_ids, desc="  分析中", ncols=70, file=sys.stdout):
        df = _read_cache(rq_id, cache_key)
        if df is None or len(df) < 26:
            analyze_fail += 1
            continue

        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values
        latest_date = df["date"].iloc[-1]
        n = len(close)

        if close[-1] == 0:
            continue

        # ===== T+1 安全过滤 =====
        if n >= 2 and close[-1] / close[-2] - 1 > 0.095:
            continue
        day_avg = (high[-1] + low[-1] + close[-1]) / 3
        if close[-1] > day_avg * 1.03:
            continue
        vol_prev_avg = volume[-6:-1].mean() if n >= 6 else volume[:-1].mean()
        if vol_prev_avg > 0 and volume[-1] / vol_prev_avg > 3.0:
            continue
        day_range = high[-1] - low[-1]
        close_pos = (close[-1] - low[-1]) / day_range if day_range > 0 else 0.5
        if close_pos < 0.25:
            continue

        # 计算 MACD
        dif, dea, hist = calc_macd(close)

        # 短期动量确认（与回测 v3 一致）
        ema5 = _ema(close, 5)
        if close[-1] < ema5[-1] * 0.99:
            continue

        # =====================================================
        #  反转信号检测
        # =====================================================

        below_zero_cross = False
        lookback = min(7, n - 1)
        for i in range(-lookback, -1):
            if dif[i] <= dea[i] and dif[i + 1] > dea[i + 1] and dif[i + 1] < 0:
                below_zero_cross = True
                break

        bottom_divergence = False
        if n >= 20:
            price_low_recent = low[-10:].min()
            price_low_prev = low[-20:-10].min()
            dif_low_recent = dif[-10:].min()
            dif_low_prev = dif[-20:-10].min()
            if price_low_recent <= price_low_prev and dif_low_recent > dif_low_prev:
                bottom_divergence = True

        hist_shrinking = False
        shrink_days = 0
        if n >= 5:
            for i in range(-1, -6, -1):
                if hist[i] > hist[i - 1] and hist[i - 1] < 0:
                    shrink_days += 1
                else:
                    break
            hist_shrinking = shrink_days >= 2

        hist_cross_zero = n >= 2 and hist[-2] <= 0 < hist[-1]

        dif_turning = n >= 3 and dif[-1] > dif[-2] and (dif[-2] <= dif[-3] or dif[-1] < 0)

        vol_ratio = 1.0
        vol_expand = False
        if n >= 13:
            vol_recent = volume[-3:].mean()
            vol_prev = volume[-13:-3].mean()
            vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1.0
            vol_expand = vol_ratio > 1.3

        second_cross = False
        if below_zero_cross and n >= 20:
            cross_count = 0
            for i in range(-20, -1):
                if dif[i] <= dea[i] and dif[i + 1] > dea[i + 1] and dif[i + 1] < 0:
                    cross_count += 1
            second_cross = cross_count >= 2

        # ===== 至少满足一个主要反转信号 =====
        has_signal = (
            below_zero_cross
            or bottom_divergence
            or (hist_shrinking and dif_turning)
            or hist_cross_zero
        )
        if not has_signal:
            continue

        # =====================================================
        #  综合打分
        # =====================================================

        # 底背离（最强反转信号）
        s_diverge = 0.25 if bottom_divergence else 0

        # 零轴下方金叉
        s_cross = 0.20 if below_zero_cross else 0

        # 二次金叉（更可靠）
        s_second = 0.10 if second_cross else 0

        # MACD 柱由负转正
        s_zero_cross = 0.12 if hist_cross_zero else 0

        # 绿柱缩脚（缩脚天数越多越好）
        s_shrink = min(shrink_days, 4) / 4 * 0.10

        # DIF 拐头
        s_dif_turn = 0.08 if dif_turning else 0

        # 放量反弹
        s_vol = 0.08 if vol_expand else 0

        # DIF 离零轴的距离（越接近零越好，说明快要翻正）
        dif_pct = dif[-1] / close[-1] * 100
        # dif_pct 通常在 -3 到 0 之间，越接近 0 越好
        s_near_zero = max(0, 1 + dif_pct / 3) * 0.07 if dif_pct < 0 else 0.07

        # T+1 收盘位置
        s_close_pos = close_pos * 0.05

        # 成交额因子：大成交额的信号更可靠（与回测一致）
        amount = df["amount"].values
        avg_amount = amount[-5:].mean()
        s_turnover = min(avg_amount / 3e8, 1.0) * 0.05

        # 相对强度因子：近 5 日涨幅 vs 沪深 300（与回测 v3 一致）
        if n >= 5:
            stk_ret5 = close[-1] / close[-5] - 1
            rs = stk_ret5 - idx_ret5
            s_rs = min(max(rs / 0.05 + 0.5, 0), 1.0) * 0.05
        else:
            s_rs = 0.025

        score = (s_diverge + s_cross + s_second + s_zero_cross + s_shrink
                 + s_dif_turn + s_vol + s_near_zero + s_close_pos + s_turnover + s_rs)

        # 状态标签
        if bottom_divergence and below_zero_cross:
            status = "背离金叉"
        elif bottom_divergence:
            status = "底背离"
        elif second_cross:
            status = "二次金叉"
        elif below_zero_cross:
            status = "零下金叉"
        elif hist_cross_zero:
            status = "柱过零轴"
        elif hist_shrinking:
            status = "绿柱缩脚"
        else:
            status = "拐头中"

        candidates.append({
            "id": rq_id,
            "name": "",
            "price": close[-1],
            "date": latest_date,
            "dif": dif[-1],
            "dea": dea[-1],
            "hist": hist[-1],
            "status": status,
            "diverge": bottom_divergence,
            "vol_ratio": vol_ratio,
            "score": score,
        })

    print(f"  通过筛选: {len(candidates)} 只 (数据缺失: {analyze_fail})")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    if top:
        print(f"  获取 Top {len(top)} 名称...")
        for c in top:
            time.sleep(0.3)
            bs_code = rq_to_bs(c["id"])
            c["name"] = fetch_stock_name(bs_code)

    return top


# =====================================================================
#  输出
# =====================================================================

def output_result(top):
    if not top:
        print("\n  没有符合条件的股票。")
        return

    print()
    print("=" * 112)
    print(f"  MACD 反转选股 TOP 10 (数据截至 {top[0]['date']})")
    print("=" * 112)
    print(
        f"{'#':>3s}  {'代码':<14s} {'名称':<8s} {'现价':>7s} "
        f"{'DIF':>7s} {'DEA':>7s} {'MACD柱':>7s} "
        f"{'状态':<10s} {'背离':>4s} {'量比':>5s} {'得分':>5s}"
    )
    print("-" * 112)

    ids = []
    for i, c in enumerate(top):
        diverge = "是" if c["diverge"] else "否"
        print(
            f"{i+1:>3d}  {c['id']:<14s} {c['name']:<8s} {c['price']:>7.2f} "
            f"{c['dif']:>7.3f} {c['dea']:>7.3f} {c['hist']:>7.3f} "
            f"{c['status']:<10s} {diverge:>4s} {c['vol_ratio']:>5.2f} {c['score']:>5.3f}"
        )
        ids.append(f"{c['id']} {c['name']}")

    print("-" * 112)
    print()
    print("  MACD 反转选股逻辑（买在底部拐点）:")
    print("  ┌─ 必要条件 ─────────────────────────────────────────┐")
    print("  │  非涨停（T+1 安全）                                 │")
    print("  │  至少满足一个主要反转信号                            │")
    print("  ├─ 主要反转信号 ─────────────────────────────────────┤")
    print("  │  ★★ 底背离 = 股价新低 但 MACD 不新低       +0.25  │")
    print("  │  ★  零轴下方金叉 = DIF 穿上 DEA (DIF<0)    +0.20  │")
    print("  │  ★  MACD 柱由负转正 = 空转多                +0.12  │")
    print("  │      二次金叉 = 零下第二次金叉（更可靠）     +0.10  │")
    print("  ├─ 辅助信号 ─────────────────────────────────────────┤")
    print("  │      绿柱缩脚 = 卖压持续减弱                +0.10  │")
    print("  │      DIF 拐头向上                            +0.08  │")
    print("  │      放量反弹 = 资金介入                     +0.08  │")
    print("  │      DIF 接近零轴 = 快要翻多                 +0.07  │")
    print("  ├─ 过滤 + 因子（与回测 v3 一致）──────────────────────┤")
    print("  │  ✓ 短期动量确认: close ≥ EMA5 × 0.99        过滤  │")
    print("  │      相对强度 = 近5日跑赢大盘越多越好        +0.05  │")
    print("  │      成交额因子 = 大成交额信号更可靠         +0.05  │")
    print("  └────────────────────────────────────────────────────┘")
    print()
    print("  信号强度排序: 背离金叉 > 底背离 > 二次金叉 > 零下金叉 > 柱过零轴 > 绿柱缩脚")
    print()
    print("  指标说明:")
    print("    DIF = EMA12 - EMA26（快慢均线差）")
    print("    DEA = DIF 的 9 日 EMA（信号线）")
    print("    MACD柱 = (DIF - DEA) × 2（红绿柱）")
    print("    底背离 = 股价创新低但 MACD 不创新低（空方力竭，反转在即）")
    print("    零轴下方金叉 = DIF 在负值区域穿上 DEA（底部买入信号）")
    print()
    print(f"  推荐关注: {ids}")


if __name__ == "__main__":
    t0 = time.time()
    pool = local_prefilter(200)
    rq_ids = [c["id"] for c in pool]
    top = online_score(rq_ids)
    output_result(top)
    print(f"\n  总耗时: {time.time() - t0:.1f} 秒")
