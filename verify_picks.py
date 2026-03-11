# -*- coding: utf-8 -*-
"""
MACD 选股成功率验证脚本（A股 + 美股）

基于 screener_macd.py / screener_macd_us.py 的推荐结果，
在下一个交易日收盘后运行，验证前5名推荐股票的次日涨跌表现。

工作流程：
  1. 某日收盘后运行选股脚本 → 推荐保存到 .cn_picks/ 或 .us_picks/
  2. 下一个交易日收盘后运行本脚本 → 计算推荐股票的次日涨幅
  3. 结果追加到 report.csv，持续积累

运行：
  python verify_picks.py              # A股，验证最近一次
  python verify_picks.py --all        # A股，验证所有未验证的
  python verify_picks.py --stats      # A股，只看累积统计
  python verify_picks.py --us         # 美股，验证最近一次
  python verify_picks.py --us --all   # 美股，验证所有未验证的
"""
import argparse
import csv
import datetime
import io
import json
import pathlib
import time
import urllib.request

import numpy as np
import pandas as pd

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

_TOP_N = 5
_CSV_FIELDS = [
    "verify_date", "pick_date", "rank", "stock_id", "stock_name",
    "rec_price", "verify_price", "return_pct", "result",
]


# =====================================================================
#  市场配置
# =====================================================================

def _market_config(is_us):
    """根据市场返回 (picks_dir, cache_dir, report_file, market_label)。"""
    if is_us:
        picks = _SCRIPT_DIR / ".us_picks"
        cache = _SCRIPT_DIR / ".us_cache"
        return picks, cache, picks / "report.csv", "美股"
    picks = _SCRIPT_DIR / ".cn_picks"
    cache = _SCRIPT_DIR / ".cn_cache"
    return picks, cache, picks / "report.csv", "A股"


# =====================================================================
#  数据获取 — A股（BaoStock）
# =====================================================================

def _get_verify_price_cn(stock_id, pick_date, today_str, cache_dir):
    """A股：获取推荐日之后首个交易日的收盘价。"""
    from bs_utils import fetch_bars, rq_to_bs

    fname = f"{stock_id.replace('.', '_')}.csv"

    if cache_dir.exists():
        for d in sorted(cache_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            p = d / fname
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df["date"] = df["date"].astype(str)
                after = df[df["date"] > pick_date]
                if len(after) > 0:
                    row = after.iloc[0]
                    return float(row["close"]), str(row["date"])
            except Exception:
                continue

    start_dt = datetime.datetime.strptime(pick_date, "%Y-%m-%d") - datetime.timedelta(days=30)
    start_str = start_dt.strftime("%Y-%m-%d")

    bs_code = rq_to_bs(stock_id)
    time.sleep(0.3)
    df = fetch_bars(bs_code, start_str, today_str)
    if df is not None and len(df) >= 1:
        df["date"] = df["date"].astype(str)
        after = df[df["date"] > pick_date]
        if len(after) > 0:
            row = after.iloc[0]
            return float(row["close"]), str(row["date"])

    return None, None


# =====================================================================
#  数据获取 — 美股（Stooq + 缓存）
# =====================================================================

def _stooq_fetch(symbol, start_str, end_str):
    """从 Stooq 下载单只股票日线，返回 DataFrame 或 None。"""
    d1 = start_str.replace("-", "")
    d2 = end_str.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        content = resp.read().decode()
        if "Exceeded" in content or "No data" in content or len(content.strip()) < 30:
            return None
        df = pd.read_csv(io.StringIO(content))
        if len(df) < 2:
            return None
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


def _get_verify_price_us(ticker, pick_date, today_str, cache_dir):
    """美股：获取推荐日之后首个交易日的收盘价。"""
    fname = f"{ticker}.csv"

    if cache_dir.exists():
        for d in sorted(cache_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            p = d / fname
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                df["date"] = df["date"].astype(str)
                after = df[df["date"] > pick_date]
                if len(after) > 0:
                    row = after.iloc[0]
                    return float(row["close"]), str(row["date"])
            except Exception:
                continue

    start_dt = datetime.datetime.strptime(pick_date, "%Y-%m-%d") - datetime.timedelta(days=30)
    start_str = start_dt.strftime("%Y-%m-%d")

    stooq_sym = ticker.lower().replace(".", "-") + ".us"
    time.sleep(0.3)
    df = _stooq_fetch(stooq_sym, start_str, today_str)
    if df is not None and len(df) >= 1:
        df["date"] = df["date"].astype(str)
        after = df[df["date"] > pick_date]
        if len(after) > 0:
            row = after.iloc[0]
            return float(row["close"]), str(row["date"])

    return None, None


# =====================================================================
#  报告读写
# =====================================================================

def _load_report(report_file):
    if not report_file.exists():
        return []
    rows = []
    with open(report_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _save_report(rows, picks_dir, report_file):
    picks_dir.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _verified_pick_dates(report_rows):
    return {r["pick_date"] for r in report_rows}


# =====================================================================
#  单次验证
# =====================================================================

def _verify_one(picks_file, today_str, is_us, cache_dir):
    """验证一个推荐文件，返回结果行列表。"""
    with open(picks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pick_date = data["pick_date"]
    picks = data["picks"][:_TOP_N]

    # US picks 用 "ticker" 字段，CN picks 用 "id" 字段
    id_key = "ticker" if is_us else "id"
    name_key = "ticker" if is_us else "name"

    results = []
    for p in picks:
        stock_id = p[id_key]
        if is_us:
            close, close_date = _get_verify_price_us(stock_id, pick_date, today_str, cache_dir)
        else:
            close, close_date = _get_verify_price_cn(stock_id, pick_date, today_str, cache_dir)

        if close is None or close_date is None:
            continue

        rec_price = p["price"]
        ret = (close - rec_price) / rec_price * 100
        result = "涨" if ret > 0 else ("跌" if ret < 0 else "平")

        results.append({
            "verify_date": close_date,
            "pick_date": pick_date,
            "rank": str(p["rank"]),
            "stock_id": stock_id,
            "stock_name": p.get(name_key, stock_id),
            "rec_price": f"{rec_price:.2f}",
            "verify_price": f"{close:.2f}",
            "return_pct": f"{ret:.2f}",
            "result": result,
        })

    return results


# =====================================================================
#  输出
# =====================================================================

def _print_day_result(results, pick_date):
    if not results:
        return

    verify_date = results[0]["verify_date"]
    up = [r for r in results if r["result"] == "涨"]
    down = [r for r in results if r["result"] == "跌"]
    flat = [r for r in results if r["result"] == "平"]
    win_rate = len(up) / len(results) * 100
    avg_ret = np.mean([float(r["return_pct"]) for r in results])

    print()
    print(f"  ┌─ 验证结果 ({pick_date} 推荐 → {verify_date} 收盘) ─────────┐")
    print(f"  │")
    print(f"  │  成功率: {win_rate:.0f}%  ({len(up)}涨 / {len(down)}跌"
          + (f" / {len(flat)}平" if flat else "") + ")")
    print(f"  │  平均涨幅: {avg_ret:+.2f}%")
    print(f"  │")
    print(f"  │  {'#':>2s}  {'代码':<14s} {'名称':<8s} {'推荐价':>7s} {'验证价':>7s} {'涨幅':>7s}")
    print(f"  │  " + "─" * 50)

    for r in results:
        icon = "▲" if r["result"] == "涨" else ("▼" if r["result"] == "跌" else "─")
        print(f"  │  {r['rank']:>2s}  {r['stock_id']:<14s} {r['stock_name']:<8s} "
              f"{float(r['rec_price']):>7.2f} {float(r['verify_price']):>7.2f} "
              f"{float(r['return_pct']):>+6.2f}% {icon}")

    if up:
        names = ", ".join(r["stock_name"] or r["stock_id"] for r in up)
        print(f"  │")
        print(f"  │  涨: {names}")
    if down:
        names = ", ".join(r["stock_name"] or r["stock_id"] for r in down)
        print(f"  │  跌: {names}")

    print(f"  └──────────────────────────────────────────────────┘")


def _print_cumulative(all_rows):
    if not all_rows:
        print("  暂无验证记录。")
        return

    from collections import defaultdict
    by_date = defaultdict(list)
    for r in all_rows:
        by_date[r["verify_date"]].append(r)

    total = len(all_rows)
    total_up = sum(1 for r in all_rows if r["result"] == "涨")
    total_down = sum(1 for r in all_rows if r["result"] == "跌")
    total_rate = total_up / total * 100 if total > 0 else 0
    total_avg = np.mean([float(r["return_pct"]) for r in all_rows])

    print()
    print("=" * 64)
    print("  累积统计")
    print("=" * 64)
    print(f"  验证天数: {len(by_date)} 天, 共 {total} 只")
    print(f"  总成功率: {total_rate:.1f}% ({total_up}涨 / {total_down}跌)")
    print(f"  总平均涨幅: {total_avg:+.2f}%")
    print()

    print(f"  {'验证日':>10s}  {'推荐日':>10s}  {'成功率':>6s}  {'平均涨幅':>8s}  {'涨':>2s}  {'跌':>2s}  涨的股票")
    print("  " + "─" * 62)

    for vdate in sorted(by_date.keys()):
        rows = by_date[vdate]
        up_r = [r for r in rows if r["result"] == "涨"]
        dn_r = [r for r in rows if r["result"] == "跌"]
        rate = len(up_r) / len(rows) * 100
        avg = np.mean([float(r["return_pct"]) for r in rows])
        pdate = rows[0]["pick_date"]
        up_names = ", ".join(r["stock_name"] or r["stock_id"] for r in up_r) if up_r else "—"
        print(f"  {vdate:>10s}  {pdate:>10s}  {rate:>5.0f}%  {avg:>+7.2f}%  {len(up_r):>2d}  {len(dn_r):>2d}  {up_names}")

    print()


# =====================================================================
#  主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MACD 选股成功率验证（A股/美股）")
    parser.add_argument("--us", action="store_true", help="验证美股推荐（默认验证A股）")
    parser.add_argument("--all", action="store_true", help="验证所有未验证的推荐")
    parser.add_argument("--stats", action="store_true", help="只显示累积统计")
    args = parser.parse_args()

    is_us = args.us
    picks_dir, cache_dir, report_file, label = _market_config(is_us)

    print()
    print("=" * 64)
    print(f"  {label} MACD 选股验证")
    print("=" * 64)

    report_rows = _load_report(report_file)

    if args.stats:
        _print_cumulative(report_rows)
        return

    if not picks_dir.exists():
        print(f"  未找到推荐记录目录 ({picks_dir.name}/)。")
        screener = "screener_macd_us.py" if is_us else "screener_macd.py"
        print(f"  请先运行 {screener} 产生推荐。")
        return

    picks_files = sorted(picks_dir.glob("*.json"))
    if not picks_files:
        screener = "screener_macd_us.py" if is_us else "screener_macd.py"
        print(f"  未找到推荐记录。请先运行 {screener}。")
        return

    verified = _verified_pick_dates(report_rows)
    unverified = [f for f in picks_files if f.stem not in verified]

    if not unverified:
        print("  所有推荐已验证完毕。")
        _print_cumulative(report_rows)
        return

    if not args.all:
        unverified = unverified[-1:]

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    new_total = 0

    for pf in unverified:
        pick_date = pf.stem
        print(f"\n  验证: {pick_date} 的推荐 (前 {_TOP_N} 名)...")

        results = _verify_one(pf, today_str, is_us, cache_dir)
        if not results:
            print(f"  推荐日 {pick_date} 之后尚无新交易数据，请在下个交易日收盘后再试。")
            continue

        report_rows = [r for r in report_rows if r["pick_date"] != pick_date]
        report_rows.extend(results)
        new_total += len(results)

        _print_day_result(results, pick_date)

    if new_total > 0:
        _save_report(report_rows, picks_dir, report_file)
        print(f"\n  报告已更新: {report_file}")

    if len(report_rows) > _TOP_N:
        _print_cumulative(report_rows)


if __name__ == "__main__":
    main()
