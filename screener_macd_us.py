# -*- coding: utf-8 -*-
"""
MACD 反转选股脚本 — 美股版

用 MACD 指标选出处于底部、即将反转的美股：
  1. 获取股票池（支持三种范围），Tiingo 逐只获取行情
  2. 基本面过滤（价格、成交额）
  3. MACD 反转信号检测 + 综合打分 → Top 10

股票池范围（通过 --scope 参数选择）：
  index  — 三大指数并集: S&P 500 + 纳斯达克 100 + 道琼斯 30 ~516 只（默认）
  main   — 主板全部：NASDAQ 大中盘 + NYSE ~5300 只
  all    — 全美股：含 NASDAQ 小盘 + AMEX ~7300 只

MACD 反转选股逻辑（与 A 股版 screener_macd.py 一致）：
  - 零轴下方金叉（DIF 从下穿上 DEA，两者均 < 0）
  - 底背离（股价创新低，但 MACD 不创新低 = 空方力竭）
  - 绿柱缩脚（MACD 柱从大负向零收敛 = 卖压减弱）
  - 放量反弹（底部缩量 + 反弹放量 = 资金介入）
  - 短期动量确认（close ≥ EMA5 × 0.99，防止假反转）
  - 相对强度因子（近 5 日跑赢 S&P 500 越多越好）

数据源: Stooq（免费、无需注册）+ 本地缓存
  每天第一次运行从 Stooq 下载并缓存到 .us_cache/ 目录
  后续运行直接读缓存，结果稳定且不消耗 Stooq 配额
运行：
  python screener_macd_us.py                  # 默认三大指数并集
  python screener_macd_us.py --scope main     # 主板全部
  python screener_macd_us.py --scope all      # 全美股
"""
import argparse
import os
import pathlib
import time
import datetime
import urllib.request
import io
import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_env():
    """从脚本同目录的 .env 文件加载环境变量（不覆盖已有值）。"""
    env_path = pathlib.Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_env()


# =====================================================================
#  股票池获取（三种范围）
# =====================================================================

_SP500_CSV_URLS = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

_SCOPE_LABELS = {
    "index": "三大指数并集: S&P 500 + 纳斯达克 100 + 道琼斯 30 (~516 只, ~3 分钟)",
    "main":  "主板全部: NASDAQ 大中盘 + NYSE (~5300 只, ~30 分钟)",
    "all":   "全美股: 含 NASDAQ 小盘 + AMEX (~7300 只, ~40 分钟)",
}

_DJIA_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

_NDX_100_EXTRA = [
    "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "ANSS", "APP",
    "ARM", "ASML", "AZN", "BIIB", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR",
    "CMCSA", "CPRT", "CRWD", "CSGP", "CTAS", "CTSH", "DASH", "DDOG", "DLTR",
    "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GILD", "GOOG",
    "GOOGL", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC",
    "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META",
    "MNST", "MRVL", "MU", "NFLX", "NXPI", "ODFL", "ON", "ORLY", "PANW",
    "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST",
    "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK",
    "VRTX", "WDAY", "WBD", "XEL", "ZS",
]


def _fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def _get_sp500_tickers():
    """在线获取 S&P 500 成分股，失败回退到内置备用列表。"""
    for url in _SP500_CSV_URLS:
        try:
            content = _fetch_url(url)
            lines = content.strip().split("\n")
            if len(lines) < 10:
                continue
            tickers = []
            for line in lines[1:]:
                sym = line.split(",")[0].strip()
                if sym:
                    tickers.append(sym.replace(".", "-"))
            if len(tickers) >= 400:
                return tickers
        except Exception:
            continue
    return _fallback_tickers()


def _get_index_union_tickers():
    """三大指数并集：S&P 500 + 纳斯达克 100 + 道琼斯 30。"""
    pool = set(_get_sp500_tickers())
    pool.update(_DJIA_30)
    pool.update(_NDX_100_EXTRA)
    return sorted(pool)


def _get_nasdaq_ftp_tickers(include_small_cap=False):
    """
    从 NASDAQ FTP 获取全部上市普通股（排除 ETF 和测试股）。
    include_small_cap=False 时只取 NASDAQ Global Select (Q) + Global (G)。
    """
    tickers = set()

    # NASDAQ 上市
    try:
        content = _fetch_url("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
        for line in content.strip().split("\n")[1:]:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, cat, test, etf = parts[0].strip(), parts[2].strip(), parts[3].strip(), parts[6].strip()
            if test != "N" or etf != "N" or not sym or sym.startswith("File"):
                continue
            if not include_small_cap and cat == "S":
                continue
            tickers.add(sym)
    except Exception as e:
        print(f"  ⚠ NASDAQ 列表获取失败: {e}")

    # NYSE / AMEX 等其他交易所
    try:
        content = _fetch_url("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
        for line in content.strip().split("\n")[1:]:
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym, exch, etf, test = parts[0].strip(), parts[2].strip(), parts[4].strip(), parts[6].strip()
            if test != "N" or etf != "N" or not sym or sym.startswith("File"):
                continue
            if not include_small_cap and exch not in ("N",):
                continue
            tickers.add(sym)
    except Exception as e:
        print(f"  ⚠ NYSE/AMEX 列表获取失败: {e}")

    return sorted(tickers)


def get_us_tickers(scope="index"):
    """
    获取美股股票池。
      scope="index" — 三大指数并集: S&P 500 + 纳斯达克 100 + 道琼斯 30 (~516)
      scope="main"  — NASDAQ 大中盘 + NYSE (~5300)
      scope="all"   — 全美股 (~7300)
    """
    print(f"  范围: {_SCOPE_LABELS.get(scope, scope)}")

    if scope == "index":
        tickers = _get_index_union_tickers()
    elif scope == "main":
        tickers = _get_nasdaq_ftp_tickers(include_small_cap=False)
    elif scope == "all":
        tickers = _get_nasdaq_ftp_tickers(include_small_cap=True)
    else:
        print(f"  ⚠ 未知范围 '{scope}'，回退到 index")
        tickers = _get_index_union_tickers()

    if not tickers:
        print("  ⚠ 在线获取失败，使用内置备用列表")
        tickers = _fallback_tickers()

    print(f"  股票池: {len(tickers)} 只")
    return tickers


def _fallback_tickers():
    """备用列表：大型科技 + 金融 + 消费 + 医疗 + 工业 + 半导体 + SaaS。"""
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT",
        "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "PM", "TXN", "UNP",
        "RTX", "AMGN", "LOW", "HON", "IBM", "QCOM", "GE", "CAT", "SPGI",
        "BA", "DE", "AXP", "BKNG", "MDLZ", "ADI", "GILD", "SYK", "ISRG",
        "MMC", "VRTX", "REGN", "ZTS", "BDX", "CI", "SO", "DUK", "CME",
        "CL", "APD", "SHW", "ITW", "PLD", "EQIX", "NSC", "MMM", "EMR",
        "GD", "FDX", "TGT", "ADP", "PYPL", "SQ", "SHOP", "SNOW", "CRWD",
        "DDOG", "NET", "ZS", "PANW", "MRVL", "ON", "ANET", "KLAC", "LRCX",
        "ASML", "SNPS", "CDNS", "NXPI", "MCHP", "SWKS", "FTNT", "WDAY",
        "TEAM", "MNST", "IDXX", "ODFL", "CPRT", "CTAS", "FAST", "PAYX",
    ]


# =====================================================================
#  EMA / MACD 计算（与 A 股版完全一致）
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
#  Stooq 数据获取 + 本地缓存
# =====================================================================

_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".us_cache"


def _stooq_fetch(symbol, start_str, end_str, timeout=10):
    """
    从 Stooq 下载单只股票日线 CSV，返回 DataFrame 或 None。
    symbol 格式: 'aapl.us' / '^spx'
    """
    d1 = start_str.replace("-", "")
    d2 = end_str.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read().decode()
        if "Exceeded" in content or "No data" in content or len(content.strip()) < 30:
            return None
        df = pd.read_csv(io.StringIO(content))
        if len(df) < 5:
            return None
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df if len(df) >= 5 else None
    except Exception:
        return None


def _cache_path(ticker, date_str):
    """返回某只股票某天的缓存文件路径。"""
    return _CACHE_DIR / date_str / f"{ticker}.csv"


def _read_cache(ticker, date_str):
    """从本地缓存读取，返回 DataFrame 或 None。"""
    p = _cache_path(ticker, date_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if len(df) >= 5 else None
    except Exception:
        return None


def _write_cache(ticker, date_str, df):
    """将 DataFrame 写入本地缓存。"""
    p = _cache_path(ticker, date_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def _detect_cache_key(today_str):
    """从本地缓存目录推断最新 cache_key（只要目录有足够 CSV 即视为有效）。"""
    cache_dirs = sorted(_CACHE_DIR.iterdir()) if _CACHE_DIR.exists() else []
    for d in reversed(cache_dirs):
        if d.is_dir() and d.name <= today_str:
            csv_count = sum(1 for _ in d.glob("*.csv"))
            if csv_count >= 5:
                return d.name
    return None


def _find_all_cached(tickers, today_str):
    """跨所有缓存目录搜索已缓存的股票，返回 {ticker: DataFrame}。"""
    if not _CACHE_DIR.exists():
        return {}
    found = {}
    remaining = set(tickers)
    for d in sorted(_CACHE_DIR.iterdir(), reverse=True):
        if not d.is_dir() or d.name > today_str or not remaining:
            continue
        for ticker in list(remaining):
            p = d / f"{ticker}.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if len(df) >= 26:
                        found[ticker] = df
                        remaining.discard(ticker)
                except Exception:
                    pass
    return found


def _detect_latest_trading_day_online(start_str, end_str):
    """用 S&P 500 指数在线探测最新交易日，返回日期字符串或 None。"""
    df = _stooq_fetch("^spx", start_str, end_str)
    if df is not None and len(df) >= 1:
        return str(df["date"].iloc[-1])
    return None


def _purge_old_cache(keep=3):
    """保留最近 keep 个缓存目录，删除更早的。"""
    import shutil
    if not _CACHE_DIR.exists():
        return
    dirs = sorted(d for d in _CACHE_DIR.iterdir() if d.is_dir())
    if len(dirs) <= keep:
        return
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def _data_date(df):
    """从 DataFrame 中提取最新数据日期。"""
    if df is not None and "date" in df.columns and len(df) > 0:
        return str(df["date"].iloc[-1])
    return "N/A"


def fetch_us_bars(tickers, days=100):
    """
    用 Stooq 逐只下载行情数据，带本地缓存。
    优先从本地缓存读取，缺失的从网络下载，网络也失败的从所有历史缓存中搜索。
    返回 {ticker: DataFrame} 字典和 cache_key。
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # 清理旧缓存，只保留最近 3 个目录
    _purge_old_cache(keep=3)

    # 优先从本地缓存推断 cache_key
    local_key = _detect_cache_key(end_str)

    # 尝试在线探测最新交易日
    online_key = _detect_latest_trading_day_online(start_str, end_str)

    if online_key:
        cache_key = online_key
        print(f"  数据范围: {start_str} ~ {end_str}  (最新交易日: {cache_key})")
    elif local_key:
        cache_key = local_key
        print(f"  数据范围: {start_str} ~ {end_str}  (Stooq不可用, 读取缓存: {cache_key})")
    else:
        cache_key = end_str
        print(f"  数据范围: {start_str} ~ {end_str}  (无缓存且Stooq不可用, 用今天日期)")

    results = {}
    from_cache = 0
    from_net = 0
    from_old = 0
    rate_limited = False

    # 搜索当前 cache_key 目录，以及 local_key 目录（可能不同）
    cached_files = set()
    for key in set(filter(None, [cache_key, local_key])):
        cache_dir = _CACHE_DIR / key
        if cache_dir.exists():
            cached_files.update(p.stem for p in cache_dir.glob("*.csv"))

    need_download = [t for t in tickers if t not in cached_files]
    have_cache = [t for t in tickers if t in cached_files]

    print(f"  本地缓存: {len(have_cache)} 只, 需下载: {len(need_download)} 只")

    # 第 1 轮：从缓存读取
    cache_dates = set()
    for ticker in tqdm(have_cache, desc="  读缓存", ncols=70, disable=not have_cache):
        df = _read_cache(ticker, cache_key)
        if df is None and local_key and local_key != cache_key:
            df = _read_cache(ticker, local_key)
        if df is not None and len(df) >= 26:
            results[ticker] = df
            from_cache += 1
            cache_dates.add(_data_date(df))
        else:
            need_download.append(ticker)

    # 第 2 轮：从网络下载缺失的
    consecutive_fail = 0
    net_failed = []
    net_dates = set()
    for ticker in tqdm(need_download, desc="  下载中", ncols=70, disable=not need_download):
        if rate_limited:
            net_failed.append(ticker)
            continue
        stooq_sym = ticker.lower().replace(".", "-") + ".us"
        df = _stooq_fetch(stooq_sym, start_str, end_str)
        if df is not None and len(df) >= 26:
            results[ticker] = df
            _write_cache(ticker, cache_key, df)
            from_net += 1
            net_dates.add(_data_date(df))
            consecutive_fail = 0
        else:
            net_failed.append(ticker)
            consecutive_fail += 1
            if consecutive_fail >= 30 and from_net == 0 and from_cache == 0:
                tqdm.write("  ⚠ 连续 30 只失败，Stooq 可能限流，停止下载")
                tqdm.write("    请稍后重试（Stooq 限额约首次请求后 24 小时重置）")
                rate_limited = True
        time.sleep(0.3)

    # 第 3 轮：网络失败的从历史缓存兜底
    old_dates = set()
    if net_failed:
        old_data = _find_all_cached(net_failed, end_str)
        for ticker, df in old_data.items():
            results[ticker] = df
            from_old += 1
            old_dates.add(_data_date(df))

    still_missing = len(net_failed) - from_old

    # 汇总输出
    print()
    print(f"  ┌─ 数据获取汇总 ──────────────────────────────────┐")
    print(f"  │  本地缓存读取: {from_cache:>4} 只  数据日期: {', '.join(sorted(cache_dates)) or 'N/A':>10s}  │")
    print(f"  │  网络新下载:   {from_net:>4} 只  数据日期: {', '.join(sorted(net_dates)) or 'N/A':>10s}  │")
    if from_old > 0:
        print(f"  │  历史缓存兜底: {from_old:>4} 只  数据日期: {', '.join(sorted(old_dates)) or 'N/A':>10s}  │")
    print(f"  │  获取失败:     {still_missing:>4} 只                            │")
    print(f"  │  ────────────────────────────────────────────── │")
    print(f"  │  合计可用:     {len(results):>4} / {len(tickers)} 只                      │")
    print(f"  └────────────────────────────────────────────────┘")

    return results, cache_key


def fetch_sp500_index(days=20, cache_key=None):
    """获取 S&P 500 指数近期数据，返回 close 数组或 None。"""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    if cache_key is None:
        cache_key = end.strftime("%Y-%m-%d")

    df = _read_cache("_SPX_INDEX", cache_key)
    if df is None:
        df = _stooq_fetch("^spx", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is not None and len(df) >= 5:
            _write_cache("_SPX_INDEX", cache_key, df)

    if df is not None and len(df) >= 5:
        return df["close"].values
    return None


# =====================================================================
#  第 1 步：基本面预筛
# =====================================================================

def prefilter(bars_dict, top_n=200):
    """
    基本面过滤：
      - 最新收盘价 >= $5（排除仙股）
      - 近 10 日平均成交额 >= $10M（流动性）
      - 最近 3 天有成交
    """
    print()
    print("=" * 70)
    print("  第 1 步：基本面预筛")
    print("=" * 70)

    candidates = []
    for ticker, df in bars_dict.items():
        close = df["close"].values
        volume = df["volume"].values

        if close[-1] <= 0 or volume[-1] <= 0:
            continue
        if close[-1] < 5.0:
            continue

        avg_dollar_vol = (close[-10:] * volume[-10:]).mean()
        if avg_dollar_vol < 1e7:
            continue
        if any(volume[-3:] == 0):
            continue

        candidates.append({"ticker": ticker, "dollar_vol": avg_dollar_vol})

    candidates.sort(key=lambda x: x["dollar_vol"], reverse=True)
    top = candidates[:top_n]
    print(f"  基本面通过: {len(candidates)} 只 → 按成交额取 Top {min(top_n, len(candidates))}")
    return [c["ticker"] for c in top]


# =====================================================================
#  第 2 步：MACD 反转精选
# =====================================================================

def score_reversal(bars_dict, tickers, idx_ret5=0.0):
    """
    MACD 反转信号检测 + 综合打分。
    逻辑与 A 股版 online_score() 完全一致，仅去掉涨停板过滤。
    """
    print()
    print("=" * 70)
    print("  第 2 步：MACD 反转信号检测 + 综合打分")
    print("=" * 70)

    candidates = []

    for ticker in tqdm(tickers, desc="  分析中", ncols=70):
        df = bars_dict.get(ticker)
        if df is None or len(df) < 26:
            continue

        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        if close[-1] == 0:
            continue

        # ===== 安全过滤（美股无涨停，但过滤极端情况）=====
        if n >= 2 and abs(close[-1] / close[-2] - 1) > 0.25:
            continue
        day_avg = (high[-1] + low[-1] + close[-1]) / 3
        if close[-1] > day_avg * 1.05:
            continue
        vol_prev_avg = volume[-6:-1].mean() if n >= 6 else volume[:-1].mean()
        if vol_prev_avg > 0 and volume[-1] / vol_prev_avg > 5.0:
            continue
        day_range = high[-1] - low[-1]
        close_pos = (close[-1] - low[-1]) / day_range if day_range > 0 else 0.5
        if close_pos < 0.20:
            continue

        dif, dea, hist = calc_macd(close)

        ema5 = _ema(close, 5)
        if close[-1] < ema5[-1] * 0.99:
            continue

        # =====================================================
        #  反转信号检测（与 A 股版完全一致）
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

        has_signal = (
            below_zero_cross
            or bottom_divergence
            or (hist_shrinking and dif_turning)
            or hist_cross_zero
        )
        if not has_signal:
            continue

        # =====================================================
        #  综合打分（权重与 A 股版一致）
        # =====================================================

        s_diverge = 0.25 if bottom_divergence else 0
        s_cross = 0.20 if below_zero_cross else 0
        s_second = 0.10 if second_cross else 0
        s_zero_cross = 0.12 if hist_cross_zero else 0
        s_shrink = min(shrink_days, 4) / 4 * 0.10
        s_dif_turn = 0.08 if dif_turning else 0
        s_vol = 0.08 if vol_expand else 0

        dif_pct = dif[-1] / close[-1] * 100
        s_near_zero = max(0, 1 + dif_pct / 3) * 0.07 if dif_pct < 0 else 0.07

        s_close_pos = close_pos * 0.05

        dollar_vol = (close[-5:] * volume[-5:]).mean()
        s_turnover = min(dollar_vol / 5e8, 1.0) * 0.05

        if n >= 5:
            stk_ret5 = close[-1] / close[-5] - 1
            rs = stk_ret5 - idx_ret5
            s_rs = min(max(rs / 0.05 + 0.5, 0), 1.0) * 0.05
        else:
            s_rs = 0.025

        score = (s_diverge + s_cross + s_second + s_zero_cross + s_shrink
                 + s_dif_turn + s_vol + s_near_zero + s_close_pos + s_turnover + s_rs)

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

        latest_date = df["date"].iloc[-1]

        candidates.append({
            "ticker": ticker,
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

    print(f"  通过筛选: {len(candidates)} 只")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]


# =====================================================================
#  输出
# =====================================================================

def output_result(top):
    if not top:
        print("\n  没有符合条件的股票。")
        return

    print()
    print("=" * 112)
    print(f"  MACD 反转选股 TOP 10 — 美股 (数据截至 {top[0]['date']})")
    print("=" * 112)
    print(
        f"{'#':>3s}  {'Ticker':<8s} {'Price':>8s} "
        f"{'DIF':>8s} {'DEA':>8s} {'MACD':>8s} "
        f"{'状态':<10s} {'背离':>4s} {'量比':>5s} {'得分':>5s}"
    )
    print("-" * 112)

    tickers = []
    for i, c in enumerate(top):
        diverge = "是" if c["diverge"] else "否"
        print(
            f"{i+1:>3d}  {c['ticker']:<8s} {c['price']:>8.2f} "
            f"{c['dif']:>8.3f} {c['dea']:>8.3f} {c['hist']:>8.3f} "
            f"{c['status']:<10s} {diverge:>4s} {c['vol_ratio']:>5.2f} {c['score']:>5.3f}"
        )
        tickers.append(c["ticker"])

    print("-" * 112)
    print()
    print("  MACD 反转选股逻辑（买在底部拐点）:")
    print("  ┌─ 必要条件 ─────────────────────────────────────────┐")
    print("  │  极端行情过滤（单日涨跌 >25% 排除）                 │")
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
    print("  ├─ 过滤 + 因子 ──────────────────────────────────────┤")
    print("  │  ✓ 短期动量确认: close ≥ EMA5 × 0.99        过滤  │")
    print("  │      相对强度 = 近5日跑赢 S&P 500 越多越好   +0.05  │")
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
    print(f"  推荐关注: {tickers}")


# =====================================================================
#  主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="MACD 反转选股 — 美股版")
    parser.add_argument(
        "--scope", choices=["index", "main", "all"], default="index",
        help="股票池范围: index(三大指数并集~516只), main(主板~5300只), all(全美股~7300只)",
    )
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 70)
    print("  MACD 反转选股 — 美股版")
    print("  数据源: Stooq + 本地缓存")
    print("=" * 70)
    print()

    tickers = get_us_tickers(scope=args.scope)
    print()

    print("=" * 70)
    print("  下载行情数据...")
    print("=" * 70)
    bars_dict, cache_key = fetch_us_bars(tickers, days=100)

    passed = prefilter(bars_dict)

    idx_ret5 = 0.0
    idx_close = fetch_sp500_index(days=20, cache_key=cache_key)
    if idx_close is not None and len(idx_close) >= 5:
        idx_ret5 = idx_close[-1] / idx_close[-5] - 1
        print(f"\n  S&P 500 近 5 日涨幅: {idx_ret5:+.2%}")
    else:
        print("\n  S&P 500 数据获取失败，相对强度因子设为 0")

    top = score_reversal(bars_dict, passed, idx_ret5)
    output_result(top)

    print(f"\n  总耗时: {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
