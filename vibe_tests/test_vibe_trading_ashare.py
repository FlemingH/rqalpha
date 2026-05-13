#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe Vibe-Trading's A-share data, factor, options and news capabilities.

This script intentionally tests Vibe-Trading's built-in tools directly, rather
than asking an LLM agent to summarize them. That makes the result repeatable and
shows which capability is data-driven versus just prompt-driven.

Example:
    python vibe_tests/test_vibe_trading_ashare.py --code 600519.SH --name 贵州茅台
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import pathlib
import tempfile
from typing import Any

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_DIR = pathlib.Path(__file__).resolve().parent / "reports"


def normalize_ashare_code(code: str) -> str:
    """Accept 600519, 600519.SH, 600519.XSHG and normalize to Vibe format."""
    value = code.strip().upper()
    if value.endswith(".XSHG"):
        return value.replace(".XSHG", ".SH")
    if value.endswith(".XSHE"):
        return value.replace(".XSHE", ".SZ")
    if value.endswith((".SH", ".SZ", ".BJ")):
        return value
    if value.startswith(("6", "9")):
        return f"{value}.SH"
    if value.startswith(("0", "2", "3")):
        return f"{value}.SZ"
    if value.startswith(("4", "8")):
        return f"{value}.BJ"
    raise ValueError(f"无法识别 A 股代码: {code}")


def load_vibe_version() -> str:
    try:
        return importlib.metadata.version("vibe-trading-ai")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def fetch_market_data(code: str, start_date: str, end_date: str) -> dict[str, Any]:
    from mcp_server import get_market_data

    raw = get_market_data(
        codes=[code],
        start_date=start_date,
        end_date=end_date,
        source="akshare",
        interval="1D",
    )
    data = json.loads(raw)
    records = data.get(code, [])
    return {
        "status": "ok" if records else "empty",
        "source": "Vibe-Trading MCP get_market_data(source=akshare)",
        "record_count": len(records),
        "last_record": records[-1] if records else None,
        "raw_keys": list(data.keys()),
    }


def option_pricing_probe(market_result: dict[str, Any]) -> dict[str, Any]:
    from src.tools.options_pricing_tool import OptionsPricingTool

    last_record = market_result.get("last_record") or {}
    close = float(last_record.get("close") or 100.0)
    strike = round(close * 1.05, 2)

    tool = OptionsPricingTool()
    raw = tool.execute(
        spot=close,
        strike=strike,
        expiry_days=30,
        risk_free_rate=0.02,
        volatility=0.25,
        option_type="call",
    )
    result = json.loads(raw)
    result["note"] = (
        "这是 Black-Scholes 理论定价能力测试；它不代表 Vibe-Trading "
        "能够获取 A 股真实期权链或真实隐含波动率。"
    )
    return result


def single_stock_factor_probe(code: str, market_result: dict[str, Any]) -> dict[str, Any]:
    """Run Vibe's factor_analysis on one stock and capture the expected limit."""
    from src.tools.factor_analysis_tool import run_factor_analysis

    records = market_result.get("records") or []
    if not records:
        # Reconstruct from last_record-only response is impossible.
        return {
            "status": "skipped",
            "note": "行情数据未保留完整记录，无法构造单股票因子测试。",
        }

    df = pd.DataFrame(records)
    if "date" in df.columns:
        date_col = "date"
    elif "trade_date" in df.columns:
        date_col = "trade_date"
    else:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    factor = df["close"].pct_change(5).to_frame(code)
    future_return = df["close"].pct_change().shift(-1).to_frame(code)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        factor_csv = tmp_path / "single_stock_factor.csv"
        return_csv = tmp_path / "single_stock_return.csv"
        out_dir = tmp_path / "factor_output"
        factor.to_csv(factor_csv)
        future_return.to_csv(return_csv)
        raw = run_factor_analysis(str(factor_csv), str(return_csv), str(out_dir), n_groups=5)
        result = json.loads(raw)

    result["tested_scope"] = "single_stock"
    result["interpretation"] = (
        "Vibe-Trading 的 factor_analysis 是横截面 IC/IR 工具，"
        "每天至少需要多个标的共同计算 rank IC。单只股票不会得到有效 IC，"
        "也不是新闻/事件因子工具。"
    )
    return result


def news_probe(code: str, name: str, max_results: int) -> dict[str, Any]:
    from src.tools.web_search_tool import WebSearchTool

    query = f"{name} {code.split('.')[0]} 最新 新闻 公告 利好 利空"
    tool = WebSearchTool()
    raw = tool.execute(query=query, max_results=max_results)
    result = json.loads(raw)
    result["capability_note"] = (
        "这是 Vibe-Trading 的 web_search 新闻检索能力。它能获取网页新闻/公告摘要，"
        "但不会自动把新闻转成可回测的结构化因子；若要新闻因子，需要另写情绪/事件打分层。"
    )
    return result


def build_conclusion(result: dict[str, Any]) -> dict[str, Any]:
    market_ok = result["market_data"].get("status") == "ok"
    option_ok = result["options_pricing"].get("status") == "ok"
    factor_status = result["factor_analysis_single_stock"].get("status")
    news_ok = result["news_search"].get("status") == "ok"
    news_count = len(result["news_search"].get("results", []))

    return {
        "market_data_available": market_ok,
        "options_math_available": option_ok,
        "single_stock_factor_analysis_available": factor_status == "ok",
        "news_search_available": news_ok and news_count > 0,
        "answer_to_user_question": (
            "Vibe-Trading 可以通过 web_search 获取实时/近实时新闻线索；"
            "但它内置 factor_analysis 不是新闻因子，而是基于多股票横截面 CSV 的 IC/IR 工具。"
            "因此它不能直接提供'实时新闻因子'，只能作为新闻获取入口，后续需要自定义 NLP 打分。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 Vibe-Trading A股能力")
    parser.add_argument("--code", default="600519.SH", help="A股代码，如 600519.SH / 000001.SZ")
    parser.add_argument("--name", default="贵州茅台", help="股票名称，用于新闻搜索")
    parser.add_argument("--days", type=int, default=60, help="行情回看天数")
    parser.add_argument("--news-results", type=int, default=5, help="新闻搜索结果数量")
    args = parser.parse_args()

    code = normalize_ashare_code(args.code)
    end = dt.date.today()
    start = end - dt.timedelta(days=args.days)

    market_data_raw = fetch_market_data(code, start.isoformat(), end.isoformat())
    # Keep full records for the factor probe while showing a compact summary later.
    from mcp_server import get_market_data

    full_market_map = json.loads(
        get_market_data([code], start.isoformat(), end.isoformat(), source="akshare", interval="1D")
    )
    records = full_market_map.get(code, [])
    market_data = dict(market_data_raw)
    market_data["records"] = records

    result = {
        "vibe_trading_version": load_vibe_version(),
        "stock": {"code": code, "name": args.name},
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "market_data": market_data,
        "options_pricing": option_pricing_probe(market_data),
        "factor_analysis_single_stock": single_stock_factor_probe(code, market_data),
        "news_search": news_probe(code, args.name, args.news_results),
    }
    result["conclusion"] = build_conclusion(result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"vibe_trading_ashare_probe_{code.replace('.', '_')}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Vibe-Trading version: {result['vibe_trading_version']}")
    print(f"测试标的: {args.name} {code}")
    print(f"行情数据: {market_data['status']} ({market_data['record_count']} rows)")
    print(f"期权定价: {result['options_pricing'].get('status')}")
    print(f"单股因子分析: {result['factor_analysis_single_stock'].get('status')}")
    print(
        "新闻搜索: "
        f"{result['news_search'].get('status')} "
        f"({len(result['news_search'].get('results', []))} results)"
    )
    print("结论:", result["conclusion"]["answer_to_user_question"])
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()

