#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Realtime news factor for A-share hotspot discovery.

The factor uses Vibe-Trading's `web_search` tool as the news inlet, then applies
transparent keyword scoring. It is intentionally deterministic and does not need
an LLM API key.

Examples:
    python news_factor/realtime_news_factor.py --codes 000063.SZ:中兴通讯 600519.SH:贵州茅台
    python news_factor/realtime_news_factor.py --from-picks --top-n 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_DIR = pathlib.Path(__file__).resolve().parent / "reports"
PICKS_DIR = PROJECT_ROOT / ".cn_picks"


HOT_THEMES = {
    "人工智能": ["人工智能", "AI", "大模型", "算力", "智能体", "AI应用"],
    "算力光模块": ["算力", "光模块", "CPO", "液冷", "数据中心", "服务器"],
    "机器人": ["机器人", "人形机器人", "减速器", "关节模组", "伺服", "灵巧手"],
    "半导体芯片": ["半导体", "芯片", "存储", "先进封装", "国产替代", "晶圆"],
    "低空经济": ["低空经济", "eVTOL", "飞行汽车", "无人机", "通航"],
    "固态电池": ["固态电池", "锂电", "电池", "负极", "电解液", "储能"],
    "新能源车": ["新能源车", "智能驾驶", "车路云", "汽车零部件", "特斯拉"],
    "创新药": ["创新药", "医药", "临床", "获批", "新药", "BD交易"],
    "军工": ["军工", "航天", "卫星", "无人装备", "低轨"],
    "消费复苏": ["消费", "白酒", "旅游", "免税", "食品饮料", "涨价"],
    "并购重组": ["并购", "重组", "收购", "资产注入", "控制权变更"],
}


POSITIVE_KEYWORDS = {
    "涨停": 12,
    "大涨": 10,
    "突破": 8,
    "创历史新高": 12,
    "订单": 9,
    "中标": 9,
    "签约": 8,
    "合作": 6,
    "战略合作": 8,
    "获批": 8,
    "回购": 6,
    "增持": 7,
    "业绩预增": 10,
    "净利润增长": 8,
    "机构调研": 5,
    "政策支持": 8,
    "国产替代": 7,
    "新产品": 5,
}


NEGATIVE_KEYWORDS = {
    "减持": -10,
    "亏损": -10,
    "业绩下滑": -9,
    "立案": -15,
    "监管": -8,
    "问询函": -8,
    "处罚": -12,
    "风险提示": -8,
    "澄清": -5,
    "终止": -8,
    "跌停": -12,
    "大跌": -10,
    "解禁": -6,
    "诉讼": -8,
}


RECENCY_KEYWORDS = {
    "今日": 4,
    "今天": 4,
    "盘中": 4,
    "早盘": 3,
    "午后": 3,
    "最新": 3,
    "公告": 3,
    "快讯": 4,
}


SOURCE_WEIGHTS = {
    "eastmoney.com": 5,
    "10jqka.com.cn": 5,
    "sina.com.cn": 4,
    "stcn.com": 5,
    "cnstock.com": 5,
    "cs.com.cn": 5,
    "jrj.com.cn": 3,
    "xueqiu.com": 2,
    "sdyanbao.com": 2,
}

BLOCKED_SOURCE_PARTS = [
    "cloudfront.net",
    "51视频",
    "porn",
    "casino",
]

GENERIC_PAGE_KEYWORDS = [
    "行情_走势",
    "行情走势",
    "个股资金流向",
    "最新价格_行情",
    "股票价格_最新资讯_行情走势_历史数据",
]


@dataclass
class StockTarget:
    code: str
    name: str


@dataclass
class NewsItem:
    title: str
    url: str
    snippet: str
    source: str
    score: float
    matched_positive: list[str] = field(default_factory=list)
    matched_negative: list[str] = field(default_factory=list)
    matched_themes: list[str] = field(default_factory=list)


@dataclass
class NewsFactor:
    code: str
    name: str
    news_score: float
    heat_score: float
    sentiment_score: float
    theme_score: float
    risk_score: float
    source_score: float
    result_count: int
    top_themes: list[str]
    top_titles: list[str]
    query: str
    items: list[NewsItem]


def normalize_ashare_code(code: str) -> str:
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


def load_targets_from_picks(top_n: int) -> list[StockTarget]:
    files = sorted(PICKS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"未找到 .cn_picks 推荐文件: {PICKS_DIR}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    targets = []
    for item in payload.get("picks", [])[:top_n]:
        targets.append(
            StockTarget(
                code=normalize_ashare_code(str(item["id"])),
                name=str(item.get("name") or item["id"]),
            )
        )
    return targets


def parse_targets(values: list[str]) -> list[StockTarget]:
    targets = []
    for value in values:
        if ":" in value:
            code, name = value.split(":", 1)
        elif "," in value:
            code, name = value.split(",", 1)
        else:
            code, name = value, value
        targets.append(StockTarget(code=normalize_ashare_code(code), name=name.strip()))
    return targets


def _search_news(query: str, max_results: int) -> dict[str, Any]:
    from src.tools.web_search_tool import WebSearchTool

    tool = WebSearchTool()
    raw = tool.execute(query=query, max_results=max_results)
    return json.loads(raw)


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _source_weight(source: str) -> float:
    for domain, weight in SOURCE_WEIGHTS.items():
        if domain in source:
            return float(weight)
    return 1.0


def _match_weighted(text: str, keywords: dict[str, int]) -> tuple[float, list[str]]:
    score = 0.0
    matched = []
    for keyword, weight in keywords.items():
        if keyword.lower() in text.lower():
            score += weight
            matched.append(keyword)
    return score, matched


def _has_stock_mention(text: str, target: StockTarget) -> bool:
    bare_code = target.code.split(".")[0]
    return target.name in text or bare_code in text


def _match_stock_related_themes(title: str, snippet: str, target: StockTarget) -> tuple[float, list[str]]:
    """Only count themes that appear close to the stock name/code.

    Search snippets often contain site navigation such as "AI 大模型 / 机器人",
    which is not necessarily related to the article. This helper avoids giving
    theme points unless the stock and theme share a title or text segment.
    """
    bare_code = target.code.split(".")[0]
    segments = [title]
    segments.extend(re.split(r"[。；;，,\n|]", snippet))

    matched_themes = []
    score = 0.0
    for theme, keywords in HOT_THEMES.items():
        hits = []
        for segment in segments:
            if target.name not in segment and bare_code not in segment:
                continue
            hits.extend([kw for kw in keywords if kw.lower() in segment.lower()])
        if hits:
            matched_themes.append(theme)
            score += 8 + min(len(set(hits)), 3) * 2
    return score, matched_themes


def _dedupe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for item in results:
        url = item.get("url") or item.get("href") or ""
        title = item.get("title") or ""
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _looks_like_noise(title: str, url: str, snippet: str) -> bool:
    text = f"{title} {url} {snippet}".lower()
    if any(part.lower() in text for part in BLOCKED_SOURCE_PARTS):
        return True
    return False


def _looks_like_generic_quote_page(title: str, snippet: str) -> bool:
    text = f"{title} {snippet}"
    return any(keyword in text for keyword in GENERIC_PAGE_KEYWORDS)


def score_stock_news(target: StockTarget, max_results: int) -> NewsFactor:
    bare_code = target.code.split(".")[0]
    current_year = dt.date.today().year
    queries = [
        f"{target.name} {bare_code} {current_year} 今日 近一周 最新 新闻 公告 利好 利空",
        f"{target.name} {bare_code} {current_year} 热点 题材 政策 订单 AI 算力 芯片 机器人",
    ]
    raw_results = []
    per_query = max(1, math.ceil(max_results / len(queries)))
    for query in queries:
        payload = _search_news(query, max_results=per_query)
        if payload.get("status") == "ok":
            raw_results.extend(payload.get("results", []))
    results = _dedupe_results(raw_results)[:max_results]

    items: list[NewsItem] = []
    sentiment_score = 0.0
    theme_score = 0.0
    risk_score = 0.0
    source_score = 0.0
    all_themes: list[str] = []

    for item in results:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        if _looks_like_noise(title, url, snippet):
            continue
        source = _source_from_url(url)
        text = f"{title} {snippet}"
        if not _has_stock_mention(text, target):
            continue

        pos_score, positives = _match_weighted(text, POSITIVE_KEYWORDS)
        neg_score, negatives = _match_weighted(text, NEGATIVE_KEYWORDS)
        recent_score, _recency = _match_weighted(text, RECENCY_KEYWORDS)
        one_theme_score, themes = _match_stock_related_themes(title, snippet, target)
        one_source_score = _source_weight(source)
        if _looks_like_generic_quote_page(title, snippet):
            # 普通行情页可能混入全站新闻流，不代表该股票的实时热点。
            continue

        item_score = pos_score + neg_score + recent_score + one_theme_score + one_source_score
        sentiment_score += pos_score + neg_score
        theme_score += one_theme_score
        risk_score += abs(neg_score)
        source_score += one_source_score
        all_themes.extend(themes)

        items.append(
            NewsItem(
                title=title,
                url=url,
                snippet=snippet,
                source=source,
                score=round(item_score, 2),
                matched_positive=positives,
                matched_negative=negatives,
                matched_themes=themes,
            )
        )

    heat_score = min(30.0, math.log1p(len(items)) * 12)
    raw_score = heat_score + sentiment_score + theme_score + source_score - risk_score * 0.6
    news_score = max(-100.0, min(100.0, raw_score))
    theme_rank = pd.Series(all_themes).value_counts() if all_themes else pd.Series(dtype=int)

    return NewsFactor(
        code=target.code,
        name=target.name,
        news_score=round(news_score, 2),
        heat_score=round(heat_score, 2),
        sentiment_score=round(sentiment_score, 2),
        theme_score=round(theme_score, 2),
        risk_score=round(risk_score, 2),
        source_score=round(source_score, 2),
        result_count=len(items),
        top_themes=theme_rank.head(5).index.tolist(),
        top_titles=[item.title for item in items[:3]],
        query=" || ".join(queries),
        items=items,
    )


def save_outputs(factors: list[NewsFactor], output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"news_factor_{stamp}.json"
    csv_path = output_dir / f"news_factor_{stamp}.csv"

    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "Vibe-Trading web_search + deterministic keyword scoring",
        "factor_definition": {
            "news_score": "综合新闻热度、情绪、题材、来源可信度并扣除风险词后的得分，范围约 -100 到 100",
            "heat_score": "搜索结果数量带来的热度分",
            "sentiment_score": "利好/利空关键词净分",
            "theme_score": "热点题材命中分",
            "risk_score": "负面风险关键词绝对分",
        },
        "factors": [
            {
                **asdict(factor),
                "items": [asdict(item) for item in factor.items],
            }
            for factor in factors
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for factor in factors:
        rows.append(
            {
                "code": factor.code,
                "name": factor.name,
                "news_score": factor.news_score,
                "heat_score": factor.heat_score,
                "sentiment_score": factor.sentiment_score,
                "theme_score": factor.theme_score,
                "risk_score": factor.risk_score,
                "source_score": factor.source_score,
                "result_count": factor.result_count,
                "top_themes": "|".join(factor.top_themes),
                "top_titles": " / ".join(factor.top_titles),
                "query": factor.query,
            }
        )
    pd.DataFrame(rows).sort_values("news_score", ascending=False).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="A股实时新闻因子：用 Vibe-Trading 搜索新闻并打分")
    parser.add_argument("--codes", nargs="*", help="代码:名称，例如 000063.SZ:中兴通讯")
    parser.add_argument("--from-picks", action="store_true", help="从最新 .cn_picks 推荐读取股票")
    parser.add_argument("--top-n", type=int, default=10, help="从 picks 读取的数量")
    parser.add_argument("--max-results", type=int, default=6, help="每只股票搜索结果数，最大 10")
    parser.add_argument("--output-dir", default=str(REPORT_DIR), help="输出目录")
    args = parser.parse_args()

    if args.from_picks:
        targets = load_targets_from_picks(args.top_n)
    elif args.codes:
        targets = parse_targets(args.codes)
    else:
        raise SystemExit("请提供 --codes 或 --from-picks")

    factors = []
    for target in targets:
        print(f"搜索新闻: {target.code} {target.name}")
        factor = score_stock_news(target, max_results=min(args.max_results, 10))
        factors.append(factor)

    factors.sort(key=lambda item: item.news_score, reverse=True)
    json_path, csv_path = save_outputs(factors, pathlib.Path(args.output_dir))

    print()
    print("实时新闻因子排名:")
    for i, factor in enumerate(factors, start=1):
        themes = ",".join(factor.top_themes) or "-"
        print(
            f"{i:>2d}. {factor.code:<10s} {factor.name:<8s} "
            f"score={factor.news_score:>6.2f} "
            f"heat={factor.heat_score:>5.1f} "
            f"sent={factor.sentiment_score:>5.1f} "
            f"theme={factor.theme_score:>5.1f} "
            f"risk={factor.risk_score:>5.1f} "
            f"themes={themes}"
        )

    print()
    print(f"JSON报告: {json_path}")
    print(f"CSV报告:  {csv_path}")
    print("说明: 这是热点线索因子，不是投资建议；高分表示新闻/题材热度高，仍需结合价格、成交量和风险过滤。")


if __name__ == "__main__":
    main()

