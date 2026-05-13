# 实时新闻因子

这个模块用 Vibe-Trading 的 `web_search` 作为新闻入口，把搜索到的新闻、公告、研报摘要转换成可排序的 `news_score`，用于提前发现热点线索。

## 使用

```bash
python news_factor/realtime_news_factor.py --codes 000063.SZ:中兴通讯 600519.SH:贵州茅台
```

也可以直接扫描最近一次 `.cn_picks` 推荐：

```bash
python news_factor/realtime_news_factor.py --from-picks --top-n 10
```

## 输出

结果保存到 `news_factor/reports/`：

- `news_factor_*.csv`：排序后的因子表
- `news_factor_*.json`：含新闻明细、命中关键词、题材标签

## 因子含义

- `news_score`：综合分，范围约 `-100` 到 `100`
- `heat_score`：搜索结果数量带来的热度分
- `sentiment_score`：利好/利空关键词净分
- `theme_score`：热点题材命中分
- `risk_score`：负面风险词绝对分

注意：这是“热点线索因子”，不是投资建议。高分股票还要结合价格、成交量、T+1 风险和基本面过滤。

