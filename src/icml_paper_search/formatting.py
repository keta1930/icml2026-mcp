"""把检索结果渲染成紧凑文本。

MCP 的返回值直接进 agent context，所以这里对 token 敏感：只输出决策必需的
字段，摘要按配置截断，并显式给出去重统计让 agent 判断是否该换方向。
"""

from __future__ import annotations

from datetime import datetime

from .config import Settings, get_settings
from .models import Paper, SearchOutcome, SessionSummary


def format_search_outcome(
    outcome: SearchOutcome,
    settings: Settings | None = None,
    *,
    include_abstract: bool = True,
) -> str:
    """渲染一次检索的结果与去重统计。"""
    cfg = settings or get_settings()
    lines = [_outcome_header(outcome)]

    if outcome.is_empty:
        lines.append("")
        lines.append(_empty_explanation(outcome))
    for n, hit in enumerate(outcome.hits, 1):
        lines.append("")
        lines.extend(
            _paper_lines(
                hit.paper,
                index=str(n),
                score=hit.score,
                cfg=cfg,
                include_abstract=include_abstract,
            )
        )

    # 空结果时这条提示最有用，所以两种情况都要给。
    if outcome.truncated:
        lines.append("")
        lines.append(_deeper_hint(outcome, cfg))
    return "\n".join(lines)


def _deeper_hint(outcome: SearchOutcome, cfg: Settings) -> str:
    """窗口填满但新条目不够时，告诉 agent 怎么继续往深处挖。"""
    if outcome.search_depth >= cfg.max_search_depth:
        return (
            f"注：检索窗口已到深度上限 {cfg.max_search_depth}，"
            "该 query 能给的都给了；换个措辞或换子方向。"
        )
    return (
        f"注：该 query 前 {outcome.search_depth} 条相关结果已看完。"
        f"换措辞换方向通常更有效；确要在本方向深挖可把 search_depth 调到 "
        f"{min(outcome.search_depth * 2, cfg.max_search_depth)}。"
    )


def _outcome_header(outcome: SearchOutcome) -> str:
    """一行元信息，交代本轮新增 / 去重 / 会话累计。"""
    parts = [f"query={outcome.query!r}"]
    if desc := outcome.filters.describe():
        parts.append(f"filters=({desc})")
    parts.append(f"新增 {len(outcome.hits)} 条")
    parts.append(f"去重隐藏 {outcome.duplicates_suppressed} 条")
    parts.append(f"窗口 {outcome.candidates_examined}/{outcome.search_depth} 条")
    if outcome.session_id:
        parts.append(f"会话累计 {outcome.session_unique_total} 篇")
        if outcome.consecutive_empty_rounds:
            parts.append(f"连续空轮 {outcome.consecutive_empty_rounds}")
    else:
        parts.append("无会话（未去重）")
    return " | ".join(parts)


def _empty_explanation(outcome: SearchOutcome) -> str:
    """空结果的原因与下一步建议——这是会话机制最重要的信号。"""
    if outcome.candidates_examined == 0:
        # 别把"被条件筛掉了"说成"语料里没有"。
        reasons = []
        if not outcome.filters.is_empty:
            reasons.append("过滤条件可能过窄")
        if outcome.min_score is not None:
            reasons.append(f"min_score={outcome.min_score} 可能过高")
        if reasons:
            return "检索无命中：" + "，".join(reasons) + "，可放宽后重试。"
        return "检索无命中：该 query 在语料中没有匹配结果。"
    if not outcome.session_id:
        return "检索无命中。"
    base = (
        f"零新增：本轮窗口内 {outcome.candidates_examined} 条命中"
        "全部在此前轮次已返回，故不重复展示。"
    )
    if outcome.exhausted:
        return (
            base
            + f" 已连续 {outcome.consecutive_empty_rounds} 轮零新增，"
            "该方向可视为检索完毕，建议换研究主题或结束会话。"
        )
    return base + " 建议换个措辞或换子方向继续；再连续几轮为空即说明该方向已挖尽。"


def format_papers(
    papers: list[Paper],
    settings: Settings | None = None,
    *,
    missing_ids: list[str] | None = None,
) -> str:
    """渲染精确取回的论文（不截断摘要）。"""
    cfg = settings or get_settings()
    lines = [f"取回 {len(papers)} 篇论文"]
    for paper in papers:
        lines.append("")
        lines.extend(
            _paper_lines(
                paper, index=None, score=None, cfg=cfg, include_abstract=True, full=True
            )
        )
    if missing_ids:
        lines.append("")
        lines.append(f"未找到的 id：{', '.join(missing_ids)}")
    return "\n".join(lines)


def _paper_lines(
    paper: Paper,
    *,
    index: str | None,
    score: float | None,
    cfg: Settings,
    include_abstract: bool,
    full: bool = False,
) -> list[str]:
    """单篇论文的紧凑展示块。"""
    prefix = f"[{index}] " if index else ""
    head = f"{prefix}{paper.title}"
    lines = [head]

    meta = [paper.area or "Uncategorized"]
    if paper.subarea:
        meta.append(paper.subarea)
    if paper.paper_type:
        meta.append(paper.paper_type)
    if paper.spotlight:
        meta.append("spotlight")
    if score is not None:
        meta.append(f"score={score:.4f}")
    lines.append("    " + " · ".join(meta))

    if paper.authors and cfg.max_authors_shown:
        shown = list(paper.authors[: cfg.max_authors_shown])
        tail = f" 等 {len(paper.authors)} 人" if len(paper.authors) > len(shown) else ""
        lines.append("    " + ", ".join(shown) + tail)

    links = [f"id={paper.paper_id}", paper.openreview_url]
    if paper.arxiv_url:
        links.append(paper.arxiv_url)
    lines.append("    " + "  ".join(links))

    if include_abstract and paper.abstract:
        text = " ".join(paper.abstract.split())
        limit = 0 if full else cfg.abstract_preview_chars
        if limit and len(text) > limit:
            text = text[:limit].rstrip() + "…"
        if text:
            lines.append(f"    {text}")
    return lines


def format_session_summary(summary: SessionSummary, *, title: str) -> str:
    """渲染会话状态或关闭汇总。"""
    lines = [
        f"{title}：{summary.session_id}",
        f"    label={summary.label or 'untitled'}",
        f"    创建于 {_ts(summary.created_at)}，最后活跃 {_ts(summary.last_active_at)}",
        f"    query {summary.query_count} 次，去重后累计 {summary.unique_papers} 篇论文",
    ]
    if summary.default_top_k:
        lines.append(f"    默认 top_k={summary.default_top_k}")
    if summary.consecutive_empty_rounds:
        state = "已判定检索完毕" if summary.exhausted else "接近挖尽"
        lines.append(f"    连续空轮 {summary.consecutive_empty_rounds}（{state}）")

    if summary.queries:
        lines.append("")
        lines.append("query 轨迹：")
        for n, record in enumerate(summary.queries, 1):
            suffix = f"，filters=({record.filters_desc})" if record.filters_desc else ""
            lines.append(
                f"  {n}. {record.query!r} → 新增 {record.new_hits}，"
                f"去重 {record.duplicates_suppressed}{suffix}"
            )

    if summary.papers:
        lines.append("")
        lines.append(f"已检索论文（{len(summary.papers)} 篇）：")
        for n, (paper_id, paper_title) in enumerate(summary.papers, 1):
            lines.append(f"  {n}. [{paper_id}] {paper_title}")
    return "\n".join(lines)


def format_session_list(summaries: list[SessionSummary]) -> str:
    """渲染存活会话列表。"""
    if not summaries:
        return "当前没有存活的检索会话。"
    lines = [f"存活会话 {len(summaries)} 个："]
    for summary in summaries:
        flag = " [已挖尽]" if summary.exhausted else ""
        lines.append(
            f"  {summary.session_id}  label={summary.label}  "
            f"query={summary.query_count}  papers={summary.unique_papers}"
            f"  最后活跃 {_ts(summary.last_active_at)}{flag}"
        )
    return "\n".join(lines)


def _ts(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
