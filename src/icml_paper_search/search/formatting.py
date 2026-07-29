"""Render search results as compact, token-conscious text for agent context."""

from __future__ import annotations

from datetime import datetime

from ..config import Settings, get_settings
from ..domain.models import Paper, SearchOutcome, SessionSummary


def format_search_outcome(
    outcome: SearchOutcome,
    settings: Settings | None = None,
    *,
    include_abstract: bool = True,
) -> str:
    """Render one search result and its deduplication statistics."""
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

    # The continuation hint is most useful for empty results, but applies to both cases.
    if outcome.truncated:
        lines.append("")
        lines.append(_deeper_hint(outcome, cfg))
    return "\n".join(lines)


def _deeper_hint(outcome: SearchOutcome, cfg: Settings) -> str:
    """Explain how to continue when a full window yields too few new results."""
    if outcome.search_depth >= cfg.max_search_depth:
        return (
            f"Note: the search window has reached the maximum depth of "
            f"{cfg.max_search_depth}. Rephrase the query or change direction."
        )
    return (
        f"Note: the first {outcome.search_depth} relevant results for this query "
        "have been examined. Rephrasing or changing direction is usually more effective; "
        f"to search deeper in this direction, set search_depth to "
        f"{min(outcome.search_depth * 2, cfg.max_search_depth)}."
    )


def _outcome_header(outcome: SearchOutcome) -> str:
    """Render one metadata line with new, duplicate, window, and session counts."""
    parts = [f"query={outcome.query!r}"]
    if desc := outcome.filters.describe():
        parts.append(f"filters=({desc})")
    parts.append(f"new={len(outcome.hits)}")
    parts.append(f"duplicates_suppressed={outcome.duplicates_suppressed}")
    parts.append(f"window={outcome.candidates_examined}/{outcome.search_depth}")
    parts.append("score=RRF[0,1]")
    if outcome.session_id:
        parts.append(f"session_total={outcome.session_unique_total}")
        if outcome.consecutive_empty_rounds:
            parts.append(f"consecutive_empty_rounds={outcome.consecutive_empty_rounds}")
    else:
        parts.append("stateless (deduplication disabled)")
    return " | ".join(parts)


def _empty_explanation(outcome: SearchOutcome) -> str:
    """Explain an empty result and recommend the next action."""
    if outcome.candidates_examined == 0:
        # Do not describe results removed by filters as absent from the corpus.
        reasons = []
        if not outcome.filters.is_empty:
            reasons.append("the filters may be too narrow")
        if outcome.min_score is not None:
            reasons.append(f"min_score={outcome.min_score} may be too high")
        if reasons:
            return "No matches: " + "; ".join(reasons) + ". Relax them and try again."
        return "No matches: this query has no matching results in the corpus."
    if not outcome.session_id:
        return "No matches."
    base = (
        f"No new results: all {outcome.candidates_examined} hits in this window "
        "were returned in earlier rounds and have been suppressed."
    )
    if outcome.exhausted:
        return (
            base
            + f" There have been {outcome.consecutive_empty_rounds} consecutive empty "
            "rounds, so this direction is exhausted. Change the research topic or close "
            "the session."
        )
    return (
        base
        + " Rephrase the query or change direction; further consecutive empty rounds "
        "indicate that this direction is exhausted."
    )


def format_papers(
    papers: list[Paper],
    settings: Settings | None = None,
    *,
    missing_ids: list[str] | None = None,
) -> str:
    """Render exactly retrieved papers without truncating abstracts."""
    cfg = settings or get_settings()
    lines = [f"Retrieved {len(papers)} paper(s)"]
    for paper in papers:
        lines.append("")
        lines.extend(
            _paper_lines(
                paper, index=None, score=None, cfg=cfg, include_abstract=True, full=True
            )
        )
    if missing_ids:
        lines.append("")
        lines.append(f"IDs not found: {', '.join(missing_ids)}")
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
    """Render a compact block for one paper."""
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
    lines.append("    " + " | ".join(meta))

    if paper.authors and cfg.max_authors_shown:
        shown = list(paper.authors[: cfg.max_authors_shown])
        tail = (
            f" ({len(paper.authors)} authors total)"
            if len(paper.authors) > len(shown)
            else ""
        )
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


def format_session_summary(
    summary: SessionSummary, *, title: str, include_session_id: bool = True
) -> str:
    """Render a session status or closure summary."""
    heading = f"{title}: {summary.session_id}" if include_session_id else title
    lines = [
        heading,
        f"    label={summary.label or 'untitled'}",
        f"    created_at={_ts(summary.created_at)}, last_active_at={_ts(summary.last_active_at)}",
        f"    queries={summary.query_count}, unique_papers={summary.unique_papers}",
    ]
    if summary.default_top_k:
        lines.append(f"    default_top_k={summary.default_top_k}")
    if summary.consecutive_empty_rounds:
        state = "exhausted" if summary.exhausted else "nearly exhausted"
        lines.append(
            f"    consecutive_empty_rounds={summary.consecutive_empty_rounds} ({state})"
        )

    if summary.queries:
        lines.append("")
        lines.append("Search rounds:")
        for n, record in enumerate(summary.queries, 1):
            suffix = f", filters=({record.filters_desc})" if record.filters_desc else ""
            lines.append(
                f"  {n}. hash:{record.query_hash[:12]} -> new={record.new_hits}, "
                f"duplicates_suppressed={record.duplicates_suppressed}, "
                f"candidates_examined={record.candidates_examined}{suffix}"
            )
        if summary.audit_truncated:
            lines.append("  ... Search rounds were truncated at the safety limit.")

    if summary.papers:
        lines.append("")
        lines.append(f"Retrieved papers ({len(summary.papers)}):")
        for n, (paper_id, paper_title) in enumerate(summary.papers, 1):
            lines.append(f"  {n}. [{paper_id}] {paper_title}")
        if summary.papers_truncated:
            lines.append("  ... The paper list was truncated at the output limit.")
    return "\n".join(lines)


def _ts(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
