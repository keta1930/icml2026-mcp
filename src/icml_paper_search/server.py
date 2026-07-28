"""ICML 2026 论文检索 MCP server（streamable HTTP transport）。

这一层只做三件事：声明 tool schema、把参数转成领域对象、把结果渲染成文本。
业务逻辑在 `service.py`，去重机制在 `sessions.py`。
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .catalog import get_catalog
from .config import get_settings
from .formatting import (
    format_papers,
    format_search_outcome,
    format_session_list,
    format_session_summary,
)
from .models import SearchFilters
from .service import PaperSearchService
from .sessions import SessionNotFoundError

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
ICML 2026 论文库（6341 篇）的语义 + BM25 混合检索。

用检索会话来做主题调研，避免重复条目反复占用 context：
1. `open_search_session` 开一个会话，拿到 session_id；
2. 带着 session_id 反复调 `search_papers`，换措辞、换子方向多角度检索；
   已经返回过的论文不会再出现，所以每轮结果都是净增信息；
3. 某轮返回 0 条说明该 query 的命中全被之前的轮次覆盖了；连续多轮为 0
   即可判定该方向检索完毕，换主题或调 `close_search_session` 收尾拿汇总。

不传 session_id 时是普通无状态检索，不去重。\
"""

_settings = get_settings()
mcp = FastMCP(
    "icml-paper-search",
    instructions=INSTRUCTIONS,
    host=_settings.server_host,
    port=_settings.server_port,
)
_service: PaperSearchService | None = None


def get_service() -> PaperSearchService:
    """进程内单例服务（Milvus 连接与 embedding 模型都在其内部懒加载）。"""
    global _service
    if _service is None:
        _service = PaperSearchService()
    return _service


@mcp.tool(
    title="开启检索会话",
    description=(
        "创建一个论文检索会话。会话会记住已返回过的论文，"
        "后续带同一 session_id 的检索只返回新论文，避免重复条目占用 context。"
        "适合围绕一个主题做多轮、多措辞的穷尽式调研。"
    ),
)
def open_search_session(
    label: Annotated[
        str, Field(default="", description="会话主题标签，便于区分并行的多个调研方向")
    ] = "",
    default_top_k: Annotated[
        int | None,
        Field(default=None, ge=1, description="该会话默认每轮返回的新论文数，默认 10"),
    ] = None,
) -> str:
    """开启一个带去重的检索会话。"""
    summary = get_service().open_session(label=label, default_top_k=default_top_k)
    return (
        format_session_summary(summary, title="已开启检索会话")
        + "\n\n把该 session_id 传给 search_papers 即可开始去重检索。"
    )


@mcp.tool(
    title="检索论文",
    description=(
        "对 ICML 2026 论文做 jina 语义向量 + BM25 加权混合检索。"
        "传入 session_id 时自动过滤该会话已返回过的论文，只给净增结果；"
        "返回 0 条意味着本轮命中全部在此前轮次出现过。"
    ),
)
def search_papers(
    query: Annotated[str, Field(description="检索语句，自然语言描述研究主题或方法")],
    session_id: Annotated[
        str | None,
        Field(default=None, description="检索会话 id；不传则为无状态检索，不去重"),
    ] = None,
    top_k: Annotated[
        int | None, Field(default=None, ge=1, le=50, description="返回的新论文数上限")
    ] = None,
    search_depth: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "该 query 的相关性窗口深度，默认 top_k×4。去重只在窗口内进行，"
                "窗口看完即返回 0；只有确定要在同一措辞上继续深挖时才调大"
            ),
        ),
    ] = None,
    area: Annotated[
        str | None,
        Field(default=None, description="按研究领域过滤，取值见 list_paper_areas"),
    ] = None,
    paper_type: Annotated[
        str | None, Field(default=None, description="按论文类型过滤，如 Poster / Oral")
    ] = None,
    spotlight_only: Annotated[
        bool, Field(default=False, description="只要 spotlight 论文")
    ] = False,
    min_score: Annotated[
        float | None,
        Field(default=None, description="混合得分下限，用于滤掉弱相关结果"),
    ] = None,
    include_abstract: Annotated[
        bool, Field(default=True, description="是否附带摘要摘录；关掉可显著省 token")
    ] = True,
) -> str:
    """混合检索论文，可选按会话去重。"""
    filters = SearchFilters(
        area=area, paper_type=paper_type, spotlight_only=spotlight_only
    )
    try:
        outcome = get_service().search(
            query,
            session_id=session_id,
            top_k=top_k,
            search_depth=search_depth,
            filters=filters,
            min_score=min_score,
        )
    except SessionNotFoundError as exc:
        return f"{exc}。请先调用 open_search_session，或省略 session_id 做无状态检索。"
    except ValueError as exc:
        return f"参数错误：{exc}"
    return format_search_outcome(
        outcome, get_settings(), include_abstract=include_abstract
    )


@mcp.tool(
    title="关闭检索会话",
    description=(
        "关闭检索会话并返回汇总：query 轨迹、去重统计、"
        "以及本次调研累计检索到的全部论文清单。"
    ),
)
def close_search_session(
    session_id: Annotated[str, Field(description="要关闭的会话 id")],
) -> str:
    """关闭会话并交出完整调研汇总。"""
    try:
        summary = get_service().close_session(session_id)
    except SessionNotFoundError as exc:
        return str(exc)
    return format_session_summary(summary, title="已关闭检索会话")


@mcp.tool(
    title="查看会话状态",
    description=(
        "查看检索会话的进展：已跑过的 query、累计论文数、连续空轮数。"
        "不传 session_id 则列出全部存活会话。"
    ),
)
def search_session_status(
    session_id: Annotated[
        str | None, Field(default=None, description="要查看的会话 id；不传则列出全部")
    ] = None,
    include_papers: Annotated[
        bool, Field(default=False, description="是否列出已检索到的论文标题")
    ] = False,
) -> str:
    """查询会话状态或列出存活会话。"""
    service = get_service()
    if session_id is None:
        return format_session_list(service.list_sessions())
    try:
        summary = service.session_status(session_id, include_papers=include_papers)
    except SessionNotFoundError as exc:
        return str(exc)
    return format_session_summary(summary, title="检索会话状态")


@mcp.tool(
    title="按 id 取论文",
    description=(
        "按 OpenReview id 精确取回论文的完整摘要（检索结果里的摘要是截断的）。"
        "传 session_id 时这些论文会登记进会话去重集。"
    ),
)
def get_papers(
    paper_ids: Annotated[list[str], Field(description="OpenReview id 列表")],
    session_id: Annotated[
        str | None, Field(default=None, description="登记到该会话的去重集")
    ] = None,
) -> str:
    """取回指定论文的完整记录。"""
    try:
        papers = get_service().get_papers(paper_ids, session_id=session_id)
    except SessionNotFoundError as exc:
        return str(exc)
    found = {p.paper_id for p in papers}
    missing = [pid for pid in paper_ids if pid not in found]
    return format_papers(papers, get_settings(), missing_ids=missing)


@mcp.tool(
    title="列出可过滤维度",
    description="列出语料的 area 与 paper_type 取值及分布，用于构造 search_papers 的过滤条件。",
)
def list_paper_areas() -> str:
    """语料的可过滤维度取值。"""
    return get_catalog().render()


def main() -> None:
    """以 streamable HTTP 启动 MCP server，监听 `server_host:server_port`。

    长驻进程：embedding 模型与 Milvus 连接在首次 tool 调用时懒加载，之后常驻内存，
    所有 client 共享，省去 stdio 每次 spawn 进程都重新加载的开销。
    """
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "starting streamable-http MCP server on http://%s:%s/mcp",
        settings.server_host,
        settings.server_port,
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
