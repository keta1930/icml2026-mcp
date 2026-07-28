# icml-paper-search-mcp

面向 ICML 2026 论文库（6341 篇）的 MCP server，提供 **Jina 语义向量 + BM25 加权混合检索**，并通过检索会话避免多轮调研反复返回同一批论文。

检索会话会记住已经返回过的论文。agent 可以围绕同一主题持续更换 query，每轮只接收净增结果；连续多轮没有新增结果时，即可判断当前方向基本挖尽。不创建会话也可以执行普通的无状态检索。

## 功能

- 语义向量与 BM25 加权混合排序
- 会话级去重、query 轨迹和累计论文汇总
- 按 `area`、`paper_type`、spotlight 和最低得分过滤
- 按 OpenReview id 精确取回完整摘要
- Streamable HTTP MCP transport
- 建库、检索和语料信息 CLI

## 运行要求

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可访问的 Milvus 实例
- 本地模型目录或可下载的 sentence-transformers 兼容模型；默认配置使用 1024 维 `jina-embeddings-v5-text-small`

默认 Milvus 地址、模型路径和 CUDA 设备是本机开发配置。首次运行前请按实际环境修改 `.env`。

## 快速开始

在仓库根目录执行：

```bash
uv sync
cp .env.example .env

# 编辑 .env 后检查配置与本地语料
uv run icml-papers info

# 首次使用时建立 Milvus collection
uv run icml-papers build
```

`build` 默认会删除同名 collection 并全量重建。若 Milvus 中已有行数为 6341 的 `icml2026repro_papers`，可以跳过；若只想在 collection 已存在时报错，使用 `uv run icml-papers build --keep-existing`。

## 启动并接入 MCP client

server 使用 Streamable HTTP transport，是需要单独启动的长驻进程：

```bash
uv run icml-paper-search-mcp
```

默认端点为 `http://127.0.0.1:20441/mcp`。监听地址和端口分别由 `ICML_PAPERS_SERVER_HOST`、`ICML_PAPERS_SERVER_PORT` 控制。

注册到 Claude Code 项目配置：

```bash
claude mcp add --transport http icml-papers --scope project http://127.0.0.1:20441/mcp
```

等价的 `.mcp.json` 配置如下：

```json
{
  "mcpServers": {
    "icml-papers": {
      "type": "http",
      "url": "http://127.0.0.1:20441/mcp"
    }
  }
}
```

client 连接前必须先启动 server。embedding 模型和 Milvus 连接会在第一次调用相关工具时延迟加载，后续请求复用同一个进程内的实例。

## MCP 工具

| 工具 | 用途 |
|---|---|
| `open_search_session` | 开启检索会话，返回 `session_id` |
| `search_papers` | 混合检索；带 `session_id` 时只返回该会话未见过的论文 |
| `close_search_session` | 关闭会话，返回 query 轨迹和累计论文清单 |
| `search_session_status` | 查看指定会话进展，或列出全部存活会话 |
| `get_papers` | 按 OpenReview id 精确取回完整摘要 |
| `list_paper_areas` | 列出 `area`、`paper_type` 的合法取值及分布 |

典型调研流程：

```text
open_search_session(label="diffusion for LM")
  -> search_papers("discrete diffusion language models", session_id=...)      新增 10
  -> search_papers("masked diffusion text generation", session_id=...)        新增 6
  -> search_papers("non-autoregressive sequence generation", session_id=...)  新增 2
  -> search_papers("diffusion decoding strategies", session_id=...)           新增 0
  -> ...连续 3 轮零新增，当前方向视为挖尽
close_search_session(session_id=...)                                           汇总 18 篇
```

`search_papers` 支持 `area`、`paper_type`、`spotlight_only`、`min_score` 过滤。设置 `include_abstract=false` 可以减少返回内容；省略 `session_id` 则执行无状态检索。

## 固定相关性窗口

每个 query 使用固定深度的相关性窗口：`search_depth` 默认为 `top_k × 4`。会话去重只发生在这个窗口内，过滤已见论文后最多返回 `top_k` 条新结果。

```text
top_k=5, search_depth=20，同一 query 连续检索：
  round 1 -> 返回 rank 1-5
  round 2 -> 返回 rank 6-10
  round 3 -> 返回 rank 11-15
  round 4 -> 返回 rank 16-20
  round 5 -> 窗口内结果全部见过，返回 0 条
```

server 不会自动越过窗口补满结果，以免不断下探到弱相关区域。确需继续使用同一措辞深挖时，显式增大 `search_depth`；通常更推荐换措辞或子方向。每次返回都会附带新增数、去重数、窗口用量、会话累计数和连续空轮数。

## CLI

直接执行一次检索：

```bash
uv run icml-papers search discrete diffusion language models
```

进入带会话去重的交互检索：

```bash
uv run icml-papers search --session
```

可用 `--area`、`--type`、`--spotlight`、`--top-k` 和 `--depth` 调整检索。CLI 检索与 MCP server 使用同一套配置、模型和 Milvus collection。

## 数据与配置

`data/` 是随仓库保存的静态语料快照：

| 文件 | 内容 |
|---|---|
| `index.json` | 6341 篇论文的标题、作者、领域、OpenReview id、arXiv id 等元数据 |
| `abstracts.json` | OpenReview id 到摘要的映射 |
| `areas.json` | 领域名清单 |

所有运行时配置都使用 `ICML_PAPERS_` 环境变量前缀，并可通过仓库根目录的 `.env` 覆盖。完整配置项及默认值见 `src/icml_paper_search/config.py`，常用配置模板见 `.env.example`。

更换 embedding 模型或维度、更新语料后，需要重新执行 `uv run icml-papers build`，确保 Milvus collection 与当前配置一致。将 `ICML_PAPERS_EMBEDDING_DEVICE` 设为 `cpu` 可以在无 GPU 环境运行，但建库会更慢。

## 开发

核心测试使用内存假检索引擎，不需要 Milvus、embedding 模型或 GPU：

```bash
uv run python -m pytest
```

代码架构、边界约束和变更检查见 [AGENTS.md](AGENTS.md)。
