# ICML Paper Search MCP

[English](README.md) | [简体中文](README.zh-CN.md) | [文档](https://keta1930.github.io/icml2026-mcp/zh/docs/)

面向 6,341 篇 ICML 2026 论文的开源 MCP 检索服务。它把论文语料和检索索引保存在本地，以 Chroma dense retrieval 与 SQLite FTS5 BM25 做 weighted RRF 融合，并用持久化 capability session 避免多轮调研重复返回同一篇论文。

无需 Milvus、GPU、本地 embedding 模型或额外数据库服务。自托管只需要 Python、磁盘与 DashScope API key；也可以直接连接公共实例试用。

## 为什么使用它

- **为 agent 而设计**：通过 MCP 暴露检索、精确取回、过滤和会话状态工具。
- **混合检索**：语义召回负责概念相关性，BM25 保留术语和专有名词的精确命中。
- **多轮不重复**：session 在服务重启后仍可继续，同一 session 的并发检索也不会重复提交论文。
- **本地持久化**：论文快照、Chroma、FTS5 与 runtime SQLite 均由单机管理。
- **费用有边界**：建库必须显式确认 Token 上限；在线搜索有缓存、并发限制和日/月预算。
- **适合公共服务**：提供限流、排队上限、健康检查、可信代理和脱敏错误契约。

## 直接连接公共服务

公共 MCP 端点：

```text
https://icml.agent4sci.com:20441/mcp
```

注册到 Claude Code：

```bash
claude mcp add --transport http icml-papers --scope user https://icml.agent4sci.com:20441/mcp
```

等价的 MCP 客户端配置：

```json
{
  "mcpServers": {
    "icml-papers": {
      "type": "http",
      "url": "https://icml.agent4sci.com:20441/mcp"
    }
  }
}
```

`/mcp` 是 Streamable HTTP 协议端点，不是网页或普通 REST JSON API。浏览器地址栏访问时通常会得到 HTTP 406 和 `Client must accept text/event-stream`；这是预期的内容协商结果。请使用支持 Streamable HTTP 的 MCP 客户端，由 SDK 完成初始化和 transport session 管理。人工检查服务使用：

```text
https://icml.agent4sci.com:20441/health/live
https://icml.agent4sci.com:20441/health/ready
```

响应头 `Mcp-Session-Id` 是 transport session；`open_search_session` 返回的 `session_id` 是用于论文去重的业务 capability session，两者不能混用。

公共实例匿名、免费且有共享额度，不提供可用性保证。新的检索 query 会发送到 DashScope 生成 embedding；不要在 query 或 session 标签中放入密钥、个人信息或未公开内容。

## 本地快速开始

运行要求：Python 3.12+、[uv](https://docs.astral.sh/uv/) 和已开通 `qwen3.7-text-embedding` 的 DashScope API key。

```bash
uv sync
cp .env.example .env
```

在 `.env` 中配置 key。直接值与 secret 文件二选一，文件优先：

```dotenv
ICML_PAPERS_DASHSCOPE_API_KEY=your-dashscope-api-key
# ICML_PAPERS_DASHSCOPE_API_KEY_FILE=/absolute/path/to/dashscope_api_key
```

先执行零云调用预检：

```bash
uv run icml-papers info
uv run icml-papers build
```

`build` 默认只输出稳定 `build_id`、论文数、语料哈希和保守的 Token 上界。确认预算后显式建库：

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP>
```

网络或额度问题中断后，可在语料、模型、维度和 batch size 未改变时恢复：

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP> --resume
```

启动 Streamable HTTP MCP 服务：

```bash
uv run icml-paper-search-mcp
curl -fsS http://127.0.0.1:20441/health/ready
```

默认 MCP 地址是 `http://127.0.0.1:20441/mcp`。注册到当前项目：

```bash
claude mcp add --transport http icml-papers --scope project http://127.0.0.1:20441/mcp
```

完整的建库契约、本地目录和故障恢复步骤见[本地部署文档](docs/content/docs/zh/getting-started/local-deployment.mdx)。

## 推荐的调研流程

```text
open_search_session(label="diffusion for language models")
  -> search_papers("discrete diffusion language models", session_id=...)
  -> search_papers("masked diffusion text generation", session_id=...)
  -> search_session_status(session_id=...)
  -> close_search_session(session_id=...)
```

`session_id` 是至少 256-bit 的 bearer capability。拿到它的人即可访问对应会话；服务端只保存其 SHA-256 摘要，不能列举或找回原始 id。默认空闲 24 小时过期。

## MCP 工具

| 工具 | 用途 |
|---|---|
| `open_search_session` | 创建不可枚举的 capability session |
| `search_papers` | 执行混合检索；在 session 内只返回未见论文 |
| `search_session_status` | 查看调用方持有的单个 session |
| `close_search_session` | 关闭 session 并返回有界汇总 |
| `get_papers` | 按 OpenReview id 精确取回论文 |
| `list_paper_areas` | 列出 `area` 与 `paper_type` 的可用值 |

参数、上下限和错误码见 [MCP Tools Reference](docs/content/docs/zh/reference/mcp-tools.mdx)。

## 检索语义

每个 query 对 dense 与 BM25 各执行一次固定深度召回，再做 weighted RRF。默认 `search_depth = top_k × 4`，不会为了补满 `top_k` 自动下探到弱相关区域。

结果中的 `score` 是归一化到 `[0, 1]` 的融合排名分，不是余弦相似度、概率或跨 query 可比较的绝对相关度。它适合当前结果内排序和 `min_score` 粗过滤。

## CLI

```bash
uv run icml-papers info
uv run icml-papers search "discrete diffusion language models"
uv run icml-papers search --session
```

交互检索支持 `--area`、`--type`、`--spotlight`、`--top-k` 和 `--depth`。完整说明见 [CLI Reference](docs/content/docs/zh/reference/cli.mdx)。

## 数据与运行边界

- `data/` 保存 6,341 篇论文的静态元数据、摘要与领域快照。
- `.data/index/` 保存版本化 Chroma、FTS5、manifest 与原子 `current` 指针。
- `.data/state/runtime.sqlite3` 保存 session、限流、预算和 query embedding cache。
- Chroma 持久化目录只允许一个应用进程打开；不要用多个 worker 或容器共享该目录扩容。
- 服务启动与 readiness 不访问 DashScope；首次未命中缓存的 query 才生成在线 embedding。

## Docker 部署

镜像包含程序和语料，不包含 DashScope key、构建索引或 runtime state。生产参考位于 [`deploy/compose.yaml`](deploy/compose.yaml)，包含显式 `build-index` job、单进程应用、持久化目录、Docker secret 与 TLS sidecar。

```bash
docker build --build-arg VCS_REF=local -t icml-paper-search-mcp:local .
sh deploy/smoke-container.sh icml-paper-search-mcp:local
```

生产部署前必须替换示例域名、证书、Host/Origin 白名单和可信代理网段。详细步骤见[容器部署](docs/content/docs/zh/operations/docker.mdx)。

## 长期方向

后续工作聚焦两条主线：提升不同措辞、术语、过滤组合与 fixed window 下的检索稳健性；从 ICML 扩展到更多开放学术数据源，同时保留来源、venue/year 隔离和索引 manifest 契约。完整边界见[长期方向](docs/content/docs/zh/roadmap.mdx)。

## 开发

核心测试完全离线，不需要 key、网络、Docker、GPU 或外部数据库：

```bash
uv run python -m pytest
```

面向 agent 的架构边界与变更检查见 [AGENTS.md](AGENTS.md)。

## 文档站

Fumadocs 文档站位于 [`docs/`](docs/)，包含概念、教程、指南、Reference、部署运维、排障和 AI-readable delivery。

在线中文文档：<https://keta1930.github.io/icml2026-mcp/zh/docs/>

```bash
cd docs
pnpm install
pnpm dev
```

开发站点默认运行在 `http://127.0.0.1:20440`，中文文档入口为 `/zh/docs/`，同时提供 `/zh/llms.txt` 和每个页面的 `.md` 版本。

## License

本项目使用 [MIT License](LICENSE)。
