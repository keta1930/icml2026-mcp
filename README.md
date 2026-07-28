# icml-paper-search-mcp

面向 6341 篇 ICML 2026 论文的 MCP 检索服务。它使用 DashScope `qwen3.7-text-embedding` 生成 dense embedding，以本地持久化的 Chroma COSINE 与 SQLite FTS5 BM25 做加权 RRF 融合，并通过持久化检索会话避免多轮调研反复返回同一批论文。

自托管不需要 Milvus、Docker 服务、GPU 或本地模型；只需要 Python、磁盘和调用方自己的 DashScope API key。也可以直接连接下方的公共服务，无需提供 key。

## 功能

- Chroma dense + SQLite FTS5 BM25 混合检索
- 版本化本地索引、显式付费建库、断点恢复与原子发布
- 跨进程重启持久化的会话去重、query 轨迹和论文汇总
- 按 `area`、`paper_type`、spotlight 和最低得分过滤
- 按 OpenReview id 精确取回完整摘要
- Streamable HTTP MCP transport 与健康检查
- 面向匿名共享服务的有界并发、等待超时、限流、费用预算和 embedding 缓存

## 直接使用公共服务

公共 MCP 端点：

```text
https://icml.agent4sci.com:20441/mcp
```

注册到 Claude Code：

```bash
claude mcp add --transport http icml-papers --scope user https://icml.agent4sci.com:20441/mcp
```

等价配置：

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

公共服务是匿名、免费且有额度的共享实例，不提供可用性保证。调用方无需上传 DashScope key，但新的检索 query 会由服务端发送给 DashScope 生成 embedding；不要在 query 或会话标签中放入密钥、个人信息或未公开内容。

## 本地快速开始

运行要求：

- Python 3.12 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可访问 DashScope 北京地域 native API 的网络
- 已开通 `qwen3.7-text-embedding` 的 DashScope API key

在仓库根目录安装依赖并创建本机配置：

```bash
uv sync
cp .env.example .env
```

在 `.env` 中填写自己的 key。直接配置与 secret 文件二选一；两者同时存在时 secret 文件优先：

```dotenv
ICML_PAPERS_DASHSCOPE_API_KEY=your-dashscope-api-key
# ICML_PAPERS_DASHSCOPE_API_KEY_FILE=/absolute/path/to/dashscope_api_key
```

`.env` 和常见 secret 文件已被 Git 忽略，但仍应限制文件权限，并确保 key 不进入镜像、命令输出、日志或提交记录。

### 1. 零云调用预检

```bash
uv run icml-papers info
uv run icml-papers build
```

`info` 检查语料、当前索引与配置状态；不读取 key，也不访问 DashScope。`build` 默认同样只做 preflight，会打印论文数、语料哈希、稳定 `build_id` 和 `estimated_input_tokens`，不会创建索引或产生云端费用。`estimated_input_tokens` 是按每条输入的 UTF-8 byte 数再加 32 Token 固定余量计算的保守预留上界；完整建库时，`TOKEN_CAP` 至少应覆盖该值。

### 2. 显式建库

确认 preflight 输出后，为本次任务设置可接受的 Token 硬上限：

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP>
```

只有同时给出 `--execute` 和正数 `--max-input-tokens` 才会调用 DashScope。每批调用前先按保守上界预留，最后一批会按剩余额度缩小；论文按不超过 20 条的批次编码，完整批次写入后才前移 checkpoint。Chroma、FTS5、manifest 与语料指纹全部校验通过后，才会原子切换 `current` 到新版本，已有可用版本不会被失败构建替换。

若 provider 报告的实际 usage 超过调用前预留，当前批次不会写入索引或 checkpoint，staging 会记录不含论文原文的 `failure-audit.json` 并拒绝自动 resume。云调用此时已经发生，服务只能立即停止后续费用；应检查 provider Token 契约，并在移走该 staging 后重新建库。

若网络或额度问题中断了建库，在语料、模型、维度和 batch size 未改变的前提下恢复：

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP> --resume
```

`--resume` 只接受与当前 preflight 完全匹配、且没有 usage 契约故障审计的 staging/checkpoint，并跳过已经完整提交的批次。修改语料、embedding 契约或融合参数后，应重新生成并发布完整索引。

当前 manifest 要求语料文件哈希、corpus hash、完整 paper-id 指纹、Chroma collection metadata 和样本向量维度全部一致。缺少这些字段的旧索引不会被兼容读取；升级时先停止服务并把旧目录移作备份，再重新建库：

```bash
mv .data/index .data/index.pre-fingerprint
uv run icml-papers build
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP>
```

### 3. 启动服务

```bash
uv run icml-paper-search-mcp
```

默认端点为 `http://127.0.0.1:20441/mcp`。确认本地索引可用：

```bash
curl -fsS http://127.0.0.1:20441/health/live
curl -fsS http://127.0.0.1:20441/health/ready
```

`live` 只表示进程存活；`ready` 会零云调用地检查 runtime SQLite、随服务提供的 `index.json` / `abstracts.json`、当前 Chroma 版本、完整 paper-id 集合、FTS5 索引、collection metadata、样本向量维度和 manifest 是否一致。服务启动不会调用 DashScope；首次未命中缓存的搜索才会生成 query embedding。

注册本地 MCP：

```bash
claude mcp add --transport http icml-papers --scope project http://127.0.0.1:20441/mcp
```

## MCP 工具

| 工具 | 用途 |
|---|---|
| `open_search_session` | 创建不可枚举的 capability session，返回 `session_id` |
| `search_papers` | 混合检索；带 `session_id` 时只返回该会话未见过的论文 |
| `close_search_session` | 原子关闭会话并返回有界 query/论文汇总 |
| `search_session_status` | 使用 `session_id` 查看单个会话；匿名服务不支持全局列举 |
| `get_papers` | 按 OpenReview id 取回完整摘要，可登记到会话去重集 |
| `list_paper_areas` | 列出 `area`、`paper_type` 的可用值及分布 |

典型调研流程：

```text
open_search_session(label="diffusion for LM")
  -> search_papers("discrete diffusion language models", session_id=...)
  -> search_papers("masked diffusion text generation", session_id=...)
  -> search_session_status(session_id=...)
  -> close_search_session(session_id=...)
```

`session_id` 是至少 256-bit 的 bearer capability：拿到它的人即可访问对应会话。请像临时凭据一样保存，不要公开或写入日志。服务端只持久化其 SHA-256 摘要，无法列出或找回原始 id。默认空闲 24 小时过期；只要 runtime state 数据仍在，服务重启后会话仍可继续。同一 session 的并发检索通过 SQLite 原子提交，实际返回的论文不会在并发轮次间重复。

不传 `session_id` 时搜索保持无状态。`get_papers` 只有在传入 session 时才会把实际找到的论文登记为已见；关闭会话后 capability 立即失效。

## 检索窗口与 score

每个 query 只执行一次固定深度的 dense 召回和一次 BM25 召回。`search_depth` 默认是 `top_k × 4`，并受全局上限约束；会话去重只在这个窗口内进行，服务不会自动下探到弱相关区域补满 `top_k`。

```text
top_k=5, search_depth=20，同一 query 连续检索：
  round 1 -> rank 1-5
  round 2 -> rank 6-10
  round 3 -> rank 11-15
  round 4 -> rank 16-20
  round 5 -> 0 条，窗口内结果已经耗尽
```

结果中的 `score` 是 dense 排名与 BM25 排名按默认 `0.8 / 0.2` 权重计算后，再归一化到 `[0, 1]` 的 weighted-RRF 分数。它不是余弦相似度、概率或跨 query 校准后的绝对相关度；适合当前结果内排序和 `min_score` 粗过滤，不宜直接比较不同 query 的数值。

## CLI

单次检索：

```bash
uv run icml-papers search "discrete diffusion language models"
```

带会话去重的交互检索：

```bash
uv run icml-papers search --session
```

可使用 `--area`、`--type`、`--spotlight`、`--top-k` 和 `--depth` 调整检索。CLI 与 MCP server 使用同一个本地索引、runtime state、DashScope 配置、embedding cache 和 Token 预算；按来源 IP 的限流与 admission 只应用于 MCP server。

## 公共实例的容量与限制

默认配置面向约 25–40 名间歇使用者的单机共享场景，而不是 25–40 个持续占满的并行搜索。慢速 provider、Chroma 和轻量 SQLite 状态操作使用分离的工作池；相同 query 的并发 cache miss 会合并为一次云调用。

公共实例当前采用以下默认边界：

| 边界 | 默认值 |
|---|---:|
| 每来源 IP 平均搜索速率 | 6 次/分钟 |
| 每来源 IP 突发容量 | 3 次 |
| 每来源 IP 每 UTC 日 | 200 次搜索 |
| 每来源 IP 创建 session | 3 次/分钟，突发 2 次，每 UTC 日 20 次 |
| 全局搜索速率 | 120 次/分钟 |
| active + waiting 搜索总数 | 最多 64；超额立即返回 `server_busy` |
| 搜索 worker | 默认 4；已 admission 的等待最长 10 秒 |
| DashScope 并发 | 4 |
| Chroma 并发 | 2 |
| 在线 embedding 日/月 Token 预算 | 500,000 / 10,000,000 |
| 活跃 session 总数 | 256 |

新建 session 的来源限流会在写入 session 前执行，不能通过耗尽全局 session 容量绕过。在线 query embedding 会按 UTF-8 byte 数加 16 Token 固定余量，在调用 provider 前同时预留日/月额度；请求明确未发送时释放预留，请求可能已发送但 usage 未知时按预留值计费。若实际 usage 超过预留，差额会持久记录，当前 UTC 日和月的后续在线调用都会 fail closed。

超出边界时工具返回稳定的 `rate_limited`、`server_busy`、`budget_exhausted` 等错误码。共享 NAT 下的调用方会共用 IP 配额。公共服务的限制可能随运行成本调整；需要稳定额度时请自托管。

Chroma 本地持久化目录只应由一个应用进程打开。不要通过多个 worker 或多个容器共享同一目录来扩容；更高吞吐或多节点需求应迁移到支持服务端并发的向量数据库。

## 数据、配置与持久化

`data/` 是随仓库保存的静态语料快照：

| 文件 | 内容 |
|---|---|
| `index.json` | 6341 篇论文的标题、作者、领域、OpenReview id、arXiv id 等元数据 |
| `abstracts.json` | OpenReview id 到摘要的映射 |
| `areas.json` | 领域名清单 |

默认运行数据位于：

| 路径 | 内容 |
|---|---|
| `.data/index/versions/<build_id>/chroma/` | Chroma dense 索引 |
| `.data/index/versions/<build_id>/bm25.sqlite3` | FTS5 BM25 索引 |
| `.data/index/versions/<build_id>/manifest.json` | 模型、维度、语料文件/corpus/paper-id 指纹与融合契约 |
| `.data/index/current` | 当前已发布版本指针 |
| `.data/state/runtime.sqlite3` | session、限流、预算和 query embedding 缓存 |

运行时配置集中在 `Settings`，环境变量统一使用 `ICML_PAPERS_` 前缀。常用配置和安全默认值见 [`.env.example`](.env.example)，完整字段约束见 [`config.py`](src/icml_paper_search/config.py)。

## 容器部署

[`Dockerfile`](Dockerfile) 生成非 root、只读根文件系统友好的运行镜像；镜像包含语料和程序，但不包含 DashScope key、构建后的索引或 runtime state。[`deploy/compose.yaml`](deploy/compose.yaml) 提供单进程应用、显式 `build-index` job、持久化 index/state、Docker secret 与独立 TLS sidecar 的生产参考。

构建镜像后先运行不需要真实 key 的容器 smoke；它会验证两个 console script、固定 `/opt/venv` shebang、非 root uid/gid、服务启动路径和 runtime secret 权限：

```bash
docker build --build-arg VCS_REF=local -t icml-paper-search-mcp:local .
sh deploy/smoke-container.sh icml-paper-search-mcp:local
```

部署时需要准备：

- 不可变 git-SHA 镜像标签 `ICML_MCP_IMAGE`
- 仅宿主机可读的 `ICML_PAPERS_DASHSCOPE_SECRET_FILE`
- 分离的 index 与 state 持久化目录
- `build-index` 的 `ICML_PAPERS_BUILD_MAX_INPUT_TOKENS`
- 与实际域名匹配的证书、Host/Origin 白名单和可信代理网段

先以 `build` profile 显式建库，再启动 `icml-mcp` 与 TLS sidecar。服务容器以只读方式挂载 index，只有独立建库 job 可以写入它。Compose 让 Docker 动态分配 `backend` 与 `provider-egress` bridge subnet，不预占固定网段；`ICML_PAPERS_TRUSTED_PROXY_NETWORKS` 只控制应用信任哪些代理来源，并不分配 Docker 网络。部署前应把它收敛到实际 TLS sidecar 所在的 backend 网段。

Compose 和 Nginx 示例包含项目公共域名，复制到其他服务器前必须替换域名、证书路径和 Host/Origin 白名单。

## 备份与恢复

索引和 runtime state 必须作为一组备份。为避免 Chroma 与 SQLite 快照不一致，先停止 MCP 进程或容器，再复制完整的 `.data/index` 与 `.data/state` 目录：

```bash
tar -C .data -czf icml-paper-search-backup.tgz index state
```

恢复也应在服务停止时进行，并保留原目录结构和文件权限。恢复后先运行 `uv run icml-papers info`，再启动服务并检查 `/health/ready`。若 embedding 模型、维度、语料哈希或 RRF 配置与 manifest 不一致，readiness 会拒绝服务；不要只恢复 Chroma 或只复制 `runtime.sqlite3` 而遗漏同目录的 WAL 文件。

容器部署遵循同一原则：停止单进程应用后，同时快照 index/state bind mount，再在同一配置契约下恢复。

## 安全与隐私

- 自托管 key 只通过环境 secret 或只读文件提供；secret 文件优先，公开 client 永远不需要也不应获得服务端 key。
- DashScope 会接收用于生成 embedding 的论文文本或搜索 query。不要把本服务用于敏感内容。
- runtime state 不保存原始 session id、原始 IP 或原始 query；会话标签会按原样保存，因此标签也不应包含敏感内容。
- 公共 endpoint 是匿名服务，不等同于用户鉴权。`session_id` 本身就是访问会话的 capability。
- DNS rebinding 防护默认开启。公网自托管必须使用精确 `allowed_hosts` / `allowed_origins`，只在应用确实位于受信私网代理之后时启用 forwarded client IP。
- provider 错误和未知异常只映射为稳定公开错误码，日志不应记录 provider body、key、原始 query 或 capability。

若 key 曾出现在聊天记录、终端历史、日志或提交中，应立即在 DashScope 控制台轮换。

## 开发与测试

完整离线测试不需要 DashScope key、网络、Milvus、Docker、GPU 或本地模型：

```bash
uv run python -m pytest
```

测试覆盖配置边界、provider payload 与重试脱敏、Chroma/FTS5 检索和过滤、保守建库预留与 checkpoint、完整索引指纹、fixed-window 语义、并发 session 原子去重、session-open 限流、64 请求 admission 上限、40-client 混合调用、持久预算/cache、异步 MCP schema、可信代理和错误边界。容器入口由上一节的 smoke 脚本单独验证。

代码架构、实现不变式和变更检查见 [AGENTS.md](AGENTS.md)。
