# AGENTS.md

## 项目目标

本仓库实现 6,341 篇 ICML 2026 论文的单节点 MCP 检索服务。核心行为是 DashScope query/document embedding、Chroma COSINE 与 SQLite FTS5 BM25 的 weighted RRF，以及跨多轮 query 的持久化 capability session 去重。

面向使用者的安装、运行与工具说明以 `README.md` 和 `docs/content/docs/` 为准。本文件约束实现与验证。

## 仓库结构

```text
src/icml_paper_search/   Python 服务、索引与 CLI
tests/                   完全离线的行为与并发测试
data/                    随仓库发布的 ICML 2026 静态语料快照
deploy/                  容器入口、Compose、Nginx 与 smoke
docs/                    Fumadocs 文档站子项目
```

Python 依赖方向保持为 `server.py` / `cli.py` → `service.py` → 检索、编码、session 与 state。MCP schema、client identity 和公开错误转换属于 transport 层；可复用业务行为放在 service 或更低层。

关键模块：

| 模块 | 职责 |
|---|---|
| `config.py` | `Settings`、路径、secret 解析和组合校验 |
| `catalog.py` | 合并语料、生成索引记录与过滤统计 |
| `embedding.py` | `TextEncoder` 协议与 DashScope 实现 |
| `bm25.py` / `retrieval.py` | FTS5、Chroma 与 weighted RRF |
| `manifest.py` / `ingest.py` | 版本契约、预检、checkpoint 和原子发布 |
| `state.py` / `sessions.py` | runtime schema、事务和 session 生命周期 |
| `governance.py` | 限流、预算、cache、single-flight 与并发控制 |
| `server.py` | FastMCP、Streamable HTTP、安全与健康检查 |
| `cli.py` | `info`、`build` 与 `search` 命令 |

## 检索不变式

- 每个 query 对 dense 与 BM25 各执行一次固定深度召回。默认窗口为 `top_k × depth_multiplier`，上限为 `max_search_depth`。
- 不自动扩召回补满 `top_k`。窗口内新结果不足时通过 `truncated` 和输出提示交给调用方处理。
- dense 使用 Chroma COSINE，sparse 使用 FTS5 `bm25()`；只按 manifest 的权重与 `rrf_k` 融合。
- `SearchHit.score` 保持 `[0, 1]` 且越高越好。不得描述为余弦相似度、概率或跨 query 校准分。
- `area`、`paper_type`、spotlight 在两路召回中语义一致；`min_score` 在融合后过滤。
- query、`top_k`、`search_depth` 与 `paper_ids` 始终执行 `Settings` 上限；空 query 必须拒绝。

## Session 与 runtime state

- 无 `session_id` 的搜索保持无状态；session 只隐藏该 capability 已返回或经 `get_papers` 登记的论文。
- 只有实际返回的论文能登记为 seen。同一 session 的过滤与提交必须在一个 SQLite 原子事务中完成。
- raw `session_id` 至少 256-bit，只返回给调用方；数据库只保存 capability hash。匿名接口不得枚举或恢复 session。
- 容量满时拒绝新 session，不淘汰其他活跃 session。TTL、轮次、seen 数与汇总始终有界。
- session-open 限流在任何 session 写入前执行，并按来源持久化；重启不能重置。
- runtime SQLite 启用 WAL、foreign keys、busy timeout 与短事务。不得保存原始 IP、原始 query 或 raw capability。

## Embedding 与费用

- `qwen3.7-text-embedding` 使用 DashScope 北京地域 native API 与 1,024 维输出。query 传 `text_type=query`，论文传 `text_type=document`，单批最多 20 条。
- 校验 provider status、结果数、`text_index`、维度、有限数值和 `total_tokens`。只重试网络、429、408 与 5xx，并受总 deadline 限制。
- 配额耗尽、配置错误与瞬态故障保持不同公开错误码。日志和异常不得包含 provider body/message、key、输入或向量。
- key 只由 `Settings.resolve_dashscope_api_key()` 解析，secret 文件优先。真实 key 不得进入代码、测试、镜像、文档、日志或命令参数。
- 在线 query embedding 必须经过持久 cache、single-flight、Token 预算和 provider semaphore。调用前预留，调用后结算；usage 不确定时按预留值计费。
- actual usage 超过 reserved 时记录 overrun，并使当日和当月后续在线调用 fail closed。

## 索引与建库

- `data/index.json` 与 `data/abstracts.json` 通过 OpenReview id 合并；缺少摘要时仍以标题入库。
- Chroma 只接受显式 embedding，禁止隐式下载模型。Chroma 与 FTS5 必须拥有相同 paper-id 集合和 corpus hash。
- manifest 绑定 schema、语料文件哈希、corpus hash、完整 ID 指纹、模型、维度、text type、metric、tokenizer、融合参数和 collection name。
- readiness 使用随部署提供的原始语料核验 manifest、collection metadata、完整 ID 集合与样本维度，全程零云调用。
- `icml-papers build` 默认只做 preflight。provider 调用不得出现在 import、server startup、readiness 或测试路径。
- 付费建库同时要求 `--execute` 和正数 Token cap；按完整 batch 写 staging/checkpoint，完整验证后才原子切换 `current`。
- usage 超过预留时，在写索引与 checkpoint 前停止，写入不含原文的 failure audit，并禁止该 staging 自动 resume。
- resume 只接受 build id、语料、模型、维度和 batch size 匹配且 completed ids 是严格前缀的 checkpoint。
- 旧或不完整 manifest 不兼容读取，必须显式重建；失败构建不得替换已有 `current`。

## 并发、transport 与部署

- DashScope、Chroma、FTS5 与 runtime database 延迟初始化或按线程创建。构造 FastMCP 与启动进程不得读取 key 或访问 provider。
- 同步 provider、Chroma 和 SQLite 工作从 async handler offload；慢搜索与轻量 state 使用分离 limiter。
- admission 在首个 `await` 前预留 active + waiting 容量。超额立即返回 `server_busy`；超时、取消与异常都释放名额。
- Chroma 持久化目录只允许一个应用进程打开。不得用多 worker、多个应用容器或并行 writer 共享目录。
- 容器服务只读挂载 canonical index；入口验证并复制完整版本到 state 下的可写 `runtime-index`。应用与 readiness 只打开副本。
- `runtime-index` 可再生且不进入冷备。备份在应用停止时同时处理 canonical index 与非派生 state。
- 公网部署使用精确 Host/Origin。只有 peer 位于显式可信网段时才接受代理身份 header。
- `live` 只表示进程存活；`ready` 只检查本地 state/index。未知异常统一映射为脱敏公开错误。

## 配置与输出

- 配置只由 `Settings` 读取，环境变量统一使用 `ICML_PAPERS_` 前缀。其他模块不得重复读取环境变量。
- 配置变更同步更新字段约束、组合校验、`.env.example`、测试和面向用户的 Reference。
- `formatting.py` 输出直接进入 agent context。新增字段前确认决策价值，并保持摘要裁剪、空结果解释与 session 汇总有界。
- 当前数据模型仍是 ICML 2026 单源。不得在文档或接口中把调研中的多源方案描述为已实现功能。

## 文档站约束

- `docs/` 是独立的 Next.js 16 + Fumadocs 子项目；Python 运行时不得依赖 Node.js 包。
- 文档按用户任务组织，公开接口行为以实现、schema 和测试为准。README 只保留入口与快速开始，细节放入文档站。
- `content/docs/` 是内容权威源；`.source/`、`.next/` 与 `out/` 是生成物，不提交。
- 修改 MDX 后运行 `cd docs && pnpm build`。新增页面同时更新对应 `meta.json`，并验证 `/llms.txt` 与页面 `.md` 路由覆盖。
- 架构概念图可使用项目资产；精确流程优先 Mermaid。单页不超过 300 行，代码片段保持短小。
- 文档不得包含真实 secret、内部服务器凭据、原始 session id、原始 query 或 provider 响应。

## 验证矩阵

代码变更完成后运行：

```bash
uv run python -m pytest
```

按变更类型补充验证：

| 变更 | 最低验证 |
|---|---|
| 检索 | 双路召回、fixed window、过滤一致性、RRF 范围、稳定排序 |
| Session/state | 并发原子提交、隔离、TTL/容量、重启持久化、隐私不落库 |
| Governance | session-open 限流、budget overrun、cache、single-flight、admission 取消释放 |
| 建库/index | 零云 preflight、cap、checkpoint、resume、失败不切换、完整指纹 |
| MCP/安全 | async schema、event loop、Host/Origin、可信代理、脱敏错误、健康检查 |
| 容器 | 非 root/只读约束、secret mount、动态网络、console-script smoke |
| 文档 | `cd docs && pnpm build`、导航、站内链接、llms 与逐页 Markdown |

只修改任务需要的文件。发现无关问题时报告，不顺带重构。
