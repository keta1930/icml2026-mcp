# AGENTS.md

## 项目目标

本项目为 6341 篇 ICML 2026 论文提供单节点 MCP 检索服务。核心能力是 DashScope dense embedding、Chroma COSINE 与 SQLite FTS5 BM25 的加权 RRF 检索，以及跨多轮 query 的持久化会话去重。

面向用户的安装、配置、运行、部署和工具说明以 `README.md` 为准。实施变更时遵守以下架构边界与不变式。

## 代码结构

```text
src/icml_paper_search/
├── config.py       pydantic-settings 配置、路径、secret 解析和边界校验
├── errors.py       稳定公开错误码与脱敏边界
├── models.py       论文、过滤条件、命中、轮次和会话结果模型
├── catalog.py      合并本地语料、组装索引记录与统计过滤维度
├── embedding.py    TextEncoder 协议与 DashScopeEncoder
├── bm25.py         SQLite FTS5 BM25 建库和只读查询
├── manifest.py     版本化索引契约、验证和原子 current 切换
├── ingest.py       零云预检、显式建库、checkpoint 和 staging 发布
├── retrieval.py    SearchEngine 协议与 ChromaHybridSearchEngine
├── state.py        runtime SQLite schema、事务和隐私 hash
├── sessions.py     SessionStore 协议与 capability session 生命周期
├── governance.py   限流、预算、cache、single-flight 和并发控制
├── service.py      检索、过滤与会话去重的业务编排
├── formatting.py   面向 agent context 的紧凑文本渲染
├── server.py       async FastMCP 工具、transport 安全和健康检查
└── cli.py          preflight/build、检索和本地信息 CLI
```

依赖方向保持为 `server.py` / `cli.py` 调用 `service.py`，再由 service 依赖检索、编码与会话协议。MCP schema、client identity 解析和异常转为工具文本属于 transport 层；可复用业务行为放在 service 或更低层。部署资产位于 `deploy/`，不得把云服务器专属域名、证书或网络默认值写入库级 `Settings`。

## 核心不变式

### 检索与 score

- 每个 query 对 dense 与 BM25 各执行一次固定深度召回。默认窗口为 `top_k × depth_multiplier`，并受 `max_search_depth` 限制。
- 不自动扩召回补满 `top_k`。窗口填满但新结果不足时，通过 `truncated` 和渲染提示让调用方决定增大 `search_depth` 或更换措辞。
- dense 使用 Chroma COSINE，sparse 使用 SQLite FTS5 `bm25()`；两路只通过 manifest 中的权重和 `rrf_k` 做 weighted RRF。
- `SearchHit.score` 必须保持 `[0, 1]` 且越高越好。它是归一化 RRF 排名分，不得描述成余弦相似度、概率或跨 query 校准分。
- `area`、`paper_type` 和 spotlight 过滤必须在两路召回中语义一致。`min_score` 在融合后过滤。
- `top_k`、`search_depth`、query 长度和 `paper_ids` 数量必须继续执行配置上限；空 query 必须拒绝。

### 会话与持久状态

- 会话去重只隐藏当前 capability session 已返回或经 `get_papers` 登记的论文；无 `session_id` 的检索必须保持无状态。
- 只有实际返回给调用方的命中才能登记为已见。不同 session 的去重集合必须隔离，同一 session 的并发轮次必须在一个原子事务内过滤和提交，不能先读后写。
- raw `session_id` 至少包含 256-bit 随机性，只返回给调用方；数据库仅保存 capability hash。匿名接口不得全局枚举、恢复或泄露其他 session。
- 会话容量满时拒绝新建，不得淘汰仍活跃的其他调用方会话。空闲 TTL、最大轮次、最大 seen 数和有界汇总不可绕过。
- MCP 新建 session 必须在任何 session 写入前执行持久化来源限流；分钟速率、burst 和 UTC 日上限统一来自 `Settings`，重启不得重置。
- 连续空轮计数在出现新命中时归零，达到配置阈值后才标记 `exhausted`。关闭会话必须原子返回有界摘要并立即使 capability 失效。
- runtime SQLite 使用 WAL、foreign keys、busy timeout 和短事务。查询审计只保存规范化 hash，不保存原始 query；来源身份使用部署私有 salt 后摘要。

### Embedding 与费用边界

- `qwen3.7-text-embedding` 固定使用 DashScope 北京地域 native API、1024 维 dense 输出。query 必须传 `text_type=query`，论文必须传 `text_type=document`，单批不得超过 20 条。
- provider 响应必须校验 status、数量、`text_index` 顺序、维度、有限数值和 `total_tokens`。只重试临时网络、429、408 和 5xx；重试受总 deadline 和次数限制。
- provider 错误分类只允许读取精确 allowlist 的 status/code；403 `AllocationQuota.FreeTierOnly` 和 400 `Arrearage` 必须首轮映射为不可重试的 `embedding_quota_exhausted`。401/普通 403 仍是配置错误，429 与 5xx 仍按瞬态故障处理；禁止记录 provider message/body、key、输入或 vector。
- key 统一通过 `Settings.resolve_dashscope_api_key()` 解析，secret 文件优先。禁止在代码、测试、镜像、文档、日志、异常字符串或命令行参数中写入真实 key。
- 在线 query embedding 必须经过持久 cache、single-flight、Token 预算预留/结算和 provider semaphore。调用前按 UTF-8 byte 数加固定协议开销保守预留；cache key 不得包含原始 query，只有成功向量可以缓存。
- 限流、日/月预算和缓存状态必须持久化；重启不能重置公共服务边界。请求明确未发送时释放预留，可能已发送但 usage 未知时按预留值结算；actual 超过 reserved 时持久记录 overrun，并使当日及当月后续调用 fail closed。

### 索引与建库

- `data/index.json` 和 `data/abstracts.json` 通过 OpenReview id 合并；缺少摘要时仍用标题入库。修改语料字段时同步核对 catalog、Chroma metadata、FTS5 schema、领域模型和 fetch 反序列化。
- Chroma 只保存显式生成的 embedding，禁止隐式下载或调用 Chroma embedding function。FTS5 与 Chroma 必须使用同一批 paper id 和 corpus hash。
- 索引版本必须包含 manifest，并绑定 schema、语料文件哈希、corpus hash、完整 paper-id 指纹、模型、维度、document text type、dense metric、BM25 tokenizer、融合权重和 collection name。Chroma collection metadata、完整 ID 集合和样本向量维度也必须匹配；readiness 必须用随部署提供的原始语料快照重新核对这些指纹，全程零云调用。
- 缺少当前 manifest 或 collection 指纹字段的索引不得兼容读取，必须显式重建；不得通过跳过 readiness 降级为信任索引自声明。
- `icml-papers build` 默认只做零云调用 preflight。不得把 provider 调用、建库或重建加入 import、server startup、readiness、测试或其他自动执行路径。
- 付费建库必须同时要求显式 execute 和正数 Token cap；每批调用前按 UTF-8 byte 数加单输入固定开销保守预留，必要时按剩余额度缩小批次。写入 staging，按完整批次前移 checkpoint，完整验证后再原子发布版本并切换 `current`。
- provider actual usage 超过建库预留时，必须在索引写入和 checkpoint 前 fail closed，持久化不含原文的故障审计，并禁止对该 staging 自动 resume。
- resume 只接受 build id、语料哈希、模型、维度和 batch size 匹配的 checkpoint，且 completed ids 必须是当前语料的严格前缀。失败构建不得替换已有 `current`。

### 并发、资源与 transport

- `TextEncoder`、`SearchEngine`、`SessionStore` 是替换实现与测试替身的协议边界；业务层不得直接绑定 transport 或具体测试类。
- DashScope SDK、Chroma client、FTS5 connection 和 runtime database 必须延迟初始化或按线程创建。导入包、构造 FastMCP 和启动进程不得读取 key 或访问 provider。
- 所有同步 provider、Chroma 和 SQLite 工作都必须从 async MCP handler offload，不能阻塞 event loop。慢搜索与轻量状态操作使用分离 limiter。
- admission 的 active + waiting 总数不得超过 `search_queue_size`；容量预留必须在首个 await 前完成，超额立即返回 `server_busy`。已 admission 的请求再等待有界 worker，超时、取消和异常路径都必须释放名额。
- Chroma 本地持久化目录由单个应用进程打开。不得让多个 worker、应用容器或并行 build/serve writer 共享同一目录。
- 服务容器只读挂载已发布 index，runtime state 单独读写；建库 job 才能写 index。备份和恢复必须在应用停止时同时处理 index 与 state。
- Docker builder 与 runtime 的虚拟环境绝对路径必须一致，两个 console script 必须通过真实镜像 smoke。Compose 网络由 Docker 动态分配，禁止恢复固定 subnet；`trusted_proxy_networks` 只表示信任边界，不得用于网络分配。
- 默认使用 HTTP peer 作为限流身份。只有 peer 位于显式可信网段时才接受固定代理 header；公网部署必须设置精确 Host/Origin 并保持 DNS rebinding 防护。
- `live` 只检查进程；`ready` 只检查本地 state/index，不读取 key 或访问 provider。未知异常和 provider 响应只能映射为脱敏公开错误码。

### 输出与配置

- `formatting.py` 的输出直接进入 agent context。新增字段前确认决策价值，并保持摘要截断、`include_abstract`、窗口统计、空结果解释和有界 session 汇总。
- 运行时配置集中在 `Settings`，环境变量统一使用 `ICML_PAPERS_` 前缀，并由根目录 `.env` 覆盖。其他模块不得重复读取环境变量。
- 新增或修改配置时同步更新类型约束、组合校验、`.env.example`、相关测试，以及 README 中需要面向用户公开的行为。

## 开发与验证

- 使用 Python 3.12+ 和 uv 管理项目环境。命令入口与面向用户流程以 `README.md` 为准。
- 核心测试必须完全离线，不需要 DashScope key、网络、Docker、GPU 或外部数据库。通过 `SearchEngine`、`TextEncoder`、`SessionStore` 和 provider call 注入替身。
- 检索语义变更至少覆盖双路单次召回、fixed window、过滤一致性、RRF 范围与稳定排序、会话隔离、重复抑制、空轮耗尽、上限收敛和无状态检索。
- 会话或治理变更至少覆盖同 session 并发提交、重启持久化、TTL/容量、session-open 容量攻击、secret/原文不落库、限流原子性、预算预留/overrun 熔断、cache TTL/LRU、single-flight、admission 超额立即拒绝和取消释放。
- 建库或索引变更至少覆盖零云 preflight、保守费用预留、execute/cap 门槛、batch checkpoint、provider usage 契约漂移、resume 校验、失败不切 current、源语料与 paper-id 指纹、collection metadata/维度、Chroma/FTS5 count 与基本查询。
- MCP 或部署安全变更至少覆盖 async schema、event-loop 不阻塞、40-client 混合调用、Host/Origin、可信代理、公开错误脱敏、只读 index、secret mount、动态 Docker network、真实镜像 console-script smoke 和单进程约束。
- 完成代码变更后运行 README 中的完整离线测试命令；若未运行或无法运行，在交付中明确说明。
