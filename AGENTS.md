# AGENTS.md

## 项目目标

本项目为 6341 篇 ICML 2026 论文提供 MCP 检索服务。核心能力是 Jina dense embedding 与 Milvus BM25 的加权混合检索，以及跨多轮 query 的会话级去重。

面向用户的安装、配置、运行和工具说明以 `README.md` 为准。

## 代码结构

```text
src/icml_paper_search/
├── config.py       pydantic-settings 配置与项目路径
├── models.py       论文、过滤条件、命中和会话结果模型
├── catalog.py      读取本地数据并组装入库记录、统计过滤维度
├── embedding.py    TextEncoder 协议与 JinaEncoder
├── ingest.py       Milvus schema、索引和全量建库
├── retrieval.py    SearchEngine 协议与 Milvus 混合检索实现
├── sessions.py     会话模型、SessionStore 协议和生命周期管理
├── service.py      检索、过滤与会话去重的业务编排
├── formatting.py   面向 agent context 的紧凑文本渲染
├── server.py       FastMCP 工具 schema 和 Streamable HTTP 入口
└── cli.py          建库、检索和信息查询 CLI
```

依赖方向保持为 `server.py` / `cli.py` 调用 `service.py`，再由 service 依赖检索与会话协议。MCP 参数声明、异常转成工具文本等 transport 逻辑留在 `server.py`；可复用的业务行为放在 service 或更低层。

## 核心不变式

### 检索与去重

- 每个 query 只执行一次固定深度的检索。默认窗口为 `top_k × depth_multiplier`，并受 `max_search_depth` 限制。
- 不自动扩召回补满 `top_k`。窗口填满但新结果不足时，通过 `truncated` 和渲染提示让调用方决定是否增大 `search_depth`。
- 会话去重只隐藏当前会话已经返回或经 `get_papers` 登记的论文；无 `session_id` 的检索必须保持无状态。
- 只有实际返回给调用方的命中才能由 `record_round` 登记为已见。不同会话的去重集合必须隔离。
- 连续空轮计数在出现新命中时归零，达到配置阈值后才标记 `exhausted`。
- `top_k` 和 `search_depth` 必须继续执行当前的上下限收敛，空 query 必须拒绝。

### 分层与资源

- `TextEncoder`、`SearchEngine`、`SessionStore` 是替换实现的协议边界。业务层依赖协议，不直接绑定测试替身或 transport。
- torch、sentence-transformers、pymilvus 及昂贵对象保持延迟导入或延迟初始化。导入包和启动 MCP server 不应立即加载模型或连接 Milvus。
- 默认会话存储是进程内状态；并发访问继续由 `SessionManager` 的锁保护。会话容量限制和空闲 TTL 回收不可绕过。
- `formatting.py` 的输出直接进入 agent context。新增字段前确认其决策价值，并保持摘要截断和 `include_abstract` 行为。

### 数据与 Milvus

- `data/index.json` 和 `data/abstracts.json` 通过 OpenReview id 合并；缺少摘要时仍需用标题入库。
- `ingest.py` 的 collection schema、`catalog.py` 的入库字段、`models.py` 的反序列化字段和 `retrieval.py` 的 `OUTPUT_FIELDS` 必须同步变更。
- dense 检索使用 COSINE，sparse 检索使用 Milvus BM25 Function，两路结果通过配置权重交给 `WeightedRanker`。
- embedding 维度必须与 collection schema 一致。修改模型、维度或语料后需要全量重建 collection。
- `icml-papers build` 默认删除同名 collection。不得把重建操作加入测试或自动执行路径。

## 配置约束

- 运行时配置集中在 `Settings`，环境变量统一使用 `ICML_PAPERS_` 前缀，并由根目录 `.env` 覆盖。
- 新增配置项时同时考虑类型约束、默认值、`.env.example` 和 README 中是否需要面向用户说明。
- 业务代码通过注入的 `Settings` 工作；使用进程默认配置时调用 `get_settings()`，不要在其他模块重复读取环境变量。

## 开发与验证

- 使用 Python 3.12+ 和 uv 管理环境，命令入口及本地服务依赖见 `README.md`。
- 核心单元测试必须可在没有 Milvus、模型和 GPU 的环境运行。通过 `SearchEngine`、`TextEncoder` 或 `SessionStore` 注入替身。
- 检索语义变更至少覆盖固定窗口、会话隔离、重复抑制、空轮耗尽、上限收敛和无状态检索相关用例。
- Milvus schema 或检索实现变更需要核对入库字段、输出字段、filter 表达式和建库流程。
- 完成代码变更后运行 README 中的完整离线测试命令；若未运行或无法运行，在交付中明确说明。
