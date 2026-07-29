# ICML Paper Search MCP

[English](README.md) | [简体中文](README.zh-CN.md) | [中文文档](https://keta1930.github.io/icml2026-mcp/zh/docs/)

一个用于检索 6,341 篇 ICML 2026 论文的开源 MCP 服务。项目提供六个工具，覆盖论文检索、论文读取、语料信息和搜索会话。同一会话已经返回过的论文不会再次出现。

## 添加公共 MCP

Claude Code：

```bash
claude mcp add --transport http icml-papers --scope user \
  https://icml.agent4sci.com:20441/mcp
```

通用 Streamable HTTP 配置：

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

公共实例可以免费使用，额度由所有用户共享。

## 工具

| 工具 | 用途 |
|---|---|
| `open_search_session` | 创建搜索会话 |
| `search_papers` | 搜索论文，可在会话内去重 |
| `search_session_status` | 查看会话状态 |
| `close_search_session` | 关闭会话 |
| `get_papers` | 按 OpenReview ID 读取论文 |
| `list_paper_areas` | 列出研究领域和论文类型 |

## 使用 Docker 本地部署

需要 Docker Compose，以及能够调用 `qwen3.7-text-embedding` 的 DashScope key。

```bash
mkdir -p deploy/secrets
printf '%s' 'YOUR_DASHSCOPE_KEY' > deploy/secrets/dashscope_api_key
chmod 600 deploy/secrets/dashscope_api_key

docker compose -f deploy/compose.yaml -f deploy/milvus.compose.yaml \
  up -d etcd minio standalone
docker compose -f deploy/compose.yaml -f deploy/milvus.compose.yaml \
  --profile build run --rm build-index
docker compose -f deploy/compose.yaml -f deploy/milvus.compose.yaml \
  up -d icml-mcp
```

本地 MCP 地址是 `http://127.0.0.1:20442/mcp`。Compose 已包含 Milvus Standalone、etcd 和 MinIO，索引与运行状态保存在 Docker volume 中。

直接使用 Python 运行的方法和配置项见[本地部署](https://keta1930.github.io/icml2026-mcp/zh/docs/getting-started/local-deployment/)。

## 检索

Milvus 对同一个查询分别执行向量 COSINE 检索和原生 BM25 检索，再用标准 RRF 合并两份排名。DashScope 负责生成查询和论文向量，SQLite 保存会话、请求限制和 embedding 缓存。

## 后续方向

项目的长期工作是提高检索稳健性，并在许可和元数据质量可确认的前提下增加开放学术数据源。

## 开发

```bash
uv sync
uv run python -m pytest
```

## License

[MIT](LICENSE)
