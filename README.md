# ICML Paper Search MCP

[English](README.md) | [简体中文](README.zh-CN.md) | [Documentation](https://keta1930.github.io/icml2026-mcp/en/docs/)

An open-source MCP service for searching 6,341 ICML 2026 papers. It provides six tools for search, paper lookup, corpus metadata, and search sessions. Within a session, papers already returned are not returned again.

## Add the public MCP

Claude Code:

```bash
claude mcp add --transport http icml-papers --scope user \
  https://icml.agent4sci.com:20441/mcp
```

Generic Streamable HTTP configuration:

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

The public instance is free to use while its shared quota is available.

## Tools

| Tool | Purpose |
|---|---|
| `open_search_session` | Start a session |
| `search_papers` | Search papers and optionally deduplicate within a session |
| `search_session_status` | Read session status |
| `close_search_session` | Close a session |
| `get_papers` | Fetch papers by OpenReview ID |
| `list_paper_areas` | List research areas and paper types |

## Self-host with Docker

Requirements: Docker Compose and a DashScope key with access to `qwen3.7-text-embedding`.

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

The local MCP endpoint is `http://127.0.0.1:20442/mcp`. The Compose files include Milvus Standalone, etcd, and MinIO. Index data and runtime state are stored in Docker volumes.

See [Local deployment](https://keta1930.github.io/icml2026-mcp/en/docs/getting-started/local-deployment/) for direct Python setup and configuration.

## Retrieval

Milvus runs dense COSINE search and native BM25 search over each query, then combines the two rankings with standard RRF. DashScope supplies the query and document embeddings. SQLite stores session state, request limits, and the embedding cache.

## Direction

The long-term work is to make retrieval more robust and add open scholarly data sources whose licensing and metadata quality can be verified.

## Development

```bash
uv sync
uv run python -m pytest
```

## License

[MIT](LICENSE)
