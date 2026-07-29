# ICML Paper Search MCP

[English](README.md) | [简体中文](README.zh-CN.md) | [Documentation](https://keta1930.github.io/icml2026-mcp/en/docs/)

An open-source MCP retrieval service for 6,341 ICML 2026 papers. It keeps the corpus and indexes local, combines Chroma dense retrieval with SQLite FTS5 BM25 through weighted RRF, and uses persistent capability sessions to avoid returning the same paper across research rounds.

Self-hosting requires no Milvus, GPU, local embedding model, or external database service—only Python, disk space, and a DashScope API key. You can also connect directly to the public instance.

## Why use it

- **Built for agents**: MCP tools cover search, exact retrieval, filtering, and session status.
- **Hybrid retrieval**: dense search captures conceptual relevance while BM25 preserves exact terms and specialized names.
- **No repeats across rounds**: sessions survive restarts, and concurrent searches in one session cannot commit the same paper twice.
- **Local persistence**: paper snapshots, Chroma, FTS5, and runtime SQLite remain on one machine.
- **Bounded cost**: index builds require an explicit Token cap; online search has caching, concurrency limits, and daily/monthly budgets.
- **Public-service controls**: persistent rate limits, bounded queues, health checks, trusted proxies, and redacted errors are included.

## Connect to the public service

Public MCP endpoint:

```text
https://icml.agent4sci.com:20441/mcp
```

Register it with Claude Code:

```bash
claude mcp add --transport http icml-papers --scope user https://icml.agent4sci.com:20441/mcp
```

Equivalent MCP client configuration:

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

`/mcp` is a Streamable HTTP protocol endpoint, not a web page or conventional REST JSON API. Opening it in a browser normally returns HTTP 406 with `Client must accept text/event-stream`; that is expected content negotiation. Use a Streamable HTTP MCP client whose SDK manages initialization and the transport session. For manual service checks, use:

```text
https://icml.agent4sci.com:20441/health/live
https://icml.agent4sci.com:20441/health/ready
```

The response-header `Mcp-Session-Id` identifies the transport session. The `session_id` returned by `open_search_session` is a business capability session for paper deduplication. Do not interchange them.

The public instance is anonymous and free to try, with a shared quota and no availability guarantee. New search queries are sent to DashScope for embedding; never include secrets, personal data, or unpublished material in queries or session labels.

## Local quickstart

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a DashScope API key with access to `qwen3.7-text-embedding`.

```bash
uv sync
cp .env.example .env
```

Set either a direct key or a secret file in `.env`; the file takes precedence:

```dotenv
ICML_PAPERS_DASHSCOPE_API_KEY=your-dashscope-api-key
# ICML_PAPERS_DASHSCOPE_API_KEY_FILE=/absolute/path/to/dashscope_api_key
```

Run a zero-cloud preflight first:

```bash
uv run icml-papers info
uv run icml-papers build
```

By default, `build` prints a stable `build_id`, paper count, corpus hash, and conservative Token upper bound. After reviewing the budget, build explicitly:

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP>
```

After a network or quota interruption, resume only if the corpus, model, dimension, and batch size have not changed:

```bash
uv run icml-papers build --execute --max-input-tokens <TOKEN_CAP> --resume
```

Start the Streamable HTTP MCP service:

```bash
uv run icml-paper-search-mcp
curl -fsS http://127.0.0.1:20441/health/ready
```

The default MCP URL is `http://127.0.0.1:20441/mcp`. Register it for the current project:

```bash
claude mcp add --transport http icml-papers --scope project http://127.0.0.1:20441/mcp
```

See [Local Deployment](docs/content/docs/en/getting-started/local-deployment.mdx) for the complete build contract, paths, and recovery process.

## Recommended research flow

```text
open_search_session(label="diffusion for language models")
  -> search_papers("discrete diffusion language models", session_id=...)
  -> search_papers("masked diffusion text generation", session_id=...)
  -> search_session_status(session_id=...)
  -> close_search_session(session_id=...)
```

`session_id` is a bearer capability with at least 256 bits of randomness. Anyone holding it can access that session. The server stores only a SHA-256 hash and cannot list or recover the raw ID. The default idle TTL is 24 hours.

## MCP tools

| Tool | Purpose |
|---|---|
| `open_search_session` | Create an unenumerable capability session |
| `search_papers` | Run hybrid retrieval and return only unseen papers in a session |
| `search_session_status` | Inspect one session held by the caller |
| `close_search_session` | Close a session and return a bounded summary |
| `get_papers` | Fetch papers exactly by OpenReview ID |
| `list_paper_areas` | List valid `area` and `paper_type` values |

See the [MCP Tools Reference](docs/content/docs/en/reference/mcp-tools.mdx) for parameters, bounds, and error codes.

## Retrieval semantics

Each query performs one dense and one BM25 retrieval at a fixed depth, then applies weighted RRF. The default is `search_depth = top_k × 4`; the service does not descend into weaker results merely to fill `top_k`.

The result `score` is a fused rank normalized to `[0, 1]`. It is not cosine similarity, a probability, or an absolute relevance measure comparable across queries. Use it for ordering within the current result set and coarse `min_score` filtering.

## CLI

```bash
uv run icml-papers info
uv run icml-papers search "discrete diffusion language models"
uv run icml-papers search --session
```

Interactive search supports `--area`, `--type`, `--spotlight`, `--top-k`, and `--depth`. See the [CLI Reference](docs/content/docs/en/reference/cli.mdx).

## Data and runtime boundaries

- `data/` contains static metadata, abstracts, and taxonomy snapshots for 6,341 papers.
- `.data/index/` contains versioned Chroma, FTS5, manifests, and the atomic `current` pointer.
- `.data/state/runtime.sqlite3` contains sessions, rate limits, budgets, and the query-embedding cache.
- Only one application process may open a persistent Chroma directory. Do not scale with multiple workers or containers sharing it.
- Startup and readiness do not contact DashScope. The first uncached query creates an online embedding.

## Docker deployment

The image contains the application and corpus, not a DashScope key, built index, or runtime state. [`deploy/compose.yaml`](deploy/compose.yaml) provides an explicit `build-index` job, one application process, persistent mounts, Docker secrets, and a TLS sidecar.

```bash
docker build --build-arg VCS_REF=local -t icml-paper-search-mcp:local .
sh deploy/smoke-container.sh icml-paper-search-mcp:local
```

Before production, replace example domains, certificates, Host/Origin allowlists, and trusted proxy networks. See [Docker Deployment](docs/content/docs/en/operations/docker.mdx).

## Roadmap

Long-term work has two tracks: improve retrieval robustness across paraphrases, terminology, filter combinations, and fixed windows; and expand from ICML to more open scholarly sources while preserving provenance, venue/year isolation, and manifest contracts. See the full [Roadmap](docs/content/docs/en/roadmap.mdx).

## Development

Core tests are fully offline and need no key, network, Docker, GPU, or external database:

```bash
uv run python -m pytest
```

See [AGENTS.md](AGENTS.md) for architecture boundaries and change requirements.

## Documentation site

The Fumadocs site lives in [`docs/`](docs/) and covers concepts, tutorials, guides, Reference, operations, troubleshooting, and AI-readable delivery.

English documentation: <https://keta1930.github.io/icml2026-mcp/en/docs/>

```bash
cd docs
pnpm install
pnpm dev
```

The development server runs at `http://127.0.0.1:20440` by default. English docs are under `/en/docs/`, with `/en/llms.txt` and a clean `.md` form for every page.

## License

This project is licensed under the [MIT License](LICENSE).
