# AGENTS.md

## Project Goal

This repository implements a single-node MCP search service for 6,341 ICML 2026 papers. Its core behavior combines DashScope query/document embeddings, Chroma COSINE, and SQLite FTS5 BM25 through weighted RRF, with persistent capability-session deduplication across search rounds.

Use `README.md` and `docs/content/docs/` as the user-facing sources for installation, operation, and tool usage. This file defines implementation and validation constraints.

## Repository Structure

```text
src/icml_paper_search/   Python service, index, and CLI
tests/                   Fully offline behavior and concurrency tests
data/                    Static ICML 2026 corpus snapshots shipped with the repository
deploy/                  Container entrypoint, Compose, Nginx, and smoke tests
docs/                    Fumadocs documentation subproject
```

Keep Python dependencies flowing from `server.py` / `cli.py` to `service.py`, then to retrieval, encoding, session, and state modules. MCP schema, client identity, and public error translation belong to the transport layer. Reusable business behavior belongs in the service layer or below.

Key modules:

| Module | Responsibility |
|---|---|
| `config.py` | `Settings`, paths, secret resolution, and combined validation |
| `catalog.py` | Corpus merging, index records, and filter statistics |
| `embedding.py` | `TextEncoder` protocol and DashScope implementation |
| `bm25.py` / `retrieval.py` | FTS5, Chroma, and weighted RRF |
| `manifest.py` / `ingest.py` | Version contract, preflight, checkpoints, and atomic publication |
| `state.py` / `sessions.py` | Runtime schema, transactions, and session lifecycle |
| `governance.py` | Rate limits, budgets, cache, single-flight, and concurrency control |
| `server.py` | FastMCP, Streamable HTTP, security, and health checks |
| `cli.py` | `info`, `build`, and `search` commands |

## Retrieval Invariants

- Run dense and BM25 retrieval exactly once per query at a fixed depth. The default window is `top_k × depth_multiplier`, capped by `max_search_depth`.
- Do not automatically expand retrieval to fill `top_k`. When the window yields too few new results, expose `truncated` and an output hint so the caller decides what to do.
- Dense retrieval uses Chroma COSINE and sparse retrieval uses FTS5 `bm25()`. Fuse only through manifest weights and `rrf_k`.
- Keep `SearchHit.score` within `[0, 1]`, with higher values better. Do not describe it as cosine similarity, probability, or a score calibrated across queries.
- Keep `area`, `paper_type`, and spotlight semantics consistent across both channels. Apply `min_score` after fusion.
- Always enforce `Settings` limits for query, `top_k`, `search_depth`, and `paper_ids`. Reject empty queries.

## Sessions and Runtime State

- Searches without `session_id` remain stateless. A session hides only papers already returned to that capability or recorded through `get_papers`.
- Record only papers actually returned. Filter and commit results for one session in a single atomic SQLite transaction.
- Raw `session_id` values contain at least 256 bits of entropy and are returned only to the caller. Store only capability hashes. Anonymous interfaces must not enumerate or recover sessions.
- Reject new sessions at capacity without evicting other active sessions. TTL, round count, seen count, and summaries remain bounded.
- Apply persistent source-based session-open rate limits before any session write. Restarts must not reset them.
- Enable WAL, foreign keys, busy timeout, and short transactions for runtime SQLite. Never store raw IP addresses, raw queries, or raw capabilities.

## Embeddings and Cost

- `qwen3.7-text-embedding` uses the DashScope Beijing-region native API with 1,024-dimensional output. Pass `text_type=query` for queries and `text_type=document` for papers, with at most 20 inputs per batch.
- Validate provider status, result count, `text_index`, dimensions, finite values, and `total_tokens`. Retry only network failures, 429, 408, and 5xx within the total deadline.
- Keep quota exhaustion, configuration errors, and transient failures on distinct public error contracts. Logs and exceptions must not contain provider bodies/messages, keys, inputs, or vectors.
- Resolve keys only through `Settings.resolve_dashscope_api_key()`, preferring the secret file. Real keys must never enter code, tests, images, documentation, logs, or command arguments.
- Route online query embeddings through persistent cache, single-flight, token budgets, and the provider semaphore. Reserve before the call and reconcile afterward; charge the reservation when sent usage is unknown.
- When actual usage exceeds the reservation, record an overrun and fail closed for subsequent online calls in the current day and month.

## Indexing and Builds

- Merge `data/index.json` and `data/abstracts.json` by OpenReview ID. Index a title even when the abstract is missing.
- Give Chroma only explicit embeddings; never download an implicit model. Chroma and FTS5 must contain identical paper-ID sets and corpus hashes.
- Bind the manifest to schema, source-file hashes, corpus hash, complete ID fingerprint, model, dimensions, text type, metric, tokenizer, fusion parameters, and collection name.
- Readiness validates the manifest, collection metadata, complete ID set, and sample dimensions against shipped source snapshots without cloud calls.
- `icml-papers build` runs preflight only by default. Provider calls must never occur during imports, server startup, readiness, or tests.
- Paid builds require both `--execute` and a positive token cap. Write complete batches to staging/checkpoint and atomically switch `current` only after complete validation.
- If usage exceeds the reservation, stop before writing the index or checkpoint, record a failure audit without source text, and prevent automatic resume of that staging directory.
- Resume only checkpoints whose build ID, corpus, model, dimensions, and batch size match and whose completed IDs are a strict corpus prefix.
- Old or incomplete manifests are incompatible and require an explicit rebuild. Failed builds must not replace the existing `current` version.

## Concurrency, Transport, and Deployment

- Initialize DashScope, Chroma, FTS5, and the runtime database lazily or per thread. Constructing FastMCP and starting a process must not read keys or access the provider.
- Offload synchronous provider, Chroma, and SQLite work from async handlers. Use separate limiters for slow search and lightweight state work.
- Reserve active-plus-waiting admission capacity before the first `await`. Return `server_busy` immediately when full, and release capacity on timeouts, cancellations, and exceptions.
- Allow only one application process to open the Chroma persistence directory. Do not share it across multiple workers, application containers, or concurrent writers.
- Mount the canonical index read-only in service containers. The entrypoint validates and copies one complete version into the writable `runtime-index` under state. The application and readiness open only that copy.
- Treat `runtime-index` as reproducible data and exclude it from cold backups. Back up the canonical index and non-derived state while the application is stopped.
- Public deployments use exact Host and Origin values. Accept proxy identity headers only when the peer belongs to an explicitly trusted network.
- `live` reports process liveness only. `ready` checks local state and index only. Map unknown exceptions to sanitized public errors.

## Configuration and Output

- Read configuration only through `Settings`, using the `ICML_PAPERS_` environment-variable prefix. Other modules must not read environment variables independently.
- Configuration changes update field constraints, combined validation, `.env.example`, tests, and the user-facing Reference together.
- `formatting.py` output enters agent context directly. Add fields only when they provide decision value, and keep abstract previews, empty-result explanations, and session summaries bounded.
- The current data model remains single-source ICML 2026. Do not describe research into multi-source support as implemented behavior in documentation or interfaces.

## Documentation Constraints

- `docs/` is an independent Next.js 16 and Fumadocs subproject. The Python runtime must not depend on Node.js packages.
- Organize documentation by user task, with public behavior derived from implementation, schema, and tests. Keep only entry points and quick starts in README; put details in the documentation site.
- `content/docs/` is the content source of truth. `.source/`, `.next/`, and `out/` are generated and must not be committed.
- Run `cd docs && pnpm build` after changing MDX. Update the matching `meta.json` for new pages and validate `/llms.txt` plus per-page `.md` routes.
- Architecture concept diagrams may use project assets; prefer Mermaid for exact processes. Keep each page under 300 lines and code snippets short.
- Documentation must not contain real secrets, internal server credentials, raw session IDs, raw queries, or provider responses.

## Validation Matrix

After code changes, run:

```bash
uv run python -m pytest
```

Add validation by change type:

| Change | Minimum validation |
|---|---|
| Retrieval | Dual-channel recall, fixed window, filter consistency, RRF range, stable ordering |
| Session/state | Atomic concurrent commits, isolation, TTL/capacity, restart persistence, no private data at rest |
| Governance | Session-open limits, budget overrun, cache, single-flight, admission cancellation release |
| Build/index | Offline preflight, cap, checkpoint, resume, failed-build preservation, complete fingerprints |
| MCP/security | Async schema, event loop, Host/Origin, trusted proxy, sanitized errors, health checks |
| Container | Non-root/read-only boundaries, secret mount, dynamic network, console-script smoke |
| Documentation | `cd docs && pnpm build`, navigation, internal links, llms, and per-page Markdown |

Modify only files required by the task. Report unrelated issues without opportunistic refactoring.
