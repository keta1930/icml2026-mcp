# AGENTS.md

## Scope

This repository serves hybrid search over 6,341 ICML 2026 papers through six MCP tools. DashScope produces 1,024-dimensional embeddings. Milvus 2.5 stores the paper records and performs native COSINE and BM25 retrieval. Milvus combines both rankings with standard RRF. SQLite stores sessions, rate limits, provider-minute usage, and the query-embedding cache.

## Package structure

```text
src/icml_paper_search/
├── cli/          command-line entry point
├── corpus/       source-snapshot loading and catalog data
├── domain/       models and public errors
├── embeddings/   DashScope client
├── indexing/     Milvus collection builder and manifest
├── milvus/       schema and repository
├── runtime/      SQLite sessions, limits, quotas, and cache
├── search/       application service and output formatting
└── transport/    FastMCP and HTTP health routes
```

Dependencies flow from `transport` and `cli` toward `search`, then through domain-specific interfaces. Runtime state must not leak into the domain models. The Milvus SDK stays behind `milvus/` and `indexing/`.

## Retrieval

- Each query issues one dense COSINE request and one native BM25 request to the same Milvus collection.
- Combine the two rankings with `RRFRanker()` and no weights.
- Apply `area`, `paper_type`, and `spotlight` filters to both requests.
- Search one fixed candidate window. Do not expand it automatically to fill a deduplicated result page.
- Keep the returned score in `[0, 1]`. It is useful only for ordering one query.
- Validate the current manifest, full paper-ID set, schema, BM25 function, and indexes without contacting DashScope.

## Sessions and runtime state

- A request without `session_id` is stateless. A session suppresses papers already returned or fetched within that capability.
- Store only hashes of session IDs, client addresses, and queries.
- Commit session deduplication and returned papers in one SQLite transaction.
- Enforce session capacity, TTL, round count, seen-paper count, and output limits.
- Use the current SQLite schema only. A database with another schema must be rejected.

## Embeddings and provider limits

- `qwen3.7-text-embedding` uses the DashScope Beijing endpoint and 1,024 dimensions.
- Use `text_type=query` for searches and `text_type=document` for index builds. A document batch contains at most 20 inputs.
- Validate status, result count, input positions, dimensions, finite values, and provider token usage.
- Retry only retryable transport and provider failures within the configured deadline.
- Query calls pass through the persistent cache, single-flight, a 64-call semaphore, and atomic RPM/TPM reservations.
- Default provider limits are 24,000 RPM and 1,000,000 TPM. Do not add daily or monthly application quotas.
- Resolve keys only through `Settings`; never expose keys, provider bodies, input text, or vectors in logs and public errors.

## Index builds

- Join `data/index.json` and `data/abstracts.json` by OpenReview ID.
- `icml-papers build` is an offline preflight. Network calls require `--execute` and a positive `--max-input-tokens` value.
- Create the dense field, analyzed document field, BM25 function, sparse field, COSINE index, and BM25 sparse index through pymilvus.
- Record progress after complete insert batches. Resume only when the checkpoint and collection exactly match the current build.
- Publish `current.json` only after schema, row count, full ID set, vector dimension, and hybrid-search validation succeed.

## Transport and deployment

- FastMCP construction, imports, liveness, and readiness must not call DashScope.
- Offload synchronous search and SQLite work from the event loop with bounded worker limiters.
- Release admission capacity on success, failure, timeout, and cancellation.
- Public errors and MCP output are English and must not include internal exception text.
- The reference deployment uses the bundled Milvus 2.5 Standalone Compose stack and one application process.
- Public proxy headers are accepted only from configured trusted networks. Use exact public Host and Origin values.

## Validation

Run the relevant checks before committing:

```bash
uv run python -m pytest
docker compose -f deploy/compose.yaml -f deploy/milvus.compose.yaml config --quiet
cd docs && pnpm build
```

Tests must stay offline unless an integration test is explicitly marked and isolated. Update English and Chinese documentation together when public behavior changes.
