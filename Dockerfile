ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ARG UV_VERSION=0.9.29

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM ${PYTHON_IMAGE} AS runtime

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="ICML 2026 Paper Search MCP" \
      org.opencontainers.image.description="Single-process Chroma and DashScope MCP service" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV HOME=/tmp \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ICML_PAPERS_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && install --directory --owner app --group app \
        /var/lib/icml-paper-search/index \
        /var/lib/icml-paper-search/state \
        /run/icml-runtime-secrets

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=root:root data ./data
COPY --chown=root:root --chmod=0555 deploy/container-entrypoint.sh /usr/local/bin/icml-container-entrypoint

EXPOSE 20441

USER 10001:10001

ENTRYPOINT ["icml-container-entrypoint"]
CMD ["icml-paper-search-mcp"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:20441/health/ready', timeout=4).read()"]

STOPSIGNAL SIGTERM
