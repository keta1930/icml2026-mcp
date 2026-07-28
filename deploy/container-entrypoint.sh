#!/bin/sh
set -eu

bootstrap_secret="${ICML_BOOTSTRAP_DASHSCOPE_KEY_FILE:-/run/secrets/dashscope_api_key}"
runtime_secret="${ICML_PAPERS_DASHSCOPE_API_KEY_FILE:-/run/icml-runtime-secrets/dashscope_api_key}"
runtime_secret_dir="$(dirname -- "$runtime_secret")"

# Compose file secrets preserve host ownership. Root only prepares tmpfs, then drops itself.
if [ "$(id -u)" -eq 0 ]; then
    mkdir -p "$runtime_secret_dir"
    chown root:root "$runtime_secret_dir"
    chmod 0700 "$runtime_secret_dir"
    if [ -f "$bootstrap_secret" ]; then
        install -m 0400 "$bootstrap_secret" "$runtime_secret"
        chown app:app "$runtime_secret"
    fi
    chown app:app "$runtime_secret_dir"
    exec gosu app:app "$@"
fi

if [ -r "$bootstrap_secret" ]; then
    install -m 0400 "$bootstrap_secret" "$runtime_secret"
fi

exec "$@"
