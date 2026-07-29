#!/bin/sh
set -eu

bootstrap_secret="${ICML_BOOTSTRAP_DASHSCOPE_KEY_FILE:-/run/secrets/dashscope_api_key}"
runtime_secret="${ICML_PAPERS_DASHSCOPE_API_KEY_FILE:-/run/icml-runtime-secrets/dashscope_api_key}"
runtime_secret_dir="$(dirname -- "$runtime_secret")"

if [ "$(id -u)" -eq 0 ]; then
    mkdir -p "$runtime_secret_dir"
    chown root:root "$runtime_secret_dir"
    chmod 0700 "$runtime_secret_dir"
    if [ -f "$bootstrap_secret" ]; then
        install -m 0400 "$bootstrap_secret" "$runtime_secret"
        chown app:app "$runtime_secret"
    fi
    chown app:app "$runtime_secret_dir"

    for writable_dir in ${ICML_WRITABLE_DIRS:-}; do
        case "$writable_dir" in
            /var/lib/icml-paper-search/index|/var/lib/icml-paper-search/state) ;;
            *) echo "invalid writable directory" >&2; exit 1 ;;
        esac
        mkdir -p "$writable_dir"
        chown app:app "$writable_dir"
    done
    exec gosu app:app "$0" "$@"
fi

exec "$@"
