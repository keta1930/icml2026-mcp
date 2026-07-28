#!/bin/sh
set -eu

bootstrap_secret="${ICML_BOOTSTRAP_DASHSCOPE_KEY_FILE:-/run/secrets/dashscope_api_key}"
runtime_secret="${ICML_PAPERS_DASHSCOPE_API_KEY_FILE:-/run/icml-runtime-secrets/dashscope_api_key}"
runtime_secret_dir="$(dirname -- "$runtime_secret")"
runtime_index_generation=""
runtime_index_backup=""
runtime_root=""
runtime_had_previous=0
runtime_publish_started=0

path_exists() {
    [ -e "$1" ] || [ -L "$1" ]
}

rollback_runtime_index() {
    set +e
    if [ "$runtime_publish_started" -eq 1 ]; then
        if [ "$runtime_had_previous" -eq 1 ] \
            && path_exists "$runtime_index_backup"; then
            if path_exists "$runtime_root"; then
                rm -rf -- "$runtime_root"
            fi
            mv -- "$runtime_index_backup" "$runtime_root"
        elif [ "$runtime_had_previous" -eq 0 ] && path_exists "$runtime_root"; then
            rm -rf -- "$runtime_root"
        fi
    fi
    if [ -n "$runtime_index_generation" ] \
        && path_exists "$runtime_index_generation"; then
        rm -rf -- "$runtime_index_generation"
    fi
}

handle_exit() {
    status="$1"
    trap - EXIT HUP INT TERM
    rollback_runtime_index
    exit "$status"
}

fail_runtime_index() {
    echo "runtime index preparation failed" >&2
    exit 1
}

validate_version_shape() {
    version_path="$1"
    if [ ! -d "$version_path" ] || [ -L "$version_path" ] \
        || [ ! -f "$version_path/manifest.json" ] \
        || [ -L "$version_path/manifest.json" ] \
        || [ ! -d "$version_path/chroma" ] \
        || [ -L "$version_path/chroma" ] \
        || [ ! -f "$version_path/bm25.sqlite3" ] \
        || [ -L "$version_path/bm25.sqlite3" ]; then
        fail_runtime_index
    fi
    version_symlink="$(find "$version_path" -type l -print -quit)" \
        || fail_runtime_index
    if [ -n "$version_symlink" ]; then
        fail_runtime_index
    fi
}

validate_runtime_generation() {
    copied_version="$1"
    validate_version_shape "$copied_version"
    if ! python - "$copied_version" >/dev/null 2>&1 <<'PY'
import sys
from pathlib import Path

from icml_paper_search.config import get_settings
from icml_paper_search.manifest import validate_index_version

validate_index_version(Path(sys.argv[1]), settings=get_settings())
PY
    then
        fail_runtime_index
    fi
}

recover_runtime_state() {
    state_root="$1"
    runtime_index_backup="$state_root/.runtime-index.backup"
    if path_exists "$runtime_index_backup"; then
        if [ -L "$runtime_index_backup" ]; then
            fail_runtime_index
        fi
        if path_exists "$runtime_root"; then
            rm -rf -- "$runtime_index_backup" || fail_runtime_index
        else
            mv -- "$runtime_index_backup" "$runtime_root" \
                || fail_runtime_index
        fi
    fi
    for stale_generation in "$state_root"/.runtime-index.generation.*; do
        if ! path_exists "$stale_generation"; then
            continue
        fi
        case "$stale_generation" in
            "$state_root"/.runtime-index.generation.*) ;;
            *) fail_runtime_index ;;
        esac
        rm -rf -- "$stale_generation" || fail_runtime_index
    done
    runtime_index_backup=""
}

prepare_runtime_index() {
    canonical_root="${ICML_BOOTSTRAP_INDEX_ROOT:-}"
    runtime_root="${ICML_PAPERS_INDEX_ROOT:-}"
    state_db="${ICML_PAPERS_STATE_DB_PATH:-}"

    case "$canonical_root:$runtime_root:$state_db" in
        /*:/*:/*) ;;
        *) fail_runtime_index ;;
    esac

    state_root="$(dirname -- "$state_db")"
    if [ "$state_root" = "/" ] || [ "$runtime_root" != "$state_root/runtime-index" ]; then
        fail_runtime_index
    fi
    lock_path="$state_root/.runtime-index.lock"
    if [ -L "$lock_path" ]; then
        fail_runtime_index
    fi
    exec 9>>"$lock_path" || fail_runtime_index
    flock -n 9 || fail_runtime_index
    recover_runtime_state "$state_root"

    current_path="$canonical_root/current"
    if [ ! -f "$current_path" ] || [ -L "$current_path" ]; then
        fail_runtime_index
    fi
    build_id="$(cat -- "$current_path")" || fail_runtime_index
    case "$build_id" in
        ""|.*|*[!A-Za-z0-9._-]*) fail_runtime_index ;;
        [A-Za-z0-9]*) ;;
        *) fail_runtime_index ;;
    esac
    if [ "${#build_id}" -gt 128 ]; then
        fail_runtime_index
    fi

    canonical_version="$canonical_root/versions/$build_id"
    validate_version_shape "$canonical_version"

    runtime_index_generation="$(
        mktemp -d "$state_root/.runtime-index.generation.XXXXXX"
    )" || fail_runtime_index
    case "$runtime_index_generation" in
        "$state_root"/.runtime-index.generation.*) ;;
        *) fail_runtime_index ;;
    esac
    if ! mkdir -p "$runtime_index_generation/versions"; then
        fail_runtime_index
    fi
    if ! cp -a -- \
        "$canonical_version" "$runtime_index_generation/versions/$build_id"; then
        fail_runtime_index
    fi
    if ! printf '%s\n' "$build_id" >"$runtime_index_generation/current"; then
        fail_runtime_index
    fi

    if ! chmod -R u+rwX "$runtime_index_generation"; then
        fail_runtime_index
    fi
    validate_runtime_generation "$runtime_index_generation/versions/$build_id"
    if [ "$(cat -- "$current_path")" != "$build_id" ]; then
        fail_runtime_index
    fi

    runtime_index_backup="$state_root/.runtime-index.backup"
    if path_exists "$runtime_root"; then
        runtime_had_previous=1
        runtime_publish_started=1
        if ! mv -- "$runtime_root" "$runtime_index_backup"; then
            fail_runtime_index
        fi
    else
        runtime_had_previous=0
        runtime_publish_started=1
    fi
    if ! mv -- "$runtime_index_generation" "$runtime_root"; then
        fail_runtime_index
    fi
    runtime_index_generation=""
    runtime_publish_started=0
    if [ "$runtime_had_previous" -eq 1 ]; then
        if ! rm -rf -- "$runtime_index_backup"; then
            echo "runtime index backup cleanup deferred" >&2
        fi
    fi
    runtime_index_backup=""
    runtime_had_previous=0
    flock -u 9 || fail_runtime_index
    exec 9>&-
}

trap 'handle_exit $?' EXIT
trap 'exit 1' HUP INT TERM

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
    exec gosu app:app "$0" "$@"
fi

if [ -r "$bootstrap_secret" ]; then
    install -m 0400 "$bootstrap_secret" "$runtime_secret"
fi

if [ -n "${ICML_BOOTSTRAP_INDEX_ROOT:-}" ]; then
    prepare_runtime_index
fi

exec "$@"
