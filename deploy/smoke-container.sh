#!/bin/sh
set -eu

image="${1:?usage: smoke-container.sh IMAGE}"

image_user="$(docker image inspect --format '{{.Config.User}}' "$image")"
if [ "$image_user" != "10001:10001" ]; then
    echo "unexpected image user: $image_user" >&2
    exit 1
fi

docker run --rm --entrypoint icml-papers "$image" --help >/dev/null
docker run --rm --entrypoint icml-papers "$image" info >/dev/null

set +e
server_output="$(
    timeout --signal=TERM --kill-after=5s 3s \
        docker run --rm \
        --env ICML_PAPERS_INDEX_ROOT=/var/lib/icml-paper-search/index \
        --env ICML_PAPERS_STATE_DB_PATH=/var/lib/icml-paper-search/state/smoke.sqlite3 \
        --entrypoint icml-paper-search-mcp "$image" --help 2>&1
)"
server_status=$?
set -e

case "$server_status" in
    0 | 124 | 137 | 143) ;;
    *)
        printf '%s\n' "$server_output" | sed -n '1,80p' >&2
        exit "$server_status"
        ;;
esac

if ! printf '%s\n' "$server_output" | grep -q "starting streamable-http MCP server"; then
    printf '%s\n' "$server_output" | sed -n '1,80p' >&2
    exit 1
fi

if ! printf '%s\n' "$server_output" | grep -q "Application startup complete"; then
    printf '%s\n' "$server_output" | sed -n '1,80p' >&2
    exit 1
fi

if printf '%s\n' "$server_output" | grep -q "Application startup failed"; then
    printf '%s\n' "$server_output" | sed -n '1,80p' >&2
    exit 1
fi

if printf '%s\n' "$server_output" | grep -Eq '/build/\.venv|bad interpreter|not found|No such file'; then
    printf '%s\n' "$server_output" | sed -n '1,80p' >&2
    exit 1
fi

docker run --rm --entrypoint sh "$image" -c '
    test "$(id -u):$(id -g)" = "10001:10001"
    test "$(head -n 1 /opt/venv/bin/icml-paper-search-mcp)" = "#!/opt/venv/bin/python3"
    test "$(head -n 1 /opt/venv/bin/icml-papers)" = "#!/opt/venv/bin/python3"
'

docker run --rm \
    --user 0:0 \
    --cap-drop ALL \
    --cap-add CHOWN \
    --cap-add SETGID \
    --cap-add SETUID \
    --security-opt no-new-privileges:true \
    --entrypoint sh \
    "$image" -c '
        mkdir -p /run/secrets
        install -m 0400 /dev/null /run/secrets/dashscope_api_key
        exec /usr/local/bin/icml-container-entrypoint sh -c \
            "test \"\$(id -u):\$(id -g)\" = \"10001:10001\" &&
             test \"\$(stat -c %U:%G /run/icml-runtime-secrets/dashscope_api_key)\" = \"app:app\" &&
             test \"\$(stat -c %a /run/icml-runtime-secrets/dashscope_api_key)\" = \"400\""
    '
