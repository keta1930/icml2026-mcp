#!/bin/sh
set -eu

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
expected_lineage="/etc/letsencrypt/live/icml.agent4sci.com"
renewed_lineage="${RENEWED_LINEAGE:-}"
container_name="icml-mcp-tls"

[ -n "$renewed_lineage" ] || exit 0
[ -d "$expected_lineage" ] || exit 0
[ -d "$renewed_lineage" ] || exit 0

if [ "$(readlink -f -- "$renewed_lineage")" != "$(readlink -f -- "$expected_lineage")" ]; then
    exit 0
fi

if [ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != "true" ]; then
    exit 0
fi

docker exec "$container_name" nginx -t
docker kill --signal=HUP "$container_name" >/dev/null
