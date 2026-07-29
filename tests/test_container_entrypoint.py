"""Validate the container entrypoint and bundled Milvus deployment contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _compose_config() -> dict:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "deploy/compose.yaml",
            "-f",
            "deploy/milvus.compose.yaml",
            "--profile",
            "build",
            "--profile",
            "production",
            "--profile",
            "tools",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_compose_bundles_pinned_milvus_standalone_stack():
    services = _compose_config()["services"]
    assert services["standalone"]["image"] == "milvusdb/milvus:v2.5.27"
    assert services["etcd"]["image"] == "quay.io/coreos/etcd:v3.5.18"
    assert services["minio"]["image"] == "minio/minio:RELEASE.2024-05-28T17-19-04Z"
    assert services["attu"]["image"] == "zilliz/attu:v2.5.10"
    assert (
        services["standalone"]["depends_on"]["etcd"]["condition"] == "service_healthy"
    )
    assert (
        services["standalone"]["depends_on"]["minio"]["condition"] == "service_healthy"
    )


def test_application_and_builder_use_the_same_milvus_and_manifest_volume():
    services = _compose_config()["services"]
    app = services["icml-mcp"]
    builder = services["build-index"]
    assert app["environment"]["ICML_PAPERS_MILVUS_URI"] == "http://standalone:19530"
    assert builder["environment"]["ICML_PAPERS_MILVUS_URI"] == "http://standalone:19530"
    assert app["depends_on"]["standalone"]["condition"] == "service_healthy"
    assert builder["depends_on"]["standalone"]["condition"] == "service_healthy"
    assert any(
        mount["target"] == "/var/lib/icml-paper-search/index" and mount["read_only"]
        for mount in app["volumes"]
    )
    assert any(
        mount["target"] == "/var/lib/icml-paper-search/index"
        and not mount.get("read_only", False)
        for mount in builder["volumes"]
    )


def test_public_service_capacity_matches_runtime_settings():
    nginx = (ROOT / "deploy/nginx.conf").read_text(encoding="utf-8")
    assert "rate=600r/m" in nginx
    assert "burst=100 nodelay" in nginx
    assert "limit_conn mcp_global_connections 512" in nginx


def test_entrypoint_only_prepares_secret_and_explicit_writable_directories():
    script = (ROOT / "deploy/container-entrypoint.sh").read_text(encoding="utf-8")
    assert "ICML_WRITABLE_DIRS" in script
    assert "/var/lib/icml-paper-search/index" in script
    assert "/var/lib/icml-paper-search/state" in script
    assert "exec gosu app:app" in script
    assert "cp -a" not in script
    subprocess.run(
        ["sh", "-n", str(ROOT / "deploy/container-entrypoint.sh")], check=True
    )
