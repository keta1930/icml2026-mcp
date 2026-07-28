"""验证容器启动时的只读 canonical 与可写 runtime 索引边界。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from icml_paper_search.bm25 import BM25IndexBuilder
from icml_paper_search.catalog import (
    PaperRecord,
    corpus_hash,
    data_file_hashes,
    paper_ids_hash,
)
from icml_paper_search.config import Settings
from icml_paper_search.manifest import (
    IndexManifest,
    IndexPaths,
    collection_metadata_for,
    collection_name_for,
    promote_staging,
    resolve_current,
    validate_index_version,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "deploy" / "container-entrypoint.sh"
DIMENSION = 1024


def _run_entrypoint(
    canonical_root: Path,
    state_root: Path,
    command: str = "true",
    *,
    prepare_runtime: bool = True,
    runtime_root: Path | None = None,
    real_validation: bool = False,
    command_stubs: dict[str, str] | None = None,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """用测试目录执行真实容器入口。"""
    environment = _entrypoint_environment(
        canonical_root,
        state_root,
        prepare_runtime=prepare_runtime,
        runtime_root=runtime_root,
        real_validation=real_validation,
        command_stubs=command_stubs,
        extra_environment=extra_environment,
    )
    return subprocess.run(
        [str(ENTRYPOINT), "sh", "-c", command],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _entrypoint_environment(
    canonical_root: Path,
    state_root: Path,
    *,
    prepare_runtime: bool = True,
    runtime_root: Path | None = None,
    real_validation: bool = False,
    command_stubs: dict[str, str] | None = None,
    extra_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """构造可注入确定性 shell 命令的入口测试环境。"""
    environment = os.environ.copy()
    target_root = runtime_root or state_root / "runtime-index"
    if not prepare_runtime:
        target_root = canonical_root
    environment.update(
        {
            "ICML_PAPERS_INDEX_ROOT": str(target_root),
            "ICML_PAPERS_STATE_DB_PATH": str(state_root / "runtime.sqlite3"),
            "ICML_BOOTSTRAP_DASHSCOPE_KEY_FILE": str(state_root / "missing-key"),
            "ICML_PAPERS_DASHSCOPE_API_KEY_FILE": str(
                state_root / "runtime-secrets" / "missing-key"
            ),
        }
    )
    if prepare_runtime:
        environment["ICML_BOOTSTRAP_INDEX_ROOT"] = str(canonical_root)
    else:
        environment.pop("ICML_BOOTSTRAP_INDEX_ROOT", None)
    stubs = dict(command_stubs or {})
    if not real_validation:
        stubs.setdefault("python", "#!/bin/sh\nexit 0\n")
    if stubs:
        stub_dir = state_root / "test-bin"
        stub_dir.mkdir(exist_ok=True)
        for name, content in stubs.items():
            path = stub_dir / name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)
        environment["PATH"] = f"{stub_dir}:{environment['PATH']}"
    if extra_environment:
        environment.update(extra_environment)
    return environment


def _write_minimal_version(root: Path, *, build_id: str = "v1-test") -> Path:
    """写出用于入口边界测试的完整版本目录形状。"""
    version = root / "versions" / build_id
    (version / "chroma").mkdir(parents=True)
    (version / "manifest.json").write_text(
        json.dumps({"build_id": build_id}), encoding="utf-8"
    )
    (version / "bm25.sqlite3").write_bytes(b"bm25")
    (version / "chroma" / "chroma.sqlite3").write_bytes(b"chroma")
    (root / "current").write_text(f"{build_id}\n", encoding="utf-8")
    return version


def _tree_fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    """记录普通文件的 mode、mtime 与内容摘要。"""
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        metadata = path.stat()
        result[str(path.relative_to(root))] = (
            stat.S_IMODE(metadata.st_mode),
            metadata.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result


def _make_real_index(tmp_path: Path) -> tuple[Settings, Path]:
    """创建无需 provider 的单论文 Chroma/FTS5 测试索引。"""
    import chromadb

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical_root = tmp_path / "canonical"
    settings = Settings(
        _env_file=None,
        data_dir=data_dir,
        index_root=canonical_root,
        state_db_path=tmp_path / "state" / "runtime.sqlite3",
    )
    fields = {
        "orid": "paper-1",
        "i": 1,
        "title": "Runtime index",
        "abstract": "local chroma readiness",
        "text": "Runtime index\nlocal chroma readiness",
        "authors": json.dumps(["Alice"]),
        "area": "Systems",
        "sub": "",
        "type": "Poster",
        "spot": False,
        "pid": "1",
        "arxiv": "",
    }
    record = PaperRecord(fields=fields, text=fields["text"])
    (data_dir / "index.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "i": 1,
                        "pid": "1",
                        "orid": "paper-1",
                        "title": fields["title"],
                        "authors": ["Alice"],
                        "area": "Systems",
                        "sub": "",
                        "type": "Poster",
                        "spot": False,
                        "arxiv": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "abstracts.json").write_text(
        json.dumps({"paper-1": fields["abstract"]}), encoding="utf-8"
    )

    digest = corpus_hash([record])
    build_id = f"v1-{digest[:16]}-{DIMENSION}d"
    paths = IndexPaths.for_root(canonical_root)
    staging = paths.staging(build_id)
    staging.mkdir(parents=True)
    collection_name = collection_name_for(build_id)
    manifest = IndexManifest.create(
        build_id=build_id,
        corpus_hash=digest,
        paper_ids_hash=paper_ids_hash([record.paper_id]),
        record_count=1,
        data_file_hashes=data_file_hashes(settings),
        settings=settings,
        collection_name=collection_name,
    )
    client = chromadb.PersistentClient(path=str(staging / "chroma"))
    collection = client.create_collection(
        name=collection_name,
        metadata=collection_metadata_for(manifest),
        embedding_function=None,
    )
    vector = [0.0] * DIMENSION
    vector[0] = 1.0
    collection.add(
        ids=[record.paper_id],
        embeddings=[vector],
        documents=[record.text],
        metadatas=[record.fields],
    )
    with BM25IndexBuilder(staging / "bm25.sqlite3") as builder:
        builder.add_records([record])
        builder.finalize(corpus_hash=digest)
    write_manifest(staging, manifest)
    promote_staging(canonical_root, build_id)
    return settings, canonical_root


def _set_tree_read_only(root: Path, read_only: bool) -> None:
    """切换测试 canonical 树的写权限。"""
    directory_mode = 0o555 if read_only else 0o755
    file_mode = 0o444 if read_only else 0o644
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(directory_mode if path.is_dir() else file_mode)
    root.chmod(directory_mode)


def _write_previous_runtime(state_root: Path) -> Path:
    """写出可识别的上一代 runtime。"""
    runtime_root = state_root / "runtime-index"
    runtime_root.mkdir(parents=True)
    (runtime_root / "previous-generation").write_text("keep", encoding="utf-8")
    return runtime_root


def _assert_no_publish_residue(state_root: Path) -> None:
    """断言发布事务没有遗留 generation 或 backup。"""
    assert not any(state_root.glob(".runtime-index.generation.*"))
    assert not (state_root / ".runtime-index.backup").exists()


def test_read_only_canonical_prepares_valid_runtime_without_mutation(tmp_path):
    settings, canonical_root = _make_real_index(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    _set_tree_read_only(canonical_root, True)
    before = _tree_fingerprint(canonical_root)

    try:
        result = _run_entrypoint(
            canonical_root,
            state_root,
            real_validation=True,
            extra_environment={"ICML_PAPERS_DATA_DIR": str(settings.data_dir)},
        )
        assert result.returncode == 0, result.stderr

        runtime_root = state_root / "runtime-index"
        runtime_settings = Settings(
            _env_file=None,
            data_dir=settings.data_dir,
            index_root=runtime_root,
            state_db_path=settings.state_db_path,
        )
        manifest = validate_index_version(
            resolve_current(runtime_root), settings=runtime_settings
        )
        assert manifest.record_count == 1
        assert _tree_fingerprint(canonical_root) == before
    finally:
        _set_tree_read_only(canonical_root, False)


def test_repeated_start_rebuilds_clean_runtime_copy(tmp_path):
    canonical_root = tmp_path / "canonical"
    canonical_version = _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()

    first = _run_entrypoint(canonical_root, state_root)
    assert first.returncode == 0, first.stderr
    runtime_marker = state_root / "runtime-index" / "versions" / "v1-test" / "drift"
    runtime_marker.write_text("chroma changed this generation", encoding="utf-8")
    (canonical_version / "canonical-only").write_text("fresh", encoding="utf-8")

    second = _run_entrypoint(canonical_root, state_root)
    assert second.returncode == 0, second.stderr
    assert not runtime_marker.exists()
    assert (
        state_root
        / "runtime-index"
        / "versions"
        / "v1-test"
        / "canonical-only"
    ).read_text(encoding="utf-8") == "fresh"


def test_invalid_version_fails_closed_without_replacing_previous_runtime(tmp_path):
    canonical_root = tmp_path / "canonical"
    version = _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    old_runtime = state_root / "runtime-index"
    old_runtime.mkdir(parents=True)
    marker = old_runtime / "previous-generation"
    marker.write_text("keep", encoding="utf-8")
    (version / "escape").symlink_to(tmp_path / "outside-secret")

    result = _run_entrypoint(canonical_root, state_root)

    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not any(state_root.glob(".runtime-index.tmp.*"))


@pytest.mark.parametrize("current", ["../secret", "v1/../../secret", ".", ""])
def test_current_rejects_path_traversal_and_does_not_copy_secret(tmp_path, current):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    (canonical_root / "current").write_text(current, encoding="utf-8")
    sentinel = "sentinel-runtime-copy-secret"
    (tmp_path / "secret").write_text(sentinel, encoding="utf-8")
    state_root = tmp_path / "state"
    state_root.mkdir()

    result = _run_entrypoint(canonical_root, state_root)

    assert result.returncode != 0
    assert not (state_root / "runtime-index").exists()
    assert sentinel not in result.stdout
    assert sentinel not in result.stderr


def test_copy_contains_only_current_version_unit_not_adjacent_secret(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    sentinel = "sentinel-adjacent-secret"
    (canonical_root / "do-not-copy.secret").write_text(sentinel, encoding="utf-8")
    state_root = tmp_path / "state"
    state_root.mkdir()

    result = _run_entrypoint(canonical_root, state_root)

    assert result.returncode == 0, result.stderr
    runtime_root = state_root / "runtime-index"
    assert not (runtime_root / "do-not-copy.secret").exists()
    assert sentinel not in "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in runtime_root.rglob("*")
        if path.is_file()
    )


def test_runtime_target_must_be_the_fixed_state_child(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    unsafe_target = tmp_path / "outside-runtime"
    unsafe_target.mkdir()
    marker = unsafe_target / "keep"
    marker.write_text("untouched", encoding="utf-8")

    result = _run_entrypoint(
        canonical_root, state_root, runtime_root=unsafe_target
    )

    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_runtime_symlink_is_replaced_without_touching_its_target(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    outside = tmp_path / "outside-runtime"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("untouched", encoding="utf-8")
    (state_root / "runtime-index").symlink_to(outside, target_is_directory=True)

    result = _run_entrypoint(canonical_root, state_root)

    runtime_root = state_root / "runtime-index"
    assert result.returncode == 0, result.stderr
    assert not runtime_root.is_symlink()
    assert (runtime_root / "current").read_text(encoding="utf-8") == "v1-test\n"
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_concurrent_start_fails_closed_while_first_holds_state_lock(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    gate = state_root / "gate"
    gate.mkdir()
    ready = gate / "ready"
    release = gate / "release"
    os.mkfifo(ready)
    os.mkfifo(release)
    cp_stub = """#!/bin/sh
if mkdir "$ICML_TEST_GATE_OWNER" 2>/dev/null; then
    printf ready >"$ICML_TEST_READY_FIFO"
    IFS= read -r _release <"$ICML_TEST_RELEASE_FIFO"
fi
exec /usr/bin/cp "$@"
"""
    environment = _entrypoint_environment(
        canonical_root,
        state_root,
        command_stubs={"cp": cp_stub},
        extra_environment={
            "ICML_TEST_GATE_OWNER": str(gate / "owner"),
            "ICML_TEST_READY_FIFO": str(ready),
            "ICML_TEST_RELEASE_FIFO": str(release),
        },
    )
    first = subprocess.Popen(
        [str(ENTRYPOINT), "true"],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert ready.read_text(encoding="utf-8") == "ready"
        second = subprocess.run(
            [str(ENTRYPOINT), "true"],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert second.returncode != 0
    finally:
        release.write_text("continue\n", encoding="utf-8")
    _stdout, stderr = first.communicate(timeout=10)

    assert first.returncode == 0, stderr
    assert (state_root / "runtime-index" / "current").is_file()
    _assert_no_publish_residue(state_root)


def test_state_lock_is_released_before_application_exec(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    lock_path = state_root / ".runtime-index.lock"

    result = _run_entrypoint(
        canonical_root,
        state_root,
        f'exec 8>>"{lock_path}"; flock -n 8',
    )

    assert result.returncode == 0, result.stderr


def test_second_publish_rename_failure_restores_previous_runtime(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    runtime_root = _write_previous_runtime(state_root)
    counter = state_root / "mv-counter"
    mv_stub = """#!/bin/sh
count=0
if [ -f "$ICML_TEST_MV_COUNTER" ]; then
    count="$(cat "$ICML_TEST_MV_COUNTER")"
fi
count=$((count + 1))
printf '%s\n' "$count" >"$ICML_TEST_MV_COUNTER"
if [ "$count" -eq 2 ]; then
    exit 1
fi
exec /usr/bin/mv "$@"
"""

    result = _run_entrypoint(
        canonical_root,
        state_root,
        command_stubs={"mv": mv_stub},
        extra_environment={"ICML_TEST_MV_COUNTER": str(counter)},
    )

    assert result.returncode != 0
    assert (runtime_root / "previous-generation").read_text() == "keep"
    _assert_no_publish_residue(state_root)


def test_signal_after_backup_rename_restores_previous_runtime(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    runtime_root = _write_previous_runtime(state_root)
    counter = state_root / "mv-counter"
    mv_stub = """#!/bin/sh
count=0
if [ -f "$ICML_TEST_MV_COUNTER" ]; then
    count="$(cat "$ICML_TEST_MV_COUNTER")"
fi
count=$((count + 1))
printf '%s\n' "$count" >"$ICML_TEST_MV_COUNTER"
/usr/bin/mv "$@" || exit 1
if [ "$count" -eq 1 ]; then
    kill -TERM "$PPID"
fi
"""

    result = _run_entrypoint(
        canonical_root,
        state_root,
        command_stubs={"mv": mv_stub},
        extra_environment={"ICML_TEST_MV_COUNTER": str(counter)},
    )

    assert result.returncode != 0
    assert (runtime_root / "previous-generation").read_text() == "keep"
    _assert_no_publish_residue(state_root)


def test_partial_backup_cleanup_failure_keeps_committed_runtime(tmp_path):
    settings, canonical_root = _make_real_index(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    _write_previous_runtime(state_root)
    backup = state_root / ".runtime-index.backup"
    app_marker = state_root / "app-ran"
    rm_stub = """#!/bin/sh
if [ "${3:-}" = "$ICML_TEST_BACKUP_PATH" ]; then
    /usr/bin/rm -f "$ICML_TEST_BACKUP_PATH/previous-generation"
    exit 1
fi
exec /usr/bin/rm "$@"
"""
    validation_environment = {
        "ICML_PAPERS_DATA_DIR": str(settings.data_dir),
        "ICML_TEST_BACKUP_PATH": str(backup),
        "ICML_TEST_APP_MARKER": str(app_marker),
    }

    result = _run_entrypoint(
        canonical_root,
        state_root,
        'printf ran >"$ICML_TEST_APP_MARKER"',
        real_validation=True,
        command_stubs={"rm": rm_stub},
        extra_environment=validation_environment,
    )

    assert result.returncode == 0, result.stderr
    assert app_marker.read_text(encoding="utf-8") == "ran"
    runtime_settings = Settings(
        _env_file=None,
        data_dir=settings.data_dir,
        index_root=state_root / "runtime-index",
        state_db_path=settings.state_db_path,
    )
    assert validate_index_version(
        resolve_current(runtime_settings.index_root), settings=runtime_settings
    ).record_count == 1
    assert backup.is_dir()
    assert not any(state_root.glob(".runtime-index.generation.*"))

    retry = _run_entrypoint(
        canonical_root,
        state_root,
        real_validation=True,
        extra_environment={"ICML_PAPERS_DATA_DIR": str(settings.data_dir)},
    )
    assert retry.returncode == 0, retry.stderr
    _assert_no_publish_residue(state_root)


@pytest.mark.parametrize("signal_name", ["TERM", "INT"])
def test_signal_during_backup_cleanup_keeps_committed_runtime(
    tmp_path, signal_name
):
    settings, canonical_root = _make_real_index(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    _write_previous_runtime(state_root)
    backup = state_root / ".runtime-index.backup"
    rm_stub = """#!/bin/sh
if [ "${3:-}" = "$ICML_TEST_BACKUP_PATH" ]; then
    /usr/bin/rm -f "$ICML_TEST_BACKUP_PATH/previous-generation"
    kill -"$ICML_TEST_CLEANUP_SIGNAL" "$PPID"
    exit 1
fi
exec /usr/bin/rm "$@"
"""
    validation_environment = {
        "ICML_PAPERS_DATA_DIR": str(settings.data_dir),
        "ICML_TEST_BACKUP_PATH": str(backup),
        "ICML_TEST_CLEANUP_SIGNAL": signal_name,
    }

    result = _run_entrypoint(
        canonical_root,
        state_root,
        real_validation=True,
        command_stubs={"rm": rm_stub},
        extra_environment=validation_environment,
    )

    assert result.returncode != 0
    runtime_settings = Settings(
        _env_file=None,
        data_dir=settings.data_dir,
        index_root=state_root / "runtime-index",
        state_db_path=settings.state_db_path,
    )
    assert validate_index_version(
        resolve_current(runtime_settings.index_root), settings=runtime_settings
    ).record_count == 1
    assert backup.is_dir()
    assert not any(state_root.glob(".runtime-index.generation.*"))

    retry = _run_entrypoint(
        canonical_root,
        state_root,
        real_validation=True,
        extra_environment={"ICML_PAPERS_DATA_DIR": str(settings.data_dir)},
    )
    assert retry.returncode == 0, retry.stderr
    _assert_no_publish_residue(state_root)


def test_current_drift_preserves_previous_runtime(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root, build_id="v1-first")
    _write_minimal_version(canonical_root, build_id="v1-next")
    (canonical_root / "current").write_text("v1-first\n", encoding="utf-8")
    state_root = tmp_path / "state"
    state_root.mkdir()
    runtime_root = _write_previous_runtime(state_root)
    cp_stub = """#!/bin/sh
/usr/bin/cp "$@" || exit 1
printf '%s\n' "$ICML_TEST_NEXT_BUILD" >"$ICML_TEST_CURRENT_PATH"
"""

    result = _run_entrypoint(
        canonical_root,
        state_root,
        command_stubs={"cp": cp_stub},
        extra_environment={
            "ICML_TEST_CURRENT_PATH": str(canonical_root / "current"),
            "ICML_TEST_NEXT_BUILD": "v1-next",
        },
    )

    assert result.returncode != 0
    assert (runtime_root / "previous-generation").read_text() == "keep"
    _assert_no_publish_residue(state_root)


def test_complete_validation_failure_preserves_previous_runtime(tmp_path):
    settings, canonical_root = _make_real_index(tmp_path)
    state_root = tmp_path / "state"
    state_root.mkdir()
    validation_environment = {"ICML_PAPERS_DATA_DIR": str(settings.data_dir)}
    first = _run_entrypoint(
        canonical_root,
        state_root,
        real_validation=True,
        extra_environment=validation_environment,
    )
    assert first.returncode == 0, first.stderr
    runtime_root = state_root / "runtime-index"
    before = _tree_fingerprint(runtime_root)
    (resolve_current(canonical_root) / "manifest.json").write_text(
        "not-json", encoding="utf-8"
    )

    second = _run_entrypoint(
        canonical_root,
        state_root,
        real_validation=True,
        extra_environment=validation_environment,
    )

    assert second.returncode != 0
    assert _tree_fingerprint(runtime_root) == before
    _assert_no_publish_residue(state_root)
    assert settings.data_dir.is_dir()


def test_success_cleans_stale_generation_and_backup(tmp_path):
    canonical_root = tmp_path / "canonical"
    _write_minimal_version(canonical_root)
    state_root = tmp_path / "state"
    state_root.mkdir()
    _write_previous_runtime(state_root)
    stale_generation = state_root / ".runtime-index.generation.stale"
    stale_generation.mkdir()
    (stale_generation / "partial").write_text("partial", encoding="utf-8")
    stale_backup = state_root / ".runtime-index.backup"
    stale_backup.mkdir()
    (stale_backup / "old").write_text("old", encoding="utf-8")

    result = _run_entrypoint(canonical_root, state_root)

    assert result.returncode == 0, result.stderr
    _assert_no_publish_residue(state_root)


def test_build_mode_uses_canonical_directly_without_runtime_copy(tmp_path):
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()

    result = _run_entrypoint(
        canonical_root,
        state_root,
        'printf built > "$ICML_PAPERS_INDEX_ROOT/build-marker"',
        prepare_runtime=False,
    )

    assert result.returncode == 0, result.stderr
    assert (canonical_root / "build-marker").read_text(encoding="utf-8") == "built"
    assert not (state_root / "runtime-index").exists()


def test_compose_separates_app_runtime_from_build_canonical():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    app = services["icml-mcp"]
    build = services["build-index"]
    tls = services["icml-mcp-tls"]

    assert app["environment"]["ICML_BOOTSTRAP_INDEX_ROOT"].endswith("/index")
    assert app["environment"]["ICML_PAPERS_INDEX_ROOT"].endswith(
        "/state/runtime-index"
    )
    assert "ICML_BOOTSTRAP_INDEX_ROOT" not in build["environment"]
    assert build["environment"]["ICML_PAPERS_INDEX_ROOT"].endswith("/index")

    app_mounts = {volume["target"]: volume for volume in app["volumes"]}
    build_mounts = {volume["target"]: volume for volume in build["volumes"]}
    canonical = "/var/lib/icml-paper-search/index"
    state = "/var/lib/icml-paper-search/state"
    assert app_mounts[canonical]["read_only"] is True
    assert app_mounts[state].get("read_only", False) is False
    assert build_mounts[canonical].get("read_only", False) is False
    assert build_mounts[state].get("read_only", False) is False
    assert [name for name, service in services.items() if service.get("ports")] == [
        "icml-mcp-tls"
    ]
    assert tls["ports"] == ["${ICML_PAPERS_PUBLIC_PORT:-20441}:20441/tcp"]
    assert tls["read_only"] is True
    assert set(tls["tmpfs"]) == {
        "/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777",
        "/var/cache/nginx:rw,noexec,nosuid,nodev,size=8m,mode=0755",
    }
