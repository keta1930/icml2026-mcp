"""Generate bilingual reference fragments from runtime schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
OUTPUT = DOCS_ROOT / "generated"
CATALOG_PATH = DOCS_ROOT / "i18n" / "reference.zh.json"
LANGUAGES = ("en", "zh")
sys.path.insert(0, str(ROOT / "src"))

from icml_paper_search.config import Settings
from icml_paper_search.transport.mcp import create_mcp

ENGLISH_LABELS = {
    "parameter": "Parameter",
    "type": "Type",
    "required": "Required",
    "default": "Default",
    "constraints": "Constraints",
    "description": "Description",
    "yes": "yes",
    "no": "no",
    "no_parameters": "No parameters",
    "return_note": "Returns MCP text content; failures use stable `error[code]` text.",
    "environment_variable": "Environment variable",
    "unset": "unset",
    "min_chars": "min chars",
    "max_chars": "max chars",
    "min_items": "min items",
    "max_items": "max items",
}


def _schema_type(schema: dict[str, Any]) -> str:
    """Compress a JSON Schema type into table-friendly text."""
    if "anyOf" in schema:
        values = [_schema_type(value) for value in schema["anyOf"]]
        return " | ".join(dict.fromkeys(values))
    if schema.get("type") == "array":
        return f"{_schema_type(schema.get('items', {}))}[]"
    return str(schema.get("type", "any"))


def _constraints(schema: dict[str, Any], labels: dict[str, str]) -> str:
    """Extract boundary constraints relevant to callers."""
    target = next(
        (item for item in schema.get("anyOf", []) if item.get("type") != "null"),
        schema,
    )
    constraint_labels = {
        "minimum": "≥",
        "exclusiveMinimum": ">",
        "maximum": "≤",
        "exclusiveMaximum": "＜",
        "minLength": labels["min_chars"],
        "maxLength": labels["max_chars"],
        "minItems": labels["min_items"],
        "maxItems": labels["max_items"],
    }
    return (
        ", ".join(
            f"{label} {target[key]}"
            for key, label in constraint_labels.items()
            if key in target
        )
        or "—"
    )


def _default(value: Any, labels: dict[str, str]) -> str:
    """Render a stable, sanitized default value."""
    if value is None:
        return labels["unset"]
    if isinstance(value, bool):
        return str(value).lower()
    if value == "":
        return '`""`'
    if isinstance(value, (list, tuple)):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    return f"`{value}`"


def _assert_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    """Reject missing and unknown translation keys."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise SystemExit(
            f"invalid translation catalog at {context}: "
            f"missing={missing}, unknown={unknown}"
        )


def _load_chinese_catalog(tools: list[Any]) -> dict[str, Any]:
    """Load and validate the Chinese catalog against live tool schemas."""
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot load {CATALOG_PATH.relative_to(ROOT)}: {error}"
        ) from error
    if not isinstance(catalog, dict):
        raise SystemExit("Chinese reference catalog must be a JSON object")
    _assert_exact_keys(catalog, {"labels", "tools"}, "root")

    labels = catalog["labels"]
    translated_tools = catalog["tools"]
    if not isinstance(labels, dict) or not isinstance(translated_tools, dict):
        raise SystemExit("Chinese reference catalog labels and tools must be objects")
    _assert_exact_keys(labels, set(ENGLISH_LABELS), "labels")
    _assert_exact_keys(translated_tools, {tool.name for tool in tools}, "tools")

    for key, value in labels.items():
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"empty Chinese translation at labels.{key}")
    for tool in tools:
        translation = translated_tools[tool.name]
        if not isinstance(translation, dict):
            raise SystemExit(f"translation for tools.{tool.name} must be an object")
        _assert_exact_keys(
            translation,
            {"title", "description", "parameters"},
            f"tools.{tool.name}",
        )
        parameters = translation["parameters"]
        if not isinstance(parameters, dict):
            raise SystemExit(f"tools.{tool.name}.parameters must be an object")
        expected_parameters = set(tool.parameters.get("properties", {}))
        _assert_exact_keys(
            parameters,
            expected_parameters,
            f"tools.{tool.name}.parameters",
        )
        strings = [
            translation["title"],
            translation["description"],
            *parameters.values(),
        ]
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise SystemExit(f"empty Chinese translation under tools.{tool.name}")
    return catalog


def _tool_reference(tools: list[Any], language: str, catalog: dict[str, Any]) -> str:
    """Generate one locale's parameter tables from FastMCP tool schemas."""
    labels = ENGLISH_LABELS if language == "en" else catalog["labels"]
    sections = ["{/* Generated by docs/scripts/generate_reference.py. */}"]
    for tool in tools:
        translation = catalog["tools"][tool.name] if language == "zh" else None
        title = translation["title"] if translation else tool.title
        description = translation["description"] if translation else tool.description
        parameter_copy = translation["parameters"] if translation else {}
        schema = tool.parameters
        required = set(schema.get("required", []))
        sections.extend(
            [
                "",
                f"### `{tool.name}` — {title}",
                "",
                description or "—",
                "",
                "| "
                + " | ".join(
                    labels[key]
                    for key in (
                        "parameter",
                        "type",
                        "required",
                        "default",
                        "constraints",
                        "description",
                    )
                )
                + " |",
                "|---|---|---:|---|---|---|",
            ]
        )
        properties = schema.get("properties", {})
        if not properties:
            sections.append(f"| — | — | — | — | — | {labels['no_parameters']} |")
        for name, prop in properties.items():
            default = "—" if name in required else _default(prop.get("default"), labels)
            raw_description = parameter_copy.get(name, prop.get("description", "—"))
            parameter_description = str(raw_description).replace("|", "\\|")
            value_type = _schema_type(prop).replace("|", "\\|")
            sections.append(
                f"| `{name}` | `{value_type}` | "
                f"{labels['yes'] if name in required else labels['no']} | {default} | "
                f"{_constraints(prop, labels)} | {parameter_description} |"
            )
        sections.extend(["", labels["return_note"]])
    return "\n".join(sections) + "\n"


def _configuration_reference(language: str, catalog: dict[str, Any]) -> str:
    """Generate one locale's environment-variable table from Settings."""
    labels = ENGLISH_LABELS if language == "en" else catalog["labels"]
    schema = Settings.model_json_schema()
    path_defaults = {
        "data_dir": "data",
        "index_root": ".data/index",
        "state_db_path": ".data/state/runtime.sqlite3",
    }
    lines = [
        "{/* Generated by docs/scripts/generate_reference.py. */}",
        "",
        "| "
        + " | ".join(
            labels[key]
            for key in ("environment_variable", "type", "default", "constraints")
        )
        + " |",
        "|---|---|---|---|",
    ]
    for name, prop in schema.get("properties", {}).items():
        env_name = f"ICML_PAPERS_{name.upper()}"
        value = path_defaults.get(name, prop.get("default"))
        if name in {"dashscope_api_key", "dashscope_api_key_file"}:
            value = None
        value_type = _schema_type(prop).replace("|", "\\|")
        lines.append(
            f"| `{env_name}` | `{value_type}` | {_default(value, labels)} | "
            f"{_constraints(prop, labels)} |"
        )
    return "\n".join(lines) + "\n"


def _expected_outputs() -> dict[Path, str]:
    """Validate all inputs before constructing the complete output set."""
    tools = list(create_mcp(Settings(_env_file=None))._tool_manager.list_tools())
    catalog = _load_chinese_catalog(tools)
    outputs: dict[Path, str] = {}
    for language in LANGUAGES:
        locale_root = OUTPUT / language
        outputs[locale_root / "mcp-tools.mdx"] = _tool_reference(
            tools, language, catalog
        )
        outputs[locale_root / "configuration.mdx"] = _configuration_reference(
            language, catalog
        )
    return outputs


def _synchronize(outputs: dict[Path, str], *, check: bool) -> int:
    """Write the exact generated tree or report any drift in check mode."""
    existing = set(OUTPUT.rglob("*.mdx")) if OUTPUT.exists() else set()
    expected = set(outputs)
    stale = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    unexpected = sorted(existing - expected)
    if check and (stale or unexpected):
        details = [
            *(f"stale: {path.relative_to(ROOT)}" for path in stale),
            *(f"unexpected: {path.relative_to(ROOT)}" for path in unexpected),
        ]
        raise SystemExit("generated reference drift:\n" + "\n".join(details))
    if check:
        return 0

    for path in unexpected:
        path.unlink()
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    for directory in sorted(OUTPUT.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return len(stale) + len(unexpected)


def main() -> int:
    """Generate or validate the bilingual reference fragments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = _synchronize(_expected_outputs(), check=args.check)
    if not args.check:
        print(f"generated reference files: {changed} changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
