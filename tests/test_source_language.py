"""Enforce English source explanations while preserving arbitrary business data."""

from __future__ import annotations

import ast
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

import pytest

from icml_paper_search.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "src", ROOT / "tests", ROOT / "docs" / "scripts")
COMMENT_FILES = (ROOT / ".env.example", ROOT / ".gitignore", ROOT / "deploy" / "compose.yaml")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True, slots=True)
class LanguageViolation:
    """Describe one disallowed source-language occurrence."""

    path: Path
    line: int
    category: str
    text: str

    def render(self) -> str:
        """Render a compact failure with a repository-relative location."""
        excerpt = " ".join(self.text.strip().split())[:120]
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.category}: {excerpt}"


def iter_python_files() -> list[Path]:
    """Return Python sources from the fixed runtime, test, and docs-script roots."""
    return sorted(path for root in PYTHON_ROOTS for path in root.rglob("*.py"))


def find_docstrings(path: Path) -> list[LanguageViolation]:
    """Find Han characters in module, class, and function docstrings through the AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[LanguageViolation] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body:
            continue
        expression = node.body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue
        if HAN.search(expression.value.value):
            violations.append(
                LanguageViolation(
                    path,
                    expression.lineno,
                    "docstring",
                    expression.value.value,
                )
            )
    return violations


def find_comments(path: Path) -> list[LanguageViolation]:
    """Find Han characters in Python comments through encoding-aware tokenization."""
    with tokenize.open(path) as source:
        tokens = tokenize.generate_tokens(source.readline)
        return [
            LanguageViolation(path, token.start[0], "comment", token.string)
            for token in tokens
            if token.type == tokenize.COMMENT and HAN.search(token.string)
        ]


def find_runtime_literals(path: Path) -> list[LanguageViolation]:
    """Find Han characters in non-docstring literals in runtime source files."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.body:
            expression = node.body[0]
            if (
                isinstance(expression, ast.Expr)
                and isinstance(expression.value, ast.Constant)
                and isinstance(expression.value.value, str)
            ):
                docstring_nodes.add(id(expression.value))

    return [
        LanguageViolation(path, node.lineno, "runtime string", node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
        and HAN.search(node.value)
    ]


def _find_inline_hash_comment(line: str, *, dotenv: bool) -> str | None:
    """Return an unquoted hash comment using dotenv or YAML escape semantics."""
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            if character == "\\" and (quote == '"' or dotenv):
                index += 2
                continue
            if character == quote:
                if (
                    not dotenv
                    and quote == "'"
                    and index + 1 < len(line)
                    and line[index + 1] == "'"
                ):
                    index += 2
                    continue
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and (index == 0 or line[index - 1].isspace()):
            return line[index:]
        index += 1
    return None


def _configuration_comment(path: Path, line: str) -> str | None:
    """Extract a comment according to the supported root-file format."""
    if path.name == ".gitignore":
        stripped = line.lstrip()
        return stripped if stripped.startswith("#") else None
    if path.name == ".env.example":
        return _find_inline_hash_comment(line, dotenv=True)
    if path.suffix in {".yaml", ".yml"}:
        return _find_inline_hash_comment(line, dotenv=False)
    raise AssertionError(f"unsupported configuration comment format: {path}")


def find_configuration_comments(path: Path) -> list[LanguageViolation]:
    """Find Han characters in format-aware root configuration comments."""
    violations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        comment = _configuration_comment(path, line)
        if comment is not None and HAN.search(comment):
            violations.append(
                LanguageViolation(path, line_number, "configuration comment", comment)
            )
    return violations


def test_python_comments_and_docstrings_are_english():
    violations = [
        violation
        for path in iter_python_files()
        for violation in (*find_docstrings(path), *find_comments(path))
    ]

    assert not violations, "\n" + "\n".join(item.render() for item in violations)


def test_runtime_string_literals_are_english():
    violations = [
        violation
        for path in sorted((ROOT / "src").rglob("*.py"))
        for violation in find_runtime_literals(path)
    ]

    assert not violations, "\n" + "\n".join(item.render() for item in violations)


def test_root_configuration_comments_are_english():
    violations = [
        violation
        for path in COMMENT_FILES
        for violation in find_configuration_comments(path)
    ]

    assert not violations, "\n" + "\n".join(item.render() for item in violations)


@pytest.mark.parametrize("name", [".env.example", "compose.yaml"])
def test_configuration_guard_rejects_full_line_han_comments(tmp_path, name):
    path = tmp_path / name
    path.write_text("# 中文整行注释\n", encoding="utf-8")

    violations = find_configuration_comments(path)

    assert [(item.line, item.category) for item in violations] == [
        (1, "configuration comment")
    ]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (".env.example", "VALUE=enabled  # 中文行内注释\n"),
        ("compose.yaml", "enabled: true  # 中文行内注释\n"),
    ],
)
def test_configuration_guard_rejects_inline_han_comments(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    violations = find_configuration_comments(path)

    assert len(violations) == 1
    assert violations[0].line == 1


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (
            ".env.example",
            "DOUBLE=\"literal # 中文值\"\nSINGLE='literal # 中文值'\n",
        ),
        (
            "compose.yaml",
            "double: \"literal # 中文值\"\nsingle: 'literal # 中文值'\n",
        ),
    ],
)
def test_configuration_guard_ignores_hashes_inside_quoted_values(
    tmp_path, name, content
):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    assert find_configuration_comments(path) == []


def test_dotenv_single_quote_escape_keeps_hash_inside_value_and_finds_real_comment(
    tmp_path,
):
    path = tmp_path / ".env.example"
    path.write_text(
        r"VALUE='literal \' # 中文值'" "\n"
        r"CLOSED='literal \''  # 中文注释" "\n",
        encoding="utf-8",
    )

    violations = find_configuration_comments(path)

    assert [(item.line, item.text) for item in violations] == [(2, "# 中文注释")]


def test_yaml_quote_escapes_keep_hashes_inside_values_and_find_real_comment(tmp_path):
    path = tmp_path / "compose.yaml"
    path.write_text(
        "single: 'literal '' # 中文值'\n"
        'double: "literal \\" # 中文值"\n'
        "closed: 'literal '' value'  # 中文注释\n",
        encoding="utf-8",
    )

    violations = find_configuration_comments(path)

    assert [(item.line, item.text) for item in violations] == [(3, "# 中文注释")]


def test_cli_help_uses_the_english_contract(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "Local hybrid indexing and search for ICML 2026" in output
    assert "Preflight or explicitly build a versioned local index" in output
    assert "Hybrid search (omit query for interactive mode)" in output
