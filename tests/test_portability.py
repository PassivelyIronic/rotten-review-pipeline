"""Guards against defects that only surface on a non-UTF-8 locale.

Python falls back to `locale.getpreferredencoding()` when text I/O is opened
without an explicit encoding. That is UTF-8 on the CI runners and on macOS, and
cp1252 on a default Windows install — so code that reads a file containing an em
dash works everywhere the tests normally run and raises `UnicodeDecodeError` on a
contributor's laptop. These tests fail on any platform, which is the point.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"]

TEXT_IO_NAMES = {"open", "read_text", "write_text"}
BINARY_MODES = {"rb", "wb", "ab", "r+b", "w+b", "br", "bw"}


def _python_files() -> list[Path]:
    return sorted(path for directory in SOURCE_DIRS for path in directory.rglob("*.py"))


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_binary(node: ast.Call) -> bool:
    """open(path, 'rb') and friends take no encoding, so they are not offenders."""
    for arg in node.args[1:]:
        if isinstance(arg, ast.Constant) and arg.value in BINARY_MODES:
            return True
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value in BINARY_MODES
    return False


def test_no_text_io_without_explicit_encoding():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node) not in TEXT_IO_NAMES or _is_binary(node):
                continue
            if any(keyword.arg == "encoding" for keyword in node.keywords):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, (
        "text I/O without encoding='utf-8' breaks on non-UTF-8 locales (Windows cp1252): "
        + ", ".join(offenders)
    )


def test_render_results_round_trip(tmp_path):
    """The README renderer must survive non-ASCII content in the file it rewrites."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title — with an em dash, R² and ±1 point\n\n"
        "<!-- results:start -->\n\nplaceholder\n\n<!-- results:end -->\n\ntail\n",
        encoding="utf-8",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(
        json.dumps(
            {
                "sentiment_en": {"accuracy": 0.812, "n_eval": 100},
                "score_regressor": {"mae": 1.2, "r2": 0.4, "within_1_point": 0.6, "n_eval": 90},
            }
        ),
        encoding="utf-8",
    )

    script = (REPO_ROOT / "scripts" / "render_results.py").read_text(encoding="utf-8")
    script = script.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]", f'REPO_ROOT = Path(r"{tmp_path}")'
    )
    patched = tmp_path / "render_results.py"
    patched.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr

    rewritten = readme.read_text(encoding="utf-8")
    assert "em dash, R² and ±1 point" in rewritten, "non-ASCII content outside the block was lost"
    assert "0.812" in rewritten and "1.200" in rewritten
    assert rewritten.endswith("tail\n")


def test_render_results_without_metrics_fails_cleanly(tmp_path):
    script = (REPO_ROOT / "scripts" / "render_results.py").read_text(encoding="utf-8")
    script = script.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[1]", f'REPO_ROOT = Path(r"{tmp_path}")'
    )
    patched = tmp_path / "render_results.py"
    patched.write_text(script, encoding="utf-8")
    (tmp_path / "README.md").write_text("nothing here", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(patched)], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 1
    assert "not found" in result.stderr


@pytest.mark.parametrize("path", [REPO_ROOT / "README.md", REPO_ROOT / "AUDIT.md"])
def test_docs_are_valid_utf8(path):
    path.read_text(encoding="utf-8")
