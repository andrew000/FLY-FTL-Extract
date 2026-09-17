"""The small commands of the drop-in: ``stub``/``check`` (exit 2), ``config sample``, errors."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fly_ftl_extract import __version__
from fly_ftl_extract.cli import NOT_TRAINED_EXIT_CODE, NOT_TRAINED_MSG
from fly_ftl_extract.cli.sample import SAMPLE_CHECK, SAMPLE_EXTRACT, SAMPLE_STUB, sample_text


def run_ftl(*argv: str, cwd: Path | None = None) -> tuple[int, bytes, bytes]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    result = subprocess.run(
        [sys.executable, "-m", "fly_ftl_extract", *argv],
        cwd=cwd,
        capture_output=True,
        check=False,
        env=env,
        timeout=300,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ("stub",),
        ("stub", "locales/en", "stub.pyi"),
        ("check",),
        ("check", "locales", "app", "-l", "uk"),
    ],
)
def test_stub_and_check_are_not_trained(argv: tuple[str, ...]) -> None:
    code, out, err = run_ftl(*argv)
    assert code == NOT_TRAINED_EXIT_CODE
    assert out == b""
    assert err.decode("utf-8").strip() == NOT_TRAINED_MSG


def test_config_sample_matches_the_original_text() -> None:
    """Byte-identical to ``ftl config sample`` of 0.12.1 (captured in ``cli/sample.py``;
    ``scripts/compare_with_reference.py`` re-checks it against the real binary)."""
    code, out, _ = run_ftl("config", "sample")
    assert code == 0
    assert out == (SAMPLE_EXTRACT + SAMPLE_STUB + SAMPLE_CHECK).encode("utf-8")
    assert b"\r\n" not in out  # LF only, like the Rust binary
    for command, text in (
        ("extract", SAMPLE_EXTRACT),
        ("stub", SAMPLE_STUB),
        ("check", SAMPLE_CHECK),
    ):
        code, out, _ = run_ftl("config", "sample", "--command", command)
        assert code == 0
        assert out == text.encode("utf-8")
        assert sample_text(command) == text
        assert text.endswith("\n\n")


def test_version_and_missing_paths(tmp_path: Path) -> None:
    code, out, _ = run_ftl("--version")
    assert code == 0
    assert out.decode("utf-8").strip() == f"ftl {__version__}"
    code, out, err = run_ftl("extract", cwd=tmp_path)
    assert code == 2
    assert out == b""
    assert err.decode("utf-8") == (
        "[ERROR cli] Configuration error: Missing code path. Pass it as an argument or set "
        "tool.ftl-extract.extract.code-path\n"
    )


def test_deprecated_comment_junks_key_warns_like_the_original(tmp_path: Path) -> None:
    """Seen on a real project: 0.12.1 prints this WARN before ``Code path`` and goes on."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ftl-extract.extract]\ncode-path = "app"\nlocales-path = "locales"\n'
        "comment-junks = true\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    code, out, err = run_ftl("extract", "--fly-no-tui", cwd=tmp_path)
    assert code == 0
    assert out == b""
    lines = err.decode("utf-8").splitlines()
    assert lines[0] == (
        "[WARN  cli] comment-junks has no effect and will be removed in 0.13: syntax errors "
        "in .ftl files abort the run"
    )
    assert lines[1].startswith("[INFO  cli] Code path: ")


def test_nonexistent_code_path_is_an_empty_run(tmp_path: Path) -> None:
    """Like the original: no files, statistics of zeros, exit 0."""
    code, out, err = run_ftl("extract", "nonexistent", "locales", "--fly-no-tui", cwd=tmp_path)
    assert code == 0
    assert out == b""
    text = err.decode("utf-8")
    assert "[INFO  cli]   - Py files count: 0\n" in text
    assert '[INFO  cli]   - FTL keys stored: {"en": 0}\n' in text
    assert "[INFO  cli] ✅ Done in " in text
    assert "[INFO  fly]   - Trials: 0 " in text
