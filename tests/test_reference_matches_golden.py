"""reference/ (ast teacher) + ftl/ (merge & writer) must reproduce ftl 0.12.1 byte-for-byte.

Every ``tests/golden/<fixture>/<run>`` was produced by the original binary
(``scripts/gen_golden.py``).  For each run the fixture is copied to a temp dir, the same
argv is replayed through ``reference.extract_code`` + ``ftl.pipeline.run_extract`` and the
resulting tree, exit code, stdout and stderr are compared.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from fly_ftl_extract.cli.config import ConfigError, load_pyproject, resolve_options
from fly_ftl_extract.ftl.pipeline import EXIT_CONFIG_ERROR, LogLine, done_line, run_extract
from fly_ftl_extract.reference.extractor import extract_code

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "scripts"))
from _extract_argv import parse_extract_argv  # noqa: E402
from gen_golden import scrub  # noqa: E402


def _runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for fixture in sorted(FIXTURES.iterdir()):
        if not fixture.is_dir():
            continue
        spec = json.loads((fixture / "args.json").read_text(encoding="utf-8"))
        runs.extend((fixture.name, run) for run in spec["runs"])
    return runs


def replay(work: Path, argv: list[str]) -> tuple[int, str, str]:
    """Run reference + pipeline in ``work`` with the given argv; return (exit, stdout, stderr)."""
    overrides, config = parse_extract_argv(argv)
    started = time.perf_counter()
    try:
        options = resolve_options(overrides, load_pyproject(config, str(work)))
    except ConfigError as err:
        line = LogLine("ERROR", "cli", f"Configuration error: {err}")
        return EXIT_CONFIG_ERROR, "", line.render() + "\n"
    outcome = run_extract(extract_code(options), options)
    stderr = outcome.render_logs()
    if outcome.exit_code == 0:
        stderr += done_line(time.perf_counter() - started).render() + "\n"
    return outcome.exit_code, "", stderr


def _normalize(text: str) -> str:
    return text if os.name == "nt" else text.replace("\\", "/")


def _tree_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix != ".py" and p.name != "args.json"
    }


@pytest.mark.parametrize(("fixture", "run"), _runs())
def test_reference_matches_golden(
    fixture: str, run: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = json.loads((FIXTURES / fixture / "args.json").read_text(encoding="utf-8"))
    argv = spec["runs"][run]
    work = tmp_path / f"{fixture}__{run}"
    shutil.copytree(FIXTURES / fixture, work)
    monkeypatch.chdir(work)

    exit_code, stdout, stderr = replay(work, argv)
    golden = GOLDEN / fixture / run

    assert exit_code == int((golden / "exit_code.txt").read_text(encoding="utf-8").strip())
    assert scrub(stdout, work) == (golden / "stdout.txt").read_text(encoding="utf-8", newline="")
    expected_stderr = (golden / "stderr.txt").read_text(encoding="utf-8", newline="")
    assert _normalize(scrub(stderr, work)) == _normalize(expected_stderr)
    assert _tree_files(work) == _tree_files(golden / "tree")
