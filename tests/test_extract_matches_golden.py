"""``ftl extract`` *through the fly* must reproduce ftl 0.12.1 byte-for-byte on every fixture.

Every ``tests/golden/<fixture>/<run>`` was produced by the original binary.  Here the same
argv is replayed through **our CLI** in a fresh interpreter (``python -m fly_ftl_extract
extract … --fly-no-tui``, cwd = a temporary copy of the fixture) and compared:

* exit code;
* stdout (empty, like the original);
* stderr up to and including the ``✅ Done`` line == golden (timings scrubbed); every line
  after it must be one of ours (``[… fly] …``) — the fly may speak only after the
  original has finished;
* the locale tree.

The subprocess runs without ``PYTHONUTF8`` so the CLI itself has to write UTF-8 into a
redirected stream (``✅``).  A second block checks determinism: two runs give the same tree
and the same margins (``-v``), and a 2-process pool gives the same tree as one process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "scripts"))
from gen_golden import scrub  # noqa: E402

DONE_PREFIX = "[INFO  cli] ✅ Done in "
FLY_LINE = re.compile(r"^\[(?:INFO |DEBUG|WARN |ERROR) fly(?:::\w+)?\] ")
MARGIN_LINE = re.compile(r"^\[DEBUG fly\] .* margin [+-]\d+\.\d+ \(")


def _runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for fixture in sorted(FIXTURES.iterdir()):
        if not fixture.is_dir():
            continue
        spec = json.loads((fixture / "args.json").read_text(encoding="utf-8"))
        runs.extend((fixture.name, run) for run in spec["runs"])
    return runs


def run_cli(work: Path, argv: list[str], *extra: str) -> tuple[int, str, str]:
    """``ftl extract <argv> --fly-no-tui`` in ``work``; raw (untranslated) stdout/stderr."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    result = subprocess.run(
        [sys.executable, "-m", "fly_ftl_extract", "extract", *argv, "--fly-no-tui", *extra],
        cwd=work,
        capture_output=True,
        check=False,
        env=env,
        timeout=900,
    )
    return result.returncode, result.stdout.decode("utf-8"), result.stderr.decode("utf-8")


def split_at_done(stderr: str) -> tuple[str, list[str]]:
    """``(original part incl. the Done line, our lines after it)``."""
    lines = stderr.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(DONE_PREFIX):
            head = "\n".join(lines[: i + 1]) + "\n"
            tail = [line for line in lines[i + 1 :] if line]
            return head, tail
    return stderr, []


def _normalize(text: str) -> str:
    return text if os.name == "nt" else text.replace("\\", "/")


def _tree_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix != ".py" and p.name != "args.json"
    }


def _copy(fixture: str, run: str, tmp_path: Path) -> Path:
    work = tmp_path / f"{fixture}__{run}"
    shutil.copytree(FIXTURES / fixture, work)
    return work


@pytest.mark.parametrize(("fixture", "run"), _runs())
def test_extract_matches_golden(fixture: str, run: str, tmp_path: Path) -> None:
    spec = json.loads((FIXTURES / fixture / "args.json").read_text(encoding="utf-8"))
    argv = spec["runs"][run]
    work = _copy(fixture, run, tmp_path)
    golden = GOLDEN / fixture / run

    exit_code, stdout, stderr = run_cli(work, argv)
    expected_exit = int((golden / "exit_code.txt").read_text(encoding="utf-8").strip())
    assert exit_code == expected_exit, stderr
    assert scrub(stdout, work) == (golden / "stdout.txt").read_text(encoding="utf-8", newline="")

    head, fly_lines = split_at_done(stderr)
    expected_stderr = (golden / "stderr.txt").read_text(encoding="utf-8", newline="")
    assert _normalize(scrub(head, work)) == _normalize(expected_stderr)
    if expected_exit == 0:
        assert fly_lines, "the fly statistics block is missing after ✅ Done"
    offenders = [line for line in fly_lines if not FLY_LINE.match(line)]
    assert not offenders, offenders
    assert _tree_files(work) == _tree_files(golden / "tree")


def _basic_argv() -> list[str]:
    spec = json.loads((FIXTURES / "basic" / "args.json").read_text(encoding="utf-8"))
    return list(spec["runs"]["default"])


def test_two_runs_are_identical_tree_and_margins(tmp_path: Path) -> None:
    """Seeds come from the file bytes: the same code gives the same spikes, twice."""
    results = []
    for n in (1, 2):
        work = _copy("basic", f"det{n}", tmp_path)
        exit_code, _, stderr = run_cli(work, _basic_argv(), "-v")
        assert exit_code == 0, stderr
        margins = [line for line in stderr.split("\n") if MARGIN_LINE.match(line)]
        assert len(margins) > 20
        results.append((_tree_files(work), margins))
    assert results[0] == results[1]


def test_batch_16_equals_batch_256(tmp_path: Path) -> None:
    """The TUI batch (16 trials per brain call) changes no spike: same tree, same margins."""
    results = []
    for batch in ("16", "256"):
        work = _copy("basic", f"batch{batch}", tmp_path)
        exit_code, _, stderr = run_cli(work, _basic_argv(), "-v", "--fly-batch", batch)
        assert exit_code == 0, stderr
        assert f"batch {batch})" in stderr
        margins = [line for line in stderr.split("\n") if MARGIN_LINE.match(line)]
        assert len(margins) > 20
        results.append((_tree_files(work), margins))
    assert results[0] == results[1]


_POOL_PROBE = """
import sys
from fly_ftl_extract.cli.config import ExtractOverrides
from fly_ftl_extract.cli.extract import FlyOptions, run_command
sys.exit(run_command(
    ExtractOverrides(code_path="app", locales_path="locales"), None,
    FlyOptions(workers=2, tui=False, inprocess_threshold=1),
))
"""


def test_process_pool_gives_the_same_tree(tmp_path: Path) -> None:
    """Every trial has its own seed: splitting files over processes changes no spike."""
    single = _copy("basic", "single", tmp_path)
    exit_code, _, stderr = run_cli(single, _basic_argv())
    assert exit_code == 0, stderr
    pooled = _copy("basic", "pooled", tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _POOL_PROBE],
        cwd=pooled,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    assert "(2 processes, batch 64)" in result.stderr
    assert _tree_files(single) == _tree_files(pooled)
