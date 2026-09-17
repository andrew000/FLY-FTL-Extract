"""``--fly-audit``: the teacher next to the fly — 0 differences on every fixture, exit as golden."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fly_ftl_extract.audit import compare_extractions, report_lines
from fly_ftl_extract.ftl.merge import CodeExtraction
from fly_ftl_extract.ftl.rustorder import RustMap
from fly_ftl_extract.reference.extractor import extract_code

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_names, fixture_options  # noqa: E402

AUDIT_OK = "[INFO  fly::audit] Audit: 0 differences from the ast reference."


def _first_run(fixture: str) -> tuple[str, list[str]]:
    spec = json.loads((FIXTURES / fixture / "args.json").read_text(encoding="utf-8"))
    run = next(iter(spec["runs"]))
    return run, spec["runs"][run]


@pytest.mark.parametrize("fixture", fixture_names())
def test_audit_reports_no_differences_on_fixtures(fixture: str, tmp_path: Path) -> None:
    run, argv = _first_run(fixture)
    work = tmp_path / fixture
    shutil.copytree(FIXTURES / fixture, work)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    result = subprocess.run(
        [sys.executable, "-m", "fly_ftl_extract", "extract", *argv, "--fly-no-tui", "--fly-audit"],
        cwd=work,
        capture_output=True,
        check=False,
        env=env,
        timeout=900,
    )
    stderr = result.stderr.decode("utf-8")
    expected_exit = int((GOLDEN / fixture / run / "exit_code.txt").read_text(encoding="utf-8"))
    assert result.returncode == expected_exit, stderr
    assert AUDIT_OK in stderr.splitlines(), stderr
    assert "[WARN  fly::audit]" not in stderr


def test_compare_reports_a_planted_difference() -> None:
    _, options = fixture_options("basic")
    cwd = os.getcwd()
    os.chdir(FIXTURES / "basic")
    try:
        teacher = extract_code(options)
    finally:
        os.chdir(cwd)
    assert compare_extractions(teacher, teacher) == []
    assert [line.level for line in report_lines([])] == ["INFO"]

    keys: RustMap = RustMap()
    for name, key in teacher.keys.items():
        if name != "hello-user":
            keys.insert(name, key)
    fly = CodeExtraction(
        keys, list(teacher.diagnostics), teacher.py_files_count, teacher.py_files_with_keys
    )
    differences = compare_extractions(fly, teacher)
    assert len(differences) == 1
    assert differences[0].startswith("key `hello-user` missed by the fly")
    levels = [line.level for line in report_lines(differences)]
    assert levels == ["WARN", "ERROR"]
