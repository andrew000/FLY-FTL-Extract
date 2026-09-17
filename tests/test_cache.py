"""``--cache``: the second run does not call the brain; a changed file is sniffed alone.

The fixture is ``exclude_dirs`` (three judged files whose output re-reads cleanly; ``basic``
would not do — its ``dotted.key.name`` is invalid Fluent, so a second run of the *original*
aborts on the re-read too, ``docs/FORMAT.md`` §1).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fly_ftl_extract import __version__
from fly_ftl_extract.cli.cache import CACHE_SCHEMA, DEFAULT_CACHE_DIR, MAGIC

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "projects" / "exclude_dirs"
GOLDEN_TREE = REPO / "tests" / "golden" / "exclude_dirs" / "default" / "tree"
CACHE_NAME = f"extract-{__version__}-v{CACHE_SCHEMA}.bin"

TRIALS = re.compile(r"^\[INFO  fly\]   - Trials: (\d+) ", re.MULTILINE)
FILES = re.compile(
    r"^\[INFO  fly\]   - Files sniffed: (\d+) \(from cache: (\d+), "
    r"without i18n names: (\d+), walked: (\d+)\)",
    re.MULTILINE,
)


def run_cli(work: Path, *extra: str) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fly_ftl_extract",
            "extract",
            "app",
            "locales",
            "--fly-no-tui",
            *extra,
        ],
        cwd=work,
        capture_output=True,
        check=False,
        env=env,
        timeout=900,
    )
    return result.returncode, result.stderr.decode("utf-8")


def counters(stderr: str) -> tuple[int, int, int]:
    """``(trials, files sniffed, files from cache)`` of the fly statistics block."""
    trials = TRIALS.search(stderr)
    files = FILES.search(stderr)
    assert trials is not None, stderr
    assert files is not None, stderr
    return int(trials.group(1)), int(files.group(1)), int(files.group(2))


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*.ftl"))}


def _copy(tmp_path: Path) -> Path:
    work = tmp_path / "exclude_dirs"
    shutil.copytree(FIXTURE, work)
    return work


def test_second_run_uses_the_cache_and_a_changed_file_is_resniffed(tmp_path: Path) -> None:
    work = _copy(tmp_path)

    code, err = run_cli(work, "--cache")
    assert code == 0, err
    trials, sniffed, cached = counters(err)
    assert trials > 0
    assert sniffed == 3
    assert cached == 0
    cache_file = work / DEFAULT_CACHE_DIR / CACHE_NAME
    assert cache_file.exists()
    assert cache_file.read_bytes().startswith(MAGIC)
    first_tree = _tree(work / "locales")
    assert first_tree == _tree(GOLDEN_TREE / "locales")

    # second run: nothing changed → the brain is never called
    code, err = run_cli(work, "--cache")
    assert code == 0, err
    trials, sniffed, cached = counters(err)
    assert trials == 0
    assert sniffed == 0
    assert cached == 3
    assert _tree(work / "locales") == first_tree

    # touch one file (a comment without any i18n name): only that file is sniffed again
    target = work / "app" / "main.py"
    target.write_bytes(target.read_bytes() + b"\n# touched by the cache test\n")
    code, err = run_cli(work, "--cache")
    assert code == 0, err
    trials, sniffed, cached = counters(err)
    assert trials > 0
    assert sniffed == 1
    assert cached == 2
    assert _tree(work / "locales") == first_tree

    # --clear-cache: everything is sniffed again, into a fresh cache
    code, err = run_cli(work, "--clear-cache")
    assert code == 0, err
    trials, sniffed, cached = counters(err)
    assert trials > 0
    assert sniffed == 3
    assert cached == 0
    assert cache_file.exists()


def test_cache_path_directory_and_file(tmp_path: Path) -> None:
    work = _copy(tmp_path)
    code, err = run_cli(work, "--cache-path", "mycache")
    assert code == 0, err
    assert (work / "mycache" / CACHE_NAME).exists()
    code, err = run_cli(work, "--cache-path", "mycache")
    assert code == 0, err
    trials, _, cached = counters(err)
    assert trials == 0
    assert cached == 3

    flat = tmp_path / "flat.bin"
    code, err = run_cli(work, "--cache-path", str(flat))
    assert code == 0, err
    assert flat.exists()
    code, err = run_cli(work, "--cache-path", str(flat))
    assert code == 0, err
    trials, _, cached = counters(err)
    assert trials == 0
    assert cached == 3


def test_damaged_cache_is_ignored(tmp_path: Path) -> None:
    work = _copy(tmp_path)
    cache_dir = work / DEFAULT_CACHE_DIR
    cache_dir.mkdir()
    (cache_dir / CACHE_NAME).write_bytes(b"not a cache")
    code, err = run_cli(work, "--cache")
    assert code == 0, err
    trials, sniffed, cached = counters(err)
    assert trials > 0
    assert sniffed == 3
    assert cached == 0
    assert (cache_dir / CACHE_NAME).read_bytes().startswith(MAGIC)
