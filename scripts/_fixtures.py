"""Walk the test fixtures the way a run of ``ftl extract`` would (helper for scripts/tests).

For every fixture project the argv of its first run (``args.json``) is resolved into
:class:`ExtractOptions` and every Python file of the code tree is yielded with its source.
Files the reference cannot parse (deliberately broken fixtures) are yielded too; callers
decide what to do with them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from fly_ftl_extract.cli.config import load_pyproject, resolve_options
from fly_ftl_extract.files import find_py_files
from fly_ftl_extract.ftl.model import ExtractOptions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract_argv import parse_extract_argv

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"


@dataclass(frozen=True)
class FixtureFile:
    """One Python file of a fixture project."""

    fixture: str
    run: str
    path: str
    """Display path relative to the fixture root (as the walker produces it)."""
    source: str | None
    """Decoded UTF-8 source, ``None`` when the file is not valid UTF-8."""
    options: ExtractOptions


def fixture_names() -> list[str]:
    return sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())


def fixture_options(name: str, run: str | None = None) -> tuple[str, ExtractOptions]:
    """``(run, options)`` of a fixture, resolved with the fixture root as cwd."""
    root = FIXTURES / name
    spec = json.loads((root / "args.json").read_text(encoding="utf-8"))
    run = run or next(iter(spec["runs"]))
    overrides, config = parse_extract_argv(spec["runs"][run])
    cwd = os.getcwd()
    os.chdir(root)
    try:
        options = resolve_options(overrides, load_pyproject(config, str(root)))
    finally:
        os.chdir(cwd)
    return run, options


def fixture_files(name: str, run: str | None = None) -> list[FixtureFile]:
    """Every Python file of the fixture's code tree with the run's options."""
    root = FIXTURES / name
    run, options = fixture_options(name, run)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        paths = find_py_files(options.code_path, options.exclude_dirs)
        out = []
        for path in paths:
            data = Path(path).read_bytes()
            try:
                source: str | None = data.decode("utf-8")
            except UnicodeDecodeError:
                source = None
            out.append(FixtureFile(name, run, path, source, options))
    finally:
        os.chdir(cwd)
    return out


def all_fixture_files() -> list[FixtureFile]:
    return [f for name in fixture_names() for f in fixture_files(name)]
