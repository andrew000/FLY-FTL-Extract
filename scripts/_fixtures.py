"""Walk the test fixtures the way a run of ``ftl extract`` would (helper for scripts/tests).

For every fixture project the argv of its first run (``args.json``) is resolved into
:class:`ExtractOptions` and every Python file of the code tree is yielded with its source.
:func:`fixture_candidates` adds the tokenizer's candidates with the teacher's labels, and
:func:`fixture_odors` encodes them — the *real* odours used for calibration, the odour
report and the sparsity tests.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.cli.config import load_pyproject, resolve_options
from fly_ftl_extract.files import find_py_files
from fly_ftl_extract.ftl.model import ExtractOptions, FluentKey
from fly_ftl_extract.odor.encoder import encode_many
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.reference.labels import label_candidates
from fly_ftl_extract.tokenizer.candidates import Candidate, iter_candidates

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


@dataclass(frozen=True)
class LabeledCandidate:
    """A candidate of a fixture file with the teacher's verdict."""

    file: FixtureFile
    index: int
    """Position among the file's candidates (part of the production seed)."""
    candidate: Candidate
    positive: bool
    occurrence: FluentKey | None
    """The teacher's key when the candidate is positive."""


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


def fixture_candidates(names: list[str] | None = None) -> list[LabeledCandidate]:
    """Labeled candidates of every parsable, UTF-8 fixture file (all fixtures by default)."""
    out: list[LabeledCandidate] = []
    for name in names or fixture_names():
        for f in fixture_files(name):
            if f.source is None:
                continue
            try:
                occurrences = key_occurrences(f.path, f.source, f.options)
            except SyntaxError:
                continue  # deliberately broken fixture files
            candidates = list(iter_candidates(f.source, f.options))
            labels, positive_index = label_candidates(candidates, occurrences)
            occurrence_of = dict(zip(positive_index, occurrences, strict=True))
            out.extend(
                LabeledCandidate(f, i, c, labels[i], occurrence_of.get(i))
                for i, c in enumerate(candidates)
            )
    return out


def fixture_odors(items: list[LabeledCandidate] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(odors, positive)`` for the labeled candidates (encoded with each file's options)."""
    items = fixture_candidates() if items is None else items
    if not items:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=bool)
    odors = np.concatenate([encode_many([it.candidate.window], it.file.options) for it in items])
    return odors, np.array([it.positive for it in items])
