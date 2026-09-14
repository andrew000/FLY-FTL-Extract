"""The generated corpus: reproducible, compilable, labelled, and disjoint from the fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import FIXTURES  # noqa: E402
from make_dataset import DATASET_DIR, generate_snippets, rows_of_snippet  # noqa: E402


def fixture_sources() -> set[str]:
    return {p.read_text(encoding="utf-8", errors="replace") for p in FIXTURES.rglob("*.py")}


def test_generation_is_deterministic() -> None:
    a, _ = generate_snippets(20, seed=7)
    b, _ = generate_snippets(20, seed=7)
    assert [s.source for s in a] == [s.source for s in b]
    assert [s.config for s in a] == [s.config for s in b]
    c, _ = generate_snippets(20, seed=8)
    assert [s.source for s in a] != [s.source for s in c]


def test_snippets_compile_have_candidates_and_labels() -> None:
    snippets, _ = generate_snippets(40, seed=11)
    n_keys = n_pos = n_kwargs = 0
    for s in snippets:
        compile(s.source, "<snippet>", "exec", dont_inherit=True)
        assert 1 <= s.source.count("\n") <= 60
        result = rows_of_snippet(s)
        if result is None:
            continue
        keys, kwargs = result
        n_keys += len(keys)
        n_pos += sum(r.label for r in keys)
        n_kwargs += len(kwargs)
        for r in keys:
            assert r.odor.shape == (124,)
            assert 0 <= r.odor.min() <= r.odor.max() < 1
    assert n_keys > 40
    assert n_pos > 5
    assert n_kwargs > 5


def test_generated_snippets_never_equal_a_fixture_file() -> None:
    snippets, _ = generate_snippets(200, seed=5)
    fixtures = fixture_sources()
    assert not any(s.source in fixtures for s in snippets)


def test_dataset_on_disk_is_disjoint_from_fixtures() -> None:
    path = DATASET_DIR / "snippets.jsonl"
    if not path.exists():
        pytest.skip("dataset not generated (scripts/make_dataset.py)")
    fixtures = fixture_sources()
    stripped = {f.strip() for f in fixtures}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            source = json.loads(line)["source"]
            assert source not in fixtures
            assert source.strip() not in stripped
