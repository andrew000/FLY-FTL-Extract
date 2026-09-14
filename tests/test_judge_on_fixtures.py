"""The whole fly on the golden fixtures (holdout): tokenizer → odour → brain → judge.

For every parsable fixture file the set of ``(call position, key name, placeable kwargs)``
the fly produces must equal what the ast teacher produces.  This is the Phase 5 gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fly_ftl_extract.dopamine import Judge, WeightsMismatchError, WeightsMissingError
from fly_ftl_extract.ftl.model import kwargs_from_key
from fly_ftl_extract.reference.extractor import key_occurrences

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_files, fixture_names  # noqa: E402

Key = tuple[tuple[int, int], str, tuple[str, ...]]


@pytest.mark.parametrize("fixture", fixture_names())
def test_fly_matches_teacher_on_fixture(fixture: str) -> None:
    judge: Judge | None = None
    checked = 0
    for f in fixture_files(fixture):
        if f.source is None:
            continue
        try:
            occurrences = key_occurrences(f.path, f.source, f.options)
        except SyntaxError:
            continue
        if judge is None:
            try:
                judge = Judge(f.options)
            except (WeightsMissingError, WeightsMismatchError) as err:
                pytest.skip(f"no trained fly: {err}")
        expected: set[Key] = {
            (
                (k.source_location.line, k.source_location.column),
                k.key,
                tuple(sorted(kwargs_from_key(k))),
            )
            for k in occurrences
            if k.source_location is not None
        }
        judged = judge.judge_source(f.source, f.source.encode("utf-8"))
        got: set[Key] = {
            (k.call_position, k.key_name, tuple(sorted(k.placeable))) for k in judged.keys
        }
        assert got == expected, (
            f"{fixture}/{f.path}: missing {sorted(expected - got)!r}, "
            f"extra {sorted(got - expected)!r}"
        )
        checked += 1
    if fixture in ("basic", "prefix", "keys_lazy", "ignore_attrs"):
        assert checked > 0
