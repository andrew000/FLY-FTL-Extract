"""The labelling rule: exactly one positive candidate per key occurrence of the teacher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.reference.labels import LabelError, label_candidates, label_kwargs
from fly_ftl_extract.tokenizer.candidates import iter_candidates

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_candidates, fixture_files, fixture_names  # noqa: E402

OPTS = ExtractOptions(code_path="app", locales_path="locales")


def labelled(source: str, options: ExtractOptions = OPTS) -> list[tuple[str, str | None, bool]]:
    cands = list(iter_candidates(source, options))
    labels, _ = label_candidates(cands, key_occurrences("x.py", source, options))
    return [(c.kind, c.key_name, lab) for c, lab in zip(cands, labels, strict=True)]


@pytest.mark.parametrize("fixture", fixture_names())
def test_one_positive_per_occurrence(fixture: str) -> None:
    n_occurrences = 0
    for f in fixture_files(fixture):
        if f.source is None:
            continue
        try:
            occurrences = key_occurrences(f.path, f.source, f.options)
        except SyntaxError:
            continue
        labels, positives = label_candidates(
            list(iter_candidates(f.source, f.options)), occurrences
        )
        assert sum(labels) == len(occurrences) == len(positives)
        assert len(set(positives)) == len(positives)
        n_occurrences += len(occurrences)
    if fixture in ("basic", "prefix", "keys_lazy", "ignore_attrs"):
        assert n_occurrences > 0


def test_get_like_call_marks_the_string() -> None:
    assert labelled('i18n.get("k", a=1)\n') == [("chain", None, False), ("string", "k", True)]
    assert labelled('L("lazy")\n') == [("chain", None, False), ("string", "lazy", True)]


def test_attribute_call_marks_the_chain_even_when_the_string_matches() -> None:
    got = labelled('i18n.core.get("core-get")\n')
    assert got == [("chain", "core-get", True), ("string", "core-get", False)]
    assert labelled("i18n.some.key_1(_path='w')\n") == [("chain", "some-key_1", True)]


def test_negatives_stay_negative() -> None:
    got = labelled('i18n.get(var)\nprint("x")\ni18n.set_locale("uk")\nother.i18n.get("n")\n')
    assert all(lab is False for _, _, lab in got)


def test_kwarg_labels() -> None:
    opts = ExtractOptions(
        code_path="app", locales_path="locales", ignore_kwargs=frozenset({"when"})
    )
    source = 'i18n.get("k", a=1, _path="w/x.ftl", when=0, b=2)\n'
    cands = list(iter_candidates(source, opts))
    occurrences = key_occurrences("x.py", source, opts)
    _, positives = label_candidates(cands, occurrences)
    string = cands[positives[0]]
    assert [k.name for k in string.kwargs] == ["a", "_path", "when", "b"]
    assert label_kwargs(string, occurrences[0]) == [True, False, False, True]


def test_label_error_when_no_candidate_matches() -> None:
    source = 'i18n.get("k")\n'
    cands = [c for c in iter_candidates(source, OPTS) if c.kind == "chain"]
    with pytest.raises(LabelError):
        label_candidates(cands, key_occurrences("x.py", source, OPTS))


def test_fixture_candidates_carry_occurrences() -> None:
    items = fixture_candidates(["basic"])
    positives = [it for it in items if it.positive]
    assert positives
    assert all(
        it.occurrence is not None and it.occurrence.key == it.candidate.key_name for it in positives
    )
    assert all(it.occurrence is None for it in items if not it.positive)
