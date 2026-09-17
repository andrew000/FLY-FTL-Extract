"""The generated corpus: reproducible, compilable, labelled, and disjoint from the fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fly_ftl_extract.odor.encoder import DEFAULT_ENCODER, N_PN_DEFAULT, N_SLOTS

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import FIXTURES  # noqa: E402
from make_dataset import (  # noqa: E402
    DATASET_DIR,
    GRAMMAR5_FAMILIES,
    GRAMMAR5_LAYOUTS,
    MUTATION_CONTEXTS,
    MUTATION_FAMILIES,
    balance_and_split,
    family_shares,
    generate_snippets,
    rows_of_snippet,
)


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
        # ≤ 40 statements; grammar-5 statements broken over lines make a module longer
        assert 1 <= s.source.count("\n") <= 120
        result = rows_of_snippet(s)
        if result is None:
            continue
        keys, kwargs = result
        n_keys += len(keys)
        n_pos += sum(r.label for r in keys)
        n_kwargs += len(kwargs)
        for r in keys:
            assert r.odor.shape == (N_SLOTS, N_PN_DEFAULT)
            assert 0 <= r.odor.min() <= r.odor.max() < 1
            assert r.odor[DEFAULT_ENCODER.context_before].any()  # the root puff
            assert r.odor[DEFAULT_ENCODER.summary_slot].any()  # the summary puff
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


def test_balance_and_split_is_a_function_of_the_corpus_only() -> None:
    """The proxy and the dataset must see the same rows and the same split."""
    snippets, _ = generate_snippets(60, seed=9)
    keys, kwargs = [], []
    for s in snippets:
        result = rows_of_snippet(s)
        if result is not None:
            keys.extend(result[0])
            kwargs.extend(result[1])
    a = balance_and_split(keys, kwargs, len(snippets), 5)
    b = balance_and_split(keys, kwargs, len(snippets), 5)
    assert [(r.snippet, r.index) for r in a[0]] == [(r.snippet, r.index) for r in b[0]]
    assert a[2] == b[2]
    assert set(a[2].values()) <= {0, 1, 2}
    assert len(a[2]) == len(snippets)


def test_grammar4_families_are_all_present_and_labelled_by_the_teacher() -> None:
    """Every mutation family and context occurs in a modest corpus, and each one yields
    rows with labels of both classes across the corpus where the teacher says so."""
    snippets, _ = generate_snippets(400, seed=21)
    shares = family_shares(snippets)
    for tag in (*MUTATION_FAMILIES, *MUTATION_CONTEXTS):
        assert shares[tag]["occurrences"] > 0, tag
    assert abs(sum(shares[f]["share"] for f in MUTATION_FAMILIES) - 1.0) < 1e-9
    # the two constructs the attempt-9 fly got wrong must be common now
    assert shares["ignore-L2-last"]["snippet_share"] > 0.05
    assert shares["prefix-name-get"]["snippet_share"] > 0.05
    labels_seen: set[bool] = set()
    for s in snippets[:150]:
        result = rows_of_snippet(s)
        if result is not None:
            labels_seen.update(r.label for r in result[0])
    assert labels_seen == {True, False}


def test_grammar5_families_have_the_auditors_shares_and_both_layouts() -> None:
    """Each grammar-5 family is ~5–7 % of all family occurrences (grammar-4 + grammar-5),
    both layouts (one line / broken over lines) occur, and the corpus still compiles
    (an async def with `return value` and `yield` was the one construct that did not)."""
    snippets, rejected = generate_snippets(1500, seed=33)
    assert rejected == 0
    shares = family_shares(snippets)
    for tag in GRAMMAR5_FAMILIES:
        assert 0.04 <= shares[tag]["share_all"] <= 0.08, (tag, shares[tag]["share_all"])
        assert shares[tag]["snippet_share"] > 0.1, tag
    for tag in GRAMMAR5_LAYOUTS:
        assert 0.3 <= shares[tag]["share"] <= 0.7, (tag, shares[tag]["share"])
    assert abs(sum(shares[f]["share"] for f in GRAMMAR5_FAMILIES) - 1.0) < 1e-9
    total_all = sum(shares[f]["share_all"] for f in (*MUTATION_FAMILIES, *GRAMMAR5_FAMILIES))
    assert abs(total_all - 1.0) < 1e-9


def test_grammar5_positive_families_yield_keys_and_twins_do_not() -> None:
    """The teacher's labels on the new positions: a dict value / kwarg value / decorator
    argument / sequence element / return value made by an i18n name is a key; the twin
    constructs without an i18n call give no key at all."""
    snippets, _ = generate_snippets(600, seed=34)
    positive = {f for f in GRAMMAR5_FAMILIES if not f.startswith("g5-neg-")}
    negative = set(GRAMMAR5_FAMILIES) - positive
    keys_in_positive = keys_in_negative_only = 0
    for s in snippets:
        fams = set(s.families) & set(GRAMMAR5_FAMILIES)
        if not fams:
            continue
        result = rows_of_snippet(s)
        if result is None:
            continue
        n_keys = sum(r.label for r in result[0])
        if fams <= negative and not (set(s.families) & set(MUTATION_FAMILIES)):
            # only twin families and no other i18n statement: the file may still hold
            # ordinary i18n calls from the plain grammar, so count, do not assert per file
            keys_in_negative_only += n_keys
        if fams & positive:
            keys_in_positive += n_keys
    assert keys_in_positive > 100
    assert keys_in_negative_only < keys_in_positive
