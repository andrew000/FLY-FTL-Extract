"""The odour encoder: deterministic, sensitive to the window and to the options.

``fly-odor-4`` is a temporal code: one PN vector per slot (6 before, focus, 3 after).
"""

from __future__ import annotations

import numpy as np

from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor.encoder import (
    DEFAULT_ENCODER,
    ENCODER_VERSION,
    N_PN_DEFAULT,
    N_SLOTS,
    bucket,
    encode,
    encode_many,
    normalize_window,
    slot_features,
    token_depths,
)
from fly_ftl_extract.tokenizer.candidates import Candidate, iter_candidates

OPTS = ExtractOptions(code_path="app", locales_path="locales")
SELF_OPTS = ExtractOptions(
    code_path="app", locales_path="locales", i18n_keys_prefix=frozenset({"self"})
)

SOURCE = """
class H:
    def run(self, user):
        self.i18n.get("self-key", name=user.name)
        i18n.get("plain-key")
        self.i18n.set_locale("uk")
        return other(1, "not-a-key")
"""


def cands(source: str = SOURCE, options: ExtractOptions = OPTS) -> list[Candidate]:
    return list(iter_candidates(source, options))


def test_encoder_is_deterministic() -> None:
    a = [encode(c.window, OPTS) for c in cands()]
    b = [encode(c.window, OPTS) for c in cands()]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert np.array_equal(encode_many([c.window for c in cands()], OPTS), np.stack(a))


def test_vector_shape_and_range() -> None:
    for c in cands():
        vec = encode(c.window, OPTS)
        assert vec.shape == (N_SLOTS, N_PN_DEFAULT)
        assert vec.dtype == np.float32
        assert vec.min() >= 0.0
        assert vec.max() < 1.0
        # the focus slot always smells of something; a context slot lights a few PNs
        assert (vec[DEFAULT_ENCODER.context_before] > 0).sum() >= 4
        for i, feats in enumerate(slot_features(c.window, OPTS)):
            assert ((vec[i] > 0).sum() > 0) == bool(feats)


def test_missing_context_slots_are_silent_puffs() -> None:
    """A candidate on the first line has empty leading slots (zero rows), and the tokens
    it does have sit right-aligned to the focus."""
    c = next(c for c in cands('i18n.get("hello")\n') if c.kind == "string")
    vec = encode(c.window, OPTS)
    n_before = len(c.window.before)
    assert n_before < DEFAULT_ENCODER.context_before
    silent = DEFAULT_ENCODER.context_before - n_before
    assert not vec[:silent].any()
    assert vec[silent:].any(axis=1).all()
    slots = slot_features(c.window, OPTS)
    assert [bool(sl) for sl in slots[:silent]] == [False] * silent
    assert any(f == "tok@-1:(" for f, _ in slots[DEFAULT_ENCODER.context_before - 1])


def test_token_depths_walk_the_brackets() -> None:
    source = 'i18n.get("k", name=f(user.name))\n'
    c = next(c for c in cands(source) if c.text == "f")  # ``name=f(user.name)`` in the call
    before, after = token_depths(c.window)
    tokens = [t for t, _ in normalize_window(c.window, OPTS)]
    assert tokens == ["get", "(", "<STR>", ",", "<NAME>", "=", "<NAME>", "(", "<NAME>", "."]
    assert before == [0, 0, 1, 1, 1, 1]  # ``get (`` are outside the call, the rest inside
    assert after == [1, 2, 2]  # ``(`` opens one more level for ``user .``


def test_slot_order_carries_the_position() -> None:
    """The same token in another slot lights other PNs (its distance is in the hash), and
    the sequence read backwards is another odour."""
    c = next(c for c in cands() if c.text == '"plain-key"')
    vec = encode(c.window, OPTS)
    assert not np.array_equal(vec, vec[::-1])
    slots = slot_features(c.window, OPTS)
    names = {f for f, _ in slots[DEFAULT_ENCODER.context_before - 1]}
    assert "tok@-1:(" in names
    assert "tok:(" in names
    assert "type@-1:OP" in names


def test_different_windows_give_different_vectors() -> None:
    vecs = [encode(c.window, OPTS).reshape(-1) for c in cands()]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            assert not np.array_equal(vecs[i], vecs[j])


def test_same_text_in_different_files_gives_the_same_vector() -> None:
    a = cands("x = 1\n" + SOURCE)  # a different file with the same statements later on
    b = cands(SOURCE)
    # windows that do not reach the differing first line are identical
    for ca, cb in zip(a[1:], b[1:], strict=True):
        if ca.window.before == cb.window.before:
            assert np.array_equal(encode(ca.window, OPTS), encode(cb.window, OPTS))
    # and the very same source twice, of course
    for ca, cb in zip(cands(SOURCE), cands(SOURCE), strict=True):
        assert np.array_equal(encode(ca.window, OPTS), encode(cb.window, OPTS))


def test_prefix_option_changes_only_windows_that_contain_self() -> None:
    changed = 0
    for c in cands():
        plain, with_prefix = encode(c.window, OPTS), encode(c.window, SELF_OPTS)
        encoded = [t for t, _ in normalize_window(c.window, OPTS)]
        raw = normalize_window(c.window, OPTS)
        has_self = any(
            tok.text == "self"
            for tok in (
                *c.window.before[max(0, len(c.window.before) - DEFAULT_ENCODER.context_before) :],
                *c.window.focus,
                *c.window.after[: DEFAULT_ENCODER.context_after],
            )
        )
        assert len(encoded) == len(raw)
        if has_self:
            assert not np.array_equal(plain, with_prefix)
            changed += 1
        else:
            assert np.array_equal(plain, with_prefix)
    assert changed > 0


def test_normalization_classes() -> None:
    c = next(c for c in cands() if c.text == '"self-key"')
    tokens = [t for t, _ in normalize_window(c.window, SELF_OPTS)]
    assert "<PREFIX>" in tokens
    assert "<I18N>" in tokens
    assert "get" in tokens
    assert "<STR>" in tokens
    assert "<NAME>" in tokens
    assert "self" not in tokens
    assert "user" not in tokens
    c2 = next(c for c in cands() if c.text == "self.i18n.set_locale")
    assert "<IGNORE>" in [t for t, _ in normalize_window(c2.window, OPTS)]


def test_short_before_context_keeps_its_leading_tokens() -> None:
    """A candidate on the first line of a file still smells of ``<I18N> . get (``."""
    c = next(c for c in cands('i18n.get("hello")\n') if c.kind == "string")
    assert len(c.window.before) < DEFAULT_ENCODER.context_before
    tokens = [t for t, _ in normalize_window(c.window, OPTS)]
    assert tokens[:4] == ["<I18N>", ".", "get", "("]


def test_focus_positions() -> None:
    c = next(c for c in cands() if c.text == '"plain-key"')
    seq = normalize_window(c.window, OPTS)
    positions = [p for _, p in seq]
    assert positions == sorted(positions)
    assert positions.count(0) == 1
    assert seq[positions.index(0)][0] == "<STR>"


def test_bucket_is_stable_and_in_range() -> None:
    assert bucket("pos:1:0:<STR>", N_PN_DEFAULT) == bucket("pos:1:0:<STR>", N_PN_DEFAULT)
    assert 0 <= bucket("anything", N_PN_DEFAULT) < N_PN_DEFAULT
    assert bucket("a", 124) != bucket("b", 124) or bucket("a", 123) != bucket("b", 123)
    assert ENCODER_VERSION
