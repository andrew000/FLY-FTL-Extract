"""Turn a token :class:`Window` into an odour *sequence*: one PN activity vector per slot.

Normalisation (CLAUDE.md, «Odour»): string literals → ``<STR>``, numbers → ``<NUM>``,
identifiers → ``<NAME>``, except the names the options make meaningful, which keep a
class token of their own so the fly can smell them: ``--i18n-keys`` → ``<I18N>``,
``-p`` prefixes → ``<PREFIX>``, ``--ignore-attributes`` → ``<IGNORE>``, ``--ignore-kwargs``
→ ``<IGNORE_KW>``.  ``get`` and ``_path`` are constants of the original, so they stay
literal; Python keywords stay literal; operators stay literal.  The options therefore change
*how a window smells*, never what the fly decides about the smell.

Temporal coding (``fly-odor-4``, auditor's decision after Phase 5): the window is a sequence
of ``N_SLOTS`` = 6 + 1 + 3 slots — six tokens before the focus (right-aligned to it), the
focus itself, three tokens after.  Every slot becomes its own ``n_pn`` vector — one *puff*
of odour that the mushroom body receives for ``BrainParams.puff_ms`` before the next slot's
puff arrives (``Brain.simulate_sequence``).  A slot's features are the normalised token
together with its role: signed distance to the focus, ``tokenize`` type and bracket depth
(derived lexically by walking the brackets outward from the focus).  The focus slot hashes
every focus token with its index in the focus (a chain ``i18n.a.b`` is several tokens but
one slot) plus the window's scalars (kind, first positional, in kwargs, depth).  A slot with
no token (a candidate on the first line of a file) is a zero vector: a silent puff.  Every
feature is hashed with ``blake2b`` (salt = :data:`ENCODER_VERSION`) into ``n_pn`` buckets,
``hashes_per_feature`` times; the bucket sums its feature weights and the sum goes through
``tanh``.

Why: with 124 PN buckets a single vector of positional n-grams collides (the Phase 5
proxy in docs/METRICS.md §6 topped at F1 0.973 for context 6/3 while the same features in
1024 buckets reached 0.997).  Spreading the window over time gives every slot its own 124
buckets and lets the Kenyon cells' membrane carry the previous tokens into the next puff.
Proxy on the grammar-3 test split (docs/METRICS.md §6): fly-odor-3 0.9870, slots 0.9921,
slots + bigrams 0.9973, slots + bigrams with 2 hashes per feature 0.9985 (the default).
What remains is beyond a 6/3 window: ``obj.self.i18n.get("k", …)`` — the object that makes
it a non-key sits seven tokens before the string.
:func:`ngram_features` is the ``fly-odor-3`` feature set, kept only so that the proxy can
put the two encoders side by side.
"""

from __future__ import annotations

import hashlib
import keyword
import math
from dataclasses import dataclass

import numpy as np

from fly_ftl_extract.ftl.model import GET_ATTR, PATH_KWARG, ExtractOptions
from fly_ftl_extract.tokenizer.candidates import Tok, Window

ENCODER_VERSION = "fly-odor-4"
"""Salt of the feature hash.  Bump on *any* change of the normalisation, the feature set,
the weights, the slot layout or the bucket count: trained MBON weights are only valid for
one version."""

N_PN_DEFAULT = 124
"""Number of PN buckets = cholinergic uniglomerular PNs in ``data/mb_fafb783.npz``."""

_OPENERS = frozenset("([{")
_CLOSERS = frozenset(")]}")


@dataclass(frozen=True)
class EncoderParams:
    """Knobs of the encoder (all part of :data:`ENCODER_VERSION`)."""

    context_before: int = 6
    """Slots before the focus (the tokenizer's window may hold more tokens; the last
    ``context_before`` are used, right-aligned to the focus)."""
    context_after: int = 3
    """Slots after the focus."""
    max_depth: int = 5
    """Bracket depth is clipped here before it becomes a feature."""
    hashes_per_feature: int = 2
    """How many buckets one feature lights (a Bloom-filter style spread: more active PNs
    per puff without more features).  Proxy (docs/METRICS.md §6): 1 → F1 0.9973,
    2 → 0.9985 with bigrams; ~10 active PNs per puff."""
    bigrams: bool = True
    """Add ``(previous token, token)`` to every context slot.  Proxy: without bigrams the
    slot code tops at F1 0.9921 (0.9930 even with 1024 buckets), with them 0.9973–0.9985 —
    the limit was the linear readout's lack of pair interactions, not the hash."""
    # --- fly-odor-3 knobs, used by :func:`ngram_features` only -------------------------
    max_ngram: int = 3
    """Longest positional n-gram (fly-odor-3)."""
    distance_tau: float = 6.0
    """fly-odor-3: weight of a token ``d`` positions from the focus is ``exp(-d / tau)``."""
    bag_weight: float = 0.0
    """fly-odor-3: weight multiplier for position-free n-grams (0 = none)."""

    @property
    def n_slots(self) -> int:
        """Puffs per window: before + focus + after."""
        return self.context_before + 1 + self.context_after


DEFAULT_ENCODER = EncoderParams()
N_SLOTS = DEFAULT_ENCODER.n_slots
"""Puffs per window with the default parameters (6 + 1 + 3)."""

_CLASS_TOKENS = {
    "STRING": "<STR>",
    "NUMBER": "<NUM>",
    "NEWLINE": "<NL>",
    "FSTRING_START": "<FSTR>",
    "FSTRING_MIDDLE": "<FSTR_MID>",
    "FSTRING_END": "<FSTR_END>",
    "TSTRING_START": "<TSTR>",
    "TSTRING_MIDDLE": "<TSTR_MID>",
    "TSTRING_END": "<TSTR_END>",
}


def normalize_token(tok: Tok, options: ExtractOptions) -> str:
    """The class or literal a token contributes to the smell."""
    if tok.type == "NAME":
        name = tok.text
        if keyword.iskeyword(name):
            return name
        if name in options.i18n_keys:
            return "<I18N>"
        if name in options.i18n_keys_prefix:
            return "<PREFIX>"
        if name in options.ignore_attributes:
            return "<IGNORE>"
        if name in options.ignore_kwargs:
            return "<IGNORE_KW>"
        if name in (GET_ATTR, PATH_KWARG):
            return name
        return "<NAME>"
    if tok.type == "OP":
        return tok.text
    if tok.type == "STRING" and tok.text[:1].lower() == "b":
        return "<BYTES>"
    return _CLASS_TOKENS.get(tok.type, f"<{tok.type}>")


def _context(window: Window, params: EncoderParams) -> tuple[tuple[Tok, ...], tuple[Tok, ...]]:
    """The before/after tokens that are encoded.

    The last ``context_before`` tokens before and the first ``context_after`` after.
    NB: a plain ``before[len - n:]`` goes negative for short contexts (a candidate on the
    first line of a file) and silently drops the leading tokens: fly-odor-2 lost
    ``<I18N> .`` there and missed every one-line file; fly-odor-3 fixed it.
    """
    before = window.before[max(0, len(window.before) - params.context_before) :]
    after = window.after[: params.context_after]
    return before, after


def normalize_window(
    window: Window, options: ExtractOptions, params: EncoderParams = DEFAULT_ENCODER
) -> list[tuple[str, int]]:
    """``(token, position)`` pairs; position 0 is the first focus token, before < 0.

    Only the last ``params.context_before`` tokens before and the first
    ``params.context_after`` after the focus are kept.
    """
    before, after = _context(window, params)
    out: list[tuple[str, int]] = []
    n_before = len(before)
    for i, tok in enumerate(before):
        out.append((normalize_token(tok, options), i - n_before))
    for i, tok in enumerate(window.focus):
        out.append((normalize_token(tok, options), i))
    n_focus = len(window.focus)
    for i, tok in enumerate(after):
        out.append((normalize_token(tok, options), n_focus + i))
    return out


def token_depths(
    window: Window, params: EncoderParams = DEFAULT_ENCODER
) -> tuple[list[int], list[int]]:
    """Bracket depth *before* each encoded context token, ``(before, after)``.

    Derived from ``window.depth`` (the depth at the focus) by walking the brackets outward:
    to the left an opener means the tokens before it are one level up and a closer means
    they are one level down; to the right the mirror image.  Clipped to
    ``[0, params.max_depth]``.
    """
    before, after = _context(window, params)
    lo, hi = 0, params.max_depth
    depths_before: list[int] = []
    current = window.depth
    for tok in reversed(before):
        if tok.type == "OP" and tok.text in _OPENERS:
            current -= 1
            depths_before.append(current)
        elif tok.type == "OP" and tok.text in _CLOSERS:
            depths_before.append(current)
            current += 1
        else:
            depths_before.append(current)
    depths_before.reverse()
    depths_after: list[int] = []
    current = window.depth
    for tok in after:
        if tok.type == "OP" and tok.text in _OPENERS:
            depths_after.append(current)
            current += 1
        elif tok.type == "OP" and tok.text in _CLOSERS:
            current -= 1
            depths_after.append(current)
        else:
            depths_after.append(current)
    return (
        [min(max(d, lo), hi) for d in depths_before],
        [min(max(d, lo), hi) for d in depths_after],
    )


def slot_features(
    window: Window, options: ExtractOptions, params: EncoderParams = DEFAULT_ENCODER
) -> list[list[tuple[str, float]]]:
    """``(feature, weight)`` pairs per slot (``params.n_slots`` lists; empty = silent puff).

    Slot order is source order: ``context_before`` slots (right-aligned to the focus, the
    leading ones empty when the file starts), the focus slot, ``context_after`` slots.
    """
    before, after = _context(window, params)
    depths_before, depths_after = token_depths(window, params)
    slots: list[list[tuple[str, float]]] = [[] for _ in range(params.n_slots)]
    focus_slot = params.context_before

    def context_slot(
        tok: Tok, prev: Tok | None, distance: int, depth: int
    ) -> list[tuple[str, float]]:
        text = normalize_token(tok, options)
        out = [
            (f"tok:{text}", 1.0),
            (f"tok@{distance}:{text}", 1.0),
            (f"type@{distance}:{tok.type}", 1.0),
            (f"depth@{distance}:{depth}", 1.0),
        ]
        if params.bigrams:
            prev_text = normalize_token(prev, options) if prev is not None else "<BOF>"
            out.append((f"bi@{distance}:{prev_text}\x1f{text}", 1.0))
        return out

    n_before = len(before)
    # everything the tokenizer saw before the encoded context, for the first slot's bigram
    all_before = window.before
    offset = len(all_before) - n_before
    for i, tok in enumerate(before):
        prev = all_before[offset + i - 1] if offset + i - 1 >= 0 else None
        slots[focus_slot - n_before + i] = context_slot(tok, prev, i - n_before, depths_before[i])

    focus = window.focus
    focus_features: list[tuple[str, float]] = [
        (f"focus:{i}:{normalize_token(tok, options)}", 1.0) for i, tok in enumerate(focus)
    ]
    focus_features += [
        (f"focus_len:{len(focus)}", 1.0),
        (f"kind:{window.focus_kind}", 1.0),
        (f"first_positional:{int(window.first_positional)}", 1.0),
        (f"in_kwargs:{int(window.in_kwargs)}", 1.0),
        (f"depth:{min(max(window.depth, 0), params.max_depth)}", 1.0),
    ]
    slots[focus_slot] = focus_features

    for i, tok in enumerate(after):
        prev = focus[-1] if i == 0 else after[i - 1]
        slots[focus_slot + 1 + i] = context_slot(tok, prev, i + 1, depths_after[i])
    return slots


def ngram_features(
    window: Window, options: ExtractOptions, params: EncoderParams = DEFAULT_ENCODER
) -> list[tuple[str, float]]:
    """The ``fly-odor-3`` feature set (one vector per window; proxy comparison only).

    Positional n-grams (n = 1..``max_ngram``) of the normalised tokens, optionally the same
    n-grams without position (``bag_weight``), plus the window's scalars; tokens far from
    the focus weigh ``exp(-distance / distance_tau)``.
    """
    seq = normalize_window(window, options, params)
    n_focus = len(window.focus)
    out: list[tuple[str, float]] = []
    for n in range(1, params.max_ngram + 1):
        for i in range(len(seq) - n + 1):
            gram = seq[i : i + n]
            pos = gram[0][1]
            touches_focus = any(0 <= p < n_focus for _, p in gram)
            if touches_focus:
                distance = 0
            elif pos < 0:
                distance = -(gram[-1][1])
            else:
                distance = pos - n_focus + 1
            weight = math.exp(-distance / params.distance_tau)
            text = "\x1f".join(t for t, _ in gram)
            out.append((f"pos:{n}:{pos}:{text}", weight))
            if params.bag_weight > 0:
                out.append((f"bag:{n}:{text}", weight * params.bag_weight))
    out.append((f"depth:{min(window.depth, params.max_depth)}", 1.0))
    out.append((f"first_positional:{int(window.first_positional)}", 1.0))
    out.append((f"in_kwargs:{int(window.in_kwargs)}", 1.0))
    out.append((f"kind:{window.focus_kind}", 1.0))
    return out


def bucket(feature: str, n_pn: int, salt: str = ENCODER_VERSION, k: int = 0) -> int:
    """Hash bucket of a feature (``blake2b`` salted with :data:`ENCODER_VERSION`).

    ``k`` selects one of the ``hashes_per_feature`` independent buckets of a feature.
    """
    digest = hashlib.blake2b(
        feature.encode("utf-8") if k == 0 else f"{feature}\x1e{k}".encode(),
        digest_size=8,
        salt=salt.encode("ascii")[:16],
    ).digest()
    return int.from_bytes(digest, "little") % n_pn


def hash_features(
    features: list[tuple[str, float]], n_pn: int, *, salt: str = ENCODER_VERSION, hashes: int = 1
) -> np.ndarray:
    """One ``(n_pn,)`` float32 vector in ``[0, 1)`` from ``(feature, weight)`` pairs."""
    vec = np.zeros(n_pn, dtype=np.float64)
    for feature, weight in features:
        for k in range(hashes):
            vec[bucket(feature, n_pn, salt, k)] += weight
    return np.tanh(vec).astype(np.float32)


def encode(
    window: Window,
    options: ExtractOptions,
    n_pn: int = N_PN_DEFAULT,
    params: EncoderParams = DEFAULT_ENCODER,
    *,
    salt: str = ENCODER_VERSION,
) -> np.ndarray:
    """Odour sequence ``(n_slots, n_pn)`` float32 in ``[0, 1)``; a silent slot is a zero row."""
    slots = slot_features(window, options, params)
    out = np.zeros((params.n_slots, n_pn), dtype=np.float32)
    for i, feats in enumerate(slots):
        if feats:
            out[i] = hash_features(feats, n_pn, salt=salt, hashes=params.hashes_per_feature)
    return out


def encode_many(
    windows: list[Window],
    options: ExtractOptions,
    n_pn: int = N_PN_DEFAULT,
    params: EncoderParams = DEFAULT_ENCODER,
) -> np.ndarray:
    """``(len(windows), n_slots, n_pn)`` array of odour sequences."""
    if not windows:
        return np.zeros((0, params.n_slots, n_pn), dtype=np.float32)
    return np.stack([encode(w, options, n_pn, params) for w in windows])
