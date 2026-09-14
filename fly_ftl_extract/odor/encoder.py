"""Turn a token :class:`Window` into an odour: a PN activity vector in ``[0, 1]``.

Normalisation (CLAUDE.md, «Odour»): string literals → ``<STR>``, numbers → ``<NUM>``,
identifiers → ``<NAME>``, except the names the options make meaningful, which keep a
class token of their own so the fly can smell them: ``--i18n-keys`` → ``<I18N>``,
``-p`` prefixes → ``<PREFIX>``, ``--ignore-attributes`` → ``<IGNORE>``, ``--ignore-kwargs``
→ ``<IGNORE_KW>``.  ``get`` and ``_path`` are constants of the original, so they stay
literal; Python keywords stay literal; operators stay literal.  The options therefore change
*how a window smells*, never what the fly decides about the smell.

Features: n-grams (n = 1..3) of the normalised tokens with their position relative to the
focus (positional n-grams) plus the same n-grams without position (bag n-grams, half
weight), plus the window's scalar features (depth, first positional, in kwargs, focus
kind).  Every feature is hashed with ``blake2b`` (salt = :data:`ENCODER_VERSION`) into one
of ``n_pn`` buckets; the bucket sums its feature weights and the sum goes through ``tanh``.
Tokens far from the focus weigh less (``exp(-distance / DISTANCE_TAU)``).
"""

from __future__ import annotations

import hashlib
import keyword
import math
from dataclasses import dataclass

import numpy as np

from fly_ftl_extract.ftl.model import GET_ATTR, PATH_KWARG, ExtractOptions
from fly_ftl_extract.tokenizer.candidates import Tok, Window

ENCODER_VERSION = "fly-odor-1"
"""Salt of the feature hash.  Bump on *any* change of the normalisation, the feature set,
the weights or the bucket count: trained MBON weights are only valid for one version."""

N_PN_DEFAULT = 124
"""Number of PN buckets = cholinergic uniglomerular PNs in ``data/mb_fafb783.npz``."""


@dataclass(frozen=True)
class EncoderParams:
    """Knobs of the encoder (all part of :data:`ENCODER_VERSION`)."""

    max_ngram: int = 3
    """Longest n-gram."""
    distance_tau: float = 6.0
    """Weight of a token ``d`` positions from the focus is ``exp(-d / distance_tau)``."""
    bag_weight: float = 0.5
    """Weight multiplier for position-free n-grams."""
    max_depth: int = 5
    """Bracket depth is clipped here before it becomes a feature."""


DEFAULT_ENCODER = EncoderParams()

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


def normalize_window(window: Window, options: ExtractOptions) -> list[tuple[str, int]]:
    """``(token, position)`` pairs; position 0 is the first focus token, before < 0."""
    out: list[tuple[str, int]] = []
    n_before = len(window.before)
    for i, tok in enumerate(window.before):
        out.append((normalize_token(tok, options), i - n_before))
    for i, tok in enumerate(window.focus):
        out.append((normalize_token(tok, options), i))
    n_focus = len(window.focus)
    for i, tok in enumerate(window.after):
        out.append((normalize_token(tok, options), n_focus + i))
    return out


def features(
    window: Window, options: ExtractOptions, params: EncoderParams = DEFAULT_ENCODER
) -> list[tuple[str, float]]:
    """``(feature, weight)`` pairs of a window (before hashing)."""
    seq = normalize_window(window, options)
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
            out.append((f"bag:{n}:{text}", weight * params.bag_weight))
    out.append((f"depth:{min(window.depth, params.max_depth)}", 1.0))
    out.append((f"first_positional:{int(window.first_positional)}", 1.0))
    out.append((f"in_kwargs:{int(window.in_kwargs)}", 1.0))
    out.append((f"kind:{window.focus_kind}", 1.0))
    return out


def bucket(feature: str, n_pn: int) -> int:
    """Hash bucket of a feature (``blake2b`` salted with :data:`ENCODER_VERSION`)."""
    digest = hashlib.blake2b(
        feature.encode("utf-8"), digest_size=8, salt=ENCODER_VERSION.encode("ascii")[:16]
    ).digest()
    return int.from_bytes(digest, "little") % n_pn


def encode(
    window: Window,
    options: ExtractOptions,
    n_pn: int = N_PN_DEFAULT,
    params: EncoderParams = DEFAULT_ENCODER,
) -> np.ndarray:
    """PN activity vector (``float32``, shape ``(n_pn,)``, values in ``[0, 1)``)."""
    vec = np.zeros(n_pn, dtype=np.float64)
    for feature, weight in features(window, options, params):
        vec[bucket(feature, n_pn)] += weight
    return np.tanh(vec).astype(np.float32)


def encode_many(
    windows: list[Window],
    options: ExtractOptions,
    n_pn: int = N_PN_DEFAULT,
    params: EncoderParams = DEFAULT_ENCODER,
) -> np.ndarray:
    """``(len(windows), n_pn)`` matrix of odours."""
    if not windows:
        return np.zeros((0, n_pn), dtype=np.float32)
    return np.stack([encode(w, options, n_pn, params) for w in windows])
