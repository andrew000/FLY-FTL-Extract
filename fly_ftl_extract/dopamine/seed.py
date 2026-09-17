"""Deterministic trial seeds (CLAUDE.md, «Brain», «Determinism»).

The fly must give the same answer for the same code on every run, so the Poisson input of
a candidate's trial is seeded from the file content, the candidate's index in that file
and the encoder version.  Extra sniffs of the same candidate (``resniff``) derive their
seeds from the base seed.
"""

from __future__ import annotations

import hashlib

_SNIFF_STRIDE = 0x9E3779B97F4A7C15  # golden-ratio constant: distinct streams per sniff
_MASK = (1 << 64) - 1


def trial_seed(content: bytes, candidate_index: int, encoder_version: str) -> int:
    """64-bit seed of the first sniff of candidate ``candidate_index`` in ``content``."""
    digest = hashlib.sha256(
        content
        + b"\0"
        + str(candidate_index).encode("ascii")
        + b"\0"
        + encoder_version.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


KWARG_INDEX_BASE = 1_000_000
KWARG_INDEX_STRIDE = 1000


def kwarg_trial_index(candidate_index: int, kwarg_index: int) -> int:
    """Pseudo candidate index of the ``kwarg_index``-th keyword argument of a candidate."""
    return KWARG_INDEX_BASE + candidate_index * KWARG_INDEX_STRIDE + kwarg_index


def sniff_seed(base_seed: int, sniff: int) -> int:
    """Seed of the ``sniff``-th extra trial (``sniff = 0`` is the base seed itself)."""
    return (base_seed + _SNIFF_STRIDE * sniff) & _MASK


def salted_seed(base_seed: int, salt: int) -> int:
    """``--fly-seed``: another nose for the same code (``salt = 0`` keeps the production seed).

    The salt is hashed so that neighbouring salts give unrelated streams.
    """
    if salt == 0:
        return base_seed
    digest = hashlib.sha256(salt.to_bytes(8, "little", signed=True) + b"fly-seed").digest()
    return (base_seed ^ int.from_bytes(digest[:8], "little")) & _MASK
