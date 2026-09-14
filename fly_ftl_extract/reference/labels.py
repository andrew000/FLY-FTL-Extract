"""Labels for the fly's training data, derived from the ast teacher.

Rule (CLAUDE.md, «Dopamine»): for every key occurrence reported by
:func:`fly_ftl_extract.reference.extractor.key_occurrences` exactly one candidate at the
same call position is positive —

* *get-like* call (bare ``name(...)``, ``x.get(...)``: the chain candidate has no key name):
  the string candidate that is the first positional argument is the key, the chain is
  negative;
* *attribute* call (``i18n.some.key(...)``): the chain candidate is the key; a string
  inside the call (``i18n.core.get("core-get")``) is negative even when its text happens
  to equal the key.

Keyword arguments of a positive call are *placeable* when the teacher put them into the
message (``_path`` and ``--ignore-kwargs`` names are not).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from fly_ftl_extract.ftl.model import FluentKey, kwargs_from_key
from fly_ftl_extract.tokenizer.candidates import Candidate


class LabelError(ValueError):
    """The candidates do not offer exactly one match for an occurrence (a tokenizer bug)."""


def label_candidates(
    candidates: Sequence[Candidate], occurrences: Sequence[FluentKey]
) -> tuple[list[bool], list[int]]:
    """``(is_key per candidate, positive candidate index per occurrence)``."""
    labels = [False] * len(candidates)
    by_call: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        if c.call_position is not None:
            by_call[c.call_position].append(i)
    positives: list[int] = []
    for occ in occurrences:
        loc = occ.source_location
        if loc is None:
            msg = f"occurrence {occ.key!r} has no location"
            raise LabelError(msg)
        pos = (loc.line, loc.column)
        idxs = by_call.get(pos, [])
        chains = [i for i in idxs if candidates[i].kind == "chain"]
        if len(chains) != 1:
            msg = f"{pos}: {len(chains)} chain candidates for key {occ.key!r}"
            raise LabelError(msg)
        chain = candidates[chains[0]]
        if chain.key_name is None:
            matches = [
                i
                for i in idxs
                if candidates[i].kind == "string"
                and candidates[i].window.first_positional
                and candidates[i].key_name == occ.key
            ]
        else:
            matches = chains if chain.key_name == occ.key else []
        if len(matches) != 1:
            msg = f"{pos}: {len(matches)} candidates match key {occ.key!r}"
            raise LabelError(msg)
        if labels[matches[0]]:
            msg = f"{pos}: candidate already positive for another occurrence"
            raise LabelError(msg)
        labels[matches[0]] = True
        positives.append(matches[0])
    return labels, positives


def label_kwargs(candidate: Candidate, occurrence: FluentKey) -> list[bool]:
    """``is_placeable`` for every keyword argument of a positive candidate's call."""
    placeable = set(kwargs_from_key(occurrence))
    return [k.name in placeable for k in candidate.kwargs]
