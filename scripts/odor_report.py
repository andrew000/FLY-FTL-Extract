"""Measure the odours of every candidate in the fixtures and write ``docs/ODOR.md``.

For each fixture (first run of ``args.json``) every parsable Python file is tokenized into
candidates; a candidate is *positive* when ``reference/`` reports a key at the same call
position with the same key name.  Since ``fly-odor-4`` a candidate's odour is a *sequence*
of puffs (one PN vector per slot of the 6 / focus / 3 window).  Then:

* distribution of active PNs per puff and of silent slots per candidate;
* per-puff Kenyon-cell patterns of every sequence (two seeds, ``Brain.simulate_sequence``)
  and the Jaccard of the active (puff, KC) sets for (a) the same candidate with different
  seeds, (b) positive/negative pairs, (c) positive/positive pairs from different files,
  (d) negative/negative pairs;
* the same numbers for the encoder variants the proxy compared (docs/METRICS.md §6);
* a human-readable dump of the first candidates of fixture ``basic``.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, load
from fly_ftl_extract.odor.encoder import (
    DEFAULT_ENCODER,
    ENCODER_VERSION,
    EncoderParams,
    encode_many,
    normalize_window,
    slot_features,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixtures import LabeledCandidate as Labeled
from _fixtures import fixture_candidates as labeled_candidates

REPO = Path(__file__).resolve().parent.parent
OUT_DOC = REPO / "docs" / "ODOR.md"
ACTIVE_THRESHOLD = 0.1
DUMP_COUNT = 20
MAX_PAIRS = 3000
SEEDS = (1, 2)
SWEEP = (
    EncoderParams(bigrams=True, hashes_per_feature=2),  # fly-odor-4
    EncoderParams(bigrams=True, hashes_per_feature=1),
    EncoderParams(bigrams=False, hashes_per_feature=2),
    EncoderParams(bigrams=False, hashes_per_feature=1),
)


def jaccard_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard of the active sets per row; trailing dimensions (puff, KC) are flattened."""
    x, y = (a > 0).reshape(a.shape[0], -1), (b > 0).reshape(b.shape[0], -1)
    return (x & y).sum(1) / np.maximum((x | y).sum(1), 1)


def histogram(values: np.ndarray, edges: list[int]) -> list[tuple[str, int]]:
    return [
        (f"{lo}–{hi - 1}", int(((values >= lo) & (values < hi)).sum()))
        for lo, hi in itertools.pairwise(edges)
    ]


@dataclass(frozen=True)
class Measure:
    params: EncoderParams
    active_pn_per_puff: float
    kc_active_per_puff: float
    j_same: float
    j_pos_neg: float
    j_pos_pos: float
    j_neg_neg: float


def encode_items(items: list[Labeled], params: EncoderParams) -> np.ndarray:
    """``(n, n_slots, n_pn)`` puff sequences, encoded fixture by fixture with its options."""
    options_by_fixture = {it.file.fixture: it.file.options for it in items}
    return np.concatenate(
        [
            encode_many(
                [it.candidate.window for it in items if it.file.fixture == fx], opts, params=params
            )
            for fx, opts in options_by_fixture.items()
        ]
    )


def measure(
    items: list[Labeled], brain: Brain, params: EncoderParams, rng_seed: int = 0
) -> tuple[Measure, np.ndarray, dict[int, np.ndarray]]:
    """Puffs, per-puff KC patterns and the four Jaccard numbers for one encoder setting."""
    puffs = encode_items(items, params)
    positive = np.array([it.positive for it in items])
    files = np.array([f"{it.file.fixture}/{it.file.path}" for it in items])
    results = {s: brain.simulate_sequence(puffs, seed=s) for s in SEEDS}
    kc = {s: r.kc_counts for s, r in results.items()}
    rng = np.random.default_rng(rng_seed)
    pos_idx = [int(i) for i in np.flatnonzero(positive)]
    neg_idx = [int(i) for i in np.flatnonzero(~positive)]

    def sample(pairs: list[tuple[int, int]]) -> np.ndarray:
        arr = np.array(pairs)
        if len(arr) > MAX_PAIRS:
            arr = arr[rng.choice(len(arr), MAX_PAIRS, replace=False)]
        return arr

    pn = sample([(p, n) for p in pos_idx for n in neg_idx])
    pp = sample(
        [(a, b) for i, a in enumerate(pos_idx) for b in pos_idx[i + 1 :] if files[a] != files[b]]
    )
    nn = sample([(a, b) for i, a in enumerate(neg_idx) for b in neg_idx[i + 1 :]])
    k1 = kc[SEEDS[0]]
    non_empty = puffs.any(axis=2)
    m = Measure(
        params=params,
        active_pn_per_puff=float((puffs > ACTIVE_THRESHOLD).sum(axis=2)[non_empty].mean()),
        kc_active_per_puff=results[SEEDS[0]].kc_active_fraction_non_empty,
        j_same=float(jaccard_pairs(k1, kc[SEEDS[1]]).mean()),
        j_pos_neg=float(jaccard_pairs(k1[pn[:, 0]], k1[pn[:, 1]]).mean()),
        j_pos_pos=float(jaccard_pairs(k1[pp[:, 0]], k1[pp[:, 1]]).mean()),
        j_neg_neg=float(jaccard_pairs(k1[nn[:, 0]], k1[nn[:, 1]]).mean()),
    )
    return m, puffs, kc


def main() -> int:
    items = labeled_candidates()
    # keep items grouped by fixture: the odour matrix is built fixture by fixture
    order = {fx: i for i, fx in enumerate(dict.fromkeys(it.file.fixture for it in items))}
    items = sorted(items, key=lambda it: order[it.file.fixture])
    brain = Brain(load(), DEFAULT_PARAMS)
    sweep = [measure(items, brain, p)[0] for p in SWEEP]
    for m in sweep:
        print(
            f"bigrams {m.params.bigrams} hashes {m.params.hashes_per_feature}: "
            f"PN/puff {m.active_pn_per_puff:.1f} KC/puff {m.kc_active_per_puff:.3f} "
            f"J same {m.j_same:.3f} pos/neg {m.j_pos_neg:.3f} pos/pos {m.j_pos_pos:.3f} "
            f"neg/neg {m.j_neg_neg:.3f}",
            flush=True,
        )
    _current, puffs, kc = measure(items, brain, DEFAULT_ENCODER)
    positive = np.array([it.positive for it in items])
    non_empty = puffs.any(axis=2)
    active = (puffs > ACTIVE_THRESHOLD).sum(axis=2)  # (n, n_slots)
    silent_slots = (~non_empty).sum(axis=1)
    focus = DEFAULT_ENCODER.context_before
    results = {s: brain.simulate_sequence(puffs, seed=s) for s in SEEDS}
    frac = {s: r.kc_active_fraction_non_empty for s, r in results.items()}
    per_slot_kc = [
        float(results[SEEDS[0]].kc_active_fraction[non_empty[:, k], k].mean())
        for k in range(puffs.shape[1])
    ]
    per_slot_pn = [float(active[non_empty[:, k], k].mean()) for k in range(puffs.shape[1])]

    rng = np.random.default_rng(0)
    same = jaccard_pairs(kc[SEEDS[0]], kc[SEEDS[1]])
    pos_idx = np.flatnonzero(positive)
    neg_idx = np.flatnonzero(~positive)
    files = np.array([f"{it.file.fixture}/{it.file.path}" for it in items])
    pn_pairs = np.array([(p, n) for p in pos_idx for n in neg_idx])
    if len(pn_pairs) > MAX_PAIRS:
        pn_pairs = pn_pairs[rng.choice(len(pn_pairs), MAX_PAIRS, replace=False)]
    pp_pairs = np.array(
        [(a, b) for i, a in enumerate(pos_idx) for b in pos_idx[i + 1 :] if files[a] != files[b]]
    )
    if len(pp_pairs) > MAX_PAIRS:
        pp_pairs = pp_pairs[rng.choice(len(pp_pairs), MAX_PAIRS, replace=False)]
    k1 = kc[SEEDS[0]]
    pos_neg = jaccard_pairs(k1[pn_pairs[:, 0]], k1[pn_pairs[:, 1]])
    pos_pos = jaccard_pairs(k1[pp_pairs[:, 0]], k1[pp_pairs[:, 1]])
    nn_pairs = np.array([(a, b) for i, a in enumerate(neg_idx) for b in neg_idx[i + 1 :]])
    if len(nn_pairs) > MAX_PAIRS:
        nn_pairs = nn_pairs[rng.choice(len(nn_pairs), MAX_PAIRS, replace=False)]
    neg_neg = jaccard_pairs(k1[nn_pairs[:, 0]], k1[nn_pairs[:, 1]])

    def stats(v: np.ndarray) -> str:
        return f"{v.mean():.3f} (median {np.median(v):.3f}, min {v.min():.3f}, max {v.max():.3f})"

    print(f"candidates {len(items)}, positives {int(positive.sum())}")
    print(f"active PN per non-empty puff: mean {active[non_empty].mean():.1f}")
    print(f"KC active per puff: {frac}")
    print(f"(a) same candidate, seeds {SEEDS}: J {stats(same)}")
    print(f"(b) positive/negative: J {stats(pos_neg)}")
    print(f"(c) positive/positive, different files: J {stats(pos_pos)}")
    print(f"(d) negative/negative: J {stats(neg_neg)}")

    basic = [it for it in items if it.file.fixture == "basic" and it.file.path.endswith("start.py")]
    dump_lines = []
    for it in basic[:DUMP_COUNT]:
        c = it.candidate
        seq = " ".join(t for t, _ in normalize_window(c.window, it.file.options))
        kwargs = ", ".join(
            f"{k.name}={k.path_value!r}" if k.path_value is not None else k.name for k in c.kwargs
        )
        slots = slot_features(c.window, it.file.options)
        per_slot = " ".join(f"[{len(s)}]" if s else "[·]" for s in slots)
        dump_lines.append(
            f"{c.line}:{c.column} {c.kind:<6} {'POS' if it.positive else 'neg'} "
            f"key={c.key_name!r} kwargs=[{kwargs}]{' **' if c.kwargs_unknown else ''}\n"
            f"    {seq}\n"
            f"    features per slot: {per_slot}"
        )
    edges = [0, 4, 8, 10, 12, 14, 16, 20, 30, 125]
    hist = histogram(active[non_empty], edges)
    per_kind = {
        kind: active[np.array([it.candidate.kind == kind for it in items])][:, focus]
        for kind in ("string", "chain")
    }

    lines = [
        "# Odour: the fixture candidates, their PN vectors and KC patterns",
        "",
        f"Generated by `scripts/odor_report.py`. Encoder `{ENCODER_VERSION}` ({DEFAULT_ENCODER}),",
        (
            f"brain `BrainParams` at the defaults (puff {DEFAULT_PARAMS.puff_ms:.0f} ms × "
            f"{puffs.shape[1]} puffs + {DEFAULT_PARAMS.t_silence:.0f} ms of silence, syn_scale "
            f"{DEFAULT_PARAMS.syn_scale}, apl_scale {DEFAULT_PARAMS.apl_scale}). Candidates — from every fixture "
            "(the first run of each `args.json`),"
        ),
        "positive = the labelling rule from `reference/labels.py` (exactly one positive per teacher occurrence).",
        "",
        "## 1. Candidates",
        "",
        (
            f"- Candidates in total: {len(items)}; positive {int(positive.sum())}, negative "
            f"{int((~positive).sum())}. Recall at the candidate level = 1.0 (`tests/test_tokenizer.py`)."
        ),
        (
            f"- By kind: string {int(sum(it.candidate.kind == 'string' for it in items))}, "
            f"chain {int(sum(it.candidate.kind == 'chain' for it in items))}."
        ),
        (
            f"- Silent slots (a candidate close to the start of the file): in {int((silent_slots > 0).sum())} "
            f"candidates, on average {silent_slots.mean():.2f} of {puffs.shape[1]} slots."
        ),
        "",
        "## 2. How many PNs one puff activates",
        "",
        (
            f"An active PN = odor > {ACTIVE_THRESHOLD}. One puff is one window slot: on average "
            f"{active[non_empty].mean():.1f} of 124 ({active[non_empty].mean() / 124:.0%}) in a non-empty "
            f"puff; the candidate slot: string {per_kind['string'].mean():.1f}, chain "
            f"{per_kind['chain'].mean():.1f} (a chain hashes each of its tokens, hence denser)."
        ),
        "",
        "| active PNs in the puff | puffs |",
        "|---:|---:|",
    ]
    lines.extend(f"| {rng_} | {n} |" for rng_, n in hist)
    lines += [
        "",
        "| slot | active PN | KC active / puff (seed 1) |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {k - focus:+d} | {per_slot_pn[k]:.1f} | {per_slot_kc[k]:.3f} |"
        for k in range(puffs.shape[1])
    )
    lines += [
        "",
        "## 3. KC patterns",
        "",
        f"Simulation of all {len(items)} sequences with two seeds. The share of active KCs per non-empty puff: "
        + ", ".join(f"seed {s}: {v:.3f}" for s, v in frac.items())
        + " (the calibration on these very sequences aimed at 0.08–0.10, docs/BENCH.md §2).",
        "",
        "Jaccard is computed over the sets of active (puff, KC), i.e. the token order is part of the code.",
        "",
        "| pairs | n | Jaccard of active (puff, KC): mean (median, min, max) |",
        "|---|---:|---|",
        f"| (a) the same candidate, seed {SEEDS[0]} vs {SEEDS[1]} | {len(same)} | {stats(same)} |",
        f"| (b) positive / negative (the same seed) | {len(pos_neg)} | {stats(pos_neg)} |",
        f"| (c) positive / positive from different files | {len(pos_pos)} | {stats(pos_pos)} |",
        f"| (d) negative / negative | {len(neg_neg)} | {stats(neg_neg)} |",
        "",
        "### Encoder levers (the same candidate set, seed 1 vs 2)",
        "",
        (
            "Bigrams (`bigrams`: the pair «previous token, token» is added to every slot) and the number "
            "of hashes per feature (`hashes_per_feature`). The first row is the current parameters; for the proxy ceiling "
            "of every variant see docs/METRICS.md §6."
        ),
        "",
        "| bigrams | hashes | active PN / puff | KC active / puff | J (a) same | J (b) pos/neg | J (c) pos/pos | J (d) neg/neg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {m.params.bigrams} | {m.params.hashes_per_feature} | {m.active_pn_per_puff:.1f} "
            f"| {m.kc_active_per_puff:.3f} | {m.j_same:.3f} | {m.j_pos_neg:.3f} | {m.j_pos_pos:.3f} "
            f"| {m.j_neg_neg:.3f} |"
            for m in sweep
        ),
        "",
        f"## 4. Candidate dump of the `basic` fixture (app/handlers/start.py, first {DUMP_COUNT})",
        "",
        (
            "Format: `line:column kind POS/neg key=… kwargs=[…]`, the normalised window "
            "(6 tokens before, focus, 3 after — what goes into the slots) and the number of features in each "
            "of the 10 slots (`[·]` — a silent puff)."
        ),
        "",
        "```",
        *dump_lines,
        "```",
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
