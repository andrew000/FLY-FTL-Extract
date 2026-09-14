"""Measure the odours of every candidate in the fixtures and write ``docs/ODOR.md``.

For each fixture (first run of ``args.json``) every parsable Python file is tokenized into
candidates; a candidate is *positive* when ``reference/`` reports a key at the same call
position with the same key name.  Then:

* distribution of active PNs (odour > 0.1) and odour mass per candidate;
* Kenyon-cell patterns of every odour (two seeds) and the Jaccard of the active-KC sets for
  (a) the same candidate with different seeds, (b) positive/negative pairs,
  (c) positive/positive pairs from different files;
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
)
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.tokenizer.candidates import Candidate, iter_candidates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixtures import FixtureFile, all_fixture_files

REPO = Path(__file__).resolve().parent.parent
OUT_DOC = REPO / "docs" / "ODOR.md"
ACTIVE_THRESHOLD = 0.1
DUMP_COUNT = 20
MAX_PAIRS = 3000
SEEDS = (1, 2)
SWEEP = (
    EncoderParams(max_ngram=3, distance_tau=6.0, bag_weight=0.5),
    EncoderParams(max_ngram=3, distance_tau=6.0, bag_weight=0.0),
    EncoderParams(max_ngram=3, distance_tau=3.0, bag_weight=0.5),
    EncoderParams(max_ngram=3, distance_tau=3.0, bag_weight=0.0),
    EncoderParams(max_ngram=2, distance_tau=6.0, bag_weight=0.5),
    EncoderParams(max_ngram=2, distance_tau=3.0, bag_weight=0.0),
    EncoderParams(max_ngram=1, distance_tau=6.0, bag_weight=0.0),
)


@dataclass(frozen=True)
class Labeled:
    file: FixtureFile
    candidate: Candidate
    positive: bool


def labeled_candidates() -> list[Labeled]:
    out: list[Labeled] = []
    for f in all_fixture_files():
        if f.source is None:
            continue
        try:
            positives = key_occurrences(f.path, f.source, f.options)
        except SyntaxError:
            continue
        keys = {
            ((k.source_location.line, k.source_location.column), k.key)
            for k in positives
            if k.source_location is not None
        }
        out.extend(
            Labeled(f, c, (c.call_position, c.key_name) in keys)
            for c in iter_candidates(f.source, f.options)
        )
    return out


def jaccard_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y = a > 0, b > 0
    return (x & y).sum(1) / np.maximum((x | y).sum(1), 1)


def histogram(values: np.ndarray, edges: list[int]) -> list[tuple[str, int]]:
    return [
        (f"{lo}–{hi - 1}", int(((values >= lo) & (values < hi)).sum()))
        for lo, hi in itertools.pairwise(edges)
    ]


@dataclass(frozen=True)
class Measure:
    params: EncoderParams
    active_pn: float
    mass: float
    kc_active: float
    j_same: float
    j_pos_neg: float
    j_pos_pos: float
    j_neg_neg: float


def measure(
    items: list[Labeled], brain: Brain, params: EncoderParams, rng_seed: int = 0
) -> tuple[Measure, np.ndarray, dict[int, np.ndarray]]:
    """Odours, KC patterns and the four Jaccard numbers for one encoder setting."""
    options_by_fixture = {it.file.fixture: it.file.options for it in items}
    odors = np.concatenate(
        [
            encode_many(
                [it.candidate.window for it in items if it.file.fixture == fx], opts, params=params
            )
            for fx, opts in options_by_fixture.items()
        ]
    )
    positive = np.array([it.positive for it in items])
    files = np.array([f"{it.file.fixture}/{it.file.path}" for it in items])
    kc = {s: brain.simulate(odors, seed=s).kc_counts for s in SEEDS}
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
    m = Measure(
        params=params,
        active_pn=float((odors > ACTIVE_THRESHOLD).sum(axis=1).mean()),
        mass=float(odors.sum(axis=1).mean()),
        kc_active=float((k1 > 0).mean()),
        j_same=float(jaccard_pairs(k1, kc[SEEDS[1]]).mean()),
        j_pos_neg=float(jaccard_pairs(k1[pn[:, 0]], k1[pn[:, 1]]).mean()),
        j_pos_pos=float(jaccard_pairs(k1[pp[:, 0]], k1[pp[:, 1]]).mean()),
        j_neg_neg=float(jaccard_pairs(k1[nn[:, 0]], k1[nn[:, 1]]).mean()),
    )
    return m, odors, kc


def main() -> int:
    items = labeled_candidates()
    # keep items grouped by fixture: the odour matrix is built fixture by fixture
    order = {fx: i for i, fx in enumerate(dict.fromkeys(it.file.fixture for it in items))}
    items = sorted(items, key=lambda it: order[it.file.fixture])
    sweep = [measure(items, Brain(load(), DEFAULT_PARAMS), p)[0] for p in SWEEP]
    for m in sweep:
        print(
            f"ngram {m.params.max_ngram} tau {m.params.distance_tau} bag {m.params.bag_weight}: "
            f"PN {m.active_pn:.1f} mass {m.mass:.1f} KC {m.kc_active:.3f} "
            f"J same {m.j_same:.3f} pos/neg {m.j_pos_neg:.3f} pos/pos {m.j_pos_pos:.3f} "
            f"neg/neg {m.j_neg_neg:.3f}",
            flush=True,
        )
    options_by_fixture = {it.file.fixture: it.file.options for it in items}
    odors = np.concatenate(
        [
            encode_many([it.candidate.window for it in items if it.file.fixture == fx], opts)
            for fx, opts in options_by_fixture.items()
        ]
    )
    positive = np.array([it.positive for it in items])
    files = np.array([f"{it.file.fixture}/{it.file.path}" for it in items])
    active = (odors > ACTIVE_THRESHOLD).sum(axis=1)
    mass = odors.sum(axis=1)

    cx = load()
    brain = Brain(cx, DEFAULT_PARAMS)
    results = {s: brain.simulate(odors, seed=s) for s in SEEDS}
    kc = {s: r.kc_counts for s, r in results.items()}
    frac = {s: float(r.kc_active_fraction.mean()) for s, r in results.items()}

    rng = np.random.default_rng(0)
    same = jaccard_pairs(kc[SEEDS[0]], kc[SEEDS[1]])
    pos_idx = np.flatnonzero(positive)
    neg_idx = np.flatnonzero(~positive)
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
    print(
        f"active PN: mean {active.mean():.1f}, median {np.median(active):.0f}, min {active.min()}, max {active.max()}"
    )
    print(f"odor mass: mean {mass.mean():.2f}")
    print(f"KC active fraction: {frac}")
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
        dump_lines.append(
            f"{c.line}:{c.column} {c.kind:<6} {'POS' if it.positive else 'neg'} "
            f"key={c.key_name!r} kwargs=[{kwargs}]{' **' if c.kwargs_unknown else ''}\n"
            f"    {seq}"
        )
    edges = [0, 10, 20, 30, 40, 50, 60, 80, 100, 125]
    hist = histogram(active, edges)
    per_kind = {
        kind: active[np.array([it.candidate.kind == kind for it in items])]
        for kind in ("string", "chain")
    }

    lines = [
        "# Odour: the fixture candidates, their PN vectors and KC patterns",
        "",
        f"Generated by `scripts/odor_report.py`. Encoder `{ENCODER_VERSION}` ({DEFAULT_ENCODER}),",
        (
            f"brain `BrainParams` at the defaults (T_stim {DEFAULT_PARAMS.t_stim:.0f} ms, syn_scale "
            f"{DEFAULT_PARAMS.syn_scale}). Candidates — from every fixture (the first run of each `args.json`),"
        ),
        "positive = `reference/` finds a key with the same call position and the same name.",
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
        "",
        "## 2. How many PNs one odour activates",
        "",
        (
            f"An active PN = odor > {ACTIVE_THRESHOLD}. Mean {active.mean():.1f} of 124 "
            f"({active.mean() / 124:.0%}), median {np.median(active):.0f}, min {active.min()}, max {active.max()}; "
            f"string candidates {per_kind['string'].mean():.1f}, chain candidates {per_kind['chain'].mean():.1f}. "
            f"Mean odour mass (the vector sum) {mass.mean():.2f}, positives {mass[positive].mean():.2f}, "
            f"negatives {mass[~positive].mean():.2f}."
        ),
        "",
        "| active PNs | candidates |",
        "|---:|---:|",
    ]
    lines.extend(f"| {rng_} | {n} |" for rng_, n in hist)
    lines += [
        "",
        "## 3. KC patterns",
        "",
        f"Simulation of all {len(items)} odours with two seeds. The share of active KCs: "
        + ", ".join(f"seed {s}: {v:.3f}" for s, v in frac.items())
        + " (the calibration aimed at 0.05–0.10 on a typical odour of 30 % PNs).",
        "",
        "| pairs | n | Jaccard of active KCs: mean (median, min, max) |",
        "|---|---:|---|",
        f"| (a) the same candidate, seed {SEEDS[0]} vs {SEEDS[1]} | {len(same)} | {stats(same)} |",
        f"| (b) positive / negative (the same seed) | {len(pos_neg)} | {stats(pos_neg)} |",
        f"| (c) positive / positive from different files | {len(pos_pos)} | {stats(pos_pos)} |",
        f"| (d) negative / negative | {len(neg_neg)} | {stats(neg_neg)} |",
        "",
        "### Encoder levers (the same candidate set, seed 1 vs 2)",
        "",
        (
            "The n-gram density (`max_ngram`, `bag_weight` — the weight of position-independent n-grams) and "
            "the distance weights (`distance_tau`). The first row is the current parameters."
        ),
        "",
        "| max_ngram | distance_tau | bag_weight | active PN | mass | KC active | J (a) same | J (b) pos/neg | J (c) pos/pos | J (d) neg/neg |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {m.params.max_ngram} | {m.params.distance_tau} | {m.params.bag_weight} | {m.active_pn:.1f} "
            f"| {m.mass:.1f} | {m.kc_active:.3f} | {m.j_same:.3f} | {m.j_pos_neg:.3f} | {m.j_pos_pos:.3f} "
            f"| {m.j_neg_neg:.3f} |"
            for m in sweep
        ),
        "",
        f"## 4. Candidate dump of the `basic` fixture (app/handlers/start.py, first {DUMP_COUNT})",
        "",
        (
            "Format: `line:column kind POS/neg key=… kwargs=[…]` and the normalised window "
            "(12 tokens before, focus, 6 after)."
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
