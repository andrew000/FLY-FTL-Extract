"""Calibrate ``BrainParams.syn_scale`` on real odours and measure KC-code reliability.

Odours (auditor's decision after Phase 4): every candidate of every fixture, tokenized and
encoded with that fixture's options — no synthetic odour any more.  For every candidate
``syn_scale`` the script runs the same odour set with APL, without APL (APL→* weights
zeroed) and without KC→KC edges, and prints:

* the active Kenyon-cell fraction (two denominators: all KC, and KC that have a PN input);
* the highest firing rate of any neuron and the median rate of the *active* Kenyon
  cells (spikes / (T_stim + T_silence));
* mean APL and MBON spike counts.

Target: 8–10 % active KC with APL, > 30 % without; runaway criterion (auditor, after
Phase 3): no neuron above 1 / t_refractory and median rate of active KC < 50 Hz.  The chosen
value is written to ``docs/calibration.json`` and the whole curve to the ``calibration``
section of ``docs/BENCH.md``; ``BrainParams.syn_scale`` is then set by hand to the chosen
value (``tests/test_lif.py`` checks that the two agree).

The script also reports the PLAN's Jaccard numbers on the real odours (same candidate with
two seeds vs different candidates) for several stimulus lengths, and how many active PNs
a candidate's odour has versus how many of a Kenyon cell's claws it reaches.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fly_ftl_extract.brain import Brain, BrainParams, Connectome, load
from fly_ftl_extract.odor.encoder import ENCODER_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docsection import update_section
from _fixtures import fixture_candidates, fixture_odors

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "calibration.json"
OUT_DOC = REPO / "docs" / "BENCH.md"

SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
ACTIVE_PN_THRESHOLD = 0.1
TARGET_WITH_APL = (0.08, 0.10)
TARGET_WITHOUT_APL = 0.30
MAX_ACTIVE_KC_MEDIAN_HZ = 50.0
JACCARD_STIMS = (50.0, 75.0, 100.0)
VOTE_MAJORITY = 2  # of 3 sniffs
JACCARD_THRESHOLD = 0.5
PAIR_SAMPLE = 3000

SKELETON = "# Brain: parameters, calibration, speed\n"


def without_kc_kc(cx: Connectome) -> Connectome:
    """The connectome with every KC→KC edge removed (weights and syn_count alike)."""
    mask = np.zeros(cx.n_neurons, dtype=np.float32)
    mask[cx.kc_idx] = 1.0
    d = sp.diags(mask, format="csr")

    def strip(m: sp.csr_matrix) -> sp.csr_matrix:
        out = sp.csr_matrix(m - d @ m @ d)
        out.eliminate_zeros()
        return out

    return dataclasses.replace(cx, weights=strip(cx.weights), syn_count=strip(cx.syn_count))


def max_rate_hz(counts: np.ndarray, params: BrainParams) -> float:
    return float(counts.max()) * 1000.0 / (params.n_steps * params.dt)


def median_active_rate_hz(counts: np.ndarray, params: BrainParams) -> float:
    """Median firing rate over the (trial, KC) pairs that spiked at least once."""
    active = counts[counts > 0]
    if active.size == 0:
        return 0.0
    return float(np.median(active)) * 1000.0 / (params.n_steps * params.dt)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a > 0, b > 0
    return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)


def calibration_curve(cx: Connectome, odors: np.ndarray) -> list[dict[str, float]]:
    rows = []
    cx_nokk = without_kc_kc(cx)
    for scale in SCALES:
        p = BrainParams(syn_scale=scale)
        with_apl = Brain(cx, p).simulate(odors, seed=1)
        no_apl = Brain(cx, p, apl_enabled=False).simulate(odors, seed=1)
        no_kk = Brain(cx_nokk, p).simulate(odors, seed=1)
        rows.append(
            {
                "syn_scale": scale,
                "kc_active_all": float(with_apl.kc_active_fraction.mean()),
                "kc_active_pn_input": float(with_apl.kc_active_fraction_pn_input.mean()),
                "kc_max_rate_hz": max_rate_hz(with_apl.kc_counts, p),
                "kc_median_active_rate_hz": median_active_rate_hz(with_apl.kc_counts, p),
                "max_rate_any_neuron_hz": max(
                    max_rate_hz(with_apl.kc_counts, p),
                    max_rate_hz(with_apl.apl_counts, p),
                    max_rate_hz(with_apl.mbon_counts, p),
                ),
                "apl_spikes": float(with_apl.apl_counts.mean()),
                "mbon_spikes": float(with_apl.mbon_counts.mean()),
                "kc_active_no_apl": float(no_apl.kc_active_fraction.mean()),
                "kc_max_rate_no_apl_hz": max_rate_hz(no_apl.kc_counts, p),
                "kc_active_no_kc_kc": float(no_kk.kc_active_fraction.mean()),
                "kc_max_rate_no_kc_kc_hz": max_rate_hz(no_kk.kc_counts, p),
            }
        )
        r = rows[-1]
        print(
            f"syn_scale {scale:5.2f}: KC active {r['kc_active_all']:.3f} "
            f"(PN-input {r['kc_active_pn_input']:.3f}), max {r['kc_max_rate_hz']:.0f} Hz, "
            f"median active {r['kc_median_active_rate_hz']:.0f} Hz, "
            f"no APL {r['kc_active_no_apl']:.3f}, no KC->KC max {r['kc_max_rate_no_kc_kc_hz']:.0f} Hz",
            flush=True,
        )
    return rows


def choose(rows: list[dict[str, float]]) -> tuple[float, list[str]]:
    """The passing scale closest to the middle of the target band, and unmet criteria."""
    lo, hi = TARGET_WITH_APL
    mid = (lo + hi) / 2
    max_rate = BrainParams().max_rate_hz

    def unmet(r: dict[str, float]) -> list[str]:
        out = []
        if not lo <= r["kc_active_all"] <= hi:
            out.append(f"KC active with APL {r['kc_active_all']:.3f} not in [{lo}, {hi}]")
        if r["kc_active_no_apl"] <= TARGET_WITHOUT_APL:
            out.append(f"KC active without APL {r['kc_active_no_apl']:.3f} <= {TARGET_WITHOUT_APL}")
        if r["max_rate_any_neuron_hz"] > max_rate:
            out.append(f"a neuron fired at {r['max_rate_any_neuron_hz']:.0f} Hz > 1/t_refractory")
        if r["kc_median_active_rate_hz"] >= MAX_ACTIVE_KC_MEDIAN_HZ:
            out.append(
                f"median active-KC rate {r['kc_median_active_rate_hz']:.0f} Hz >= "
                f"{MAX_ACTIVE_KC_MEDIAN_HZ:.0f} Hz"
            )
        return out

    passing = [r for r in rows if not unmet(r)]
    pool = passing or [r for r in rows if len(unmet(r)) == 1] or rows
    best = min(pool, key=lambda r: abs(r["kc_active_all"] - mid))
    return best["syn_scale"], unmet(best)


def reliability(cx: Connectome, scale: float, odors: np.ndarray) -> list[dict[str, float]]:
    """Same-candidate vs different-candidate Jaccard of the active-KC sets (PLAN test)."""
    rng = np.random.default_rng(0)
    n = len(odors)
    pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    if len(pairs) > PAIR_SAMPLE:
        pairs = pairs[rng.choice(len(pairs), PAIR_SAMPLE, replace=False)]
    rows = []
    for t_stim in JACCARD_STIMS:
        p = BrainParams(syn_scale=scale, t_stim=t_stim)
        b = Brain(cx, p)
        k1, k2 = b.simulate(odors, 1).kc_counts, b.simulate(odors, 2).kc_counts
        votes_a = sum((b.simulate(odors, s).kc_counts > 0).astype(np.int8) for s in (1, 2, 3))
        votes_b = sum((b.simulate(odors, s).kc_counts > 0).astype(np.int8) for s in (4, 5, 6))
        vote_a = (np.asarray(votes_a) >= VOTE_MAJORITY).astype(np.int8)
        vote_b = (np.asarray(votes_b) >= VOTE_MAJORITY).astype(np.int8)
        rows.append(
            {
                "t_stim_ms": t_stim,
                "n_steps": p.n_steps,
                "kc_active_all": float((k1 > 0).mean()),
                "jaccard_same_odor": float(jaccard(k1, k2).mean()),
                "jaccard_same_odor_vote3": float(jaccard(vote_a, vote_b).mean()),
                "jaccard_different_odors": float(jaccard(k1[pairs[:, 0]], k1[pairs[:, 1]]).mean()),
                "kc_max_rate_hz": max_rate_hz(k1, p),
            }
        )
        r = rows[-1]
        print(
            f"t_stim {t_stim:5.0f} ms: same-candidate J {r['jaccard_same_odor']:.3f} "
            f"(vote3 {r['jaccard_same_odor_vote3']:.3f}), different {r['jaccard_different_odors']:.3f}",
            flush=True,
        )
    return rows


def claws(cx: Connectome, scale: float, odors: np.ndarray) -> dict[str, object]:
    """How many PN claws of a KC are reached by the odour, and how that predicts activity."""
    b = Brain(cx, BrainParams(syn_scale=scale))
    res = b.simulate(odors, seed=1)
    pn_kc = (cx.syn_count[cx.pn_idx][:, cx.kc_idx] > 0).astype(np.int32)  # (n_pn, n_kc)
    active_claws = (odors > ACTIVE_PN_THRESHOLD).astype(np.int32) @ pn_kc.toarray()
    total_claws = np.asarray(pn_kc.sum(axis=0)).ravel()
    active = res.kc_counts > 0
    by_k = []
    for k in range(int(active_claws.max()) + 1):
        cells = active_claws == k
        n_cells = int(cells.sum())
        if n_cells == 0:
            continue
        by_k.append(
            {
                "active_claws": k,
                "kc_trials": n_cells,
                "share_of_kc_trials": n_cells / active_claws.size,
                "p_active": float(active[cells].mean()),
                "share_of_active_kc": float((active & cells).sum() / max(active.sum(), 1)),
            }
        )
    return {
        "mean_claws_per_kc": float(total_claws.mean()),
        "mean_active_claws_per_kc": float(active_claws.mean()),
        "mean_active_pn_per_odor": float((odors > ACTIVE_PN_THRESHOLD).sum(axis=1).mean()),
        "by_active_claws": by_k,
    }


def write_doc(
    *,
    curve: list[dict[str, float]],
    chosen: float,
    unmet: list[str],
    rel: list[dict[str, float]],
    claw: dict[str, object],
    n_odors: int,
    n_positive: int,
) -> None:
    lines = [
        "## 2. Calibrating `syn_scale` on real odours",
        "",
        (
            f"Odours: all {n_odors} candidates from all fixtures ({n_positive} positive), tokenised "
            f"and encoded with the `{ENCODER_VERSION}` encoder using each fixture's options; the synthetic odour "
            "(30 % of PNs at 150 Hz) is no longer used (reviewer's decision after Phase 4). "
            "The same set for every variant, seed 1."
        ),
        "KC rate = spikes / (T_stim + T_silence). «no APL» — the APL→* weights zeroed;",
        "«no KC→KC» — the 942 KC→KC edges removed (a check for runaway through the recurrence).",
        "",
        (
            "| syn_scale | KC active (all) | KC active (with PN input) | max KC, Hz | median active KC, Hz "
            "| APL spikes | MBON spikes "
            "| no APL: KC active | no APL: max KC, Hz | no KC→KC: KC active | no KC→KC: max KC, Hz |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in curve:
        mark = " **←**" if r["syn_scale"] == chosen else ""
        lines.append(
            f"| {r['syn_scale']}{mark} | {r['kc_active_all']:.3f} | {r['kc_active_pn_input']:.3f} "
            f"| {r['kc_max_rate_hz']:.0f} | {r['kc_median_active_rate_hz']:.0f} "
            f"| {r['apl_spikes']:.1f} | {r['mbon_spikes']:.2f} "
            f"| {r['kc_active_no_apl']:.3f} | {r['kc_max_rate_no_apl_hz']:.0f} "
            f"| {r['kc_active_no_kc_kc']:.3f} | {r['kc_max_rate_no_kc_kc_hz']:.0f} |"
        )
    lines += [
        "",
        f"Criteria: with APL {TARGET_WITH_APL[0]:.0%}–{TARGET_WITH_APL[1]:.0%} of KCs active, without APL > "
        f"{TARGET_WITHOUT_APL:.0%}; runaway: no neuron above 1/t_refractory = "
        f"{BrainParams().max_rate_hz:.0f} Hz, the median rate of active KCs < {MAX_ACTIVE_KC_MEDIAN_HZ:.0f} Hz. "
        f"Chosen: **syn_scale = {chosen}**"
        + (" — every criterion met." if not unmet else " — not met: " + "; ".join(unmet) + "."),
        "",
        "### How many KC claws a real odour reaches",
        "",
        (
            f"An active PN = odor > {ACTIVE_PN_THRESHOLD}; on average {claw['mean_active_pn_per_odor']:.1f} "
            f"active PNs per odour. The mean number of PN inputs (claws) per KC: {claw['mean_claws_per_kc']:.2f}; "
            f"of them on average {claw['mean_active_claws_per_kc']:.2f} are active. The distribution (KC × trials) and"
        ),
        "the probability that the KC spikes, by the number of active claws:",
        "",
        "| active claws | KC×trials | share | P(KC active) | share among active KCs |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_k = claw["by_active_claws"]
    assert isinstance(by_k, list)
    lines.extend(
        f"| {r['active_claws']} | {r['kc_trials']} | {r['share_of_kc_trials']:.3f} | {r['p_active']:.3f} "
        f"| {r['share_of_active_kc']:.3f} |"
        for r in by_k
    )
    current = BrainParams()
    lines += [
        "",
        "### Reliability of the KC code (the Jaccard test from PLAN) on real odours",
        "",
        (
            "Jaccard of the active-KC sets: the same candidate seed 1 vs 2 (mean over all candidates), "
            "the votes of 3 trials (two independent seed triples) and random pairs of different candidates. "
            f"The row with the current T_stim = {current.t_stim:.0f} ms is marked; the table over T_stim stays as "
            "the rationale for the reviewer's decision after Phase 3 (at 50 ms the same odour gave J ≈ 0.40)."
        ),
        "",
        "| T_stim, ms | steps | KC active | J same candidate | J vote-3 | J different candidates | max KC, Hz |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rel:
        mark = " **←**" if r["t_stim_ms"] == current.t_stim else ""
        lines.append(
            f"| {r['t_stim_ms']:.0f}{mark} | {r['n_steps']} | {r['kc_active_all']:.3f} "
            f"| {r['jaccard_same_odor']:.3f} | {r['jaccard_same_odor_vote3']:.3f} "
            f"| {r['jaccard_different_odors']:.3f} | {r['kc_max_rate_hz']:.0f} |"
        )
    chosen_row = next(r for r in rel if r["t_stim_ms"] == current.t_stim)
    same_gap = chosen_row["jaccard_same_odor"] - JACCARD_THRESHOLD
    diff_gap = JACCARD_THRESHOLD - chosen_row["jaccard_different_odors"]
    lines += [
        "",
        (
            f"With the current parameters: the same candidate J = {chosen_row['jaccard_same_odor']:.3f} "
            f"(gap to {JACCARD_THRESHOLD}: {same_gap:+.3f}), different candidates J = "
            f"{chosen_row['jaccard_different_odors']:.3f} (gap {diff_gap:+.3f})."
            + (
                ""
                if same_gap > 0
                else " The criterion «same > 0.5» is not met: by the reviewer's decision after Phase 4 "
                "the corresponding test is diagnostic (xfail), the Phase 5 gate is readout accuracy."
            )
        ),
    ]
    update_section(OUT_DOC, "calibration", "\n".join(lines), SKELETON)


def main() -> int:
    cx = load()
    items = fixture_candidates()
    odors, positive = fixture_odors(items)
    print(f"real odours: {len(odors)} candidates, {int(positive.sum())} positive", flush=True)
    curve = calibration_curve(cx, odors)
    chosen, unmet = choose(curve)
    print(f"chosen syn_scale = {chosen}" + (f"; unmet: {unmet}" if unmet else ""))
    rel = reliability(cx, chosen, odors)
    claw = claws(cx, chosen, odors)
    current = BrainParams()
    chosen_row = next(r for r in rel if r["t_stim_ms"] == current.t_stim)
    payload = {
        "syn_scale": chosen,
        "unmet_criteria": unmet,
        "odors": {
            "source": "fixture candidates",
            "encoder_version": ENCODER_VERSION,
            "n": len(odors),
            "n_positive": int(positive.sum()),
        },
        "t_stim_ms": current.t_stim,
        "jaccard_gap": {
            "same_odor": chosen_row["jaccard_same_odor"] - JACCARD_THRESHOLD,
            "different_odors": JACCARD_THRESHOLD - chosen_row["jaccard_different_odors"],
        },
        "curve": curve,
        "reliability": rel,
        "claws": claw,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_doc(
        curve=curve,
        chosen=chosen,
        unmet=unmet,
        rel=rel,
        claw=claw,
        n_odors=len(odors),
        n_positive=int(positive.sum()),
    )
    print(f"wrote {OUT_JSON} and the calibration section of {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
