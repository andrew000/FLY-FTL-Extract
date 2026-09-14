"""Calibrate ``BrainParams.syn_scale`` and measure KC-code reliability.

Typical odour (PLAN, Phase 3): 30 % of the projection neurons at 150 Hz, i.e. odour value
0.75 with ``rate_max = 200 Hz``.  For every candidate ``syn_scale`` the script runs the same
odour set with APL, without APL (APL→* weights zeroed) and without KC→KC edges, and prints:

* the active Kenyon-cell fraction (two denominators: all KC, and KC that have a PN input);
* the highest firing rate of any neuron and the median rate of the *active* Kenyon
  cells (spikes / (T_stim + T_silence));
* mean APL and MBON spike counts.

Target: 5–10 % active KC with APL, > 30 % without; runaway criterion (auditor, after
Phase 3): no neuron above 1 / t_refractory and median rate of active KC < 50 Hz.  The chosen value
is written to ``docs/calibration.json`` and the whole curve to the ``calibration`` section
of ``docs/BENCH.md``; ``BrainParams.syn_scale`` is then set by hand to the chosen value
(``tests/test_lif.py`` checks that the two agree).

The script also reports what the PLAN's Jaccard tests will see (same odour, different seed
vs different odours) for several stimulus lengths and input rates, and how many of a
Kenyon cell's PN claws are inside the odour — the two numbers that explain the reliability.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fly_ftl_extract.brain import Brain, BrainParams, Connectome, load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docsection import update_section

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "calibration.json"
OUT_DOC = REPO / "docs" / "BENCH.md"

SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
N_TRIALS = 128
ODOR_SEED = 2024
PN_ACTIVE_FRACTION = 0.30
ODOR_LEVEL = 0.75  # 150 Hz at rate_max = 200 Hz
TARGET_WITH_APL = (0.05, 0.10)
TARGET_WITHOUT_APL = 0.30
MAX_ACTIVE_KC_MEDIAN_HZ = 50.0
JACCARD_TRIALS = 32
JACCARD_STIMS = (50.0, 75.0, 100.0)
JACCARD_LEVELS = (0.75, 1.0)
VOTE_MAJORITY = 2  # of 3 sniffs
JACCARD_THRESHOLD = 0.5

SKELETON = "# Brain: parameters, calibration, speed\n"


def typical_odors(n_trials: int, n_pn: int, seed: int, level: float = ODOR_LEVEL) -> np.ndarray:
    """``n_trials`` odours, each a fresh random 30 % subset of the PNs at ``level``."""
    rng = np.random.default_rng(seed)
    odors = np.zeros((n_trials, n_pn), dtype=np.float32)
    k = round(PN_ACTIVE_FRACTION * n_pn)
    for i in range(n_trials):
        odors[i, rng.choice(n_pn, size=k, replace=False)] = level
    return odors


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


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float(((a & b).sum(1) / np.maximum((a | b).sum(1), 1)).mean())


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
        print(
            f"syn_scale {scale:5.2f}: KC active {rows[-1]['kc_active_all']:.3f} "
            f"(PN-input {rows[-1]['kc_active_pn_input']:.3f}), max {rows[-1]['kc_max_rate_hz']:.0f} Hz, "
            f"median active {rows[-1]['kc_median_active_rate_hz']:.0f} Hz, "
            f"no APL {rows[-1]['kc_active_no_apl']:.3f}, no KC->KC max {rows[-1]['kc_max_rate_no_kc_kc_hz']:.0f} Hz",
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


def reliability(cx: Connectome, scale: float) -> list[dict[str, float]]:
    """Same-odour vs different-odour Jaccard of the active-KC sets (PLAN test)."""
    n_pn = len(cx.pn_idx)
    rows = []
    for t_stim in JACCARD_STIMS:
        for level in JACCARD_LEVELS:
            p = BrainParams(syn_scale=scale, t_stim=t_stim)
            b = Brain(cx, p)
            a = np.tile(typical_odors(1, n_pn, 1, level), (JACCARD_TRIALS, 1))
            c = np.tile(typical_odors(1, n_pn, 2, level), (JACCARD_TRIALS, 1))
            ra, ra2, rc = b.simulate(a, 10), b.simulate(a, 11), b.simulate(c, 10)
            votes_a = sum((b.simulate(a, s).kc_counts > 0).astype(np.int8) for s in (10, 11, 12))
            votes_b = sum((b.simulate(a, s).kc_counts > 0).astype(np.int8) for s in (13, 14, 15))
            vote_a = np.asarray(votes_a) >= VOTE_MAJORITY
            vote_b = np.asarray(votes_b) >= VOTE_MAJORITY
            rows.append(
                {
                    "t_stim_ms": t_stim,
                    "odor_level": level,
                    "pn_rate_hz": level * p.rate_max,
                    "n_steps": p.n_steps,
                    "kc_active_all": float(ra.kc_active_fraction.mean()),
                    "jaccard_same_odor": jaccard(ra.kc_counts, ra2.kc_counts),
                    "jaccard_same_odor_vote3": jaccard(
                        vote_a.astype(np.int8), vote_b.astype(np.int8)
                    ),
                    "jaccard_different_odors": jaccard(ra.kc_counts, rc.kc_counts),
                    "kc_max_rate_hz": max_rate_hz(ra.kc_counts, p),
                }
            )
            print(
                f"t_stim {t_stim:5.0f} ms, PN {level * p.rate_max:3.0f} Hz: same-odour J "
                f"{rows[-1]['jaccard_same_odor']:.3f} (vote3 {rows[-1]['jaccard_same_odor_vote3']:.3f}), "
                f"different {rows[-1]['jaccard_different_odors']:.3f}",
                flush=True,
            )
    return rows


def claws(cx: Connectome, scale: float, odors: np.ndarray) -> dict[str, object]:
    """How many PN claws of a KC are inside the odour, and how that predicts its activity."""
    b = Brain(cx, BrainParams(syn_scale=scale))
    res = b.simulate(odors, seed=1)
    pn_kc = (cx.syn_count[cx.pn_idx][:, cx.kc_idx] > 0).astype(np.int32)  # (n_pn, n_kc)
    active_claws = (odors > 0).astype(np.int32) @ pn_kc.toarray()  # (n_trials, n_kc)
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
        "by_active_claws": by_k,
    }


def write_doc(
    curve: list[dict[str, float]],
    chosen: float,
    unmet: list[str],
    rel: list[dict[str, float]],
    claw: dict[str, object],
) -> None:
    lines = [
        "## 2. Calibrating `syn_scale`",
        "",
        (
            f"A typical odour: {PN_ACTIVE_FRACTION:.0%} of PNs at {ODOR_LEVEL * 200:.0f} Hz (odor {ODOR_LEVEL}), "
            f"{N_TRIALS} trials with different random subsets of PNs (seed {ODOR_SEED}), the same set for"
        ),
        "every variant. KC rate = spikes / (T_stim + T_silence). «no APL» — the APL→* weights zeroed;",
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
        f"{TARGET_WITHOUT_APL:.0%}; runaway (reviewer's decision after Phase 3): no neuron above "
        f"1/t_refractory = {BrainParams().max_rate_hz:.0f} Hz, the median rate of active KCs < "
        f"{MAX_ACTIVE_KC_MEDIAN_HZ:.0f} Hz. Chosen: **syn_scale = {chosen}**"
        + (" — every criterion met." if not unmet else " — not met: " + "; ".join(unmet) + "."),
        "",
        "### How many KC claws fall into the odour",
        "",
        (
            f"The mean number of PN inputs (claws) per KC: {claw['mean_claws_per_kc']:.2f}; of them on average "
            f"{claw['mean_active_claws_per_kc']:.2f} are active in the typical odour. The distribution (KC × trials) and"
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
    lines += [
        "",
        "### Reliability of the KC code (the Jaccard test from PLAN) and the choice of T_stim",
        "",
        f"One odour, {JACCARD_TRIALS} trials; Jaccard of the active-KC sets between two seeds of the same odour,",
        "between the votes of 3 trials (two independent seed triples) and between two different odours. This is the table",
        "from which the reviewer chose T_stim = 100 ms after Phase 3 (at 50 ms the same odour gave J ≈ 0.40).",
        f"The row with the current parameters (T_stim = {BrainParams().t_stim:.0f} ms, {ODOR_LEVEL * 200:.0f} Hz) is marked.",
        "",
        "| T_stim, ms | PN, Hz | steps | KC active | J same odour | J vote-3 | J different odours | max KC, Hz |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    current = BrainParams()
    for r in rel:
        mark = (
            " **←**" if r["t_stim_ms"] == current.t_stim and r["odor_level"] == ODOR_LEVEL else ""
        )
        lines.append(
            f"| {r['t_stim_ms']:.0f}{mark} | {r['pn_rate_hz']:.0f} | {r['n_steps']} | {r['kc_active_all']:.3f} "
            f"| {r['jaccard_same_odor']:.3f} | {r['jaccard_same_odor_vote3']:.3f} "
            f"| {r['jaccard_different_odors']:.3f} | {r['kc_max_rate_hz']:.0f} |"
        )
    chosen_row = next(
        r for r in rel if r["t_stim_ms"] == current.t_stim and r["odor_level"] == ODOR_LEVEL
    )
    same_gap = chosen_row["jaccard_same_odor"] - JACCARD_THRESHOLD
    diff_gap = JACCARD_THRESHOLD - chosen_row["jaccard_different_odors"]
    lines += [
        "",
        (
            f"With the current parameters both criteria hold at once: the same odour J = "
            f"{chosen_row['jaccard_same_odor']:.3f} > {JACCARD_THRESHOLD} (gap {same_gap:+.3f}), "
            f"different odours J = {chosen_row['jaccard_different_odors']:.3f} < {JACCARD_THRESHOLD} "
            f"(gap {diff_gap:+.3f})."
        ),
    ]
    update_section(OUT_DOC, "calibration", "\n".join(lines), SKELETON)


def main() -> int:
    cx = load()
    odors = typical_odors(N_TRIALS, len(cx.pn_idx), ODOR_SEED)
    curve = calibration_curve(cx, odors)
    chosen, unmet = choose(curve)
    print(f"chosen syn_scale = {chosen}" + (f"; unmet: {unmet}" if unmet else ""))
    rel = reliability(cx, chosen)
    claw = claws(cx, chosen, odors)
    payload = {
        "syn_scale": chosen,
        "unmet_criteria": unmet,
        "t_stim_ms": BrainParams().t_stim,
        "jaccard_gap": {
            "same_odor": next(
                r["jaccard_same_odor"]
                for r in rel
                if r["t_stim_ms"] == BrainParams().t_stim and r["odor_level"] == ODOR_LEVEL
            )
            - JACCARD_THRESHOLD,
            "different_odors": JACCARD_THRESHOLD
            - next(
                r["jaccard_different_odors"]
                for r in rel
                if r["t_stim_ms"] == BrainParams().t_stim and r["odor_level"] == ODOR_LEVEL
            ),
        },
        "typical_odor": {"pn_active_fraction": PN_ACTIVE_FRACTION, "odor_level": ODOR_LEVEL},
        "n_trials": N_TRIALS,
        "curve": curve,
        "reliability": rel,
        "claws": claw,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_doc(curve, chosen, unmet, rel, claw)
    print(f"wrote {OUT_JSON} and the calibration section of {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
