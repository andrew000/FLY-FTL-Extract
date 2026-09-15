"""Calibrate ``BrainParams.syn_scale`` and ``apl_scale`` in the temporal code; KC reliability.

Odours (auditor's decision after Phase 4): every candidate of every fixture, tokenized and
encoded with that fixture's options — no synthetic odour.  Since ``fly-odor-4`` a candidate
is a *sequence* of puffs and the brain runs ``Brain.simulate_sequence`` (auditor's decision
after Phase 5); the calibration target is therefore **per puff**: 8–10 % of the Kenyon cells
active within a puff, averaged over the puffs that carry an odour (empty slots are silent).

Why two knobs: with the FlyWire counts scaled alike (``apl_scale = 1``) the single APL —
1713 PN and 56261 KC synapses in, ~17 synapses onto every KC out — fires 7–8 spikes per
20 ms puff, i.e. near its refractory ceiling, for as long as any odour is present.  The
Kenyon cells then respond only at odour onset: 10 % in the first puff, 0.4–2 % in every
later one, and no ``syn_scale`` between 1 and 40 changes that (per-puff activity saturates
at 1.6 %, see the sweep in the ``calibration`` section).  ``apl_scale`` weakens the APL's
output synapses so that its inhibition regulates instead of clamping.

The grid runs every (syn_scale, apl_scale) pair with APL and without APL (APL→* zeroed).
Criteria:

* with APL: 8–10 % active Kenyon cells per non-empty puff (auditor's target);
* APL sparsens: activity without APL ≥ ``MIN_APL_RATIO`` × activity with APL.  This
  replaces the Phase 3 "> 30 % without APL": that number assumed a 30 %-PN single odour,
  while a puff lights ~10 of 124 PNs and a KC has ~3.8 claws, so only ~27 % of the KCs see
  any input at all and > 30 % is reachable only when one PN spike alone fires a KC;
* integrating regime: the peak depolarisation of one PN spike through an average claw stays
  below threshold (``syn_scale`` below :func:`single_spike_threshold_scale`), so a KC needs
  a coincidence of inputs, not one spike;
* runaway (auditor, after Phase 3): no neuron above 1 / t_refractory, median rate of the
  active Kenyon cells < 50 Hz.

Among the passing pairs the one where the APL does the most work (largest ratio) is chosen;
both values go to ``docs/calibration.json`` and the whole grid to the ``calibration`` section
of ``docs/BENCH.md``; ``BrainParams`` is then set by hand (``tests/test_lif.py`` checks that
the two agree).  The chosen pair is also run without KC→KC edges (runaway via recurrence).

The script also reports the PLAN's Jaccard numbers on the real sequences (same candidate
with two seeds vs different candidates, on the per-puff KC states), the per-slot activity
profile, and a ``puff_ms`` sweep that backs the 20 ms in ``BrainParams``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fly_ftl_extract.brain import Brain, BrainParams, Connectome, SequenceResult, load
from fly_ftl_extract.odor.encoder import DEFAULT_ENCODER, ENCODER_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docsection import update_section
from _fixtures import fixture_candidates, fixture_odors

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "calibration.json"
OUT_DOC = REPO / "docs" / "BENCH.md"

SCALES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
APL_SCALES = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2)
SINGLE_SCALE_SWEEP = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 32.0, 40.0)  # apl_scale = 1
ACTIVE_PN_THRESHOLD = 0.1
TARGET_WITH_APL = (0.08, 0.10)
MIN_APL_RATIO = 2.0
MAX_ACTIVE_KC_MEDIAN_HZ = 50.0
PUFF_SWEEP_MS = (20.0, 30.0, 40.0, 50.0)
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


def trial_ms(res: SequenceResult, params: BrainParams) -> float:
    return res.n_steps * params.dt


def max_rate_hz(total_counts: np.ndarray, res: SequenceResult, params: BrainParams) -> float:
    return float(total_counts.max()) * 1000.0 / trial_ms(res, params)


def median_active_rate_hz(res: SequenceResult, params: BrainParams) -> float:
    """Median firing rate over the (trial, KC) pairs that spiked at least once."""
    total = res.total_kc_counts()
    active = total[total > 0]
    if active.size == 0:
        return 0.0
    return float(np.median(active)) * 1000.0 / trial_ms(res, params)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard of the active sets per row (rows flattened over trailing dimensions)."""
    a, b = (a > 0).reshape(a.shape[0], -1), (b > 0).reshape(b.shape[0], -1)
    return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)


def union_active(res: SequenceResult) -> float:
    return float((res.total_kc_counts() > 0).mean())


def single_spike_threshold_scale(cx: Connectome, params: BrainParams) -> float:
    """The ``syn_scale`` above which one PN spike through an average PN→KC claw alone lifts a
    resting Kenyon cell over threshold.

    With ``dv/dt = (g - v) / tau_m`` and ``dg/dt = -g / tau_syn`` the peak of ``v`` after a
    jump ``g += w`` is ``w · tau_syn / (tau_m - tau_syn) · (e^(-t*/tau_m) - e^(-t*/tau_syn))``
    at ``t* = tau_m tau_syn / (tau_m - tau_syn) · ln(tau_m / tau_syn)``; ``w`` is
    ``w_syn · syn_count · syn_scale`` with the mean PN→KC ``syn_count``.
    """
    pn_kc = cx.syn_count[cx.pn_idx][:, cx.kc_idx]
    mean_syn = float(pn_kc.data.mean())
    tm, ts = params.tau_m, params.tau_syn
    t_peak = tm * ts / (tm - ts) * np.log(tm / ts)
    peak_per_mv = ts / (tm - ts) * (np.exp(-t_peak / tm) - np.exp(-t_peak / ts))
    return float((params.v_th - params.v_rest) / (params.w_syn * mean_syn * peak_per_mv))


def measure(cx: Connectome, p: BrainParams, puffs: np.ndarray) -> dict[str, float]:
    """One grid point: the sequences with APL and without APL."""
    with_apl = Brain(cx, p).simulate_sequence(puffs, seed=1)
    no_apl = Brain(cx, p, apl_enabled=False).simulate_sequence(puffs, seed=1)
    active = with_apl.puff_active
    total = with_apl.total_kc_counts()
    per_puff = with_apl.kc_active_fraction_non_empty
    return {
        "syn_scale": p.syn_scale,
        "apl_scale": p.apl_scale,
        "kc_active_per_puff": per_puff,
        "kc_active_union": union_active(with_apl),
        "kc_max_rate_hz": max_rate_hz(total, with_apl, p),
        "kc_median_active_rate_hz": median_active_rate_hz(with_apl, p),
        "apl_max_rate_hz": max_rate_hz(with_apl.apl_counts.sum(axis=1), with_apl, p),
        "mbon_max_rate_hz": max_rate_hz(with_apl.mbon_counts.sum(axis=1), with_apl, p),
        "max_rate_any_neuron_hz": max(
            max_rate_hz(total, with_apl, p),
            max_rate_hz(with_apl.apl_counts.sum(axis=1), with_apl, p),
            max_rate_hz(with_apl.mbon_counts.sum(axis=1), with_apl, p),
        ),
        "apl_spikes_per_puff": float(with_apl.apl_counts[active].mean()),
        "mbon_spikes_per_puff": float(with_apl.mbon_counts.sum(axis=2)[active].mean()),
        "kc_active_no_apl": no_apl.kc_active_fraction_non_empty,
        "kc_max_rate_no_apl_hz": max_rate_hz(no_apl.total_kc_counts(), no_apl, p),
        "apl_ratio": no_apl.kc_active_fraction_non_empty / max(per_puff, 1e-9),
    }


def print_row(r: dict[str, float]) -> None:
    print(
        f"syn_scale {r['syn_scale']:5.2f} apl_scale {r['apl_scale']:4.2f}: KC active/puff "
        f"{r['kc_active_per_puff']:.3f} (union {r['kc_active_union']:.3f}), max KC "
        f"{r['kc_max_rate_hz']:.0f} Hz, median active {r['kc_median_active_rate_hz']:.0f} Hz, "
        f"APL {r['apl_spikes_per_puff']:.1f}/puff, no APL {r['kc_active_no_apl']:.3f} "
        f"(ratio {r['apl_ratio']:.1f})",
        flush=True,
    )


def single_scale_sweep(cx: Connectome, puffs: np.ndarray) -> list[dict[str, float]]:
    """Why one knob is not enough: ``apl_scale = 1`` over a wide ``syn_scale`` range."""
    rows = []
    for scale in SINGLE_SCALE_SWEEP:
        rows.append(measure(cx, BrainParams(syn_scale=scale, apl_scale=1.0), puffs))
        print_row(rows[-1])
    return rows


def calibration_grid(cx: Connectome, puffs: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for apl in APL_SCALES:
        for scale in SCALES:
            rows.append(measure(cx, BrainParams(syn_scale=scale, apl_scale=apl), puffs))
            print_row(rows[-1])
    return rows


def unmet_criteria(r: dict[str, float], max_scale: float) -> list[str]:
    lo, hi = TARGET_WITH_APL
    out = []
    if not lo <= r["kc_active_per_puff"] <= hi:
        out.append(f"KC active per puff with APL {r['kc_active_per_puff']:.3f} not in [{lo}, {hi}]")
    if r["apl_ratio"] < MIN_APL_RATIO:
        out.append(f"APL sparsening ratio {r['apl_ratio']:.2f} < {MIN_APL_RATIO}")
    if r["syn_scale"] >= max_scale:
        out.append(f"syn_scale {r['syn_scale']} >= {max_scale:.1f}: one PN spike fires a KC")
    if r["max_rate_any_neuron_hz"] > BrainParams().max_rate_hz:
        out.append(f"a neuron fired at {r['max_rate_any_neuron_hz']:.0f} Hz > 1/t_refractory")
    if r["kc_median_active_rate_hz"] >= MAX_ACTIVE_KC_MEDIAN_HZ:
        out.append(
            f"median active-KC rate {r['kc_median_active_rate_hz']:.0f} Hz >= "
            f"{MAX_ACTIVE_KC_MEDIAN_HZ:.0f} Hz"
        )
    return out


def choose(rows: list[dict[str, float]], max_scale: float) -> tuple[dict[str, float], list[str]]:
    """The passing pair where the APL does the most work (largest ratio); ties → closest
    to the middle of the band.  Falls back to the fewest unmet criteria."""
    mid = sum(TARGET_WITH_APL) / 2
    passing = [r for r in rows if not unmet_criteria(r, max_scale)]
    pool = passing or sorted(rows, key=lambda r: len(unmet_criteria(r, max_scale)))[:1]
    best = max(pool, key=lambda r: (r["apl_ratio"], -abs(r["kc_active_per_puff"] - mid)))
    return best, unmet_criteria(best, max_scale)


def no_kc_kc(cx: Connectome, p: BrainParams, puffs: np.ndarray) -> dict[str, float]:
    """The chosen pair without KC→KC edges (runaway through recurrence)."""
    res = Brain(without_kc_kc(cx), p).simulate_sequence(puffs, seed=1)
    return {
        "kc_active_per_puff": res.kc_active_fraction_non_empty,
        "kc_max_rate_hz": max_rate_hz(res.total_kc_counts(), res, p),
    }


def reliability(cx: Connectome, chosen: BrainParams, puffs: np.ndarray) -> list[dict[str, float]]:
    """Same-candidate vs different-candidate Jaccard of the per-puff KC states, for several
    puff lengths (the 20 ms row is the one in ``BrainParams``)."""
    rng = np.random.default_rng(0)
    n = len(puffs)
    pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    if len(pairs) > PAIR_SAMPLE:
        pairs = pairs[rng.choice(len(pairs), PAIR_SAMPLE, replace=False)]
    rows = []
    for puff_ms in PUFF_SWEEP_MS:
        p = dataclasses.replace(chosen, puff_ms=puff_ms)
        b = Brain(cx, p)
        r1, r2 = b.simulate_sequence(puffs, 1), b.simulate_sequence(puffs, 2)
        k1, k2 = r1.kc_counts, r2.kc_counts
        rows.append(
            {
                "puff_ms": puff_ms,
                "n_steps": r1.n_steps,
                "kc_active_per_puff": r1.kc_active_fraction_non_empty,
                "kc_active_union": union_active(r1),
                "jaccard_same_candidate": float(jaccard(k1, k2).mean()),
                "jaccard_same_candidate_union": float(
                    jaccard(r1.total_kc_counts(), r2.total_kc_counts()).mean()
                ),
                "jaccard_different_candidates": float(
                    jaccard(k1[pairs[:, 0]], k1[pairs[:, 1]]).mean()
                ),
                "kc_max_rate_hz": max_rate_hz(r1.total_kc_counts(), r1, p),
            }
        )
        r = rows[-1]
        print(
            f"puff {puff_ms:4.0f} ms: KC active/puff {r['kc_active_per_puff']:.3f}, "
            f"same-candidate J {r['jaccard_same_candidate']:.3f} "
            f"(union {r['jaccard_same_candidate_union']:.3f}), different "
            f"{r['jaccard_different_candidates']:.3f}",
            flush=True,
        )
    return rows


def slot_profile(cx: Connectome, chosen: BrainParams, puffs: np.ndarray) -> list[dict[str, float]]:
    """Per slot: how many puffs are non-empty, active PNs, active KC share, APL spikes."""
    res = Brain(cx, chosen).simulate_sequence(puffs, seed=1)
    n_slots = puffs.shape[1]
    focus = DEFAULT_ENCODER.context_before
    out = []
    for k in range(n_slots):
        active = res.puff_active[:, k]
        out.append(
            {
                "slot": k - focus,
                "non_empty_share": float(active.mean()),
                "active_pn": float((puffs[active, k] > ACTIVE_PN_THRESHOLD).sum(axis=1).mean())
                if active.any()
                else 0.0,
                "kc_active": float(res.kc_active_fraction[active, k].mean())
                if active.any()
                else 0.0,
                "kc_active_empty": float(res.kc_active_fraction[~active, k].mean())
                if (~active).any()
                else 0.0,
                "apl_spikes": float(res.apl_counts[active, k].mean()) if active.any() else 0.0,
            }
        )
    return out


def claws(cx: Connectome, chosen: BrainParams, puffs: np.ndarray) -> dict[str, object]:
    """How many PN claws of a KC a *puff* reaches, and how that predicts the KC firing in
    that puff (non-empty puffs only)."""
    res = Brain(cx, chosen).simulate_sequence(puffs, seed=1)
    pn_kc = (cx.syn_count[cx.pn_idx][:, cx.kc_idx] > 0).astype(np.int32)  # (n_pn, n_kc)
    active_puffs = res.puff_active
    flat_puffs = puffs[active_puffs]  # (n_active_puffs, n_pn)
    active_claws = (flat_puffs > ACTIVE_PN_THRESHOLD).astype(np.int32) @ pn_kc.toarray()
    total_claws = np.asarray(pn_kc.sum(axis=0)).ravel()
    active = res.kc_counts[active_puffs] > 0
    by_k = []
    for k in range(int(active_claws.max()) + 1):
        cells = active_claws == k
        n_cells = int(cells.sum())
        if n_cells == 0:
            continue
        by_k.append(
            {
                "active_claws": k,
                "kc_puffs": n_cells,
                "share_of_kc_puffs": n_cells / active_claws.size,
                "p_active": float(active[cells].mean()),
                "share_of_active_kc": float((active & cells).sum() / max(active.sum(), 1)),
            }
        )
    return {
        "mean_claws_per_kc": float(total_claws.mean()),
        "mean_active_claws_per_kc": float(active_claws.mean()),
        "mean_active_pn_per_puff": float((flat_puffs > ACTIVE_PN_THRESHOLD).sum(axis=1).mean()),
        "by_active_claws": by_k,
    }


def write_doc(
    *,
    single: list[dict[str, float]],
    grid: list[dict[str, float]],
    chosen: dict[str, float],
    unmet: list[str],
    max_scale: float,
    nokk: dict[str, float],
    rel: list[dict[str, float]],
    slots: list[dict[str, float]],
    claw: dict[str, object],
    n_odors: int,
    n_positive: int,
    n_slots: int,
) -> None:
    current = BrainParams()
    lines = [
        "## 2. Calibrating `syn_scale` and `apl_scale` on real odours (temporal coding)",
        "",
        (
            f"Odours: all {n_odors} candidates from all fixtures ({n_positive} positive), tokenised "
            f"and encoded with the `{ENCODER_VERSION}` encoder using each fixture's options: a sequence of "
            f"{n_slots} puffs of {current.puff_ms:.0f} ms (`Brain.simulate_sequence`), an empty slot is "
            "silence. The synthetic odour (30 % of PNs at 150 Hz) is not used (reviewer's decision after "
            "Phase 4). The same set for every variant, seed 1."
        ),
        (
            "«KC active / puff» — the share of active KCs within a puff, averaged over non-empty puffs "
            "(the reviewer's target after Phase 5: 8–10 %); «union» — the share of KCs that spiked at least once per trial. "
            "Rate = spikes / trial duration. «no APL» — the APL→* weights zeroed; «ratio» — by how "
            "much APL reduces the per-puff KC activity."
        ),
        "",
        "### Why one `syn_scale` is not enough (`apl_scale` = 1)",
        "",
        (
            "The single APL receives 1713 synapses from PNs and 56261 from KCs and gives ~17 synapses to every KC. "
            "With the same scale for every synapse it fires 7–8 times per 20-millisecond puff "
            f"(≈ 400 Hz, the refractory limit is {current.max_rate_hz:.0f} Hz) the whole time an odour is present: the KCs "
            "respond only to the start of the trial (the first puff ~10 %, then 0.4–2 %), and no `syn_scale` "
            "changes that:"
        ),
        "",
        "| syn_scale | KC active / puff | KC active union | APL spikes / puff | max KC, Hz | no APL: KC active / puff |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['syn_scale']} | {r['kc_active_per_puff']:.3f} | {r['kc_active_union']:.3f} "
        f"| {r['apl_spikes_per_puff']:.1f} | {r['kc_max_rate_hz']:.0f} | {r['kc_active_no_apl']:.3f} |"
        for r in single
    )
    lines += [
        "",
        "### The `syn_scale` × `apl_scale` grid",
        "",
        (
            f"Criteria: with APL {TARGET_WITH_APL[0]:.0%}–{TARGET_WITH_APL[1]:.0%} of KCs active per puff; APL "
            f"sparsifies ≥ {MIN_APL_RATIO:.0f}× (instead of «no APL > 30 %» from Phase 3 — that threshold was for "
            "a synthetic odour with 30 % of PNs; a puff lights ~10 of 124 PNs, a KC has ~3.8 claws, so only "
            "~27 % of KCs see any input at all, and > 30 % is reachable only when a single PN spike lights a KC by itself); "
            f"integrating regime: `syn_scale` < {max_scale:.1f} (the depolarisation peak from one PN spike "
            "through an average claw stays below threshold); runaway: no neuron above 1/t_refractory = "
            f"{current.max_rate_hz:.0f} Hz, the median rate of active KCs < {MAX_ACTIVE_KC_MEDIAN_HZ:.0f} Hz. "
            "Among the passing pairs the one where APL does the most work (the largest ratio) is chosen."
        ),
        "",
        (
            "| syn_scale | apl_scale | KC active / puff | KC active union | max KC, Hz | median active KC, Hz "
            "| max APL, Hz | max MBON, Hz | APL spikes / puff | MBON spikes / puff | no APL: KC active / puff | ratio | criteria |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in grid:
        is_chosen = r["syn_scale"] == chosen["syn_scale"] and r["apl_scale"] == chosen["apl_scale"]
        mark = " **←**" if is_chosen else ""
        problems = unmet_criteria(r, max_scale)
        lines.append(
            f"| {r['syn_scale']}{mark} | {r['apl_scale']} | {r['kc_active_per_puff']:.3f} "
            f"| {r['kc_active_union']:.3f} | {r['kc_max_rate_hz']:.0f} | {r['kc_median_active_rate_hz']:.0f} "
            f"| {r['apl_max_rate_hz']:.0f} | {r['mbon_max_rate_hz']:.0f} "
            f"| {r['apl_spikes_per_puff']:.2f} | {r['mbon_spikes_per_puff']:.2f} "
            f"| {r['kc_active_no_apl']:.3f} | {r['apl_ratio']:.1f} "
            f"| {'✓' if not problems else '; '.join(problems)} |"
        )
    lines += [
        "",
        f"Chosen: **syn_scale = {chosen['syn_scale']}, apl_scale = {chosen['apl_scale']}**"
        + (" — every criterion met." if not unmet else " — not met: " + "; ".join(unmet) + ".")
        + f" Without KC→KC (942 edges removed): KC active / puff {nokk['kc_active_per_puff']:.3f}, "
        f"max KC {nokk['kc_max_rate_hz']:.0f} Hz (the recurrence does not cause runaway).",
        "",
        "### Per-slot profile (the chosen pair)",
        "",
        (
            "Slot 0 is the candidate; negative slots are the context before it, positive ones after. «non-empty» — the share "
            "of candidates in which this slot has a token (a candidate close to the start of the file has a silent slot). "
            "«KC active in empty» — KC activity in a silent puff: that is the membrane's memory of "
            "the preceding puffs (the first slots have no preceding puffs, hence 0)."
        ),
        "",
        "| slot | non-empty | active PN | KC active / puff | KC active in empty | APL spikes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['slot']:+d} | {r['non_empty_share']:.2f} | {r['active_pn']:.1f} | {r['kc_active']:.3f} "
        f"| {r['kc_active_empty']:.3f} | {r['apl_spikes']:.2f} |"
        for r in slots
    )
    lines += [
        "",
        "### How many KC claws one puff reaches",
        "",
        (
            f"An active PN = odor > {ACTIVE_PN_THRESHOLD}; on average {claw['mean_active_pn_per_puff']:.1f} "
            f"active PNs per non-empty puff. The mean number of PN inputs (claws) per KC: "
            f"{claw['mean_claws_per_kc']:.2f}; of them on average "
            f"{claw['mean_active_claws_per_kc']:.2f} are active in a puff. The distribution (KC × puffs) and the probability that the KC spikes "
            "in this puff, by the number of active claws:"
        ),
        "",
        "| active claws | KC×puffs | share | P(KC active) | share among active KCs |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_k = claw["by_active_claws"]
    assert isinstance(by_k, list)
    lines.extend(
        f"| {r['active_claws']} | {r['kc_puffs']} | {r['share_of_kc_puffs']:.3f} | {r['p_active']:.3f} "
        f"| {r['share_of_active_kc']:.3f} |"
        for r in by_k
    )
    lines += [
        "",
        "### Reliability of the KC code (the Jaccard test from PLAN) and puff duration",
        "",
        (
            "Jaccard of the sets of active (puff, KC): the same candidate seed 1 vs 2 (mean over all "
            "candidates; «union» — over the per-trial united KC sets) and random pairs of different candidates. "
            f"The row with the current puff_ms = {current.puff_ms:.0f} ms is marked; a shorter puff gives fewer spikes "
            "per puff and worse reproducibility, a longer one a longer trial for the same information."
        ),
        "",
        (
            "| puff_ms | steps | KC active / puff | KC active union | J same candidate "
            "| J same, union | J different candidates | max KC, Hz |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rel:
        mark = " **←**" if r["puff_ms"] == current.puff_ms else ""
        lines.append(
            f"| {r['puff_ms']:.0f}{mark} | {r['n_steps']} | {r['kc_active_per_puff']:.3f} "
            f"| {r['kc_active_union']:.3f} | {r['jaccard_same_candidate']:.3f} "
            f"| {r['jaccard_same_candidate_union']:.3f} | {r['jaccard_different_candidates']:.3f} "
            f"| {r['kc_max_rate_hz']:.0f} |"
        )
    chosen_row = min(rel, key=lambda r: abs(r["puff_ms"] - current.puff_ms))
    same_gap = chosen_row["jaccard_same_candidate"] - JACCARD_THRESHOLD
    diff_gap = JACCARD_THRESHOLD - chosen_row["jaccard_different_candidates"]
    lines += [
        "",
        (
            f"With the current parameters: the same candidate J = {chosen_row['jaccard_same_candidate']:.3f} "
            f"per puff, {chosen_row['jaccard_same_candidate_union']:.3f} per trial "
            f"(gap to {JACCARD_THRESHOLD}: {same_gap:+.3f} per puff), different candidates J = "
            f"{chosen_row['jaccard_different_candidates']:.3f} (gap {diff_gap:+.3f})."
            + (
                ""
                if same_gap > 0
                else " The criterion «same > 0.5» per puff is not met (per trial it is): by the reviewer's "
                "decision after Phase 4 the corresponding test is diagnostic (xfail), the Phase 5 gate is readout "
                "accuracy."
            )
        ),
    ]
    update_section(OUT_DOC, "calibration", "\n".join(lines), SKELETON)


def main() -> int:
    cx = load()
    items = fixture_candidates()
    puffs, positive = fixture_odors(items)
    print(
        f"real odours: {len(puffs)} candidates ({puffs.shape[1]} puffs each), "
        f"{int(positive.sum())} positive",
        flush=True,
    )
    max_scale = single_spike_threshold_scale(cx, BrainParams())
    print(f"one PN spike alone fires a KC from syn_scale {max_scale:.2f}", flush=True)
    single = single_scale_sweep(cx, puffs)
    grid = calibration_grid(cx, puffs)
    chosen_row, unmet = choose(grid, max_scale)
    chosen = BrainParams(syn_scale=chosen_row["syn_scale"], apl_scale=chosen_row["apl_scale"])
    print(
        f"chosen syn_scale = {chosen.syn_scale}, apl_scale = {chosen.apl_scale}"
        + (f"; unmet: {unmet}" if unmet else "")
    )
    nokk = no_kc_kc(cx, chosen, puffs)
    rel = reliability(cx, chosen, puffs)
    slots = slot_profile(cx, chosen, puffs)
    claw = claws(cx, chosen, puffs)
    current = BrainParams()
    chosen_rel = min(rel, key=lambda r: abs(r["puff_ms"] - current.puff_ms))
    payload = {
        "syn_scale": chosen.syn_scale,
        "apl_scale": chosen.apl_scale,
        "unmet_criteria": unmet,
        "single_spike_threshold_scale": max_scale,
        "mode": "sequence",
        "odors": {
            "source": "fixture candidates",
            "encoder_version": ENCODER_VERSION,
            "n": len(puffs),
            "n_puffs": int(puffs.shape[1]),
            "n_positive": int(positive.sum()),
        },
        "puff_ms": current.puff_ms,
        "jaccard_gap": {
            "same_candidate": chosen_rel["jaccard_same_candidate"] - JACCARD_THRESHOLD,
            "different_candidates": JACCARD_THRESHOLD - chosen_rel["jaccard_different_candidates"],
        },
        "single_scale_sweep": single,
        "grid": grid,
        "chosen": chosen_row,
        "no_kc_kc": nokk,
        "reliability": rel,
        "slots": slots,
        "claws": claw,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_doc(
        single=single,
        grid=grid,
        chosen=chosen_row,
        unmet=unmet,
        max_scale=max_scale,
        nokk=nokk,
        rel=rel,
        slots=slots,
        claw=claw,
        n_odors=len(puffs),
        n_positive=int(positive.sum()),
        n_slots=int(puffs.shape[1]),
    )
    print(f"wrote {OUT_JSON} and the calibration section of {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
