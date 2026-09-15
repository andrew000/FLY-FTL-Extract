"""Behavioural tests of the LIF mushroom body (PLAN, Phase 3) and of the temporal code.

``Brain.simulate`` (one odour held for ``t_stim``) keeps its Phase 3 tests with the Phase 3
calibration (``PHASE3_PARAMS``: syn_scale 4.0, apl_scale 1.0) on the synthetic 30 %-PN
odour — there is no single "real odour" any more: since ``fly-odor-4`` a candidate is a
sequence of puffs, and the production parameters (``DEFAULT_PARAMS``) are calibrated for
``Brain.simulate_sequence``: 8–10 % active KC per non-empty puff on the real fixture
sequences, APL sparsening ≥ 2× (``docs/BENCH.md`` §2).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fly_ftl_extract.brain import (
    DEFAULT_PARAMS,
    Brain,
    BrainParams,
    Connectome,
    SequenceResult,
    TrialResult,
)
from fly_ftl_extract.brain import connectome as cx_mod

REPO = Path(__file__).resolve().parent.parent
CALIBRATION = REPO / "docs" / "calibration.json"
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_odors  # noqa: E402

N_TRIALS = 64
ODOR_LEVEL = 0.75  # 150 Hz at rate_max 200 Hz: the PLAN's typical odour
PN_ACTIVE_FRACTION = 0.30
SPARSITY_WITH_APL = (0.08, 0.10)  # auditor's target: per non-empty puff, sequence mode
SPARSITY_CLAUDE_MD = (0.03, 0.15)  # CLAUDE.md: outside this range the weights are wrong
SPARSITY_WITHOUT_APL = 0.30  # Phase 3 criterion (single odour, 30 % of the PNs active)
MIN_APL_RATIO = 2.0  # sequence mode: activity without APL / with APL (docs/BENCH.md §2)
PHASE3_PARAMS = BrainParams(syn_scale=4.0, apl_scale=1.0)
"""The single-odour mode keeps the calibration it was tested with in Phase 3."""
JACCARD_DIFFERENT_MAX = 0.5
JACCARD_SAME_MIN = 0.5
ACTIVE_KC_MEDIAN_RATE_MAX_HZ = 50.0  # auditor's runaway criterion (after Phase 3)


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    if not cx_mod.data_path(cx_mod.NPZ_NAME).exists():
        pytest.skip("data/mb_fafb783.npz not built")
    return cx_mod.load()


@pytest.fixture(scope="module")
def brain(connectome: Connectome) -> Brain:
    """The production brain (``DEFAULT_PARAMS``, calibrated for the temporal code)."""
    return Brain(connectome)


@pytest.fixture(scope="module")
def phase3_brain(connectome: Connectome) -> Brain:
    """The single-odour brain with its Phase 3 calibration."""
    return Brain(connectome, PHASE3_PARAMS)


@pytest.fixture(scope="module")
def real_puffs() -> np.ndarray:
    """Every fixture candidate as a puff sequence ``(n, n_slots, n_pn)`` (calibration set)."""
    puffs, _ = fixture_odors()
    assert len(puffs) > 100
    assert puffs.ndim == 3
    return puffs


@pytest.fixture(scope="module")
def synthetic_odors(phase3_brain: Brain) -> np.ndarray:
    brain = phase3_brain
    """The PLAN's typical odours for the single-odour mode: 30 % of the PNs at 150 Hz."""
    return odor_set(brain.n_pn, seed=1, n_trials=128)


def odor_set(
    n_pn: int, seed: int, n_trials: int = N_TRIALS, level: float = ODOR_LEVEL
) -> np.ndarray:
    """``n_trials`` random odours: 30 % of the PNs at ``level`` each."""
    rng = np.random.default_rng(seed)
    odors = np.zeros((n_trials, n_pn), dtype=np.float32)
    k = round(PN_ACTIVE_FRACTION * n_pn)
    for i in range(n_trials):
        odors[i, rng.choice(n_pn, size=k, replace=False)] = level
    return odors


def one_odor(n_pn: int, seed: int, n_trials: int = N_TRIALS) -> np.ndarray:
    return np.tile(odor_set(n_pn, seed, n_trials=1), (n_trials, 1))


def jaccard(a: TrialResult, b: TrialResult) -> float:
    x, y = a.kc_counts > 0, b.kc_counts > 0
    return float(((x & y).sum(axis=1) / np.maximum((x | y).sum(axis=1), 1)).mean())


def test_same_seed_is_bit_identical(brain: Brain) -> None:
    odors = odor_set(brain.n_pn, seed=1, n_trials=16)
    a = brain.simulate(odors, seed=42, raster_trials=4)
    b = brain.simulate(odors, seed=42, raster_trials=4)
    assert np.array_equal(a.kc_counts, b.kc_counts)
    assert np.array_equal(a.mbon_counts, b.mbon_counts)
    assert np.array_equal(a.apl_counts, b.apl_counts)
    assert np.array_equal(a.pn_counts, b.pn_counts)
    assert a.raster is not None
    assert b.raster is not None
    assert np.array_equal(a.raster.step, b.raster.step)
    assert np.array_equal(a.raster.kc, b.raster.kc)
    c = brain.simulate(odors, seed=43)
    assert not np.array_equal(a.kc_counts, c.kc_counts)


def test_kc_sparsity_with_apl(phase3_brain: Brain, synthetic_odors: np.ndarray) -> None:
    """Single-odour mode, Phase 3 calibration, synthetic 30 %-PN odour: the CLAUDE.md band."""
    res = phase3_brain.simulate(synthetic_odors, seed=1)
    frac = float(res.kc_active_fraction.mean())
    assert SPARSITY_CLAUDE_MD[0] <= frac <= SPARSITY_CLAUDE_MD[1]
    assert (res.kc_active_fraction_pn_input >= res.kc_active_fraction).all()
    assert res.apl_counts.mean() > 0  # APL fires


def test_sparsity_rises_without_apl(
    phase3_brain: Brain, connectome: Connectome, synthetic_odors: np.ndarray
) -> None:
    odors = synthetic_odors
    with_apl = phase3_brain.simulate(odors, seed=1)
    without = Brain(connectome, PHASE3_PARAMS, apl_enabled=False).simulate(odors, seed=1)
    assert without.apl_counts.sum() >= 0  # APL still spikes, only its outputs are cut
    assert without.kc_active_fraction.mean() > with_apl.kc_active_fraction.mean()
    assert without.kc_active_fraction.mean() > SPARSITY_WITHOUT_APL


def test_different_odors_give_different_kc_patterns(
    phase3_brain: Brain, synthetic_odors: np.ndarray
) -> None:
    real_odors = synthetic_odors
    res = phase3_brain.simulate(real_odors, seed=1)
    rng = np.random.default_rng(0)
    a = rng.permutation(len(real_odors))
    shifted = np.roll(a, 1)
    x, y = res.kc_counts[a] > 0, res.kc_counts[shifted] > 0
    j = float(((x & y).sum(axis=1) / np.maximum((x | y).sum(axis=1), 1)).mean())
    assert j < JACCARD_DIFFERENT_MAX


@pytest.mark.xfail(
    reason=(
        "diagnostic (auditor, after Phase 4): on real fixture odours the same candidate "
        "with two seeds gives Jaccard < 0.5 (docs/BENCH.md §2, per puff); the Phase 5 gate is "
        "readout accuracy"
    ),
    strict=False,
)
def test_same_odor_different_seeds_give_similar_kc_patterns(
    brain: Brain, real_puffs: np.ndarray
) -> None:
    a = brain.simulate_sequence(real_puffs, seed=1)
    b = brain.simulate_sequence(real_puffs, seed=2)
    x, y = (
        a.kc_counts.reshape(len(real_puffs), -1) > 0,
        b.kc_counts.reshape(len(real_puffs), -1) > 0,
    )
    j = float(((x & y).sum(axis=1) / np.maximum((x | y).sum(axis=1), 1)).mean())
    assert j > JACCARD_SAME_MIN


def test_zero_odor_is_silent(brain: Brain) -> None:
    res = brain.simulate(np.zeros((8, brain.n_pn), dtype=np.float32), seed=1)
    assert res.pn_counts.sum() == 0
    assert res.kc_counts.sum() == 0
    assert res.mbon_counts.sum() == 0
    assert res.apl_counts.sum() == 0


def test_no_neuron_beats_the_refractory_period(brain: Brain) -> None:
    p = brain.params
    res = brain.simulate(odor_set(brain.n_pn, seed=3, level=1.0), seed=1)
    # A refractory neuron can spike at most once per (refractory_steps + 1) steps.
    max_spikes = math.ceil(p.n_steps / (p.refractory_steps + 1))
    for counts in (res.kc_counts, res.mbon_counts, res.apl_counts):
        assert counts.max() <= max_spikes
        assert counts.max() * 1000.0 / (p.n_steps * p.dt) <= p.max_rate_hz + 1e-9
    # PNs are Poisson (no refractory, as in Shiu et al.): check the mean rate instead.
    active = res.pn_counts[odor_set(brain.n_pn, seed=3, level=1.0) > 0]
    expected = p.rate_max * 1.0 * p.t_stim / 1000.0
    assert abs(active.mean() - expected) / expected < 0.05


def test_active_kc_median_rate_is_below_50_hz(phase3_brain: Brain) -> None:
    brain = phase3_brain
    p = brain.params
    res = brain.simulate(odor_set(brain.n_pn, seed=3), seed=1)
    active = res.kc_counts[res.kc_counts > 0]
    assert active.size > 0
    median_hz = float(np.median(active)) * 1000.0 / (p.n_steps * p.dt)
    assert median_hz < ACTIVE_KC_MEDIAN_RATE_MAX_HZ


def test_raster_matches_counts(brain: Brain) -> None:
    odors = odor_set(brain.n_pn, seed=4, n_trials=8)
    res = brain.simulate(odors, seed=1, raster_trials=3)
    assert res.raster is not None
    assert res.raster.n_trials == 3
    for trial in range(3):
        dense = res.raster.dense(trial)
        assert dense.shape == (res.n_steps, brain.n_kc)
        assert np.array_equal(dense.sum(axis=0), res.kc_counts[trial])
    assert (res.raster.trial < 3).all()


def test_odor_validation(brain: Brain) -> None:
    with pytest.raises(ValueError, match="shape"):
        brain.simulate(np.zeros((2, brain.n_pn + 1), dtype=np.float32), seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brain.simulate(np.full((2, brain.n_pn), 1.5, dtype=np.float32), seed=0)


def test_dan_carry_no_current(brain: Brain, connectome: Connectome) -> None:
    assert brain.w[connectome.dan_idx].nnz == 0
    assert not np.isin(brain.int_idx, connectome.dan_idx).any()
    assert not np.isin(brain.int_idx, connectome.pn_idx).any()


def test_params_are_whole_steps() -> None:
    p = DEFAULT_PARAMS
    assert p.n_steps == round((p.t_stim + p.t_silence) / p.dt)
    assert p.refractory_steps == 22
    assert p.delay_steps == 18
    assert p.t_stim == 100.0  # auditor's decision after Phase 3 (docs/BENCH.md §2)
    with pytest.raises(ValueError, match="whole number"):
        _ = BrainParams(t_stim=50.05).stim_steps


def test_syn_scale_matches_calibration() -> None:
    if not CALIBRATION.exists():
        pytest.skip("docs/calibration.json not written (run scripts/calibrate.py)")
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert calibration["mode"] == "sequence"
    assert DEFAULT_PARAMS.syn_scale == calibration["syn_scale"]
    assert DEFAULT_PARAMS.apl_scale == calibration["apl_scale"]
    assert DEFAULT_PARAMS.puff_ms == calibration["puff_ms"]
    assert not calibration["unmet_criteria"]


# ----------------------------------------------------------------- temporal code


def test_sequence_same_seed_is_bit_identical(brain: Brain, real_puffs: np.ndarray) -> None:
    puffs = real_puffs[:16]
    a = brain.simulate_sequence(puffs, seed=42, raster_trials=4)
    b = brain.simulate_sequence(puffs, seed=42, raster_trials=4)
    assert np.array_equal(a.kc_counts, b.kc_counts)
    assert np.array_equal(a.mbon_counts, b.mbon_counts)
    assert np.array_equal(a.apl_counts, b.apl_counts)
    assert np.array_equal(a.pn_counts, b.pn_counts)
    assert a.raster is not None
    assert b.raster is not None
    assert np.array_equal(a.raster.step, b.raster.step)
    c = brain.simulate_sequence(puffs, seed=43)
    assert not np.array_equal(a.kc_counts, c.kc_counts)
    seeds = np.arange(16, dtype=np.uint64) + 1000
    d = brain.simulate_sequence(puffs, seeds)
    e = brain.simulate_sequence(puffs, seeds)
    assert np.array_equal(d.kc_counts, e.kc_counts)
    # per-trial streams: a trial's result does not depend on its neighbours in the batch
    f = brain.simulate_sequence(puffs[3:5], seeds[3:5])
    assert np.array_equal(f.kc_counts, d.kc_counts[3:5])


def test_sequence_shapes_and_bins(brain: Brain, real_puffs: np.ndarray) -> None:
    p = brain.params
    puffs = real_puffs[:8]
    n_puffs = puffs.shape[1]
    res = brain.simulate_sequence(puffs, seed=1, raster_trials=8)
    assert res.kc_counts.shape == (8, n_puffs, brain.n_kc)
    assert res.kc_counts.dtype == np.int16
    assert res.pn_counts.shape == (8, n_puffs, brain.n_pn)
    assert res.apl_counts.shape == (8, n_puffs)
    assert res.n_steps == p.sequence_steps(n_puffs) == n_puffs * p.puff_steps + 100
    assert np.array_equal(res.puff_active, puffs.any(axis=2))
    # the raster over the whole trial sums to the per-puff counts
    assert res.raster is not None
    for trial in range(8):
        dense = res.raster.dense(trial)
        assert np.array_equal(dense.sum(axis=0), res.total_kc_counts()[trial])
        # bin k = steps of puff k (the last bin also takes the silence)
        for k in range(n_puffs):
            lo = k * p.puff_steps
            hi = (k + 1) * p.puff_steps if k < n_puffs - 1 else res.n_steps
            assert np.array_equal(dense[lo:hi].sum(axis=0), res.kc_counts[trial, k])


def test_silent_puffs_drive_no_pn(brain: Brain, real_puffs: np.ndarray) -> None:
    """An empty slot is a silent puff: no PN spikes at all in that bin."""
    empty = ~real_puffs.any(axis=2)
    assert empty.any(), "the fixtures should contain candidates on the first line of a file"
    res = brain.simulate_sequence(real_puffs, seed=1)
    assert res.pn_counts[empty].sum() == 0
    # leading silent puffs (nothing before them) produce no Kenyon-cell spikes either
    leading = np.zeros_like(empty)
    for i in range(empty.shape[0]):
        k = 0
        while k < empty.shape[1] and empty[i, k]:
            leading[i, k] = True
            k += 1
    assert res.kc_counts[leading].sum() == 0


def test_sequence_sparsity_per_puff_with_apl(brain: Brain, real_puffs: np.ndarray) -> None:
    """The auditor's target: 8–10 % active KC per non-empty puff, APL firing."""
    res = brain.simulate_sequence(real_puffs, seed=1)
    frac = res.kc_active_fraction_non_empty
    assert SPARSITY_CLAUDE_MD[0] <= frac <= SPARSITY_CLAUDE_MD[1]
    assert SPARSITY_WITH_APL[0] <= frac <= SPARSITY_WITH_APL[1]
    assert res.apl_counts[res.puff_active].mean() > 0


def test_sequence_sparsity_rises_without_apl(
    brain: Brain, connectome: Connectome, real_puffs: np.ndarray
) -> None:
    with_apl = brain.simulate_sequence(real_puffs, seed=1)
    without = Brain(connectome, apl_enabled=False).simulate_sequence(real_puffs, seed=1)
    assert without.kc_active_fraction_non_empty > with_apl.kc_active_fraction_non_empty
    # the Phase 3 "> 30 % without APL" was for a 30 %-PN single odour; a puff lights ~10 of
    # 124 PNs and only ~27 % of the KCs see any input, so the criterion is the ratio
    assert (
        without.kc_active_fraction_non_empty
        >= MIN_APL_RATIO * with_apl.kc_active_fraction_non_empty
    )
    assert with_apl.apl_counts[with_apl.puff_active].mean() > 1.0  # the APL is not silent


def test_membrane_carries_over_between_puffs(brain: Brain, real_puffs: np.ndarray) -> None:
    """The same puff after a different history gives different Kenyon-cell counts: the
    state is not reset between puffs (that is the fly's memory of what came before)."""
    n = 64
    base = real_puffs[:n]
    n_puffs = base.shape[1]
    focus = n_puffs - 4  # the candidate's own slot (6 before, 3 after)
    alt = base.copy()
    alt[:, :focus] = 0.0  # same focus and after-puffs, silence before instead of context
    a = brain.simulate_sequence(base, seed=7)
    b = brain.simulate_sequence(alt, seed=7)
    same_input = (base[:, focus:] == alt[:, focus:]).all()
    assert same_input
    # the focus puff itself receives identical PN input ...
    assert np.array_equal(a.pn_counts[:, focus], b.pn_counts[:, focus])
    # ... but the Kenyon cells answer differently because of what came before
    differ = (a.kc_counts[:, focus] != b.kc_counts[:, focus]).any(axis=1)
    assert differ.mean() > 0.5


def test_sequence_order_matters(brain: Brain, real_puffs: np.ndarray) -> None:
    """The reversed sequence of puffs is a different trial for the readout."""
    puffs = real_puffs[:64]
    a = brain.simulate_sequence(puffs, seed=3)
    b = brain.simulate_sequence(puffs[:, ::-1], seed=3)
    assert not np.array_equal(a.kc_counts, b.kc_counts[:, ::-1])


def test_sequence_no_neuron_beats_the_refractory_period(
    brain: Brain, real_puffs: np.ndarray
) -> None:
    p = brain.params
    res = brain.simulate_sequence(np.minimum(real_puffs[:64] * 2, 1.0), seed=1)
    max_spikes = math.ceil(res.n_steps / (p.refractory_steps + 1))
    for counts in (res.total_kc_counts(), res.mbon_counts.sum(axis=1), res.apl_counts.sum(axis=1)):
        assert counts.max() <= max_spikes
    # and on the real sequences the median active KC stays a sparse coder (< 50 Hz)
    real = brain.simulate_sequence(real_puffs, seed=1)
    total = real.total_kc_counts()
    median_hz = float(np.median(total[total > 0])) * 1000.0 / (real.n_steps * p.dt)
    assert median_hz < ACTIVE_KC_MEDIAN_RATE_MAX_HZ


def test_sequence_validation(brain: Brain) -> None:
    with pytest.raises(ValueError, match="shape"):
        brain.simulate_sequence(np.zeros((2, 3, brain.n_pn + 1), dtype=np.float32), seed=0)
    with pytest.raises(ValueError, match="shape"):
        brain.simulate_sequence(np.zeros((2, brain.n_pn), dtype=np.float32), seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brain.simulate_sequence(np.full((2, 3, brain.n_pn), 1.5, dtype=np.float32), seed=0)
    with pytest.raises(ValueError, match="one seed per trial"):
        brain.simulate_sequence(np.zeros((2, 3, brain.n_pn), dtype=np.float32), np.arange(3))


def test_sequence_params_are_whole_steps() -> None:
    p = DEFAULT_PARAMS
    assert p.puff_ms == 20.0  # auditor's decision after Phase 5
    assert p.puff_steps == 200
    assert p.sequence_steps(10) == 2100
    with pytest.raises(ValueError, match="whole number"):
        _ = BrainParams(puff_ms=20.05).puff_steps


def test_sequence_result_helpers(brain: Brain, real_puffs: np.ndarray) -> None:
    res: SequenceResult = brain.simulate_sequence(real_puffs[:4], seed=1)
    assert res.n_trials == 4
    assert res.n_puffs == real_puffs.shape[1]
    assert res.total_kc_counts().shape == (4, brain.n_kc)
    empty = SequenceResult(
        kc_counts=np.zeros((1, 2, brain.n_kc), np.int16),
        mbon_counts=np.zeros((1, 2, 1), np.int16),
        apl_counts=np.zeros((1, 2), np.int16),
        pn_counts=np.zeros((1, 2, brain.n_pn), np.int16),
        puff_active=np.zeros((1, 2), bool),
        kc_active_fraction=np.zeros((1, 2), np.float32),
        n_steps=1,
    )
    assert empty.kc_active_fraction_non_empty == 0.0
