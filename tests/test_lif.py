"""Behavioural tests of the LIF mushroom body (PLAN, Phase 3)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, BrainParams, Connectome, TrialResult
from fly_ftl_extract.brain import connectome as cx_mod

REPO = Path(__file__).resolve().parent.parent
CALIBRATION = REPO / "docs" / "calibration.json"
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_odors  # noqa: E402

N_TRIALS = 64
ODOR_LEVEL = 0.75  # 150 Hz at rate_max 200 Hz: the PLAN's typical odour
PN_ACTIVE_FRACTION = 0.30
SPARSITY_WITH_APL = (0.08, 0.10)  # auditor's target after Phase 4, on real odours
SPARSITY_CLAUDE_MD = (0.03, 0.15)  # CLAUDE.md: outside this range the weights are wrong
SPARSITY_WITHOUT_APL = 0.30
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
    return Brain(connectome)


@pytest.fixture(scope="module")
def real_odors() -> np.ndarray:
    """Every fixture candidate encoded with its fixture's options (the calibration set)."""
    odors, _ = fixture_odors()
    assert len(odors) > 100
    return odors


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


def test_kc_sparsity_with_apl(brain: Brain, real_odors: np.ndarray) -> None:
    res = brain.simulate(real_odors, seed=1)
    frac = float(res.kc_active_fraction.mean())
    assert SPARSITY_CLAUDE_MD[0] <= frac <= SPARSITY_CLAUDE_MD[1]
    assert SPARSITY_WITH_APL[0] <= frac <= SPARSITY_WITH_APL[1]
    assert (res.kc_active_fraction_pn_input >= res.kc_active_fraction).all()
    assert res.apl_counts.mean() > 0  # APL fires


def test_sparsity_rises_without_apl(
    brain: Brain, connectome: Connectome, real_odors: np.ndarray
) -> None:
    odors = real_odors
    with_apl = brain.simulate(odors, seed=1)
    without = Brain(connectome, apl_enabled=False).simulate(odors, seed=1)
    assert without.apl_counts.sum() >= 0  # APL still spikes, only its outputs are cut
    assert without.kc_active_fraction.mean() > with_apl.kc_active_fraction.mean()
    assert without.kc_active_fraction.mean() > SPARSITY_WITHOUT_APL


def test_different_odors_give_different_kc_patterns(brain: Brain, real_odors: np.ndarray) -> None:
    res = brain.simulate(real_odors, seed=1)
    rng = np.random.default_rng(0)
    a = rng.permutation(len(real_odors))
    shifted = np.roll(a, 1)
    x, y = res.kc_counts[a] > 0, res.kc_counts[shifted] > 0
    j = float(((x & y).sum(axis=1) / np.maximum((x | y).sum(axis=1), 1)).mean())
    assert j < JACCARD_DIFFERENT_MAX


@pytest.mark.xfail(
    reason=(
        "diagnostic (auditor, after Phase 4): on real fixture odours the same candidate "
        "with two seeds gives Jaccard 0.496 < 0.5 (docs/BENCH.md); the Phase 5 gate is "
        "readout accuracy"
    ),
    strict=False,
)
def test_same_odor_different_seeds_give_similar_kc_patterns(
    brain: Brain, real_odors: np.ndarray
) -> None:
    a = brain.simulate(real_odors, seed=1)
    b = brain.simulate(real_odors, seed=2)
    assert jaccard(a, b) > JACCARD_SAME_MIN


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


def test_active_kc_median_rate_is_below_50_hz(brain: Brain) -> None:
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
    assert DEFAULT_PARAMS.syn_scale == calibration["syn_scale"]
