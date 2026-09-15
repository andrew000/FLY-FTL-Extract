"""Readout mechanics: features, the delta rule, resniff threshold, weights file, seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_ftl_extract.dopamine import (
    MbonWeights,
    Readout,
    SparseStates,
    TrainConfig,
    WeightsMismatchError,
    WeightsMissingError,
    dan_update,
    dan_update_sparse,
    features,
    margin_sparse,
    resniff_threshold,
    score,
    sniff_seed,
    train_readout,
    trial_seed,
    vote,
)
from fly_ftl_extract.dopamine import weights as weights_module
from fly_ftl_extract.dopamine.seed import kwarg_trial_index

N_KC = 64


def separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """KC counts where the label is 'more spikes in the first half of the cells'."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    counts = rng.poisson(0.3, size=(n, N_KC)).astype(np.uint8)
    counts[y, : N_KC // 2] += rng.poisson(1.5, size=(int(y.sum()), N_KC // 2)).astype(np.uint8)
    return counts, y


def test_feature_modes() -> None:
    counts = np.array([[0, 1, 3], [2, 0, 0]], dtype=np.uint8)
    assert features(counts, "binary").tolist() == [[0, 1, 1], [1, 0, 0]]
    assert np.allclose(features(counts, "log1p"), np.log1p(counts.astype(np.float32)))
    both = features(counts, "both")
    assert both.shape == (2, 6)
    assert both.dtype == np.float32


def test_dan_update_reduces_loss() -> None:
    counts, y = separable(2000, 1)
    readout = Readout.zeros(N_KC, "both")
    x = features(counts, "both")
    first = dan_update(readout, x, y.astype(np.float32), lr=0.1, l2=0.0)
    for _ in range(50):
        last = dan_update(readout, x, y.astype(np.float32), lr=0.1, l2=0.0)
    assert last < first
    assert score(readout.margin(counts) > 0, y).f1 > 0.9


def test_train_readout_early_stops_and_generalises() -> None:
    counts, y = separable(4000, 2)
    counts_val, y_val = separable(1000, 3)
    readout, log = train_readout(
        counts,
        y,
        counts_val,
        y_val,
        mode="binary",
        config=TrainConfig(lr=0.2, max_epochs=30, patience=3),
    )
    assert log.best_val_f1 > 0.9
    assert log.best_epoch >= 0
    assert score(readout.margin(counts_val) > 0, y_val).f1 == pytest.approx(log.best_val_f1)


def test_resniff_threshold_and_vote() -> None:
    margins = np.linspace(-1, 1, 101)
    theta = resniff_threshold(margins, 0.10)
    assert (np.abs(margins) < theta).mean() <= 0.10
    assert vote(np.array([[0.2, -0.5, 0.1]])).tolist() == pytest.approx([-0.2])


def test_weights_roundtrip_and_mismatch(tmp_path: Path) -> None:
    w = MbonWeights(
        key=Readout(np.arange(4, dtype=np.float32), 0.5, "binary"),
        kwarg=Readout(np.ones(8, dtype=np.float32), -0.25, "both"),
        theta_key=0.1,
        theta_kwarg=0.2,
        brain_hash="abc",
        encoder_version="v1",
        metrics={"f1": 1.0},
    )
    path = tmp_path / "w.npz"
    w.save(path)
    back = weights_module.load(path, brain_hash="abc", encoder_version="v1")
    assert np.array_equal(back.key.w, w.key.w)
    assert back.kwarg.mode == "both"
    assert back.metrics == {"f1": 1.0}
    assert back.theta_kwarg == pytest.approx(0.2)
    with pytest.raises(WeightsMismatchError, match="brain hash"):
        weights_module.load(path, brain_hash="other", encoder_version="v1")
    with pytest.raises(WeightsMismatchError, match="encoder version"):
        weights_module.load(path, brain_hash="abc", encoder_version="v2")
    with pytest.raises(WeightsMissingError, match="make_dataset"):
        weights_module.load(tmp_path / "missing.npz")


def test_seeds_are_deterministic_and_distinct() -> None:
    a = trial_seed(b"content", 3, "v1")
    assert a == trial_seed(b"content", 3, "v1")
    assert a != trial_seed(b"content", 4, "v1")
    assert a != trial_seed(b"content ", 3, "v1")
    assert a != trial_seed(b"content", 3, "v2")
    assert 0 <= a < 2**64
    assert sniff_seed(a, 0) == a
    assert len({sniff_seed(a, s) for s in range(4)}) == 4
    assert kwarg_trial_index(2, 5) != kwarg_trial_index(2, 6) != kwarg_trial_index(3, 5)


def test_sparse_states_match_dense_features() -> None:
    """The CSR path computes the same margins and the same delta-rule step as the dense
    one, for per-puff counts ``(n, n_puffs, n_kc)``."""
    rng = np.random.default_rng(5)
    counts = (rng.random((300, 4, 16)) < 0.1).astype(np.uint8) * rng.integers(1, 4, (300, 4, 16))
    counts = counts.astype(np.uint8)
    counts[7] = 0  # an all-silent trial
    y = (rng.random(300) < 0.4).astype(np.float32)
    states = SparseStates(counts, chunk=64)
    assert len(states) == 300
    assert states.n_states == 64
    for mode in ("binary", "log1p", "both"):
        dense = Readout.zeros(64, mode)
        sparse_r = Readout.zeros(64, mode)
        idx = np.array([3, 7, 250, 12, 99])
        pattern, logs = states.rows(idx)
        assert np.allclose(margin_sparse(sparse_r, pattern, logs), dense.margin(counts[idx]))
        for _ in range(5):
            loss_d = dan_update(dense, features(counts[idx], mode), y[idx], lr=0.3, l2=1e-3)
            loss_s = dan_update_sparse(sparse_r, pattern, logs, y[idx], lr=0.3, l2=1e-3)
            assert loss_d == pytest.approx(loss_s, rel=1e-5)
        assert np.allclose(dense.w, sparse_r.w, atol=1e-6)
        assert dense.b == pytest.approx(sparse_r.b)
        assert np.allclose(
            margin_sparse(sparse_r, pattern, logs), dense.margin(counts[idx]), atol=1e-5
        )


def test_train_readout_sparse_equals_dense() -> None:
    counts, y = separable(2000, 4)
    counts_val, y_val = separable(500, 6)
    cfg = TrainConfig(lr=0.2, max_epochs=5, patience=5)
    dense, log_d = train_readout(counts, y, counts_val, y_val, mode="both", config=cfg)
    sparse_r, log_s = train_readout(
        SparseStates(counts), y, counts_val, y_val, mode="both", config=cfg
    )
    assert log_d.best_val_f1 == pytest.approx(log_s.best_val_f1, abs=1e-6)
    assert np.allclose(dense.w, sparse_r.w, atol=1e-4)


def test_sparse_states_concat_matches_whole() -> None:
    rng = np.random.default_rng(9)
    counts = (rng.random((90, 3, 8)) < 0.2).astype(np.uint8) * 2
    whole = SparseStates(counts)
    parts = [SparseStates(counts[:30]), SparseStates(counts[30:70]), SparseStates(counts[70:])]
    joined = SparseStates.concat(parts)
    assert parts == []
    assert len(joined) == 90
    assert np.array_equal(joined.indptr, whole.indptr)
    assert np.array_equal(joined.indices, whole.indices)
    assert np.array_equal(joined.log1p, whole.log1p)
    idx = np.array([0, 29, 30, 69, 70, 89])
    a, b = joined.rows(idx), whole.rows(idx)
    assert (a[1] != b[1]).nnz == 0
