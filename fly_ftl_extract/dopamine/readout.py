"""The MBON readout: a linear layer over the Kenyon-cell state, trained with dopamine.

Features of one trial (``feature_mode``): ``binary`` = KC spiked, ``log1p`` = log1p(spike
count), ``both`` = the two concatenated (CLAUDE.md: ``[kc_spiked, log1p(kc_count)]``).
The readout is ``margin = X @ W + b``; ``margin > 0`` means *key* (or *placeable*).
Learning is the delta rule on minibatches with L2 (:func:`dan_update` — the dopamine
error signal), early stopping on the validation F1.  Numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

FeatureMode = Literal["binary", "log1p", "both"]
FEATURE_MODES: tuple[FeatureMode, ...] = ("binary", "log1p", "both")


def features(kc_counts: np.ndarray, mode: FeatureMode) -> np.ndarray:
    """Feature matrix ``(n_trials, dim)`` float32 of KC spike counts ``(n_trials, n_kc)``."""
    counts = np.asarray(kc_counts)
    if mode == "binary":
        return (counts > 0).astype(np.float32)
    if mode == "log1p":
        return np.log1p(counts.astype(np.float32))
    return np.concatenate(
        [(counts > 0).astype(np.float32), np.log1p(counts.astype(np.float32))], axis=1
    )


def feature_dim(n_kc: int, mode: FeatureMode) -> int:
    """Length of a feature vector."""
    return 2 * n_kc if mode == "both" else n_kc


@dataclass
class Readout:
    """Weights of one linear MBON readout."""

    w: np.ndarray
    b: float
    mode: FeatureMode

    @classmethod
    def zeros(cls, n_kc: int, mode: FeatureMode) -> Readout:
        """An untrained readout (all weights zero)."""
        return cls(np.zeros(feature_dim(n_kc, mode), dtype=np.float32), 0.0, mode)

    def margin(self, kc_counts: np.ndarray) -> np.ndarray:
        """``X @ W + b`` per trial, from KC spike counts."""
        return self.margin_from_features(features(kc_counts, self.mode))

    def margin_from_features(self, x: np.ndarray) -> np.ndarray:
        """``X @ W + b`` per trial, from an already built feature matrix."""
        return np.asarray(x @ self.w + np.float32(self.b), dtype=np.float32)


def dan_update(readout: Readout, x: np.ndarray, y: np.ndarray, lr: float, l2: float) -> float:
    """One delta-rule step on a minibatch; returns the mean logistic loss before the step.

    ``err = sigmoid(margin) - y`` is the dopamine error signal: it strengthens the
    KC→MBON synapses that were active when the answer was wrong, in the direction that
    would have made it right.
    """
    margin = readout.margin_from_features(x)
    p = 1.0 / (1.0 + np.exp(-margin))
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    err = (p - y).astype(np.float32)
    grad_w = x.T @ err / len(x) + np.float32(l2) * readout.w
    readout.w -= np.float32(lr) * grad_w
    readout.b -= lr * float(err.mean())
    return loss


@dataclass(frozen=True)
class Scores:
    """Precision / recall / F1 and the confusion matrix of binary decisions."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        """``tp / (tp + fp)``."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """``tp / (tp + fn)``."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def n(self) -> int:
        """Number of decisions."""
        return self.tp + self.fp + self.fn + self.tn

    def as_dict(self) -> dict[str, float | int]:
        """JSON-friendly form."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def score(decisions: np.ndarray, y: np.ndarray) -> Scores:
    """Confusion matrix of boolean ``decisions`` against boolean labels ``y``."""
    d, t = np.asarray(decisions, dtype=bool), np.asarray(y, dtype=bool)
    return Scores(
        tp=int((d & t).sum()),
        fp=int((d & ~t).sum()),
        fn=int((~d & t).sum()),
        tn=int((~d & ~t).sum()),
    )


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters of the delta-rule training (all recorded in METRICS.md)."""

    lr: float = 0.05
    l2: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 5
    seed: int = 0


@dataclass
class TrainLog:
    """What happened during training (for METRICS.md)."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_f1: float = 0.0


DEFAULT_TRAIN_CONFIG = TrainConfig()


def train_readout(
    counts_train: np.ndarray,
    y_train: np.ndarray,
    counts_val: np.ndarray,
    y_val: np.ndarray,
    *,
    mode: FeatureMode,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
) -> tuple[Readout, TrainLog]:
    """Delta-rule training with early stopping on the validation F1.

    ``counts_*`` are KC spike counts ``(n, n_kc)`` (uint8/int16); features are built per
    minibatch so that the full feature matrix never has to fit in memory.
    """
    n_kc = counts_train.shape[1]
    readout = Readout.zeros(n_kc, mode)
    best = Readout.zeros(n_kc, mode)
    log = TrainLog()
    rng = np.random.default_rng(config.seed)
    x_val = features(counts_val, mode)
    y_tr = np.asarray(y_train, dtype=np.float32)
    since_best = 0
    for epoch in range(config.max_epochs):
        order = rng.permutation(len(counts_train))
        losses = []
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            losses.append(
                dan_update(
                    readout, features(counts_train[idx], mode), y_tr[idx], config.lr, config.l2
                )
            )
        val_f1 = score(readout.margin_from_features(x_val) > 0, y_val).f1
        log.epochs.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_f1": val_f1})
        if val_f1 > log.best_val_f1:
            log.best_val_f1, log.best_epoch, since_best = val_f1, epoch, 0
            best = Readout(readout.w.copy(), readout.b, mode)
        else:
            since_best += 1
            if since_best >= config.patience:
                break
    return best, log


def resniff_threshold(margins: np.ndarray, max_fraction: float = 0.10) -> float:
    """θ such that at most ``max_fraction`` of the trials have ``|margin| < θ``."""
    if len(margins) == 0:
        return 0.0
    return float(np.quantile(np.abs(margins), max_fraction, method="lower"))


def vote(margins: np.ndarray) -> np.ndarray:
    """Decision of several sniffs of the same candidate: the sign of the summed margin.

    ``margins`` is ``(n_candidates, n_sniffs)``; returns the summed margin per candidate.
    """
    return np.asarray(margins, dtype=np.float32).sum(axis=1)
