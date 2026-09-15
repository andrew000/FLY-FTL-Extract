"""The MBON readout: a linear layer over the Kenyon-cell state, trained with dopamine.

Features of one trial (``feature_mode``): ``binary`` = KC spiked, ``log1p`` = log1p(spike
count), ``both`` = the two concatenated (CLAUDE.md: ``[kc_spiked, log1p(kc_count)]``).
With the temporal code the KC state of a trial is ``(n_puffs, n_kc)`` — the counts *per
puff* — and the features are taken per puff, so ``both`` is ``2 × n_puffs × n_kc`` long
(auditor's decision after Phase 5); any trailing dimensions are flattened.
The readout is ``margin = X @ W + b``; ``margin > 0`` means *key* (or *placeable*).
Learning is the delta rule on minibatches with L2 (:func:`dan_update` — the dopamine
error signal), early stopping on the validation F1.  Numpy only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

FeatureMode = Literal["binary", "log1p", "both"]
FEATURE_MODES: tuple[FeatureMode, ...] = ("binary", "log1p", "both")


def features(kc_counts: np.ndarray, mode: FeatureMode) -> np.ndarray:
    """Feature matrix ``(n_trials, dim)`` float32 of KC spike counts.

    ``kc_counts`` is ``(n_trials, n_kc)`` or, for the temporal code,
    ``(n_trials, n_puffs, n_kc)`` — flattened per trial.
    """
    counts = np.asarray(kc_counts)
    counts = counts.reshape(counts.shape[0], -1)
    if mode == "binary":
        return (counts > 0).astype(np.float32)
    if mode == "log1p":
        return np.log1p(counts.astype(np.float32))
    return np.concatenate(
        [(counts > 0).astype(np.float32), np.log1p(counts.astype(np.float32))], axis=1
    )


def feature_dim(n_states: int, mode: FeatureMode) -> int:
    """Length of a feature vector for ``n_states`` KC states (``n_puffs × n_kc``)."""
    return 2 * n_states if mode == "both" else n_states


@dataclass
class Readout:
    """Weights of one linear MBON readout."""

    w: np.ndarray
    b: float
    mode: FeatureMode

    @classmethod
    def zeros(cls, n_states: int, mode: FeatureMode) -> Readout:
        """An untrained readout (all weights zero) over ``n_states`` KC states."""
        return cls(np.zeros(feature_dim(n_states, mode), dtype=np.float32), 0.0, mode)

    @property
    def n_states(self) -> int:
        """KC states (``n_puffs × n_kc``) the readout expects."""
        return len(self.w) // 2 if self.mode == "both" else len(self.w)

    def margin(self, kc_counts: np.ndarray) -> np.ndarray:
        """``X @ W + b`` per trial, from KC spike counts."""
        return self.margin_from_features(features(kc_counts, self.mode))

    def margin_from_features(self, x: np.ndarray) -> np.ndarray:
        """``X @ W + b`` per trial, from an already built feature matrix."""
        return np.asarray(x @ self.w + np.float32(self.b), dtype=np.float32)

    def margin_batched(self, kc_counts: np.ndarray, batch: int = 2048) -> np.ndarray:
        """:meth:`margin` in slices of ``batch`` trials.

        The feature matrix of a whole split (``n × 2 × n_puffs × n_kc`` float32) never has
        to exist at once.
        """
        out = np.empty(len(kc_counts), dtype=np.float32)
        for start in range(0, len(kc_counts), batch):
            out[start : start + batch] = self.margin(kc_counts[start : start + batch])
        return out


def dan_update(readout: Readout, x: np.ndarray, y: np.ndarray, lr: float, l2: float) -> float:
    """One delta-rule step on a minibatch; returns the mean logistic loss before the step.

    ``err = sigmoid(margin) - y`` is the dopamine error signal: it strengthens the
    KC→MBON synapses that were active when the answer was wrong, in the direction that
    would have made it right.
    """
    margin = readout.margin_from_features(x)
    err, loss = _dopamine_error(margin, y)
    grad_w = x.T @ err / len(x) + np.float32(l2) * readout.w
    readout.w -= np.float32(lr) * grad_w
    readout.b -= lr * float(err.mean())
    return loss


def _dopamine_error(margin: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    p = 1.0 / (1.0 + np.exp(-margin))
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return (p - y).astype(np.float32), loss


class SparseStates:
    """KC spike counts of many trials as one CSR matrix, for fast readout training.

    Only ~9 % of the (puff, KC) states of a trial are non-zero, so the features of a
    minibatch are built from the non-zero entries alone: the ``binary`` part is the
    pattern with ones as data, the ``log1p`` part the same pattern with ``log1p(count)``
    as data.  The maths is exactly that of :func:`features` / :func:`dan_update`; only the
    zeros are never materialised (a dense ``512 × 51940`` float32 minibatch costs ~80 ms,
    the sparse one a few).
    """

    def __init__(self, counts: np.ndarray, chunk: int = 4096) -> None:
        n = counts.shape[0]
        self.n_states = int(np.prod(counts.shape[1:]))
        indptr = np.zeros(n + 1, dtype=np.int64)
        indices_parts: list[np.ndarray] = []
        log_parts: list[np.ndarray] = []
        for start in range(0, n, chunk):
            block = np.asarray(counts[start : start + chunk]).reshape(-1, self.n_states)
            rows, cols = np.nonzero(block)
            indices_parts.append(cols.astype(np.int32))
            log_parts.append(np.log1p(block[rows, cols].astype(np.float32)))
            indptr[start + 1 : start + 1 + len(block)] = indptr[start] + np.cumsum(
                np.bincount(rows, minlength=len(block))
            )
        self.indptr = indptr
        self.indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, np.int32)
        self.log1p = np.concatenate(log_parts) if log_parts else np.zeros(0, np.float32)
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    @classmethod
    def concat(cls, parts: list[SparseStates]) -> SparseStates:
        """Stack several :class:`SparseStates` (same ``n_states``) into one.

        The result is preallocated and the parts are released one by one, so the peak
        memory is the result plus the largest part — not twice the total.  ``parts`` is
        emptied.
        """
        if not parts:
            msg = "nothing to concatenate"
            raise ValueError(msg)
        n_states = parts[0].n_states
        if any(p.n_states != n_states for p in parts):
            msg = "all parts must have the same number of KC states"
            raise ValueError(msg)
        n = sum(p.n for p in parts)
        nnz = sum(len(p.indices) for p in parts)
        out = cls.__new__(cls)
        out.n_states, out.n = n_states, n
        out.indptr = np.zeros(n + 1, dtype=np.int64)
        out.indices = np.empty(nnz, dtype=np.int32)
        out.log1p = np.empty(nnz, dtype=np.float32)
        row, pos = 0, 0
        while parts:
            part = parts.pop(0)
            k = len(part.indices)
            out.indices[pos : pos + k] = part.indices
            out.log1p[pos : pos + k] = part.log1p
            out.indptr[row + 1 : row + 1 + part.n] = part.indptr[1:] + pos
            row += part.n
            pos += k
            del part
        return out

    def rows(self, idx: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``(pattern, log1p)`` CSR matrices of the selected trials (same sparsity)."""
        starts, ends = self.indptr[idx], self.indptr[idx + 1]
        lengths = ends - starts
        take = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends, strict=True)])
        indptr = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
        shape = (len(idx), self.n_states)
        cols = self.indices[take]
        log_data = self.log1p[take]
        pattern = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), cols, indptr), shape=shape)
        logs = sp.csr_matrix((log_data, cols, indptr), shape=shape)
        return pattern, logs


def _split_weights(readout: Readout) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Views of the binary and log1p halves of the weight vector (``None`` when absent)."""
    if readout.mode == "binary":
        return readout.w, None
    if readout.mode == "log1p":
        return None, readout.w
    n = readout.n_states
    return readout.w[:n], readout.w[n:]


def margin_sparse(readout: Readout, pattern: sp.csr_matrix, logs: sp.csr_matrix) -> np.ndarray:
    """``X @ W + b`` from the sparse minibatch (identical to the dense margin)."""
    w_bin, w_log = _split_weights(readout)
    margin = np.full(pattern.shape[0], np.float32(readout.b), dtype=np.float32)
    if w_bin is not None:
        margin += pattern @ w_bin
    if w_log is not None:
        margin += logs @ w_log
    return margin


def dan_update_sparse(  # noqa: PLR0917
    readout: Readout,
    pattern: sp.csr_matrix,
    logs: sp.csr_matrix,
    y: np.ndarray,
    lr: float,
    l2: float,
) -> float:
    """:func:`dan_update` on a sparse minibatch — the same step, zeros skipped."""
    err, loss = _dopamine_error(margin_sparse(readout, pattern, logs), y)
    n = np.float32(len(y))
    w_bin, w_log = _split_weights(readout)
    if w_bin is not None:
        w_bin -= np.float32(lr) * ((pattern.T @ err) / n + np.float32(l2) * w_bin)
    if w_log is not None:
        w_log -= np.float32(lr) * ((logs.T @ err) / n + np.float32(l2) * w_log)
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
    counts_train: np.ndarray | SparseStates,
    y_train: np.ndarray,
    counts_val: np.ndarray,
    y_val: np.ndarray,
    *,
    mode: FeatureMode,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    on_epoch: Callable[[dict[str, float]], None] | None = None,
) -> tuple[Readout, TrainLog]:
    """Delta-rule training with early stopping on the validation F1.

    ``counts_*`` are KC spike counts ``(n, n_kc)`` or ``(n, n_puffs, n_kc)`` (uint8/int16);
    features are built per minibatch so that the full feature matrix never has to fit in
    memory.  ``counts_train`` may also be a :class:`SparseStates` (the same trials as a
    CSR matrix): same delta rule, ~10× faster per epoch for the temporal code.
    """
    sparse = counts_train if isinstance(counts_train, SparseStates) else None
    dense = None if isinstance(counts_train, SparseStates) else np.asarray(counts_train)
    n_states = sparse.n_states if sparse is not None else int(np.prod(dense.shape[1:]))  # type: ignore[union-attr]
    readout = Readout.zeros(n_states, mode)
    best = Readout.zeros(n_states, mode)
    log = TrainLog()
    rng = np.random.default_rng(config.seed)
    y_tr = np.asarray(y_train, dtype=np.float32)
    since_best = 0
    for epoch in range(config.max_epochs):
        order = rng.permutation(len(counts_train))
        losses = []
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            if sparse is not None:
                pattern, logs = sparse.rows(idx)
                losses.append(
                    dan_update_sparse(readout, pattern, logs, y_tr[idx], config.lr, config.l2)
                )
            else:
                assert dense is not None  # noqa: S101 — the branch above handles the sparse case
                losses.append(
                    dan_update(readout, features(dense[idx], mode), y_tr[idx], config.lr, config.l2)
                )
        val_f1 = score(readout.margin_batched(counts_val) > 0, y_val).f1
        log.epochs.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_f1": val_f1})
        if on_epoch is not None:
            on_epoch(log.epochs[-1])
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
