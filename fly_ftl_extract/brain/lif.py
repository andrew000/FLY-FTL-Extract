"""Batched leaky integrate-and-fire simulation of the mushroom body.

One call to :meth:`Brain.simulate` runs ``n_trials`` independent trials at once: every
state array has the shape ``(n_trials, n_neurons)`` and the synaptic drive of a step is one
sparse product ``spikes @ W``.  Projection neurons are the input: they fire Poisson trains
at ``rate_max * odor`` during ``t_stim`` (Shiu et al. drive their input neurons with Poisson
events that each trigger a spike, which is the same thing).  Kenyon cells, APL and MBONs
integrate.  DANs neither receive nor give current here (they are the teaching signal of the
dopamine phase).

Determinism: the only randomness is ``numpy.random.Generator(PCG64(seed))`` for the PN
spikes; the float32 arithmetic is done in a fixed order, so equal inputs and seeds give
bit-identical results.

Implementation notes (measured, see ``docs/BENCH.md``): spikes and refractory neurons are
handled as *flat* index arrays (``np.flatnonzero`` on a boolean mask is ~30× faster than
``np.nonzero`` on a 2-D array or on an int16 array), the refractory set is a ring of the
last ``refractory_steps`` spike-index arrays instead of a per-neuron counter, and the drive
is ``S @ W`` with ``S`` a CSR spike matrix built directly from ``indptr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from fly_ftl_extract.brain.connectome import Connectome
from fly_ftl_extract.brain.params import DEFAULT_PARAMS, BrainParams

DriveMode = Literal["events", "dense"]
_ODOR_NDIM = 2
_EMPTY = np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class SpikeRaster:
    """Kenyon-cell spikes of the first trials as sparse events (for the TUI raster)."""

    n_trials: int
    n_steps: int
    n_kc: int
    trial: np.ndarray
    """Trial index of every spike, int32."""
    step: np.ndarray
    """Step index of every spike, int32."""
    kc: np.ndarray
    """Kenyon-cell index (position in ``Connectome.kc_idx``) of every spike, int32."""

    def dense(self, trial: int) -> np.ndarray:
        """``(n_steps, n_kc)`` boolean raster of one trial."""
        out = np.zeros((self.n_steps, self.n_kc), dtype=bool)
        mask = self.trial == trial
        out[self.step[mask], self.kc[mask]] = True
        return out


@dataclass(frozen=True)
class TrialResult:
    """Spike counts of one batch of trials."""

    kc_counts: np.ndarray
    """``(n_trials, n_kc)`` int16 spikes per Kenyon cell over the whole trial."""
    mbon_counts: np.ndarray
    """``(n_trials, n_mbon)`` int16."""
    apl_counts: np.ndarray
    """``(n_trials,)`` int16."""
    pn_counts: np.ndarray
    """``(n_trials, n_pn)`` int16 — the Poisson input that was actually drawn."""
    kc_active_fraction: np.ndarray
    """``(n_trials,)`` float32: share of *all* Kenyon cells that spiked at least once."""
    kc_active_fraction_pn_input: np.ndarray
    """``(n_trials,)`` float32: the same share among Kenyon cells that have a PN input
    (the others cannot be driven by an odour at all)."""
    n_steps: int
    raster: SpikeRaster | None = None

    @property
    def n_trials(self) -> int:
        """Number of trials in the batch."""
        return int(self.kc_counts.shape[0])


class Brain:
    """The mushroom body ready to be stimulated.

    Args:
        connectome: the FlyWire subgraph.
        params: LIF constants; ``params.syn_scale`` scales every weight.
        apl_enabled: with ``False`` the APL→* weights are zeroed (calibration only).
        drive: how the synaptic drive is computed each step — ``"events"`` builds a sparse
            spike matrix from the few (trial, neuron) events and multiplies it by ``W``;
            ``"dense"`` multiplies ``W.T`` (CSR) by the dense spike matrix.  Both give the
            same numbers; ``scripts/bench.py`` measures which is faster.
    """

    def __init__(
        self,
        connectome: Connectome,
        params: BrainParams = DEFAULT_PARAMS,
        *,
        apl_enabled: bool = True,
        drive: DriveMode = "events",
    ) -> None:
        self.connectome = connectome
        self.params = params
        self.apl_enabled = apl_enabled
        self.drive: DriveMode = drive
        n = connectome.n_neurons
        # Integrating neurons: KC, APL, MBON, in connectome order (KC first, contiguous).
        self.int_idx = np.sort(
            np.concatenate([connectome.kc_idx, connectome.apl_idx, connectome.mbon_idx])
        ).astype(np.int32)
        local = np.full(n, -1, dtype=np.int32)
        local[self.int_idx] = np.arange(len(self.int_idx), dtype=np.int32)
        self.kc_local = local[connectome.kc_idx]
        self.apl_local = local[connectome.apl_idx]
        self.mbon_local = local[connectome.mbon_idx]
        if not np.array_equal(self.kc_local, np.arange(len(self.kc_local))):
            msg = "Kenyon cells are expected to be the first, contiguous block of neurons"
            raise ValueError(msg)
        self.n_pn = len(connectome.pn_idx)
        self.n_kc = len(connectome.kc_idx)
        self.n_int = len(self.int_idx)

        # W: rows = every neuron (presynaptic), columns = integrating neurons (postsynaptic),
        # values in mV added to g per presynaptic spike.  DAN rows are empty by construction
        # (their edges live in connectome.dan_edges); PN/DAN columns are dropped.
        w = connectome.weights.astype(np.float32)[:, self.int_idx].tocsr()
        w = w.multiply(np.float32(params.w_syn * params.syn_scale)).tocsr()
        if w[connectome.dan_idx].nnz:
            msg = "DAN-presynaptic edges must not be in Connectome.weights"
            raise ValueError(msg)
        if not apl_enabled:
            keep = np.ones(n, dtype=np.float32)
            keep[connectome.apl_idx] = 0.0
            w = sp.diags(keep, format="csr") @ w
        w.eliminate_zeros()
        w.sort_indices()
        self.w: sp.csr_matrix = w
        self.w_t: sp.csr_matrix = w.T.tocsr()

        pn_kc = connectome.syn_count[connectome.pn_idx][:, connectome.kc_idx]
        self.kc_has_pn_input = np.asarray((pn_kc > 0).sum(axis=0)).ravel() > 0
        self.n_kc_with_pn_input = int(self.kc_has_pn_input.sum())

        # Euler coefficients (float32, computed once).
        self._decay_v = np.float32(1.0 - params.dt / params.tau_m)
        self._gain_g = np.float32(params.dt / params.tau_m)
        self._decay_g = np.float32(np.exp(-params.dt / params.tau_syn))
        self._u_th = np.float32(params.v_th - params.v_rest)
        self._u_reset = np.float32(params.v_reset - params.v_rest)

    def simulate(self, odors: np.ndarray, seed: int, *, raster_trials: int = 0) -> TrialResult:
        """Run one trial per row of ``odors`` (``(n_trials, n_pn)`` values in ``[0, 1]``).

        ``raster_trials`` says for how many leading trials the Kenyon-cell spike times are
        kept as :class:`SpikeRaster`.
        """
        odors = np.asarray(odors, dtype=np.float32)
        if odors.ndim != _ODOR_NDIM or odors.shape[1] != self.n_pn:
            msg = f"odors must have shape (n_trials, {self.n_pn}), got {odors.shape}"
            raise ValueError(msg)
        if odors.size and (odors.min() < 0 or odors.max() > 1):
            msg = "odor values must lie in [0, 1]"
            raise ValueError(msg)
        p = self.params
        n_trials = odors.shape[0]
        n_int, n_pn = self.n_int, self.n_pn
        rng = np.random.Generator(np.random.PCG64(seed))
        p_spike = odors * np.float32(p.rate_max * p.dt / 1000.0)
        pn_idx = self.connectome.pn_idx.astype(np.int32)
        int_idx = self.int_idx

        u = np.zeros((n_trials, n_int), dtype=np.float32)  # v - v_rest, mV
        g = np.zeros((n_trials, n_int), dtype=np.float32)  # synaptic variable, mV
        tmp = np.empty_like(u)
        u_flat, g_flat = u.reshape(-1), g.reshape(-1)
        counts = np.zeros(n_trials * n_int, dtype=np.int16)
        pn_counts = np.zeros(n_trials * n_pn, dtype=np.int16)
        n_steps, stim_steps = p.n_steps, p.stim_steps
        delay, refr_steps = p.delay_steps, p.refractory_steps
        # Spike events (flat indices) of the last `refr_steps` steps = the refractory set.
        refr_ring: list[np.ndarray] = [_EMPTY] * refr_steps
        # Events (trial rows, global neuron columns) waiting `delay` steps for delivery.
        queue: list[tuple[np.ndarray, np.ndarray] | None] = [None] * delay
        raster_rows: list[np.ndarray] = []
        raster_cols: list[np.ndarray] = []
        raster_steps: list[np.ndarray] = []

        for t in range(n_steps):
            # 1. deliver the spikes emitted `delay` steps ago
            pending = queue[t % delay]
            if pending is not None:
                self._deliver(g, pending, n_trials)
            # 2. integrate; refractory neurons are frozen (their g still collects input)
            frozen = np.concatenate(refr_ring)
            g_frozen = g_flat[frozen]
            np.multiply(u, self._decay_v, out=u)
            np.multiply(g, self._gain_g, out=tmp)
            np.add(u, tmp, out=u)
            np.multiply(g, self._decay_g, out=g)
            u_flat[frozen] = self._u_reset
            g_flat[frozen] = g_frozen
            # 3. threshold, reset (v and g, as in Shiu et al.), refractory
            spk = np.flatnonzero(u > self._u_th)
            u_flat[spk] = self._u_reset
            g_flat[spk] = 0
            counts[spk] += 1
            refr_ring[t % refr_steps] = spk
            rows = spk // n_int
            cols = int_idx[spk - rows * n_int]
            # 4. PN Poisson input
            if t < stim_steps:
                pn = np.flatnonzero(rng.random(p_spike.shape, dtype=np.float32) < p_spike)
                pn_counts[pn] += 1
                pn_rows = pn // n_pn
                rows = np.concatenate([rows, pn_rows])
                cols = np.concatenate([cols, pn_idx[pn - pn_rows * n_pn]])
            queue[t % delay] = (rows, cols) if len(rows) else None
            if raster_trials and len(spk):
                r_rows = spk // n_int
                r_cols = spk - r_rows * n_int
                keep = (r_rows < raster_trials) & (r_cols < self.n_kc)
                if keep.any():
                    raster_rows.append(r_rows[keep].astype(np.int32))
                    raster_cols.append(r_cols[keep].astype(np.int32))
                    raster_steps.append(np.full(int(keep.sum()), t, dtype=np.int32))

        counts2 = counts.reshape(n_trials, n_int)
        kc_counts = counts2[:, : self.n_kc].copy()
        active = kc_counts > 0
        raster = None
        if raster_trials:
            raster = SpikeRaster(
                n_trials=min(raster_trials, n_trials),
                n_steps=n_steps,
                n_kc=self.n_kc,
                trial=_concat(raster_rows),
                step=_concat(raster_steps),
                kc=_concat(raster_cols),
            )
        return TrialResult(
            kc_counts=kc_counts,
            mbon_counts=counts2[:, self.mbon_local].copy(),
            apl_counts=counts2[:, self.apl_local[0]].copy(),
            pn_counts=pn_counts.reshape(n_trials, n_pn),
            kc_active_fraction=active.mean(axis=1, dtype=np.float32),
            kc_active_fraction_pn_input=active[:, self.kc_has_pn_input].mean(
                axis=1, dtype=np.float32
            ),
            n_steps=n_steps,
            raster=raster,
        )

    def _deliver(self, g: np.ndarray, events: tuple[np.ndarray, np.ndarray], n_trials: int) -> None:
        rows, cols = events
        n_all = self.connectome.n_neurons
        if self.drive == "events":
            order = np.argsort(rows, kind="stable")
            rows, cols = rows[order], cols[order]
            indptr = np.zeros(n_trials + 1, dtype=np.int32)
            np.cumsum(np.bincount(rows, minlength=n_trials), out=indptr[1:])
            spikes = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), cols, indptr), shape=(n_trials, n_all)
            )
            drive = (spikes @ self.w).tocoo()
            if drive.nnz:
                g.reshape(-1)[drive.row * self.n_int + drive.col] += drive.data
        else:
            dense = np.zeros((n_all, n_trials), dtype=np.float32)
            dense[cols, rows] = 1.0
            g += (self.w_t @ dense).T


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def simulate(
    connectome: Connectome,
    odors: np.ndarray,
    seed: int,
    params: BrainParams = DEFAULT_PARAMS,
    *,
    raster_trials: int = 0,
) -> TrialResult:
    """One-shot convenience wrapper around :class:`Brain` (builds the matrices every call)."""
    return Brain(connectome, params).simulate(odors, seed, raster_trials=raster_trials)
