"""Batched leaky integrate-and-fire simulation of the mushroom body.

One call to :meth:`Brain.simulate` runs ``n_trials`` independent trials at once: every
state array has the shape ``(n_trials, n_neurons)`` and the synaptic drive of a step is one
sparse product ``spikes @ W``.  Projection neurons are the input: they fire Poisson trains
at ``rate_max * odor`` during ``t_stim`` (Shiu et al. drive their input neurons with Poisson
events that each trigger a spike, which is the same thing).  Kenyon cells, APL and MBONs
integrate.  DANs neither receive nor give current here (they are the teaching signal of the
dopamine phase).

Two ways to present an odour:

* :meth:`Brain.simulate` — one odour vector per trial, held for ``t_stim`` (Phase 3; kept
  for its tests and for calibration comparisons);
* :meth:`Brain.simulate_sequence` — the temporal code (auditor's decision after Phase 5):
  a trial is a *sequence* of ``n_puffs`` odour vectors, each presented for
  ``params.puff_ms`` with no silence in between, then ``t_silence``.  The membrane and
  synaptic state carry over from puff to puff (that carry-over is the memory of what came
  before), and the spikes are counted per puff: bin ``k`` is the ``puff_ms`` of puff ``k``,
  the last bin also takes the closing silence.  A zero puff (an empty slot of the token
  window) drives no PN at all.

Determinism: the only randomness is ``numpy.random.Generator(PCG64(seed))`` for the PN
spikes; the float32 arithmetic is done in a fixed order, so equal inputs and seeds give
bit-identical results.  ``seed`` is either one integer for the whole batch (one stream,
drawn step by step — calibration and training augmentation) or one integer per trial
(production: every candidate has its own seed, see ``dopamine/seed.py``; each trial's
Poisson numbers come from its own PCG64 stream — drawn up front in ``simulate``, one puff
at a time in ``simulate_sequence``, which is the definition of the stream in that mode).

Implementation notes (measured, see ``docs/BENCH.md``): spikes and refractory neurons are
handled as *flat* index arrays (``np.flatnonzero`` on a boolean mask is ~30× faster than
``np.nonzero`` on a 2-D array or on an int16 array), the refractory set is a ring of the
last ``refractory_steps`` spike-index arrays instead of a per-neuron counter, and the drive
is ``S @ W`` with ``S`` a CSR spike matrix built directly from ``indptr``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class SequenceResult:
    """Per-puff spike counts of one batch of temporal-code trials."""

    kc_counts: np.ndarray
    """``(n_trials, n_puffs, n_kc)`` int16 spikes per Kenyon cell within each puff."""
    mbon_counts: np.ndarray
    """``(n_trials, n_puffs, n_mbon)`` int16."""
    apl_counts: np.ndarray
    """``(n_trials, n_puffs)`` int16."""
    pn_counts: np.ndarray
    """``(n_trials, n_puffs, n_pn)`` int16 — the Poisson input that was actually drawn."""
    puff_active: np.ndarray
    """``(n_trials, n_puffs)`` bool: the puff had at least one non-zero PN value."""
    kc_active_fraction: np.ndarray
    """``(n_trials, n_puffs)`` float32: share of all Kenyon cells that spiked in the puff."""
    n_steps: int
    raster: SpikeRaster | None = None

    @property
    def n_trials(self) -> int:
        """Number of trials in the batch."""
        return int(self.kc_counts.shape[0])

    @property
    def n_puffs(self) -> int:
        """Puffs per trial."""
        return int(self.kc_counts.shape[1])

    @property
    def kc_active_fraction_non_empty(self) -> float:
        """Mean active-KC share over the puffs that carried an odour (calibration target)."""
        if not self.puff_active.any():
            return 0.0
        return float(self.kc_active_fraction[self.puff_active].mean())

    def total_kc_counts(self) -> np.ndarray:
        """``(n_trials, n_kc)`` spikes summed over the puffs."""
        return self.kc_counts.sum(axis=1, dtype=np.int32).astype(np.int16)


_PUFF_NDIM = 3


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
        if not apl_enabled or params.apl_scale != 1.0:
            keep = np.ones(n, dtype=np.float32)
            keep[connectome.apl_idx] = 0.0 if not apl_enabled else np.float32(params.apl_scale)
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

    @property
    def hash(self) -> str:
        """Identity of this brain: connectome file + every parameter (16 hex digits).

        Trained readouts are only valid for the brain they were trained on.
        """
        payload = json.dumps(
            {
                "connectome": self.connectome.sha256,
                "params": asdict(self.params),
                "apl_enabled": self.apl_enabled,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def simulate(
        self, odors: np.ndarray, seed: int | np.ndarray, *, raster_trials: int = 0
    ) -> TrialResult:
        """Run one trial per row of ``odors`` (``(n_trials, n_pn)`` values in ``[0, 1]``).

        ``seed`` is one integer (one random stream for the batch) or an array with one
        seed per trial.  ``raster_trials`` says for how many leading trials the
        Kenyon-cell spike times are kept as :class:`SpikeRaster`.
        """
        odors = np.asarray(odors, dtype=np.float32)
        if odors.ndim != _ODOR_NDIM or odors.shape[1] != self.n_pn:
            msg = f"odors must have shape (n_trials, {self.n_pn}), got {odors.shape}"
            raise ValueError(msg)
        _check_range(odors)
        p = self.params
        n_trials = odors.shape[0]
        p_spike = odors * np.float32(p.rate_max * p.dt / 1000.0)
        n_steps, stim_steps = p.n_steps, p.stim_steps
        seeds = _per_trial_seeds(seed, n_trials)
        if seeds is not None:
            # (stim_steps, n_trials, n_pn): every trial its own stream, drawn up front
            uniforms = np.empty((stim_steps, n_trials, self.n_pn), dtype=np.float32)
            for k in range(n_trials):
                uniforms[:, k, :] = np.random.Generator(np.random.PCG64(int(seeds[k]))).random(
                    (stim_steps, self.n_pn), dtype=np.float32
                )

            def draw(t: int) -> np.ndarray:
                return np.asarray(uniforms[t])
        else:
            rng = np.random.Generator(np.random.PCG64(int(seed)))

            def draw(t: int) -> np.ndarray:  # noqa: ARG001
                return rng.random(p_spike.shape, dtype=np.float32)

        bins = np.zeros(n_steps, dtype=np.int64)
        counts, pn_counts, raster = self._run(
            n_trials,
            n_steps,
            lambda t: p_spike if t < stim_steps else None,
            draw,
            bins,
            1,
            raster_trials,
        )
        counts2 = counts[0].reshape(n_trials, self.n_int)
        kc_counts = counts2[:, : self.n_kc].copy()
        active = kc_counts > 0
        return TrialResult(
            kc_counts=kc_counts,
            mbon_counts=counts2[:, self.mbon_local].copy(),
            apl_counts=counts2[:, self.apl_local[0]].copy(),
            pn_counts=pn_counts[0].reshape(n_trials, self.n_pn),
            kc_active_fraction=active.mean(axis=1, dtype=np.float32),
            kc_active_fraction_pn_input=active[:, self.kc_has_pn_input].mean(
                axis=1, dtype=np.float32
            ),
            n_steps=n_steps,
            raster=raster,
        )

    def simulate_sequence(
        self, puffs: np.ndarray, seed: int | np.ndarray, *, raster_trials: int = 0
    ) -> SequenceResult:
        """Run one temporal-code trial per row of ``puffs`` (``(n_trials, n_puffs, n_pn)``).

        Puff ``k`` drives the PNs for ``params.puff_ms``, immediately followed by puff
        ``k + 1``; ``t_silence`` closes the trial.  The membrane state is *not* reset between
        puffs.  ``seed`` as in :meth:`simulate`; with one seed per trial the trial's stream
        is consumed one puff at a time (``(puff_steps, n_pn)`` uniforms per puff).
        """
        puffs = np.asarray(puffs, dtype=np.float32)
        if puffs.ndim != _PUFF_NDIM or puffs.shape[2] != self.n_pn:
            msg = f"puffs must have shape (n_trials, n_puffs, {self.n_pn}), got {puffs.shape}"
            raise ValueError(msg)
        _check_range(puffs)
        p = self.params
        n_trials, n_puffs, n_pn = puffs.shape
        if n_puffs == 0:
            msg = "a sequence needs at least one puff"
            raise ValueError(msg)
        puff_steps = p.puff_steps
        stim_steps = n_puffs * puff_steps
        n_steps = p.sequence_steps(n_puffs)
        p_spike = puffs * np.float32(p.rate_max * p.dt / 1000.0)  # (n_trials, n_puffs, n_pn)
        seeds = _per_trial_seeds(seed, n_trials)
        if seeds is not None:
            gens = [np.random.Generator(np.random.PCG64(int(sd))) for sd in seeds]
            chunk = np.empty((puff_steps, n_trials, n_pn), dtype=np.float32)

            def draw(t: int) -> np.ndarray:
                step_in_puff = t % puff_steps
                if step_in_puff == 0:
                    for k, g in enumerate(gens):
                        chunk[:, k, :] = g.random((puff_steps, n_pn), dtype=np.float32)
                return np.asarray(chunk[step_in_puff])
        else:
            rng = np.random.Generator(np.random.PCG64(int(seed)))

            def draw(t: int) -> np.ndarray:  # noqa: ARG001
                return rng.random((n_trials, n_pn), dtype=np.float32)

        bins = np.minimum(np.arange(n_steps) // puff_steps, n_puffs - 1)
        counts, pn_counts, raster = self._run(
            n_trials,
            n_steps,
            lambda t: p_spike[:, t // puff_steps, :] if t < stim_steps else None,
            draw,
            bins,
            n_puffs,
            raster_trials,
        )
        # counts: (n_puffs, n_trials * n_int) -> (n_trials, n_puffs, n_int)
        counts3 = counts.reshape(n_puffs, n_trials, self.n_int).transpose(1, 0, 2)
        kc_counts = np.ascontiguousarray(counts3[:, :, : self.n_kc])
        return SequenceResult(
            kc_counts=kc_counts,
            mbon_counts=np.ascontiguousarray(counts3[:, :, self.mbon_local]),
            apl_counts=np.ascontiguousarray(counts3[:, :, self.apl_local[0]]),
            pn_counts=np.ascontiguousarray(
                pn_counts.reshape(n_puffs, n_trials, n_pn).transpose(1, 0, 2)
            ),
            puff_active=(puffs > 0).any(axis=2),
            kc_active_fraction=(kc_counts > 0).mean(axis=2, dtype=np.float32),
            n_steps=n_steps,
            raster=raster,
        )

    def _run(  # noqa: PLR0917
        self,
        n_trials: int,
        n_steps: int,
        p_spike_at: Callable[[int], np.ndarray | None],
        draw: Callable[[int], np.ndarray],
        bins: np.ndarray,
        n_bins: int,
        raster_trials: int,
    ) -> tuple[np.ndarray, np.ndarray, SpikeRaster | None]:
        """The step loop shared by both modes.

        ``p_spike_at(t)`` is the ``(n_trials, n_pn)`` Bernoulli probability of a PN spike
        at step ``t`` (``None`` = silence), ``draw(t)`` the matching uniforms, ``bins[t]``
        the count bin of step ``t``.  Returns ``(counts (n_bins, n_trials * n_int) int16,
        pn_counts (n_bins, n_trials * n_pn) int16, raster)``.
        """
        p = self.params
        n_int, n_pn = self.n_int, self.n_pn
        pn_idx = self.connectome.pn_idx.astype(np.int32)
        int_idx = self.int_idx

        u = np.zeros((n_trials, n_int), dtype=np.float32)  # v - v_rest, mV
        g = np.zeros((n_trials, n_int), dtype=np.float32)  # synaptic variable, mV
        tmp = np.empty_like(u)
        u_flat, g_flat = u.reshape(-1), g.reshape(-1)
        counts = np.zeros((n_bins, n_trials * n_int), dtype=np.int16)
        pn_counts = np.zeros((n_bins, n_trials * n_pn), dtype=np.int16)
        delay, refr_steps = p.delay_steps, p.refractory_steps
        # Spike events (flat indices) of the last `refr_steps` steps = the refractory set.
        refr_ring: list[np.ndarray] = [_EMPTY] * refr_steps
        # Events (trial rows, global neuron columns) waiting `delay` steps for delivery.
        queue: list[tuple[np.ndarray, np.ndarray] | None] = [None] * delay
        raster_rows: list[np.ndarray] = []
        raster_cols: list[np.ndarray] = []
        raster_steps: list[np.ndarray] = []

        for t in range(n_steps):
            b = bins[t]
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
            counts[b, spk] += 1
            refr_ring[t % refr_steps] = spk
            rows = spk // n_int
            cols = int_idx[spk - rows * n_int]
            # 4. PN Poisson input
            p_spike = p_spike_at(t)
            if p_spike is not None:
                pn = np.flatnonzero(draw(t) < p_spike)
                pn_counts[b, pn] += 1
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
        return counts, pn_counts, raster

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


def _check_range(odors: np.ndarray) -> None:
    if odors.size and (odors.min() < 0 or odors.max() > 1):
        msg = "odor values must lie in [0, 1]"
        raise ValueError(msg)


def _per_trial_seeds(seed: int | np.ndarray, n_trials: int) -> np.ndarray | None:
    """The per-trial seed array, or ``None`` for a single batch-wide seed."""
    if isinstance(seed, int | np.integer):
        return None
    seeds = np.asarray(seed)
    if seeds.shape != (n_trials,):
        msg = f"one seed per trial expected, got shape {seeds.shape} for {n_trials} trials"
        raise ValueError(msg)
    return seeds


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
