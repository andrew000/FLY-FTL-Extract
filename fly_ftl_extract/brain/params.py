"""Parameters of the leaky integrate-and-fire mushroom body.

Every number lives here, with its unit and its source.  The neuron and synapse constants
are those of Shiu et al. 2024 (Nature 634, 210–219, "A Drosophila computational brain model
reveals sensorimotor processing"), read from ``model.py`` of the authors' repository
https://github.com/philshiu/Drosophila_brain_model (``default_params``).  Where our model
departs from that file the field's ``doc`` says so; the full list of differences is in
``docs/BENCH.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

_STEP_TOLERANCE = 1e-6


def _p(default: float, doc: str) -> float:
    return field(default=default, metadata={"doc": " ".join(doc.split())})


@dataclass(frozen=True)
class BrainParams:
    """Constants of the LIF simulation (unit and source in each field's ``doc`` metadata).

    Shiu et al. integrate ``dv/dt = (v_0 - v + g) / t_mbr`` and ``dg/dt = -g / tau`` with
    brian2; a presynaptic spike adds ``w_syn * syn_count * sign`` to ``g`` after ``t_dly``,
    a spike resets ``v`` and ``g`` and freezes both for ``t_rfc``.  We do the same with a
    forward-Euler step of ``dt`` on a batch of trials.  ``describe()`` renders the table.
    """

    v_rest: float = _p(-52.0, "Resting potential, mV. Shiu `v_0` (Kakaria & de Bivort 2017).")
    v_reset: float = _p(-52.0, "Potential after a spike, mV. Shiu `v_rst`.")
    v_th: float = _p(-45.0, "Spike threshold, mV; a spike is `v > v_th`. Shiu `v_th`.")
    tau_m: float = _p(20.0, "Membrane time constant, ms. Shiu `t_mbr` (0.002 µF · 10 MΩ).")
    tau_syn: float = _p(
        5.0, "Decay time constant of the synaptic variable `g`, ms. Shiu `tau` (Jürgensen 2021)."
    )
    t_refractory: float = _p(
        2.2,
        """Refractory period, ms; `v` and `g` are frozen, inputs still accumulate in `g`.
        Shiu `t_rfc` (Lazar et al. 2021).""",
    )
    t_delay: float = _p(
        1.8,
        """Synaptic delay, ms: a spike reaches its targets `t_delay` later. Shiu `t_dly`
        (Paul et al. 2015). Not in CLAUDE.md; taken from the model.""",
    )
    w_syn: float = _p(
        0.275,
        """Potential added to `g` per synapse, mV (× `syn_count` × sign). Shiu `w_syn`,
        the model's one free parameter.""",
    )
    syn_scale: float = _p(
        5.0,
        """Multiplier on `w_syn` for our subgraph. Shiu simulate the whole brain with 1;
        the isolated mushroom body needs its own value. Calibrated together with
        `apl_scale` on the real odour *sequences* of every fixture candidate in the temporal
        code (`Brain.simulate_sequence`, auditor's decisions after Phases 4 and 5): target
        8-10 % active Kenyon cells per non-empty 20 ms puff with APL, APL sparsening >= 2x,
        one PN spike alone must not fire a KC (that happens from 11.7), no neuron above
        1/t_refractory, median active-KC rate < 50 Hz. With 20 ms puffs and fly-odor-4
        the grid chose (10.0, 0.3): 9.2 % per puff, ratio 3.0. With 40 ms puffs and
        fly-odor-5 (PN driven at 0.96 of rate_max) it chooses (5.0, 0.5): 8.5 % per puff,
        28.9 % without APL (ratio 3.4) — more spikes per PN need less gain per synapse.
        Phase 3's 4.0 was for the single-odour mode with encoder fly-odor-3.
        `scripts/calibrate.py`, grid in docs/BENCH.md §2, values mirrored in
        docs/calibration.json.""",
    )
    apl_scale: float = _p(
        0.5,
        """Extra multiplier on the APL's output synapses (APL→KC, APL→PN, APL→MBON…), on top
        of `syn_scale`. 1.0 = the FlyWire counts as they are (Shiu et al. scale every synapse
        alike). Introduced for the temporal code (deviation from PLAN, docs/BENCH.md §2):
        the single APL receives 1713 PN and 56261 KC synapses and, with every synapse
        scaled alike, fires 7-8 spikes per 20 ms puff (near its 455 Hz refractory limit)
        for as long as any odour is present; the Kenyon cells then respond only at odour
        onset (10 % in the first puff, 0.4-2 % in every later one) and no `syn_scale`
        between 1 and 40 lifts the per-puff activity above 1.6 %. A value below 1 lets
        the APL regulate instead of clamping: 0.3 with 20 ms puffs, 0.5 with 40 ms puffs
        and fly-odor-5 (it still cuts the KC activity 3.4x and fires ~13 spikes per 40 ms
        puff). Calibrated with `syn_scale` in `scripts/calibrate.py`.""",
    )
    dt: float = _p(0.1, "Integration step, ms. brian2's `defaultclock.dt`, as used by Shiu.")
    rate_max: float = _p(
        200.0,
        """Firing rate of a projection neuron at odour value 1.0, Hz. CLAUDE.md; Shiu drive
        their input neurons at `r_poi = 150 Hz`, which is odour value 0.75 here.""",
    )
    t_stim: float = _p(
        100.0,
        """Duration of the odour (PN Poisson input), ms. CLAUDE.md said 50 ms; the auditor
        after Phase 3 set 100 ms because at 50 ms the same odour with two seeds gave a
        Kenyon-cell Jaccard of 0.40 (< 0.5 required); at 100 ms it is > 0.5 (docs/BENCH.md §2).""",
    )
    t_silence: float = _p(
        10.0, "Silence after the odour while the last spikes propagate, ms. CLAUDE.md."
    )
    puff_ms: float = _p(
        40.0,
        """Duration of one puff in the temporal code (`Brain.simulate_sequence`), ms: every
        slot of the token window is presented for `puff_ms`, the next slot follows without
        silence, `t_silence` closes the trial. The auditor's decision after Phase 5 set
        20 ms (one `tau_m`: the previous token's depolarisation has decayed to 1/e when the
        next arrives, a token six slots back has faded to e^-6). At 20 ms a puff delivers
        only ~3 spikes per active PN and the Kenyon-cell code is not reproducible: the same
        candidate under two seeds shares 34 % of its active (puff, KC) states, and the
        readout plateaus at val F1 0.96 (attempts 7-8, docs/METRICS.md §7). 40 ms = 2
        `tau_m`: ~6 spikes per active PN, same-candidate Jaccard 0.45, and on the
        40k-row lever harness the readout goes 0.9415 -> 0.9631 (30 ms: 0.9574); the
        previous token still carries into the next puff (e^-2 = 0.14 of its peak). Cost:
        4100 instead of 2100 steps per trial.""",
    )

    def _steps(self, duration_ms: float, name: str) -> int:
        steps = duration_ms / self.dt
        if abs(steps - round(steps)) > _STEP_TOLERANCE:
            msg = f"{name} = {duration_ms} ms is not a whole number of dt = {self.dt} ms steps"
            raise ValueError(msg)
        return round(steps)

    @property
    def stim_steps(self) -> int:
        """Number of steps with odour input."""
        return self._steps(self.t_stim, "t_stim")

    @property
    def n_steps(self) -> int:
        """Number of steps in one trial (odour + silence)."""
        return self.stim_steps + self._steps(self.t_silence, "t_silence")

    @property
    def puff_steps(self) -> int:
        """Steps of one puff in the temporal code."""
        return self._steps(self.puff_ms, "puff_ms")

    def sequence_steps(self, n_puffs: int) -> int:
        """Number of steps in one temporal-code trial (`n_puffs` puffs + silence)."""
        return n_puffs * self.puff_steps + self._steps(self.t_silence, "t_silence")

    @property
    def refractory_steps(self) -> int:
        """Refractory period in steps."""
        return self._steps(self.t_refractory, "t_refractory")

    @property
    def delay_steps(self) -> int:
        """Synaptic delay in steps (at least one: the current comes from earlier spikes)."""
        steps = self._steps(self.t_delay, "t_delay")
        if steps < 1:
            msg = "t_delay must be at least one dt"
            raise ValueError(msg)
        return steps

    @property
    def max_rate_hz(self) -> float:
        """Highest firing rate a refractory neuron can reach, Hz."""
        return 1000.0 / self.t_refractory

    def describe(self) -> str:
        """Markdown table of every field with its value and documentation."""
        lines = ["| parameter | value | meaning |", "|---|---:|---|"]
        lines.extend(
            f"| `{f.name}` | {getattr(self, f.name)} | {f.metadata['doc']} |" for f in fields(self)
        )
        return "\n".join(lines)


DEFAULT_PARAMS = BrainParams()
