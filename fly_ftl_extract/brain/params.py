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
        4.0,
        """Multiplier on `w_syn` for our subgraph. Shiu simulate the whole brain with 1;
        the isolated mushroom body needs its own value. Calibrated on the real odours of
        every fixture candidate (auditor's decision after Phase 4; the synthetic 30 %-PN
        odour is no longer used): target 8-10 % active Kenyon cells with APL, > 30 % without,
        no neuron above 1/t_refractory, median active-KC rate < 50 Hz. With encoder
        fly-odor-3 (context 6/3, no bag n-grams: sparser odours) 4.0 gives
        8.6% / 52.2%; `scripts/calibrate.py`, curve in
        docs/BENCH.md, value mirrored in docs/calibration.json.""",
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
