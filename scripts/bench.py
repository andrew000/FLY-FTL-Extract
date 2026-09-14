"""Benchmark the LIF simulation: trials/s at batch 64 / 256 / 1024, drive modes, profile.

Writes the ``bench`` and ``params`` sections of ``docs/BENCH.md``.  PLAN target: >= 200
trials/s at batch 256 for ``n_steps = (T_stim + T_silence) / dt`` steps.
"""

from __future__ import annotations

import cProfile
import io
import platform
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, load
from fly_ftl_extract.brain.lif import DriveMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docsection import update_section
from _fixtures import fixture_odors
from calibrate import SKELETON

REPO = Path(__file__).resolve().parent.parent
OUT_DOC = REPO / "docs" / "BENCH.md"
BATCHES = (64, 256, 1024)
TARGET_TRIALS_PER_S = 200.0
TARGET_BATCH = 256


def time_batch(brain: Brain, odors: np.ndarray, repeats: int) -> float:
    """Best wall time of ``repeats`` runs, seconds."""
    best = float("inf")
    for i in range(repeats):
        t0 = time.perf_counter()
        brain.simulate(odors, seed=100 + i)
        best = min(best, time.perf_counter() - t0)
    return best


def profile(brain: Brain, odors: np.ndarray, top: int = 14) -> str:
    pr = cProfile.Profile()
    pr.enable()
    brain.simulate(odors, seed=5)
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    stats.print_stats(top)
    text = buf.getvalue()
    # keep only the table, shorten paths
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("ncalls"))
    out = []
    for raw in lines[start:]:
        line = raw.replace(str(REPO) + "\\", "").replace(str(REPO) + "/", "")
        if ".venv" in line:
            head, _, tail = line.rpartition("site-packages")
            line = head.split(".venv")[0] + "site-packages" + tail if tail else line
        out.append(line.rstrip())
    return "\n".join(out)


def main() -> int:
    cx = load()
    p = DEFAULT_PARAMS
    n_pn = len(cx.pn_idx)
    real, _ = fixture_odors()
    odors = np.resize(real, (max(BATCHES), n_pn))  # real fixture odours, repeated to fill
    brain = Brain(cx, p)
    brain.simulate(odors[:16], seed=0)  # warm-up

    rows = []
    for batch in BATCHES:
        repeats = 3 if batch <= TARGET_BATCH else 2
        secs = time_batch(brain, odors[:batch], repeats)
        rows.append((batch, secs, batch / secs, secs / p.n_steps * 1e3))
        print(
            f"batch {batch:5d}: {batch / secs:7.1f} trials/s  ({secs / p.n_steps * 1e3:.3f} ms/step)"
        )

    modes: list[tuple[DriveMode, float]] = []
    for mode in ("events", "dense"):
        b = Brain(cx, p, drive=mode)
        b.simulate(odors[:16], seed=0)
        secs = time_batch(b, odors[:TARGET_BATCH], 2)
        modes.append((mode, TARGET_BATCH / secs))
        print(f"drive={mode}: {TARGET_BATCH / secs:.1f} trials/s at batch {TARGET_BATCH}")
    same = np.array_equal(
        Brain(cx, p, drive="events").simulate(odors[:32], seed=3).kc_counts,
        Brain(cx, p, drive="dense").simulate(odors[:32], seed=3).kc_counts,
    )
    prof = profile(brain, odors[:TARGET_BATCH])
    print(prof)

    at_target = next(r for r in rows if r[0] == TARGET_BATCH)
    verdict = "met" if at_target[2] >= TARGET_TRIALS_PER_S else "**not met**"
    machine = (
        f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"
    )
    lines = [
        "## 3. Speed",
        "",
        (
            f"Machine: {machine}; Python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}. One thread (numpy/scipy without BLAS parallelism in these operations)."
        ),
        (
            f"A trial = {p.n_steps} steps of {p.dt} ms (T_stim {p.t_stim:.0f} + T_silence "
            f"{p.t_silence:.0f} ms), {cx.n_neurons} neurons in the graph, of which "
            f"{brain.n_int} integrate (KC + APL + MBON), W has {brain.w.nnz} edges. Odours — real "
            "candidates from the fixtures (repeated up to batch), the best of several runs."
        ),
        "",
        "| batch | s per batch | trials/s | ms per step |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {b} | {s:.3f} | {tps:.1f} | {ms:.3f} |" for b, s, tps, ms in rows)
    lines += [
        "",
        f"PLAN target ≥ {TARGET_TRIALS_PER_S:.0f} trials/s at batch {TARGET_BATCH}: {at_target[2]:.1f} — {verdict}.",
        "",
        "### Two ways to compute the synaptic current (batch 256)",
        "",
        "| drive | trials/s |",
        "|---|---:|",
    ]
    lines.extend(f"| `{m}` | {tps:.1f} |" for m, tps in modes)
    lines += [
        "",
        (
            "`events`: from the step's (trial, neuron) events a CSR spike matrix S is built and "
            "`S @ W` is computed (sparse × sparse, the result is added into `g` by flat indices). "
            "`dense`: `W.T @ spikes.T` with a dense spike matrix (n_neurons × n_trials). "
            f"The results are bit-for-bit identical: {same}. `events` is used."
        ),
        "",
        "### Profile (cProfile, batch 256, sorted by tottime)",
        "",
        "```",
        prof,
        "```",
        "",
        (
            "Reading the profile: the ufunc calls (multiply/add/compare on (n_trials, n_int) arrays) "
            "are not shown separately by cProfile — they are part of the tottime of `simulate`; `_deliver` is "
            "building the CSR spike matrix and `csr_matmat` (the product itself is a small share, the rest is "
            "scipy's constructor checks); Poisson generation (`Generator.random`) does not make "
            "the top. The bottleneck is neither matmul nor Poisson but the per-step Python overhead."
        ),
    ]
    update_section(OUT_DOC, "bench", "\n".join(lines), SKELETON)
    update_section(
        OUT_DOC, "params", "## 1a. Parameters (`BrainParams`)\n\n" + p.describe(), SKELETON
    )
    print(f"wrote the bench and params sections of {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
