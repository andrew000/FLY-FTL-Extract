"""Benchmark the LIF simulation: single-odour mode (Phase 3) and the temporal code.

Writes the ``params``, ``bench`` and ``bench_sequence`` sections of ``docs/BENCH.md`` and
``docs/bench_sequence.json`` (read by ``scripts/train.py`` for METRICS.md §7).

Phase 3 target was ≥ 200 trials/s at batch 256 for one process; the auditor lifted that
threshold for the temporal code (a trial is now ~2100 steps instead of 1100).  What matters
for the training budget is the end-to-end throughput of ``scripts/train.py``'s
``simulate_all`` (several processes, chunked, including the compressed cache write), so
that is measured here at batch 64 / 128 / 256 with 16 and 30 workers.
"""

from __future__ import annotations

import cProfile
import io
import json
import platform
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, BrainParams, Connectome, load
from fly_ftl_extract.brain.lif import DriveMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _docsection import update_section
from _fixtures import fixture_odors
from calibrate import SKELETON
from train import cached_states, simulate_all

REPO = Path(__file__).resolve().parent.parent
OUT_DOC = REPO / "docs" / "BENCH.md"
OUT_JSON = REPO / "docs" / "bench_sequence.json"
BATCHES = (64, 256, 1024)
SEQ_BATCHES = (64, 128, 256)
WORKER_COUNTS = (16, 30)
PARALLEL_TRIALS = 8192
CACHE_TRIALS = 16384
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


def time_sequence(brain: Brain, puffs: np.ndarray, repeats: int) -> float:
    best = float("inf")
    for i in range(repeats):
        t0 = time.perf_counter()
        brain.simulate_sequence(puffs, seed=100 + i)
        best = min(best, time.perf_counter() - t0)
    return best


def profile_text(pr: cProfile.Profile, top: int = 14) -> str:
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    stats.print_stats(top)
    text = buf.getvalue()
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


def profile(brain: Brain, odors: np.ndarray) -> str:
    pr = cProfile.Profile()
    pr.enable()
    brain.simulate(odors, seed=5)
    pr.disable()
    return profile_text(pr)


def profile_sequence(brain: Brain, puffs: np.ndarray) -> str:
    pr = cProfile.Profile()
    pr.enable()
    brain.simulate_sequence(puffs, seed=5)
    pr.disable()
    return profile_text(pr)


def bench_single_odour(
    cx: Connectome, p: BrainParams, brain: Brain, real_puffs: np.ndarray
) -> None:
    """The Phase 3 section: ``simulate`` on the union of a candidate's puffs."""
    n_pn = len(cx.pn_idx)
    real = real_puffs.max(axis=1)
    odors = np.resize(real, (max(BATCHES), n_pn))
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
    at_target = next(r for r in rows if r[0] == TARGET_BATCH)
    verdict = "met" if at_target[2] >= TARGET_TRIALS_PER_S else "**not met**"
    machine = (
        f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"
    )
    lines = [
        "## 3. Speed: single-odour mode (Phase 3, `Brain.simulate`)",
        "",
        (
            f"Machine: {machine}; Python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}. One thread (numpy/scipy without BLAS parallelism in these operations)."
        ),
        (
            f"A trial = {p.n_steps} steps of {p.dt} ms (T_stim {p.t_stim:.0f} + T_silence "
            f"{p.t_silence:.0f} ms), {cx.n_neurons} neurons in the graph, of which "
            f"{brain.n_int} integrate (KC + APL + MBON), W has {brain.w.nnz} edges. Odours — the union of puffs of "
            "real candidates from the fixtures (repeated up to batch), the best of several runs. "
            "This mode stays for the Phase 3 tests; production is temporal coding (§3a)."
        ),
        "",
        "| batch | s per batch | trials/s | ms per step |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {b} | {s:.3f} | {tps:.1f} | {ms:.3f} |" for b, s, tps, ms in rows)
    lines += [
        "",
        (
            f"PLAN target (Phase 3) ≥ {TARGET_TRIALS_PER_S:.0f} trials/s at batch {TARGET_BATCH}: "
            f"{at_target[2]:.1f} — {verdict}."
        ),
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
    ]
    update_section(OUT_DOC, "bench", "\n".join(lines), SKELETON)


def bench_sequence(p: BrainParams, brain: Brain, real_puffs: np.ndarray) -> None:
    """The temporal-code section: ``simulate_sequence`` alone and ``simulate_all`` end to end."""
    n_puffs = real_puffs.shape[1]
    puffs = np.resize(real_puffs, (max(SEQ_BATCHES, default=256), n_puffs, real_puffs.shape[2]))
    brain.simulate_sequence(puffs[:16], seed=0)  # warm-up
    n_steps = p.sequence_steps(n_puffs)
    single = []
    for batch in SEQ_BATCHES:
        secs = time_sequence(brain, puffs[:batch], 2)
        single.append(
            {
                "batch": batch,
                "seconds": secs,
                "trials_per_s": batch / secs,
                "ms_per_step": secs / n_steps * 1e3,
            }
        )
        print(
            f"sequence batch {batch:4d}: {batch / secs:7.1f} trials/s ({secs / n_steps * 1e3:.3f} ms/step)",
            flush=True,
        )
    prof = profile_sequence(brain, puffs[:128])

    big = np.resize(real_puffs, (CACHE_TRIALS, n_puffs, real_puffs.shape[2]))
    parallel = []
    for workers in WORKER_COUNTS:
        for batch in SEQ_BATCHES:
            t0 = time.perf_counter()
            simulate_all(brain, big[:PARALLEL_TRIALS], 7, "bench", batch=batch, workers=workers)
            secs = time.perf_counter() - t0
            parallel.append(
                {
                    "workers": workers,
                    "batch": batch,
                    "seconds": secs,
                    "trials_per_s": PARALLEL_TRIALS / secs,
                }
            )
    best = max(parallel, key=lambda r: r["trials_per_s"])

    tmp = Path(tempfile.mkdtemp(prefix="bench_cache_"))
    try:
        t0 = time.perf_counter()
        counts = cached_states(brain, big, 7, "bench cache", tmp)
        secs = time.perf_counter() - t0
        size_mb = sum(f.stat().st_size for f in tmp.iterdir()) / 1e6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cache_row = {
        "trials": CACHE_TRIALS,
        "seconds": secs,
        "trials_per_s": CACHE_TRIALS / secs,
        "uint8_gb": counts.nbytes / 1e9,
        "npz_mb": size_mb,
    }
    print(
        f"cached_states end to end: {CACHE_TRIALS} trials in {secs:.0f} s "
        f"({CACHE_TRIALS / secs:.0f} trials/s), {counts.nbytes / 1e9:.2f} GB -> {size_mb:.0f} MB",
        flush=True,
    )

    OUT_JSON.write_text(
        json.dumps(
            {
                "n_steps": n_steps,
                "n_puffs": n_puffs,
                "puff_ms": p.puff_ms,
                "syn_scale": p.syn_scale,
                "apl_scale": p.apl_scale,
                "single": single,
                "parallel": parallel,
                "cache": cache_row,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "## 3a. Speed: temporal coding (`Brain.simulate_sequence`)",
        "",
        (
            f"A trial = {n_puffs} puffs × {p.puff_ms:.0f} ms + {p.t_silence:.0f} ms of silence = {n_steps} steps "
            f"of {p.dt} ms; syn_scale {p.syn_scale}, apl_scale {p.apl_scale}. Odours — real "
            "candidate sequences from the fixtures, repeated up to the needed count. The 200 trials/s "
            "per process threshold was lifted by the reviewer's decision after Phase 5; the budget is ≤ 2 h of brain per full run of "
            "`scripts/train.py`."
        ),
        "",
        "### One process",
        "",
        "| batch | s per batch | trials/s | ms per step |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['batch']} | {r['seconds']:.3f} | {r['trials_per_s']:.1f} | {r['ms_per_step']:.3f} |"
        for r in single
    )
    lines += [
        "",
        "### Several processes (`scripts/train.py::simulate_all`, chunk = 8 batches, one seed)",
        "",
        (
            f"{PARALLEL_TRIALS} trials per configuration, including the pool start-up (every process reads the connectome). "
            "Scaling is far from linear: one process's working set (the arrays `u`, `g`, `tmp` "
            "of size batch × 2646 float32) does not fit in the cache when there are many processes, and "
            "memory limits the speed — hence a smaller batch wins with 30 processes."
        ),
        "",
        "| processes | batch | s | trials/s |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['workers']} | {r['batch']}{' **←**' if r is best else ''} | {r['seconds']:.1f} "
        f"| {r['trials_per_s']:.0f} |"
        for r in parallel
    )
    lines += [
        "",
        (
            f"Best: {best['workers']} processes × batch {best['batch']} = {best['trials_per_s']:.0f} trials/s. "
            f"`cached_states` end to end ({CACHE_TRIALS} trials, simulation + `savez_compressed`): "
            f"{cache_row['seconds']:.0f} s = {cache_row['trials_per_s']:.0f} trials/s; uint8 states "
            f"{cache_row['uint8_gb']:.2f} GB → {cache_row['npz_mb']:.0f} MB on disk."
        ),
        "",
        "### Profile (cProfile, `simulate_sequence`, batch 128, sorted by tottime)",
        "",
        "```",
        prof,
        "```",
        "",
        (
            "Reading the profile: the ufunc calls (multiply/add/compare on (n_trials, n_int) arrays) "
            "are not shown separately by cProfile — they are part of the tottime of `_run`; `_deliver` is building the "
            "CSR spike matrix and `csr_matmat`; Poisson generation per puff (`Generator.random`) is "
            "a small share. The bottleneck is the per-step Python overhead, so steps ×2 ≈ time ×2."
        ),
    ]
    update_section(OUT_DOC, "bench_sequence", "\n".join(lines), SKELETON)


def main() -> int:
    cx = load()
    p = DEFAULT_PARAMS
    real_puffs, _ = fixture_odors()
    brain = Brain(cx, p)
    bench_single_odour(cx, p, brain, real_puffs)
    bench_sequence(p, brain, real_puffs)
    update_section(
        OUT_DOC, "params", "## 1a. Parameters (`BrainParams`)\n\n" + p.describe(), SKELETON
    )
    print(f"wrote the bench, bench_sequence and params sections of {OUT_DOC} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
