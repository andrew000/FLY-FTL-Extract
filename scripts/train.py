"""Dopamine training: odours → mushroom body → KC states → two linear MBON readouts.

Reads ``data/dataset/{keys,kwargs}.npz`` (from ``scripts/make_dataset.py``), simulates the
brain (train rows with ``TRAIN_SEEDS`` seeds each as augmentation, val/test rows with their
production seeds), trains a readout per feature mode with the delta rule, keeps the mode
with the best validation F1, chooses the resniff threshold θ on val (≤ 10 % resniffs),
evaluates on test with and without resniff, runs the whole fly on the golden fixtures, and
writes ``data/mbon_weights.npz`` and ``docs/METRICS.md``.

KC states are cached under ``.cache/brain_states/`` (keyed by brain hash, encoder version
and the odours) so that training experiments do not re-run the brain.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, load
from fly_ftl_extract.dopamine import (
    FEATURE_MODES,
    FeatureMode,
    Judge,
    MbonWeights,
    Readout,
    Scores,
    TrainConfig,
    resniff_threshold,
    score,
    sniff_seed,
    train_readout,
    vote,
    weights_path,
)
from fly_ftl_extract.dopamine.readout import TrainLog
from fly_ftl_extract.ftl.model import kwargs_from_key
from fly_ftl_extract.odor.encoder import ENCODER_VERSION
from fly_ftl_extract.reference.extractor import key_occurrences

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixtures import fixture_files, fixture_names

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "fly_ftl_extract" / "data" / "dataset"
CACHE_DIR = REPO / ".cache" / "brain_states"
OUT_DOC = REPO / "docs" / "METRICS.md"
ATTEMPTS = REPO / "docs" / "attempts.jsonl"
PROXY_JSON = REPO / "docs" / "encoder_proxy.json"
TRAIN_SEEDS = (101, 102, 103)  # three sniffs of every training odour
MAX_RESNIFF = 3
RESNIFF_FRACTION = 0.10
BATCH = 256
WORST = 10


class Table:
    """One dataset table (``keys`` or ``kwargs``) with its splits."""

    def __init__(self, name: str, dataset_dir: Path) -> None:
        self.name = name
        with np.load(dataset_dir / f"{name}.npz") as z:
            self.odors = z["odors"]
            self.label = z["label"]
            self.split = z["split"]
            self.seed = z["seed"]
            self.snippet = z["snippet"]
            self.index = z["index"]
        with (dataset_dir / f"{name}.jsonl").open(encoding="utf-8") as fh:
            self.meta = [json.loads(line) for line in fh]

    def rows(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)


WORKERS = max(1, (os.cpu_count() or 2) - 2)
_WORKER_BRAIN: Brain | None = None


def _worker_brain() -> Brain:
    """One brain per worker process (the connectome is loaded once per process)."""
    global _WORKER_BRAIN  # noqa: PLW0603
    if _WORKER_BRAIN is None:
        _WORKER_BRAIN = Brain(load(), DEFAULT_PARAMS)
    return _WORKER_BRAIN


def _simulate_chunk(odors: np.ndarray, seeds: np.ndarray | int) -> np.ndarray:
    """KC counts of a chunk that is a whole number of batches (so results do not depend
    on how the work was split across processes: every batch sees the same odours and,
    in single-seed mode, the same random stream)."""
    brain = _worker_brain()
    counts = np.zeros((len(odors), brain.n_kc), dtype=np.uint8)
    for start in range(0, len(odors), BATCH):
        sl = slice(start, start + BATCH)
        seed = seeds if isinstance(seeds, int) else seeds[sl]
        counts[sl] = np.minimum(brain.simulate(odors[sl], seed).kc_counts, 255)
    return counts


def simulate_all(
    brain: Brain, odors: np.ndarray, seeds: np.ndarray | int, label: str
) -> np.ndarray:
    """KC spike counts (uint8) of every odour, batched over ``WORKERS`` processes."""
    t0 = time.perf_counter()
    chunk = BATCH * 8
    starts = list(range(0, len(odors), chunk))
    if len(starts) <= 1 or WORKERS == 1:
        counts = _simulate_chunk(odors, seeds)
    else:
        parts = [
            (odors[a : a + chunk], seeds if isinstance(seeds, int) else seeds[a : a + chunk])
            for a in starts
        ]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(
                pool.map(_simulate_chunk, [o for o, _ in parts], [sd for _, sd in parts])
            )
        counts = np.concatenate(results)
    assert counts.shape == (len(odors), brain.n_kc)
    rate = len(odors) / max(time.perf_counter() - t0, 1e-9)
    print(
        f"  {label}: {len(odors)} trials, {rate:.0f} trials/s over {WORKERS} processes", flush=True
    )
    return counts


def cached_states(
    brain: Brain, odors: np.ndarray, seeds: np.ndarray | int, label: str, cache_dir: Path
) -> np.ndarray:
    seed_bytes = (
        np.asarray(seeds).astype(np.uint64).tobytes()
        if not isinstance(seeds, int)
        else str(seeds).encode()
    )
    key = hashlib.sha256(
        brain.hash.encode() + ENCODER_VERSION.encode() + odors.tobytes() + seed_bytes
    ).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{label.replace(' ', '_')}_{key}.npz"
    if path.exists():
        with np.load(path) as z:
            return np.asarray(z["counts"])
    counts = simulate_all(brain, odors, seeds, label)
    np.savez_compressed(path, counts=counts)
    return counts


@dataclass
class Evaluation:
    """Test-split results of one readout."""

    plain: Scores
    resniffed: Scores
    resniff_fraction: float
    worst: list[dict[str, object]]


def evaluate(
    brain: Brain,
    table: Table,
    readout: Readout,
    *,
    theta: float,
    counts_test: np.ndarray,
    cache_dir: Path,
    max_resniff: int,
) -> Evaluation:
    """Test scores without and with resniff, the resniff fraction and the worst examples."""
    test = table.rows(2)
    y = table.label[test]
    margin1 = readout.margin(counts_test)
    plain = score(margin1 > 0, y)
    unsure = np.flatnonzero(np.abs(margin1) < theta)
    margins = np.zeros((len(test), 1 + max_resniff), dtype=np.float32)
    margins[:, 0] = margin1
    for sniff in range(1, max_resniff + 1):
        if len(unsure) == 0:
            break
        seeds = np.array(
            [sniff_seed(int(s), sniff) for s in table.seed[test][unsure]], dtype=np.uint64
        )
        counts = cached_states(
            brain, table.odors[test][unsure], seeds, f"{table.name} test resniff {sniff}", cache_dir
        )
        margins[unsure, sniff] = readout.margin(counts)
    summed = vote(margins)
    with_resniff = score(summed > 0, y)
    wrong = np.flatnonzero((summed > 0) != y)
    worst_idx = wrong[np.argsort(-np.abs(summed[wrong]))][:WORST]
    worst: list[dict[str, object]] = [
        {
            "row": int(test[i]),
            "label": bool(y[i]),
            "margin": float(summed[i]),
            "sniffs": int(1 + (np.abs(margin1[i]) < theta) * max_resniff),
            **{k: v for k, v in table.meta[test[i]].items() if k != "label"},
        }
        for i in worst_idx
    ]
    return Evaluation(plain, with_resniff, len(unsure) / max(len(test), 1), worst)


@dataclass
class FixturesResult:
    """The whole fly against the teacher on every parsable fixture file."""

    files_total: int
    files_ok: int
    mismatches: list[str]
    seconds: float


def fixtures_check(weights: MbonWeights) -> FixturesResult:
    """Run the whole fly on every fixture file; compare with the teacher."""
    files_ok, files_total = 0, 0
    mismatches: list[str] = []
    t0 = time.perf_counter()
    for name in fixture_names():
        judge: Judge | None = None
        for f in fixture_files(name):
            if f.source is None:
                continue
            try:
                occurrences = key_occurrences(f.path, f.source, f.options)
            except SyntaxError:
                continue
            if judge is None:
                judge = Judge(f.options, weights=weights)
            expected = {
                (
                    (k.source_location.line, k.source_location.column),
                    k.key,
                    tuple(sorted(kwargs_from_key(k))),
                )
                for k in occurrences
                if k.source_location is not None
            }
            judged = judge.judge_source(f.source, f.source.encode("utf-8"))
            got = {(k.call_position, k.key_name, tuple(sorted(k.placeable))) for k in judged.keys}
            files_total += 1
            if got == expected:
                files_ok += 1
            else:
                mismatches.append(
                    f"{name}/{f.path}: missing {sorted(expected - got)!r}, "
                    f"extra {sorted(got - expected)!r}"
                )
    return FixturesResult(files_total, files_ok, mismatches, time.perf_counter() - t0)


def md_scores(s: Scores) -> str:
    return (
        f"P {s.precision:.4f} · R {s.recall:.4f} · F1 {s.f1:.4f} "
        f"(tp {s.tp}, fp {s.fp}, fn {s.fn}, tn {s.tn})"
    )


@dataclass
class TaskResult:
    """Everything train.py learned about one task (keys or kwargs)."""

    name: str
    mode: FeatureMode
    theta: float
    readout: Readout
    val: Scores
    evaluation: Evaluation
    logs: dict[str, TrainLog]
    rows: dict[str, int]
    kc_active: dict[str, float]

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "theta": self.theta,
            "val": self.val.as_dict(),
            "test": self.evaluation.plain.as_dict(),
            "test_resniff": self.evaluation.resniffed.as_dict(),
            "resniff_fraction": self.evaluation.resniff_fraction,
            "modes": {
                m: {
                    "best_val_f1": log.best_val_f1,
                    "best_epoch": log.best_epoch,
                    "epochs": len(log.epochs),
                    "loss": log.epochs[-1]["loss"] if log.epochs else None,
                }
                for m, log in self.logs.items()
            },
            "rows": self.rows,
            "kc_active_fraction": self.kc_active,
        }


def train_task(
    name: str,
    brain: Brain,
    *,
    dataset: Path,
    cache: Path,
    modes: list[str],
    config: TrainConfig,
    max_resniff: int = MAX_RESNIFF,
) -> tuple[TaskResult, dict[str, float]]:
    table = Table(name, dataset)
    tr, va, te = table.rows(0), table.rows(1), table.rows(2)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    counts_tr = np.concatenate(
        [
            cached_states(brain, table.odors[tr], s, f"{name} train seed {s}", cache)
            for s in TRAIN_SEEDS
        ]
    )
    y_tr = np.tile(table.label[tr], len(TRAIN_SEEDS))
    counts_va = cached_states(brain, table.odors[va], table.seed[va], f"{name} val", cache)
    counts_te = cached_states(brain, table.odors[te], table.seed[te], f"{name} test", cache)
    timings["brain_s"] = time.perf_counter() - t0
    kc_active = {
        "train": float((counts_tr > 0).mean()),
        "val": float((counts_va > 0).mean()),
        "test": float((counts_te > 0).mean()),
    }
    print(f"{name}: KC active fraction {kc_active}", flush=True)

    t1 = time.perf_counter()
    logs: dict[str, TrainLog] = {}
    candidates: dict[str, Readout] = {}
    for mode in modes:
        assert mode in FEATURE_MODES
        readout, log = train_readout(
            counts_tr,
            y_tr,
            counts_va,
            table.label[va],
            mode=mode,
            config=config,
        )
        logs[mode] = log
        candidates[mode] = readout
        print(
            f"  {name}/{mode}: best val F1 {log.best_val_f1:.4f} at epoch {log.best_epoch} "
            f"({len(log.epochs)} epochs)",
            flush=True,
        )
    best_mode = max(modes, key=lambda m: logs[m].best_val_f1)
    readout = candidates[best_mode]
    timings["train_s"] = time.perf_counter() - t1
    theta = resniff_threshold(readout.margin(counts_va), RESNIFF_FRACTION)
    evaluation = evaluate(
        brain,
        table,
        readout,
        theta=theta,
        counts_test=counts_te,
        cache_dir=cache,
        max_resniff=max_resniff,
    )
    val_scores = score(readout.margin(counts_va) > 0, table.label[va])
    print(
        f"  {name}: mode {best_mode}, θ {theta:.4f}; test {md_scores(evaluation.plain)}; "
        f"with resniff {md_scores(evaluation.resniffed)} "
        f"(resniff {evaluation.resniff_fraction:.1%})",
        flush=True,
    )
    result = TaskResult(
        name=name,
        mode=readout.mode,
        theta=theta,
        readout=readout,
        val=val_scores,
        evaluation=evaluation,
        logs=logs,
        rows={
            "train": len(tr),
            "train_with_sniffs": len(y_tr),
            "val": len(va),
            "test": len(te),
            "positive_test": int(table.label[te].sum()),
        },
        kc_active=kc_active,
    )
    return result, timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=weights_path())
    parser.add_argument("--modes", nargs="+", default=list(FEATURE_MODES))
    parser.add_argument("--lr", type=float, default=TrainConfig().lr)
    parser.add_argument("--l2", type=float, default=TrainConfig().l2)
    parser.add_argument("--epochs", type=int, default=TrainConfig().max_epochs)
    parser.add_argument("--attempt", default="", help="label of this attempt for METRICS.md §5")
    parser.add_argument("--max-resniff", type=int, default=MAX_RESNIFF)
    args = parser.parse_args(argv)
    config = TrainConfig(lr=args.lr, l2=args.l2, max_epochs=args.epochs)
    max_resniff = args.max_resniff

    t_start = time.perf_counter()
    dataset_meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
    if dataset_meta["encoder_version"] != ENCODER_VERSION:
        print(
            f"dataset was encoded with {dataset_meta['encoder_version']}, "
            f"current {ENCODER_VERSION}: regenerate",
            file=sys.stderr,
        )
        return 2
    brain = Brain(load(), DEFAULT_PARAMS)
    print(f"brain {brain.hash}, encoder {ENCODER_VERSION}", flush=True)

    results: dict[str, TaskResult] = {}
    timings: dict[str, float] = {}
    for name in ("keys", "kwargs"):
        result, task_timings = train_task(
            name,
            brain,
            dataset=args.dataset,
            cache=args.cache,
            modes=args.modes,
            config=config,
            max_resniff=max_resniff,
        )
        results[name] = result
        timings.update({f"{name}_{k}": v for k, v in task_timings.items()})

    metrics: dict[str, object] = {
        "encoder_version": ENCODER_VERSION,
        "brain_hash": brain.hash,
        "train_config": asdict(config),
        "train_seeds": TRAIN_SEEDS,
        "max_resniff": max_resniff,
        "resniff_fraction_target": RESNIFF_FRACTION,
        "dataset": {
            k: dataset_meta[k]
            for k in ("generator_version", "seed", "snippets", "balanced_counts", "timing_s")
        },
        "results": {k: v.summary() for k, v in results.items()},
    }
    weights = MbonWeights(
        key=results["keys"].readout,
        kwarg=results["kwargs"].readout,
        theta_key=results["keys"].theta,
        theta_kwarg=results["kwargs"].theta,
        brain_hash=brain.hash,
        encoder_version=ENCODER_VERSION,
        metrics=metrics,
        max_resniff=max_resniff,
    )
    fixtures = fixtures_check(weights)
    metrics["fixtures"] = {
        "files_total": fixtures.files_total,
        "files_ok": fixtures.files_ok,
        "seconds": fixtures.seconds,
    }
    weights = dataclasses.replace(weights, metrics=metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights.save(args.out)
    timings["total_s"] = time.perf_counter() - t_start
    record_attempt(
        args.attempt,
        results=results,
        fixtures=fixtures,
        config=config,
        dataset_meta=dataset_meta,
        weights_resniff=max_resniff,
    )
    print(
        f"fixtures: {fixtures.files_ok}/{fixtures.files_total} files match the teacher",
        flush=True,
    )
    for m in fixtures.mismatches:
        print("  MISMATCH " + m)
    write_doc(
        results=results,
        brain_hash=brain.hash,
        fixtures=fixtures,
        timings=timings,
        dataset_meta=dataset_meta,
        out=args.out,
        config=config,
    )
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes) and {OUT_DOC}")
    return 0


def record_attempt(
    label: str,
    *,
    results: dict[str, TaskResult],
    fixtures: FixturesResult,
    config: TrainConfig,
    dataset_meta: dict[str, object],
    weights_resniff: int = MAX_RESNIFF,
) -> None:
    """Append one row to docs/attempts.jsonl (what changed -> what came out)."""
    keys, kwargs = results["keys"].evaluation, results["kwargs"].evaluation
    row = {
        "label": label or "(no label)",
        "encoder_version": ENCODER_VERSION,
        "generator_version": dataset_meta["generator_version"],
        "snippets": dataset_meta["snippets"],
        "train_config": asdict(config),
        "max_resniff": weights_resniff,
        "modes": {k: v.mode for k, v in results.items()},
        "keys_plain": [keys.plain.precision, keys.plain.recall],
        "keys_resniff": [keys.resniffed.precision, keys.resniffed.recall],
        "kwargs_resniff": [kwargs.resniffed.precision, kwargs.resniffed.recall],
        "fixtures": [fixtures.files_ok, fixtures.files_total],
    }
    with ATTEMPTS.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def attempts_table() -> list[str]:
    if not ATTEMPTS.exists():
        return []
    rows = [json.loads(line) for line in ATTEMPTS.read_text(encoding="utf-8").splitlines() if line]
    lines = [
        (
            "| # | change | encoder | snippets | keys test P/R (no resniff) | keys test P/R (resniff) "
            "| kwargs test P/R | fixtures |"
        ),
        "|---:|---|---|---:|---|---|---|---|",
    ]
    lines.extend(
        f"| {i} | {r['label']} | {r['encoder_version']} | {r['snippets']} "
        f"| {r['keys_plain'][0]:.4f} / {r['keys_plain'][1]:.4f} "
        f"| {r['keys_resniff'][0]:.4f} / {r['keys_resniff'][1]:.4f} "
        f"| {r['kwargs_resniff'][0]:.4f} / {r['kwargs_resniff'][1]:.4f} "
        f"| {r['fixtures'][0]}/{r['fixtures'][1]} |"
        for i, r in enumerate(rows, 1)
    )
    return lines


def proxy_section() -> list[str]:
    """METRICS.md §6 from docs/encoder_proxy.json (scripts/encoder_proxy.py)."""
    if not PROXY_JSON.exists():
        return []
    data = json.loads(PROXY_JSON.read_text(encoding="utf-8"))
    lines = [
        "",
        "## 6. Encoder proxy: separability of the odours themselves (no brain)",
        "",
        (
            f"`scripts/encoder_proxy.py`: logistic regression (the same `dan_update`) on the 124-dimensional "
            f"odours of {data['probe_rows']} candidates from {data['snippets']} separate snippets "
            f"(seed {data['seed']}, balance 1:3, val {data['val_rows']} rows by snippet). This is an upper bound "
            "for any readout after the noisy brain: information destroyed by the hash cannot come back."
        ),
        "",
        "| encoder variant | buckets (PN) | active PN | val F1 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['name']}{' (top-' + str(r['top_k']) + ')' if r['top_k'] else ''} | {r['n_pn']} "
        f"| {r['active_pn']:.1f} | {r['val_f1']:.4f} |"
        for r in data["rows"]
    )
    return lines


def write_doc(
    *,
    results: dict[str, TaskResult],
    brain_hash: str,
    fixtures: FixturesResult,
    timings: dict[str, float],
    dataset_meta: dict[str, object],
    out: Path,
    config: TrainConfig,
) -> None:
    ds_timing = dataset_meta["timing_s"]
    assert isinstance(ds_timing, dict)
    counts = dataset_meta["balanced_counts"]
    assert isinstance(counts, dict)
    lines = [
        "# Readout metrics (Phase 5)",
        "",
        (
            f"Generated by `scripts/train.py`. Encoder `{ENCODER_VERSION}`, brain `{brain_hash}` "
            f"(`BrainParams` at the defaults), dataset `{dataset_meta['generator_version']}` "
            f"seed {dataset_meta['seed']}: {dataset_meta['snippets']} snippets, the keys table "
            f"{counts['keys']['rows']} rows ({counts['keys']['positive']} positive), kwargs "
            f"{counts['kwargs']['rows']} ({counts['kwargs']['positive']} placeable); split by "
            f"snippet 80/10/10. Training: delta rule (`dan_update`), lr {config.lr}, L2 "
            f"{config.l2}, batch {config.batch_size}, ≤ {config.max_epochs} epochs, early stopping "
            f"on val F1 (patience {config.patience}); train rows with {len(TRAIN_SEEDS)} seeds "
            "each, val/test with the production seed."
        ),
        "",
        "## 1. Results on test",
        "",
        (
            "| task | features | val F1 | test without resniff | test with resniff (≤ 3 trials) "
            "| resniff share | θ |"
        ),
        "|---|---|---:|---|---|---:|---:|",
    ]
    for r in results.values():
        lines.append(  # noqa: PERF401
            f"| {r.name} | {r.mode} | {r.val.f1:.4f} | {md_scores(r.evaluation.plain)} "
            f"| {md_scores(r.evaluation.resniffed)} | {r.evaluation.resniff_fraction:.1%} "
            f"| {r.theta:.3f} |"
        )
    lines += [
        "",
        "PLAN gate: P and R ≥ 0.995 on test for both tasks; 100 % on the golden fixtures.",
        "",
        "### Confusion matrices (test, with resniff)",
        "",
        "| task | | yes (teacher) | no (teacher) |",
        "|---|---|---:|---:|",
    ]
    for r in results.values():
        s = r.evaluation.resniffed
        lines.append(f"| {r.name} | fly: yes | {s.tp} | {s.fp} |")
        lines.append(f"| {r.name} | fly: no | {s.fn} | {s.tn} |")
    lines += [
        "",
        "### Feature modes (chosen by val F1)",
        "",
        "| task | mode | val F1 | best epoch | epochs | loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results.values():
        for mode, log in r.logs.items():
            mark = " **←**" if mode == r.mode else ""
            loss = log.epochs[-1]["loss"] if log.epochs else float("nan")
            lines.append(
                f"| {r.name} | {mode}{mark} | {log.best_val_f1:.4f} | {log.best_epoch} "
                f"| {len(log.epochs)} | {loss:.4f} |"
            )
    lines += [
        "",
        "### Share of active KCs on the dataset",
        "",
        "| task | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r.name} | {r.kc_active['train']:.3f} | {r.kc_active['val']:.3f} "
        f"| {r.kc_active['test']:.3f} |"
        for r in results.values()
    )
    lines += [
        "",
        "## 2. Golden fixtures (holdout) through the whole fly",
        "",
        (
            "tokenizer → odor → brain → judge, comparing the set of (call_position, key_name, "
            f"kwargs) with `reference/`: **{fixtures.files_ok} / {fixtures.files_total} files "
            f"match** ({fixtures.seconds:.1f} s). The same test is "
            "`tests/test_judge_on_fixtures.py`."
        ),
    ]
    if fixtures.mismatches:
        lines += ["", "Differences:", "", *(f"- {m}" for m in fixtures.mismatches)]
    lines += [
        "",
        "## 3. The 10 worst test examples (with resniff, the largest |margin| on the wrong side)",
        "",
    ]
    for r in results.values():
        lines.append(f"### {r.name}")
        lines.append("")
        if not r.evaluation.worst:
            lines.append("No errors on test.")
        for w in r.evaluation.worst:
            kwarg = f", kwarg={w['name']!r}" if "name" in w else ""
            lines.append(
                f"- teacher **{'yes' if w['label'] else 'no'}**, margin {w['margin']:+.3f}, "
                f"trials {w['sniffs']}, key={w.get('key_name')!r}{kwarg}"
            )
            lines.append(f"  `{w['window']}`")
        lines.append("")
    lines += [
        "## 4. Time",
        "",
        "| stage | seconds |",
        "|---|---:|",
        f"| make_dataset: generation | {ds_timing['generate']:.1f} |",
        f"| make_dataset: labelling + encoding | {ds_timing['label_and_encode']:.1f} |",
        f"| brain keys (train ×{len(TRAIN_SEEDS)} + val + test) | {timings['keys_brain_s']:.1f} |",
        f"| brain kwargs | {timings['kwargs_brain_s']:.1f} |",
        f"| train keys (all modes) | {timings['keys_train_s']:.1f} |",
        f"| train kwargs | {timings['kwargs_train_s']:.1f} |",
        f"| fixtures check | {fixtures.seconds:.1f} |",
        f"| train.py total | {timings['total_s']:.1f} |",
        "",
        f"Weights: `{out.name}`, {out.stat().st_size} bytes.",
        "",
        "## 5. Attempts (what changed → what came out)",
        "",
        "<!-- attempts:start -->",
        *attempts_table(),
        "<!-- attempts:end -->",
        *proxy_section(),
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
