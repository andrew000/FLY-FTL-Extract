"""``ftl extract`` through the fly.

Walk (``files.py``) → per file: prefilter, UTF-8 decode, syntax check (``ftl/pyerrors.py``)
→ candidates (``tokenizer/``) → :class:`~fly_ftl_extract.dopamine.Judge` (odour → mushroom
body → MBON readout) → :class:`~fly_ftl_extract.ftl.merge.FileExtraction` → the same merge
and ``.ftl`` pipeline the golden tests already pass.  No ``ast`` and no teacher anywhere in
this module: switch the fly off and no key is found.

The brain runs in a process pool (``cli/workers.py``) when there is enough to sniff; small
runs stay in-process because a spawned worker needs seconds to import numpy/scipy and load
the connectome.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol

from fly_ftl_extract.brain import Connectome
from fly_ftl_extract.brain import connectome as connectome_module
from fly_ftl_extract.brain.connectome import ConnectomeInvalidError, ConnectomeMissingError
from fly_ftl_extract.cli.cache import ExtractCache
from fly_ftl_extract.cli.config import (
    ConfigError,
    ExtractOverrides,
    LoadedConfig,
    load_pyproject,
    resolve_options,
)
from fly_ftl_extract.cli.workers import FileJob, chunk_jobs, init_worker, judge_chunk
from fly_ftl_extract.dopamine.judge import BATCH, BatchEvent, Judge, JudgedFile
from fly_ftl_extract.dopamine.weights import WeightsMismatchError, WeightsMissingError
from fly_ftl_extract.files import find_py_files, mentions_any_name
from fly_ftl_extract.ftl.merge import CodeExtraction, FileExtraction, merge_extractions
from fly_ftl_extract.ftl.model import (
    CodeLocation,
    DiagnosticKind,
    ExtractOptions,
    FluentKey,
    code_ftl_path,
    code_message,
    file_diagnostic,
)
from fly_ftl_extract.ftl.pipeline import (
    EXIT_CONFIG_ERROR,
    ExtractOutcome,
    LogLine,
    done_line,
    run_extract,
)
from fly_ftl_extract.ftl.pyerrors import check_syntax, ruff_style_syntax_error, utf8_error_message
from fly_ftl_extract.odor.encoder import ENCODER_VERSION
from fly_ftl_extract.tokenizer.candidates import CandidateError, iter_candidates

FLY_TARGET = "fly"
"""Log target of our own lines: ``[INFO  fly] …`` — printed only after the original's
``✅ Done`` line, so everything before it is byte-for-byte the original's output."""

EXIT_FLY_ERROR = 2
"""No connectome / no weights / weights for another fly: the run cannot start (like a
configuration error)."""

DEFAULT_WORKERS = max(1, min((os.cpu_count() or 2) - 2, 16))
"""Processes for the brain.  ``docs/bench_sequence.json``: 16 and 30 processes give the
same throughput (memory-bound), so more than 16 is never useful."""

INPROCESS_THRESHOLD = 128
"""Runs with fewer windows than this stay in one process: a spawned worker spends ~2–3 s
importing numpy/scipy and building its brain, which is more than sniffing 128 windows
(~1.3 s at 100 trials/s)."""

POOL_BATCH = 64
"""Trials per brain call inside a worker (``docs/bench_sequence.json`` §parallel: 64 beats
128 and 256 when many processes share the memory bandwidth)."""

TUI_BATCH = 16
"""Trials per brain call when the TUI is watching an in-process run (auditor's decision
after Phase 6): every batch is published as soon as it is done, so the dashboard moves
every ~0.5 s instead of once per file chunk.  Costs throughput (more calls of ~5700 steps);
``--fly-batch`` overrides."""


@dataclass(frozen=True)
class FlyOptions:
    """The ``--fly-*`` options."""

    workers: int = DEFAULT_WORKERS
    """``--fly-workers``: processes for the brain (1 = in-process)."""
    batch: int | None = None
    """``--fly-batch``: trials per brain call (default: 256 in-process, 64 in a pool)."""
    trials: int = 1
    """``--fly-trials``: sniffs every window gets before the resniff rule."""
    seed: int = 0
    """``--fly-seed``: salt mixed into every trial seed (0 = production seeds)."""
    tui: bool = True
    """TUI wanted (still needs a tty on stdout)."""
    audit: bool = False
    """``--fly-audit``: run the ast teacher next to the fly and report differences."""
    verbose: bool = False
    """``-v``: one ``[DEBUG fly]`` line per judged window after the statistics."""
    inprocess_threshold: int = INPROCESS_THRESHOLD


@dataclass
class FlyStats:
    """What the fly did in one run (the block after ``✅ Done``)."""

    files_walked: int = 0
    files_prefiltered: int = 0
    """Files whose bytes mention no i18n name (never tokenized, like the original)."""
    files_judged: int = 0
    files_cached: int = 0
    candidates: int = 0
    keys: int = 0
    trials: int = 0
    resniffs: int = 0
    brain_seconds: float = 0.0
    workers: int = 1
    batch: int = BATCH

    @property
    def trials_per_second(self) -> float:
        """Throughput of the brain phase (0 when nothing was sniffed)."""
        return self.trials / self.brain_seconds if self.brain_seconds > 0 else 0.0


class Observer(Protocol):
    """Where the run reports progress (the TUI, or nothing)."""

    live: bool
    """Wants every brain batch as it happens (then in-process runs use ``TUI_BATCH``)."""

    def on_jobs(self, jobs: list[FileJob], stats: FlyStats) -> None:
        """All files that will be sniffed are known."""

    def on_batch(self, jobs: list[FileJob], event: BatchEvent, stats: FlyStats) -> None:
        """One brain call of the chunk ``jobs`` is done (in-process runs only)."""

    def on_judged(self, job: FileJob, judged: JudgedFile, stats: FlyStats) -> None:
        """One file has its verdicts."""


class NullObserver:
    """Reports nothing."""

    live = False

    def on_jobs(self, jobs: list[FileJob], stats: FlyStats) -> None:
        """Ignore."""

    def on_batch(self, jobs: list[FileJob], event: BatchEvent, stats: FlyStats) -> None:
        """Ignore."""

    def on_judged(self, job: FileJob, judged: JudgedFile, stats: FlyStats) -> None:
        """Ignore."""


# ------------------------------------------------------------------ per-file preparation


def prepare_file(index: int, path: str, options: ExtractOptions) -> FileExtraction | FileJob:
    """Read, prefilter, decode, syntax-check and tokenize one file.

    Mirrors the original's per-file steps (read error → ``read-error``, bytes without any
    i18n name → nothing, bad UTF-8 → ``invalid-utf8``, unparsable → ``parse-error``);
    what remains is a :class:`FileJob` for the fly.
    """
    result = FileExtraction(path)
    try:
        if os.path.getsize(path) == 0:
            return result
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as err:
        result.diagnostics.append(
            file_diagnostic(
                DiagnosticKind.READ_ERROR, path, f"Failed to read Python file: {err}", 1, 1
            )
        )
        return result
    if not mentions_any_name(data, options.i18n_keys | options.i18n_keys_prefix):
        return result
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as err:
        message, line, column = utf8_error_message(data, err)
        result.diagnostics.append(
            file_diagnostic(DiagnosticKind.INVALID_UTF8, path, message, line, column)
        )
        return result
    syntax = check_syntax(source, path)
    if syntax is not None:
        message, line, column = ruff_style_syntax_error(
            syntax.msg, syntax.lineno, syntax.offset, source
        )
        result.diagnostics.append(
            file_diagnostic(
                DiagnosticKind.PARSE_ERROR,
                path,
                f"Failed to parse Python file: {message}",
                line,
                column,
            )
        )
        return result
    try:
        candidates = list(iter_candidates(source, options))
    except CandidateError as err:
        result.diagnostics.append(
            file_diagnostic(
                DiagnosticKind.PARSE_ERROR, path, f"Failed to parse Python file: {err}", 1, 1
            )
        )
        return result
    return FileJob(index, path, data, candidates)


def file_extraction_from_judged(
    path: str, judged: JudgedFile, options: ExtractOptions
) -> FileExtraction:
    """The keys the fly found in one file, merged per key like the original's matcher."""
    result = FileExtraction(path)
    for judged_key in judged.keys:
        line, column = judged_key.call_position
        location = CodeLocation(path, line, column)
        result.add(
            FluentKey(
                judged_key.key_name,
                code_message(judged_key.key_name, list(judged_key.placeable)),
                code_ftl_path(judged_key.path_value, options.default_ftl_file),
                source_location=location,
                kwargs_unknown=location if judged_key.candidate.kwargs_unknown else None,
            )
        )
    return result


# ------------------------------------------------------------------------------ the run


@dataclass
class FlyRun:
    """Everything one extraction produced, for the CLI, the TUI and the audit."""

    extraction: CodeExtraction
    stats: FlyStats
    judged: dict[int, tuple[FileJob, JudgedFile]] = field(default_factory=dict)
    """Verdicts per walker index (files the fly sniffed in this run)."""
    seconds: float = 0.0


def _judge_kwargs(fly: FlyOptions, batch: int) -> dict[str, int]:
    return {"base_trials": fly.trials, "seed_salt": fly.seed, "batch": batch}


def judge_jobs(
    jobs: list[FileJob],
    judge: Judge,
    fly: FlyOptions,
    stats: FlyStats,
    observer: Observer,
) -> dict[int, JudgedFile]:
    """Run the brain over ``jobs``: in-process, or in a spawn pool for big runs.

    Files are judged in chunks of about one batch of windows (``workers.chunk_jobs``), so
    each of the judge's four brain rounds is a full batch instead of a per-file call.
    """
    results: dict[int, JudgedFile] = {}
    if not jobs:
        return results
    windows = sum(job.windows for job in jobs)
    n_workers = min(fly.workers, len(jobs))
    use_pool = n_workers > 1 and windows >= fly.inprocess_threshold
    live = observer.live and not use_pool
    stats.workers = n_workers if use_pool else 1
    stats.batch = fly.batch or (TUI_BATCH if live else POOL_BATCH if use_pool else BATCH)
    by_index = {job.index: job for job in jobs}
    chunks = chunk_jobs(jobs, stats.batch)

    def collect(pairs: list[tuple[int, JudgedFile]]) -> None:
        for index, judged in pairs:
            results[index] = judged
            _account(by_index[index], judged, fly, stats)
            observer.on_judged(by_index[index], judged, stats)

    t0 = time.perf_counter()
    if not use_pool:
        judge.batch = stats.batch
        try:
            for chunk in chunks:
                if live:

                    def publish(event: BatchEvent, chunk: list[FileJob] = chunk) -> None:
                        observer.on_batch(chunk, event, stats)

                    judge.trial_observer = publish
                judged = judge.judge_many([(job.candidates, job.content) for job in chunk])
                collect(list(zip((job.index for job in chunk), judged, strict=True)))
        finally:
            judge.trial_observer = None
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(judge.options, _judge_kwargs(fly, stats.batch)),
        ) as pool:
            futures = [pool.submit(judge_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                collect(future.result())
    stats.brain_seconds = time.perf_counter() - t0
    return results


def _account(job: FileJob, judged: JudgedFile, fly: FlyOptions, stats: FlyStats) -> None:
    stats.files_judged += 1
    stats.candidates += len(job.candidates)
    stats.keys += len(judged.keys)
    stats.trials += judged.trials
    stats.resniffs += judged.resniffs(fly.trials)


def extract_with_fly(
    options: ExtractOptions,
    fly: FlyOptions,
    judge: Judge,
    observer: Observer | None = None,
    cache: ExtractCache | None = None,
) -> FlyRun:
    """Walk ``code_path`` and let the fly extract every file; merged like the original.

    With ``cache`` an unchanged file (same bytes on disk, same fly) is not even read: its
    stored occurrences are replayed instead of sniffed.
    """
    observer = observer if observer is not None else NullObserver()
    t0 = time.perf_counter()
    stats = FlyStats()
    paths = find_py_files(options.code_path, options.exclude_dirs)
    stats.files_walked = len(paths)
    files: list[FileExtraction | None] = [None] * len(paths)
    jobs: list[FileJob] = []
    for index, path in enumerate(paths):
        hit = cache.lookup(path, options) if cache is not None else None
        if hit is not None:
            files[index], meta = hit
            stats.files_cached += 1
            stats.candidates += meta.candidates
            stats.keys += meta.keys
            continue
        prepared = prepare_file(index, path, options)
        if isinstance(prepared, FileExtraction):
            files[index] = prepared
            if not prepared.diagnostics:
                stats.files_prefiltered += 1
            continue
        jobs.append(prepared)
    observer.on_jobs(jobs, stats)
    judged = judge_jobs(jobs, judge, fly, stats, observer)
    judged_by_index: dict[int, tuple[FileJob, JudgedFile]] = {}
    for job in jobs:
        files[job.index] = file_extraction_from_judged(job.path, judged[job.index], options)
        judged_by_index[job.index] = (job, judged[job.index])
        if cache is not None:
            cache.store(job.path, judged[job.index])
    complete = [f for f in files if f is not None]
    assert len(complete) == len(paths)  # noqa: S101 — every walked file has a result
    return FlyRun(
        merge_extractions(complete, len(paths)),
        stats,
        judged_by_index,
        time.perf_counter() - t0,
    )


# ------------------------------------------------------------------------------ logging


def neuron_line(connectome: Connectome) -> str:
    """``2935 (PN 124, KC 2597, APL 1, MBON 48, DAN 165)`` — counted in the loaded npz."""
    groups = (
        ("PN", connectome.pn_idx),
        ("KC", connectome.kc_idx),
        ("APL", connectome.apl_idx),
        ("MBON", connectome.mbon_idx),
        ("DAN", connectome.dan_idx),
    )
    inner = ", ".join(f"{name} {len(idx)}" for name, idx in groups)
    return f"{connectome.n_neurons} ({inner})"


def fly_log_lines(run: FlyRun, judge: Judge, fly: FlyOptions) -> list[LogLine]:
    """Our lines after ``✅ Done``: verbose verdicts (``-v``) and the fly statistics."""
    lines: list[LogLine] = []
    if fly.verbose:
        for index in sorted(run.judged):
            job, judged = run.judged[index]
            lines.extend(_verbose_lines(job, judged))
    s = run.stats
    weights_id = judge.weights.brain_hash
    lines.append(LogLine("INFO", FLY_TARGET, "Fly statistics:"))
    lines.append(
        LogLine("INFO", FLY_TARGET, f"  - Neurons online: {neuron_line(judge.brain.connectome)}")
    )
    lines.append(
        LogLine(
            "INFO",
            FLY_TARGET,
            f"  - Encoder {ENCODER_VERSION}, brain {weights_id}, "
            f"puffs/trial {judge.weights.key.n_states // judge.n_kc}",
        )
    )
    lines.append(
        LogLine(
            "INFO",
            FLY_TARGET,
            f"  - Files sniffed: {s.files_judged} (from cache: {s.files_cached}, "
            f"without i18n names: {s.files_prefiltered}, walked: {s.files_walked})",
        )
    )
    lines.append(
        LogLine(
            "INFO",
            FLY_TARGET,
            f"  - Candidates: {s.candidates} (key occurrences: {s.keys}, merged keys: "
            f"{len(run.extraction.keys)})",
        )
    )
    lines.append(
        LogLine(
            "INFO",
            FLY_TARGET,
            f"  - Trials: {s.trials} (resniffs: {s.resniffs}, base trials per window: "
            f"{fly.trials}, θ {judge.weights.theta_key:.2f} / {judge.weights.theta_kwarg:.2f})",
        )
    )
    lines.append(
        LogLine(
            "INFO",
            FLY_TARGET,
            f"  - Trials/s: {s.trials_per_second:.1f} ({s.workers} "
            f"{'process' if s.workers == 1 else 'processes'}, batch {s.batch})",
        )
    )
    lines.append(LogLine("INFO", FLY_TARGET, f"  - Brain wall time: {s.brain_seconds:.3f}s"))
    return lines


def _verbose_lines(job: FileJob, judged: JudgedFile) -> list[LogLine]:
    lines: list[LogLine] = []
    keys_by_candidate = {id(k.candidate): k for k in judged.keys}
    for candidate, verdict in zip(judged.candidates, judged.verdicts, strict=True):
        if verdict is None:
            continue
        answer = "KEY" if verdict.is_positive else "not a key"
        lines.append(
            LogLine(
                "DEBUG",
                FLY_TARGET,
                f"{job.path}:{candidate.line}:{candidate.column} {candidate.kind} "
                f"{candidate.text!r} → {answer} margin {verdict.margin:+.2f} "
                f"({verdict.sniffs} {'sniff' if verdict.sniffs == 1 else 'sniffs'}, "
                f"KC {verdict.kc_active_fraction * 100:.1f} %)",
            )
        )
        judged_key = keys_by_candidate.get(id(candidate))
        if judged_key is None:
            continue
        for kwarg, kv in judged_key.kwarg_verdicts:
            answer = "placeable" if kv.is_positive else "ignored"
            lines.append(
                LogLine(
                    "DEBUG",
                    FLY_TARGET,
                    f"{job.path}:{kwarg.line}:{kwarg.column} kwarg {kwarg.name}= of "
                    f"{judged_key.key_name} → {answer} margin {kv.margin:+.2f} "
                    f"({kv.sniffs} {'sniff' if kv.sniffs == 1 else 'sniffs'})",
                )
            )
    return lines


# ------------------------------------------------------------------------------ command


def emit(text: str) -> None:
    """Print one log block to stderr (the original logs to stderr, stdout stays empty)."""
    sys.stderr.write(text)
    sys.stderr.flush()


def load_judge(options: ExtractOptions, fly: FlyOptions) -> Judge:
    """The fly for this run (connectome + weights checked); raises the fly errors."""
    connectome = connectome_module.load()
    return Judge(
        options,
        connectome=connectome,
        base_trials=fly.trials,
        seed_salt=fly.seed,
        batch=fly.batch or BATCH,
    )


def run_command(
    overrides: ExtractOverrides,
    config_path: str | None,
    fly: FlyOptions,
    *,
    cwd: str | None = None,
) -> int:
    """The whole ``ftl extract``: resolve options, run the fly, write, log; returns exit code."""
    started = time.perf_counter()
    cwd = cwd if cwd is not None else os.getcwd()
    try:
        loaded = load_pyproject(config_path, cwd)
        options = resolve_options(overrides, loaded)
    except ConfigError as err:
        emit(LogLine("ERROR", "cli", f"Configuration error: {err}").render() + "\n")
        return EXIT_CONFIG_ERROR
    preamble = "".join(line.render() + "\n" for line in config_warnings(loaded))
    try:
        judge = load_judge(options, fly)
    except (
        ConnectomeMissingError,
        ConnectomeInvalidError,
        WeightsMissingError,
        WeightsMismatchError,
    ) as err:
        emit(LogLine("ERROR", FLY_TARGET, str(err)).render() + "\n")
        return EXIT_FLY_ERROR

    cache = ExtractCache.open(options, judge) if options.cache else None
    observer = _make_observer(fly, judge, options)
    try:
        run = extract_with_fly(options, fly, judge, observer, cache=cache)
    finally:
        close = getattr(observer, "close", None)
        if callable(close):
            close()
    if cache is not None:
        cache.save()
    outcome: ExtractOutcome = run_extract(run.extraction, options, extraction_seconds=run.seconds)
    text = preamble + outcome.render_logs()
    if outcome.exit_code == 0:
        text += done_line(time.perf_counter() - started).render() + "\n"
        text += "".join(line.render() + "\n" for line in fly_log_lines(run, judge, fly))
    emit(text)
    if fly.audit:
        return max(outcome.exit_code, _run_audit(options, run))
    return outcome.exit_code


DEPRECATED_CONFIG_KEYS: dict[str, str] = {
    "comment-junks": (
        "comment-junks has no effect and will be removed in 0.13: syntax errors in .ftl "
        "files abort the run"
    ),
}
"""``[WARN  cli]`` the original prints, before ``Code path``, for config keys it still
accepts but ignores (seen on a real project's ``pyproject.toml``)."""


def config_warnings(loaded: LoadedConfig | None) -> list[LogLine]:
    """The original's warnings about deprecated keys in ``[tool.ftl-extract.extract]``."""
    if loaded is None:
        return []
    return [
        LogLine("WARN", "cli", message)
        for key, message in DEPRECATED_CONFIG_KEYS.items()
        if key in loaded.section
    ]


def _make_observer(fly: FlyOptions, judge: Judge, options: ExtractOptions) -> Observer:
    """The TUI when wanted and stdout is a terminal; otherwise nothing (plain logs)."""
    if not fly.tui or not sys.stdout.isatty():
        return NullObserver()
    from fly_ftl_extract.tui import FlyTui  # noqa: PLC0415 — rich only when it is shown

    return FlyTui.start(judge, options)


def _run_audit(options: ExtractOptions, run: FlyRun) -> int:
    """``--fly-audit``: the teacher next to the fly; 1 when they disagree."""
    # CLAUDE.md rule 2b: the only import of the audit package, only in this branch.
    from fly_ftl_extract import audit  # noqa: PLC0415

    differences = audit.compare(options, run.extraction)
    emit("".join(line.render() + "\n" for line in audit.report_lines(differences)))
    return 1 if differences else 0
