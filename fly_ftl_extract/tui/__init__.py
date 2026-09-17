"""The fly on screen: ``rich`` Live dashboard of an ``ftl extract`` run (CLAUDE.md, «TUI»).

Layout (120 columns)::

    DIPTERA_  CONNECTOME · FAFB v783 · MB-R   NEURONS ONLINE 2 935   HUMAN INPUT NONE  REC 00:00:12
    ┌ SPIKE RASTER (one dot = one Kenyon cell, row = trial) ─┐ ┌ REGION DRIVE ────────────┐
    │ ⠀⠀⠁⠀⠈⠀⠀⢀⠀⠀⠐⠀…                                 │ │ PN     ▇▇▇▇▇▇▁▁▁▁  61 %  │
    │                                                       │ │ KC     ▇▇▇▇▇▁▁▁▁▁   8.7 %│
    │                                                       │ │ APL    ▇▇▇▇▇▇▁▁▁▁  fires │
    │ KC 0–149 of 2597 · summary puff · last 64 trials      │ │ MBON   ▇▇▇▇▇▇▇▇▇▁  p 0.94│
    └───────────────────────────────────────────────────────┘ └──────────────────────────┘
    ┌ EVENT LOG ─────────────────────────────────────────────────────────────────────────┐
    │ 12.4ms  ODOR   app/handlers/start.py:10  0x7f3a1c  window 19 tok                   │
    │ 61.0ms  MBON   margin +7.36 → KEY  hello-user { $name }                            │
    └────────────────────────────────────────────────────────────────────────────────────┘
     ftl extract app locales · files 3/12 · candidates 41 · keys 17 · trials 47 · 53 trials/s

Every number is measured: neuron counts come from the loaded connectome, each raster dot is
one Kenyon cell that spiked in the summary puff of that trial (``Verdict.kc_pattern``; the
panel shows as many cells as it has columns and says which in its last line), the drive
bars are the PN/KC/APL/MBON activity of the last trial, the MBON value is the readout's
``σ(margin)``.  Nothing here is drawn from a random source.

:func:`render` is pure (state → renderable) so tests can draw it into a recording console;
:class:`FlyTui` is the ``Live`` driver that implements the extractor's ``Observer`` protocol.
Off automatically when stdout is not a tty, and with ``--fly-no-tui``.
"""

from __future__ import annotations

import io
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from fly_ftl_extract.cli.extract import FlyStats
from fly_ftl_extract.cli.workers import FileJob
from fly_ftl_extract.dopamine.judge import Judge, JudgedFile, Verdict
from fly_ftl_extract.dopamine.seed import trial_seed
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor.encoder import ENCODER_VERSION, N_SLOTS

REFRESH_PER_SECOND = 15
RASTER_TRIALS = 64
"""Trials kept for the raster (the last 64, newest at the bottom)."""
EVENT_LOG_SIZE = 200
BAR_WIDTH = 10
BAR_FULL, BAR_EMPTY = "▇", "▁"
KC_BAR_FULL = 0.15
"""The KC bar is full at 15 % active cells — the upper end of CLAUDE.md's healthy 3–15 %."""
NARROW = 100
"""Below this many columns the footer and titles use their short forms."""

_BRAILLE_BASE = 0x2800
_DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
"""Braille dot bit for (row 0–3, column 0–1) of one cell."""


@dataclass(frozen=True)
class Event:
    """One event-log line."""

    t_ms: float
    kind: str
    text: str
    style: str = ""


@dataclass
class TuiState:
    """Everything the dashboard shows; updated by the observer, read by :func:`render`."""

    command: str
    dataset: str
    hemisphere: str
    n_neurons: int
    n_pn: int
    n_kc: int
    n_apl: int
    n_mbon: int
    n_dan: int
    theta_key: float
    theta_kwarg: float
    apl_max_spikes: float
    """Spikes the APL could fire in one trial at its refractory limit (bar scale)."""
    started: float = field(default_factory=time.perf_counter)
    files_total: int = 0
    files_done: int = 0
    candidates: int = 0
    keys: int = 0
    trials: int = 0
    resniffs: int = 0
    brain_started: float | None = None
    last: Verdict | None = None
    raster: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=RASTER_TRIALS))
    events: deque[Event] = field(default_factory=lambda: deque(maxlen=EVENT_LOG_SIZE))
    finished: bool = False

    @property
    def elapsed(self) -> float:
        """Seconds since the dashboard started."""
        return time.perf_counter() - self.started

    @property
    def trials_per_second(self) -> float:
        """Brain throughput so far (0 before the first trial)."""
        if self.brain_started is None or self.trials == 0:
            return 0.0
        return self.trials / max(time.perf_counter() - self.brain_started, 1e-9)

    def log(self, kind: str, text: str, style: str = "") -> None:
        """Append an event stamped with the real elapsed time."""
        self.events.append(Event(self.elapsed * 1000.0, kind, text, style))

    @classmethod
    def from_judge(cls, judge: Judge, command: str) -> TuiState:
        """A state whose header numbers come from the judge's connectome and weights."""
        cx = judge.brain.connectome
        p = judge.brain.params
        dataset = str(cx.meta.get("dataset", "FlyWire FAFB v783")).removeprefix("FlyWire ")
        return cls(
            command=command,
            dataset=dataset,
            hemisphere=cx.hemisphere,
            n_neurons=cx.n_neurons,
            n_pn=len(cx.pn_idx),
            n_kc=len(cx.kc_idx),
            n_apl=len(cx.apl_idx),
            n_mbon=len(cx.mbon_idx),
            n_dan=len(cx.dan_idx),
            theta_key=judge.weights.theta_key,
            theta_kwarg=judge.weights.theta_kwarg,
            apl_max_spikes=p.sequence_steps(N_SLOTS) * p.dt / p.t_refractory,
        )


# ------------------------------------------------------------------------------ drawing


def group_thousands(n: int) -> str:
    """``2935`` → ``2 935`` (thin grouping like the mock-up)."""
    return f"{n:,}".replace(",", " ")


def braille_rows(matrix: np.ndarray, width: int, height: int) -> list[str]:
    """Draw a boolean ``(trials, cells)`` matrix as braille: one dot per (trial, cell).

    Rows are trials (newest last, bottom-aligned), columns are cells in their order; the
    first ``2 × width`` cells fit (the rest is cropped), 4 trials per character row.
    """
    width, height = max(width, 1), max(height, 1)
    rows_avail, cols_avail = 4 * height, 2 * width
    grid = np.zeros((rows_avail, cols_avail), dtype=bool)
    shown = matrix[-rows_avail:, :cols_avail]
    if shown.size:
        grid[rows_avail - shown.shape[0] :, : shown.shape[1]] = shown
    lines: list[str] = []
    for r in range(height):
        chars = []
        for c in range(width):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    if grid[4 * r + dy, 2 * c + dx]:
                        bits |= _DOT_BITS[dy][dx]
            chars.append(chr(_BRAILLE_BASE + bits))
        lines.append("".join(chars))
    return lines


def bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """``▇▇▇▁▁▁▁▁▁▁`` for a value in [0, 1]."""
    filled = round(min(max(fraction, 0.0), 1.0) * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def sigmoid(x: float) -> float:
    """The readout's probability of «key» for a margin."""
    return 1.0 / (1.0 + math.exp(-x))


class _Raster:
    """Renderable that fills whatever box the layout gives it with the KC raster.

    The last line names the cells actually drawn (``KC 0–149 of 2597``): two per column,
    so the range follows the terminal width.
    """

    def __init__(self, patterns: list[np.ndarray], n_kc: int) -> None:
        self.patterns = patterns
        self.n_kc = n_kc

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(options.max_width, 1)
        height = max((options.height or 8) - 1, 1)
        shown = min(2 * width, self.n_kc)
        matrix = np.stack(self.patterns) if self.patterns else np.zeros((0, self.n_kc), dtype=bool)
        for line in braille_rows(matrix, width, height):
            yield Text(line, style="cyan", no_wrap=True, overflow="crop")
        yield Text(
            f"KC 0–{shown - 1} of {self.n_kc} · summary puff · last {RASTER_TRIALS} trials",
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )


def _header(state: TuiState, width: int) -> Text:
    rec = time.strftime("%H:%M:%S", time.gmtime(state.elapsed))
    side = "MB-R" if state.hemisphere == "right" else "MB-L"
    text = Text(no_wrap=True, overflow="crop")
    text.append("DIPTERA_", style="bold magenta")
    text.append(f"  CONNECTOME · {state.dataset} · {side}", style="bold")
    text.append(f"    NEURONS ONLINE {group_thousands(state.n_neurons)}", style="bold green")
    if width >= NARROW:
        text.append("   HUMAN INPUT NONE", style="dim")
    text.append(f"   REC {rec}", style="dim" if state.finished else "bold red")
    return text


def _drive(state: TuiState, width: int) -> Text:
    wide = width >= NARROW
    bar_w = BAR_WIDTH if wide else 6
    text = Text(no_wrap=True, overflow="crop")
    v = state.last
    if v is None:
        for name in ("PN", "KC", "APL", "MBON"):
            text.append(f"{name:<7}", style="bold")
            text.append(bar(0.0, bar_w) + "\n")
        text.append("waiting for the first odour", style="dim")
        return text
    p_key = sigmoid(v.margin)
    of_pn = f" of {state.n_pn}" if wide else ""
    of_kc = f" of {state.n_kc}" if wide else ""
    apl = ("fires " if v.apl_spikes else "silent ") if wide else ""
    margin = f"{v.margin:+.2f}" if wide else f"{v.margin:+.1f}"
    text.append("PN     ", style="bold")
    text.append(bar(v.pn_active_fraction, bar_w), style="yellow")
    text.append(f"  {v.pn_active_fraction * 100:4.1f}%{of_pn}\n")
    text.append("KC     ", style="bold")
    text.append(bar(v.kc_active_fraction / KC_BAR_FULL, bar_w), style="cyan")
    text.append(f"  {v.kc_active_fraction * 100:4.1f}%{of_kc}\n")
    text.append("APL    ", style="bold")
    text.append(bar(v.apl_spikes / state.apl_max_spikes, bar_w), style="red")
    text.append(f"  {apl}{v.apl_spikes} spk\n")
    text.append("MBON   ", style="bold")
    text.append(bar(p_key, bar_w), style="green" if v.is_positive else "magenta")
    text.append(f"  p {p_key:.2f} {margin}\n")
    text.append(
        f"MBON   {v.mbon_active_fraction * 100:.0f}% of {state.n_mbon} · "
        f"{v.sniffs} sniff{'s' if v.sniffs != 1 else ''}",
        style="dim",
    )
    return text


def _event_log(state: TuiState, height: int) -> Text:
    text = Text(no_wrap=True, overflow="ellipsis")
    events = list(state.events)[-max(height, 1) :]
    for i, e in enumerate(events):
        text.append(f"{e.t_ms:8.1f}ms  ", style="dim")
        text.append(f"{e.kind:<8}", style="bold")
        text.append(e.text, style=e.style)
        if i < len(events) - 1:
            text.append("\n")
    return text


def _footer(state: TuiState, width: int) -> Text:
    rate = state.trials_per_second
    text = Text(no_wrap=True, overflow="crop", style="dim")
    if width >= NARROW:
        text.append(
            f" {state.command} · files {state.files_done}/{state.files_total} · candidates "
            f"{state.candidates} · keys {state.keys} · trials {state.trials} "
            f"(resniffs {state.resniffs}) · {rate:.0f} trials/s"
        )
    else:
        text.append(
            f" files {state.files_done}/{state.files_total} · keys {state.keys} · "
            f"trials {state.trials} · {rate:.0f} trials/s"
        )
    if state.finished:
        text.append(" · done", style="bold green")
    return text


def render(state: TuiState, width: int = 120, height: int = 40) -> RenderableType:
    """The whole dashboard for the current state and terminal size (pure)."""
    layout = Layout(name="root")
    log_height = 6 if height <= 24 else 8  # noqa: PLR2004 — 80×24 terminals
    raster_title = (
        "SPIKE RASTER (one dot = one Kenyon cell, row = trial)"
        if width >= NARROW
        else "SPIKE RASTER"
    )
    layout.split_column(
        Layout(_header(state, width), name="header", size=1),
        Layout(name="body", ratio=1),
        Layout(
            Panel(_event_log(state, log_height), title="EVENT LOG", title_align="left"),
            name="log",
            size=log_height + 2,
        ),
        Layout(_footer(state, width), name="footer", size=1),
    )
    layout["body"].split_row(
        Layout(
            Panel(_Raster(list(state.raster), state.n_kc), title=raster_title, title_align="left"),
            name="raster",
            ratio=2,
        ),
        Layout(
            Panel(_drive(state, width), title="REGION DRIVE", title_align="left"),
            name="drive",
            ratio=1,
            minimum_size=30,
        ),
    )
    return layout


# ----------------------------------------------------------------------------- observer


def describe_key(judged: JudgedFile, index: int) -> str:
    """``hello-user { $name }`` for a positive candidate, its text otherwise."""
    for key in judged.keys:
        if key.candidate is judged.candidates[index]:
            placeables = "".join(f" {{ ${p} }}" for p in sorted(key.placeable))
            return f"{key.key_name}{placeables}"
    return judged.candidates[index].text


class FlyTui:
    """``Live`` driver: implements the extractor's ``Observer`` protocol."""

    def __init__(self, state: TuiState, console: Console | None = None) -> None:
        self.state = state
        self.console = console if console is not None else Console(file=sys.stdout)
        self._lock = threading.Lock()
        self._live: Live | None = None
        self.frames: list[str] = []
        """Text of every frame drawn on an event (for recordings; empty when live)."""
        self.record_frames = False

    @classmethod
    def start(
        cls,
        judge: Judge,
        options: ExtractOptions,
        command: str | None = None,
        console: Console | None = None,
    ) -> FlyTui:
        """Build the state from the judge and open the live display (stdout by default)."""
        del options
        command = command if command is not None else "ftl " + " ".join(sys.argv[1:])
        tui = cls(TuiState.from_judge(judge, command), console=console)
        tui.state.log("DAN", "(inference, no plasticity)", "dim")
        tui._live = Live(
            get_renderable=tui._renderable,
            console=tui.console,
            refresh_per_second=REFRESH_PER_SECOND,
            transient=False,
        )
        tui._live.start()
        return tui

    def _renderable(self) -> RenderableType:
        size = self.console.size
        with self._lock:
            return render(self.state, size.width, size.height)

    def snapshot(self, width: int = 120, height: int = 40) -> str:
        """The current frame as plain text (a recording console, no ANSI)."""
        console = Console(record=True, width=width, height=height, file=io.StringIO())
        with self._lock:
            console.print(render(self.state, width, height))
        return console.export_text()

    # -- Observer -------------------------------------------------------------------

    def on_jobs(self, jobs: list[FileJob], stats: FlyStats) -> None:
        """All files that will be sniffed are known."""
        with self._lock:
            self.state.files_total = len(jobs)
            self.state.brain_started = time.perf_counter()
            self.state.log(
                "WALK",
                f"{stats.files_walked} .py files, {len(jobs)} to sniff, "
                f"{stats.files_prefiltered} without i18n names, {stats.files_cached} cached",
            )

    def on_judged(self, job: FileJob, judged: JudgedFile, stats: FlyStats) -> None:
        """One file has its verdicts: raster rows, drive, event lines."""
        with self._lock:
            s = self.state
            s.files_done += 1
            s.candidates = stats.candidates
            s.keys = stats.keys
            s.trials = stats.trials
            s.resniffs = stats.resniffs
            for i, (candidate, verdict) in enumerate(
                zip(judged.candidates, judged.verdicts, strict=True)
            ):
                if verdict is None:
                    continue
                seed = trial_seed(job.content, i, ENCODER_VERSION)
                window = len(candidate.window.tokens())
                s.log(
                    "ODOR",
                    f"{job.path}:{candidate.line}:{candidate.column}  0x{seed >> 40:06x}  "
                    f"window {window} tok  {candidate.kind} {candidate.text}",
                )
                if verdict.sniffs > 1:
                    s.log("RESNIFF", f"×{verdict.sniffs}  |margin| < θ {s.theta_key:.2f}", "yellow")
                answer = (
                    f"→ KEY  {describe_key(judged, i)}"
                    if verdict.is_positive
                    else f"→ NOT A KEY  ({candidate.text})"
                )
                s.log(
                    "MBON",
                    f"margin {verdict.margin:+.2f} {answer}",
                    "bold green" if verdict.is_positive else "magenta",
                )
                s.raster.append(verdict.kc_bits(s.n_kc))
                s.last = verdict
            for key in judged.keys:
                for kwarg, kv in key.kwarg_verdicts:
                    answer = "placeable" if kv.is_positive else "ignored"
                    s.log(
                        "MBON",
                        f"kwarg {kwarg.name}= of {key.key_name}: margin {kv.margin:+.2f} "
                        f"→ {answer}",
                        "green" if kv.is_positive else "dim",
                    )
                    s.raster.append(kv.kc_bits(s.n_kc))
                    s.last = kv
        if self.record_frames:
            self.frames.append(self.snapshot())

    def close(self) -> None:
        """Stop the live display (the final frame stays on screen)."""
        with self._lock:
            self.state.finished = True
        if self._live is not None:
            self._live.refresh()
            self._live.stop()
            self._live = None


__all__ = [
    "Event",
    "FlyTui",
    "TuiState",
    "bar",
    "braille_rows",
    "describe_key",
    "group_thousands",
    "render",
    "sigmoid",
]
