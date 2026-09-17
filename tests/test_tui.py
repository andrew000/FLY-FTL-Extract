"""The TUI renders real numbers at 120×40 and 80×24, and never touches ``random``."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from fly_ftl_extract.cli.extract import FlyStats, prepare_file
from fly_ftl_extract.cli.workers import FileJob
from fly_ftl_extract.dopamine import Judge, JudgedFile, WeightsMismatchError, WeightsMissingError
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.tui import FlyTui, TuiState, bar, braille_rows, group_thousands, render

REPO = Path(__file__).resolve().parent.parent
TUI_SOURCE = REPO / "fly_ftl_extract" / "tui" / "__init__.py"
FIXTURE_FILE = REPO / "tests" / "fixtures" / "projects" / "basic" / "app" / "handlers" / "second.py"
OPTIONS = ExtractOptions(code_path="app", locales_path="locales")


def test_no_random_in_the_tui() -> None:
    source = TUI_SOURCE.read_text(encoding="utf-8")
    assert "import random" not in source
    assert "np.random" not in source
    assert "randint" not in source


def test_braille_rows_encodes_dots_exactly() -> None:
    # 4 trials × 2 cells: the first trial lights cell 0, the last lights cell 1
    matrix = np.zeros((4, 2), dtype=bool)
    matrix[0, 0] = True
    matrix[3, 1] = True
    (row,) = braille_rows(matrix, width=1, height=1)
    assert row == chr(0x2800 + 0x01 + 0x80)  # dot 1 (row 0, left) + dot 8 (row 3, right)
    empty = braille_rows(np.zeros((0, 10), dtype=bool), width=3, height=2)
    assert empty == ["⠀" * 3] * 2
    # more trials than rows: the newest trials win (bottom-aligned)
    tall = np.zeros((10, 1), dtype=bool)
    tall[-1, 0] = True
    (row,) = braille_rows(tall, width=1, height=1)
    assert row == chr(0x2800 + 0x40)  # dot 7 = bottom-left


def test_bar_and_grouping() -> None:
    assert bar(0.0) == "▁" * 10
    assert bar(1.0) == "▇" * 10
    assert bar(0.5) == "▇" * 5 + "▁" * 5
    assert group_thousands(2935) == "2 935"


@pytest.fixture(scope="module")
def judge() -> Judge:
    try:
        return Judge(OPTIONS)
    except (WeightsMissingError, WeightsMismatchError) as err:
        pytest.skip(f"no trained fly: {err}")


@pytest.fixture(scope="module")
def judged_file(judge: Judge) -> tuple[FileJob, JudgedFile]:
    prepared = prepare_file(0, str(FIXTURE_FILE), OPTIONS)
    assert isinstance(prepared, FileJob)
    return prepared, judge.judge_candidates(prepared.candidates, prepared.content)


def _stats(job: FileJob, judged: JudgedFile) -> FlyStats:
    stats = FlyStats(files_walked=1, candidates=len(job.candidates), keys=len(judged.keys))
    stats.trials = judged.trials
    stats.resniffs = judged.resniffs(1)
    return stats


@pytest.fixture(scope="module")
def judged_state(judge: Judge, judged_file: tuple[FileJob, JudgedFile]) -> TuiState:
    job, judged = judged_file
    tui = FlyTui(
        TuiState.from_judge(judge, "ftl extract app locales"), console=Console(file=io.StringIO())
    )
    tui.on_jobs([job], _stats(job, judged))
    tui.on_judged(job, judged, _stats(job, judged))
    tui.state.finished = True
    return tui.state


@pytest.mark.parametrize(("width", "height"), [(120, 40), (80, 24)])
def test_render_shows_real_numbers(judged_state: TuiState, width: int, height: int) -> None:
    console = Console(record=True, width=width, height=height, file=io.StringIO())
    console.print(render(judged_state, width, height))
    text = console.export_text()
    n_neurons = judged_state.n_neurons
    assert n_neurons > 2000
    assert f"NEURONS ONLINE {group_thousands(n_neurons)}" in text
    assert "SPIKE RASTER" in text
    assert re.search(rf"KC 0–\d+ of {judged_state.n_kc}", text)
    assert "REGION DRIVE" in text
    mbon_lines = [line for line in text.splitlines() if "MBON" in line and "margin" in line]
    assert mbon_lines, text
    assert re.search(r"margin [+-]\d+\.\d+", text)
    if width >= 100:  # the narrow footer drops the command line
        assert "ftl extract app locales" in text
    assert "files 1/1" in text
    assert "trials/s" in text
    # the raster carries the real spikes: some dots set, most not (sparse KC code)
    cells = [c for c in text if 0x2800 <= ord(c) <= 0x28FF]
    lit = [c for c in cells if ord(c) > 0x2800]
    assert lit, "no Kenyon-cell spikes drawn"
    assert len(lit) < len(cells), "every raster cell is full — that is not a sparse code"
    assert all(len(line) <= width for line in text.splitlines())


def test_live_driver_starts_updates_and_stops(
    judge: Judge, judged_file: tuple[FileJob, JudgedFile]
) -> None:
    job, judged = judged_file
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=True, width=120, height=40)
    tui = FlyTui.start(judge, OPTIONS, command="ftl extract app locales", console=console)
    try:
        tui.on_jobs([job], _stats(job, judged))
        tui.on_judged(job, judged, _stats(job, judged))
    finally:
        tui.close()
    out = sink.getvalue()
    assert "NEURONS ONLINE" in out
    assert "MBON" in out
    assert tui.state.files_done == 1
    assert len(tui.state.raster) == len(judged.all_verdicts())
