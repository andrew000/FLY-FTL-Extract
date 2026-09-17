"""Record one ``ftl extract`` run with the TUI on a fixture — frames as text, final frame as SVG.

No asciinema/gif tooling is assumed on the machine, so the recording is what ``rich`` can
export itself: every frame the dashboard drew on an observer event (``docs/tui_<fixture>_frames.txt``,
frames separated by a ruler), the final frame as plain text (``docs/tui_<fixture>.txt``) and
as SVG with colours (``docs/tui_<fixture>.svg``).  Numbers are those of the real run.

Usage::

    uv run python scripts/record_tui.py [--fixture basic] [--width 120] [--height 40]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from fly_ftl_extract.cli.config import load_pyproject, resolve_options
from fly_ftl_extract.cli.extract import FlyOptions, extract_with_fly, load_judge
from fly_ftl_extract.tui import FlyTui, TuiState, render

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extract_argv import parse_extract_argv

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"
DOCS = REPO / "docs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="basic")
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=40)
    ns = parser.parse_args()
    spec = json.loads((FIXTURES / ns.fixture / "args.json").read_text(encoding="utf-8"))
    run, argv = next(iter(spec["runs"].items()))
    with tempfile.TemporaryDirectory(prefix="fly-ftl-tui-") as tmp:
        work = Path(tmp) / ns.fixture
        shutil.copytree(FIXTURES / ns.fixture, work)
        cwd = os.getcwd()
        os.chdir(work)
        try:
            overrides, config = parse_extract_argv(argv)
            options = resolve_options(overrides, load_pyproject(config, str(work)))
            fly = FlyOptions(tui=True, workers=1)
            judge = load_judge(options, fly)
            command = "ftl extract " + " ".join(argv)
            console = Console(
                record=True,
                width=ns.width,
                height=ns.height,
                file=io.StringIO(),
                force_terminal=True,
            )
            tui = FlyTui(TuiState.from_judge(judge, command), console=console)
            tui.record_frames = True
            tui.state.log("DAN", "(inference, no plasticity)", "dim")
            result = extract_with_fly(options, fly, judge, tui)
            tui.state.finished = True
            final_text = tui.snapshot(ns.width, ns.height)
            console.print(render(tui.state, ns.width, ns.height))
            svg = console.export_svg(title=f"fly-ftl-extract · {command}")
        finally:
            os.chdir(cwd)
    ruler = "\n" + "=" * ns.width + "\n"
    (DOCS / f"tui_{ns.fixture}_frames.txt").write_text(
        ruler.join([*tui.frames, final_text]), encoding="utf-8", newline="\n"
    )
    (DOCS / f"tui_{ns.fixture}.txt").write_text(final_text, encoding="utf-8", newline="\n")
    (DOCS / f"tui_{ns.fixture}.svg").write_text(svg, encoding="utf-8", newline="\n")
    print(final_text)
    print(
        f"{len(tui.frames) + 1} frames → docs/tui_{ns.fixture}_frames.txt, final frame → "
        f"docs/tui_{ns.fixture}.txt / .svg; keys {len(result.extraction.keys)}, "
        f"trials {result.stats.trials}, {result.stats.brain_seconds:.2f} s brain ({run})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
