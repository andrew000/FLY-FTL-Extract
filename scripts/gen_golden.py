"""Generate (or check) golden outputs with the ORIGINAL ftl-extract 0.12.1 binary.

Never edit golden files by hand.  For every fixture in ``tests/fixtures/projects/<name>``
and every run in its ``args.json`` the fixture is copied to a temporary directory, the
reference ``ftl extract <args>`` is executed there, and the resulting tree (everything except
Python sources and the fixture metadata) plus ``stdout.txt`` / ``stderr.txt`` / ``exit_code.txt``
is stored under ``tests/golden/<name>/<run>/``.

Timings in the logs are replaced by ``<t>`` so the output is reproducible.

Usage::

    uv run python scripts/gen_golden.py            # (re)generate
    uv run python scripts/gen_golden.py --check    # fail if committed golden differs
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reference_bin import reference_ftl_command  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "projects"
GOLDEN = REPO / "tests" / "golden"
FIXTURE_ONLY = {"args.json"}
TIMING_RE = re.compile(r"in \d+\.\d+s\.")


def scrub(text: str, work: Path) -> str:
    """Make the captured log reproducible: timings -> ``<t>``, the temp work dir -> ``<cwd>``.

    The reference prints paths from ``pyproject.toml`` joined to the (absolute) config
    directory, so that directory shows up verbatim in the log.
    """
    text = TIMING_RE.sub("in <t>s.", text)
    return text.replace(str(work), "<cwd>")


def assert_no_pyproject_above(path: Path) -> None:
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            msg = f"{parent} contains a pyproject.toml; the reference would pick it up"
            raise SystemExit(msg)


def run_fixture(name: str, run: str, args: list[str], out_dir: Path, work_root: Path) -> None:
    work = work_root / f"{name}__{run}"
    shutil.copytree(FIXTURES / name, work)
    if not (work / "pyproject.toml").exists():
        assert_no_pyproject_above(work)
    cmd = [*reference_ftl_command(), "extract", *args]
    result = subprocess.run(  # noqa: S603
        cmd, cwd=work, capture_output=True, text=True, encoding="utf-8", check=False, timeout=300
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for src in sorted(work.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(work)
        if src.suffix == ".py" or rel.name in FIXTURE_ONLY:
            continue
        dst = out_dir / "tree" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    stdout = scrub(result.stdout, work)
    stderr = scrub(result.stderr, work)
    (out_dir / "stdout.txt").write_text(stdout, encoding="utf-8", newline="")
    (out_dir / "stderr.txt").write_text(stderr, encoding="utf-8", newline="")
    (out_dir / "exit_code.txt").write_text(f"{result.returncode}\n", encoding="utf-8", newline="")
    (out_dir / "command.txt").write_text(
        "ftl extract " + " ".join(args) + "\n", encoding="utf-8", newline=""
    )


def generate(target: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fly-ftl-golden-") as tmp:
        for fixture in sorted(FIXTURES.iterdir()):
            if not fixture.is_dir():
                continue
            spec = json.loads((fixture / "args.json").read_text(encoding="utf-8"))
            for run, args in spec["runs"].items():
                run_fixture(fixture.name, run, args, target / fixture.name / run, Path(tmp))
                print(f"{fixture.name}/{run}: ftl extract {' '.join(args)}")


def tree_diff(a: Path, b: Path) -> list[str]:
    diffs: list[str] = []
    cmp = filecmp.dircmp(a, b)

    def walk(c: filecmp.dircmp, prefix: str) -> None:
        diffs.extend(f"only in golden: {prefix}{x}" for x in c.left_only)
        diffs.extend(f"only in fresh:  {prefix}{x}" for x in c.right_only)
        diffs.extend(f"differs:        {prefix}{x}" for x in c.diff_files)
        for sub, subcmp in c.subdirs.items():
            walk(subcmp, f"{prefix}{sub}/")

    walk(cmp, "")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against committed golden")
    ns = parser.parse_args()
    version = subprocess.run(  # noqa: S603
        [*reference_ftl_command(), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"reference: {version}")
    if version != "ftl 0.12.1":
        print("expected ftl 0.12.1", file=sys.stderr)
        return 2
    if not ns.check:
        if GOLDEN.exists():
            shutil.rmtree(GOLDEN)
        generate(GOLDEN)
        return 0
    with tempfile.TemporaryDirectory(prefix="fly-ftl-golden-check-") as tmp:
        fresh = Path(tmp) / "golden"
        generate(fresh)
        diffs = tree_diff(GOLDEN, fresh)
    if diffs:
        print("golden differs from a fresh reference run:")
        print("\n".join(diffs))
        return 1
    print("golden up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
