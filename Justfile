# fly-ftl-extract — shortcuts for playing with the fly.
#   just                      list recipes
#   just sniff app locales -l en -l uk
#
# On Windows `just` uses PowerShell (set below); elsewhere `sh`. Every recipe is
# written to work in both: multi-step recipes are one command per line (each line is
# its own shell and `just` stops at the first failure), so no `&&` is needed.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

root    := justfile_directory()
scratch := root / ".cache" / "scratch"

default:
    @just --list --unsorted

# ─── Sniff ───────────────────────────────────────────────────────────────────

# `ftl extract` through the fly, with the TUI. Example: just sniff app locales -l en
sniff *ARGS:
    uv run ftl extract {{ARGS}}

# Same, but plain log with the margin of every window after "✅ Done" (no TUI)
sniff-v *ARGS:
    uv run ftl extract -v --fly-no-tui {{ARGS}}

# Fly + AST reference side by side; differences go to stderr, exit 1 if any
audit *ARGS:
    uv run ftl extract --fly-audit --fly-no-tui {{ARGS}}

# Write nothing, just show what would happen (audit + margins)
dry *ARGS:
    uv run ftl extract --dry-run --fly-audit -v --fly-no-tui {{ARGS}}

# "Another nose": the same fly with a different seed salt — does the answer change?
nose SEED *ARGS:
    uv run ftl extract --fly-seed {{SEED}} --fly-audit -v --fly-no-tui {{ARGS}}

# Sniff every window N times before the resniff rule kicks in
trials N *ARGS:
    uv run ftl extract --fly-trials {{N}} --fly-audit -v --fly-no-tui {{ARGS}}

# A fixture from tests/ through the fly with the TUI (just fx basic; just fx prefix -p self -p cls)
fx NAME *ARGS:
    uv run ftl extract tests/fixtures/projects/{{NAME}}/app {{scratch}}/fx-{{NAME}}/locales -l en {{ARGS}}

# ─── Any real project, read-only ─────────────────────────────────────────────
# Runs from the project root (so its pyproject [tool.ftl-extract.extract] applies),
# --dry-run, cache redirected to .cache/scratch, full log with every margin saved there.
#   just project C:/path/to/bot        (or ./relative/path)

# Dry-run audit of a project; prints only audit lines and the statistics
project DIR:
    uv run python -c "import os; os.makedirs(r'{{scratch}}', exist_ok=True)"
    cd "{{DIR}}"; uv run --project "{{root}}" python -c "import subprocess; subprocess.call(['uv', 'run', '--project', r'{{root}}', 'ftl', 'extract', '--dry-run', '--fly-audit', '-v', '--fly-no-tui', '--cache-path', r'{{scratch}}/project-cache.bin'], stdout=open(r'{{scratch}}/project_run.txt', 'w', encoding='utf-8'), stderr=subprocess.STDOUT)"
    uv run python -c "import re; [print(l.rstrip()) for l in open(r'{{scratch}}/project_run.txt', encoding='utf-8') if re.search(r'fly::audit|Fly statistics|Trials|keys in code|Done in', l)]"

# Only the audit differences from the last `just project` run
project-diff:
    uv run python -c "[print(l.rstrip()) for l in open(r'{{scratch}}/project_run.txt', encoding='utf-8') if 'fly::audit' in l]"

# A project with the TUI (slower: batch 16), nothing written
project-tui DIR:
    cd "{{DIR}}"; uv run --project "{{root}}" ftl extract --dry-run --cache-path "{{scratch}}/project-cache.bin"

# ─── Checks ──────────────────────────────────────────────────────────────────

# ruff + mypy + pytest (~3 min); stops at the first failure
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy fly_ftl_extract
    uv run pytest -q

# Fast tests only (skips golden-through-the-fly and fixtures-through-the-fly)
test-fast:
    uv run pytest -q -x --deselect tests/test_extract_matches_golden.py --deselect tests/test_judge_on_fixtures.py

# Real ftl 0.12.1 vs ours on every fixture → docs/COMPARE.md
compare *ARGS:
    uv run python scripts/compare_with_reference.py {{ARGS}}

# Golden files unchanged (regenerated with the real binary and compared)
golden-check:
    uv run python scripts/gen_golden.py --check

# The fly vs the teacher on every fixture (the Phase 5 gate)
fixtures:
    uv run pytest -q tests/test_judge_on_fixtures.py

# The project's guarantee: a plain extract never loads ast / reference / audit
no-ast:
    uv run pytest -v tests/test_no_ast_in_hot_path.py

# ─── Brain and training ──────────────────────────────────────────────────────

# Dataset from the current grammar → data/dataset/ (~4 min)
dataset *ARGS:
    uv run python scripts/make_dataset.py {{ARGS}}

# Full training (brain with state cache + readout). just train "attempt 13"
train LABEL *ARGS:
    uv run python scripts/train.py --attempt "{{LABEL}}" {{ARGS}}

# Quick readout estimate: 2 seeds, 150k rows, patience 5 (brain from cache when present)
train-quick LABEL:
    uv run python scripts/train.py --attempt "{{LABEL}}" --train-seeds 2 --train-rows 150000 --patience 5

# Linear probe on the odours alone, no brain — the encoder's ceiling
proxy *ARGS:
    uv run python scripts/encoder_proxy.py {{ARGS}}

# Twin diagnostics (core.internal vs nested.internal): probe, PN buckets, KC Jaccard, margins
twins *ARGS:
    uv run python scripts/twin_diagnostics.py {{ARGS}}

# Brain speed (trials/s) → docs/BENCH.md
bench:
    uv run python scripts/bench.py

# syn_scale / apl_scale calibration grid on real candidates
calibrate *ARGS:
    uv run python scripts/calibrate.py {{ARGS}}

# Odour statistics on the fixtures → docs/ODOR.md
odor:
    uv run python scripts/odor_report.py

# Rebuild the connectome from .cache/flywire/ (needs the `connectome` extra)
connectome:
    uv run --extra connectome python scripts/build_connectome.py

# TUI frames on a fixture → docs/tui_basic.{txt,svg}
record-tui *ARGS:
    uv run python scripts/record_tui.py {{ARGS}}

# ─── Release and cleanup ─────────────────────────────────────────────────────

# wheel + sdist; list the data/ files inside the wheel
build:
    uv build
    uv run python -c "import zipfile,glob; z=zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1]); [print(i.filename, i.file_size) for i in z.infolist() if '/data/' in i.filename]"

# Remove extractor caches and scratch (brain states in .cache/brain_states are kept)
clean:
    uv run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in (r'{{scratch}}', '.ftl-extract-cache', '.pytest_cache', '.ruff_cache', '.mypy_cache')]"

# Also drop the brain state cache (the next train recomputes ~1.5 h of brain)
clean-brain:
    uv run python -c "import shutil; shutil.rmtree('.cache/brain_states', ignore_errors=True)"
