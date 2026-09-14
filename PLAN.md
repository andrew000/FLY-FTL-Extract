# fly-ftl-extract development plan

Every phase ends with a **Check** list (commands that must be green) and an **Audit** list
(what to show the reviewer: the diff between tags, concrete files/numbers). The order of the
phases matters — the brain and the dataset come before the CLI, because if the fly cannot
learn, building the CLI makes no sense.

The time estimates are a guide for the agent, not a deadline.

---

## Phase 0 — Skeleton (≈ 1 h)

- `uv init`, `pyproject.toml`: name `fly-ftl-extract`, python `>=3.14,<3.15`, deps
  `numpy scipy rich click`, extra `connectome = [pyarrow, pandas]`, dev `pytest ruff mypy`.
  Entry points `ftl` and `fly-ftl` → `fly_ftl_extract.cli:main`.
- Empty packages with `__init__.py` following the layout in CLAUDE.md. `data/` with `.gitkeep`.
- `tests/test_no_ast_in_hot_path.py`: imports `fly_ftl_extract.cli`, then checks that
  `fly_ftl_extract.reference` is **not** in `sys.modules`, and that no file outside
  `reference/`, `scripts/`, `tests/` contains `import ast` / `from ast` /
  `fly_ftl_extract.reference`.
- `ruff.toml` (or in pyproject): `select = ["ALL"]` with sensible ignores; `mypy --strict`.
- Install the reference into a separate tool env: `uv tool install ftl-extract==0.12.1` — so
  that the original's `ftl` binary is available as `uv tool run --from ftl-extract ftl`. Our
  `ftl` lives in the project venv, theirs in the tool env; do not mix them up. In
  `scripts/compare_with_reference.py` the path to the reference binary comes from the env
  `FTL_REFERENCE_BIN` or `uv tool run`.

**Check**
```
uv run ruff check . && uv run mypy fly_ftl_extract && uv run pytest -q
uv tool run --from ftl-extract==0.12.1 ftl --version
```
**Audit**: tag `phase-0`; pyproject; the test that forbids ast.

---

## Phase 1 — Reference and golden fixtures (≈ 2 h)

Goal: to have a truth to compare the fly against.

- `tests/fixtures/projects/`: at least 8 mini projects, each a directory `app/` + an empty
  `locales/` + (optionally) a `pyproject.toml` with `[tool.ftl-extract.extract]`. Cover:
  1. basic: `i18n.get("k")`, `i18n.a.b()`, kwargs, `_path=`;
  2. prefix `self.i18n` with `-p`;
  3. `-k LF -K LazyProxy`, `I18nFormat(..., when=...)` with `--ignore-kwargs when`;
  4. ignore attributes (`set_locale`, custom ones via `-i/-I`);
  5. existing locales with translations + unused keys (`comment` and `warn` modes);
  6. `# ftl-extract: ignore stale/all` markers;
  7. two files with a conflicting `_path=` for one key (the original aborts — so do we);
  8. a file with a syntax error (with and without `--allow-parse-errors`);
  9. `exclude-dirs`, nested directories, `__pycache__`, `.venv`;
  10. several languages `-l en -l uk`, `--line-endings crlf`.
- `scripts/gen_golden.py`: for every fixture runs the **real** `ftl extract` with the
  arguments from `fixture/args.json`, copies the result into `tests/golden/<name>/` together
  with stdout (without timings) in `stdout.txt`. The golden files are committed.
- `fly_ftl_extract/reference/`: an ast extractor that reproduces the original's semantics
  (used as the teacher). Test `tests/test_reference_matches_golden.py`: reference + our
  `ftl/` writer → byte for byte == golden for every fixture. That is, the writer/merge/comment
  logic (`ftl/`) is written here already and checked against the reference BEFORE the fly.

**Check**
```
uv run python scripts/gen_golden.py --check   # golden files unchanged
uv run pytest tests/test_reference_matches_golden.py -q
```
**Audit**: tag `phase-1`; the list of fixtures with a short description; the diffs, if
`ftl/` had to be bent to some odd behaviour of the original (that is the most valuable
knowledge).

---

## Phase 2 — Connectome (≈ 2 h + download)

- The user downloads into `.cache/flywire/`: `proofread_connections_783.feather` (Zenodo
  10676866) and `Supplemental_file1_neuron_annotations.tsv` (GitHub flyconnectome). The script
  prints clear instructions if the files are missing.
- `scripts/build_connectome.py`:
  1. read the annotations, print `value_counts()` over `cell_class` and `super_class`
     (save them in `docs/CONNECTOME.md` — the reviewer needs to see the real names);
  2. select the neurons: PN (uniglomerular ALPN), KC (all), APL, MBON, DAN, one hemisphere;
  3. filter the edges between them, `syn_count >= 5`, sign by `nt_type`;
  4. save `data/mb_fafb783.npz` (csr `indptr/indices/data`, arrays `root_id`,
     `cell_class`, `cell_type`, `side`, group indices `pn_idx`, `kc_idx`, `apl_idx`,
     `mbon_idx`, `dan_idx`) + `meta.json` (counts, hash, date, licence CC-BY 4.0,
     citations Dorkenwald et al. 2024 / Schlegel et al. 2024).
- `fly_ftl_extract/brain/connectome.py`: `load()` → dataclass `Connectome` with validation
  (KC ≥ 1500, PN ≥ 50, exactly 1 APL, MBON ≥ 20; otherwise an error with an explanation).
- Sanity tests on the data: PN→KC edges only positive; APL→KC only negative;
  the mean number of PN inputs per KC within 3–12 (biologically ~6 claws).

**Check**
```
uv run python scripts/build_connectome.py
uv run pytest tests/test_connectome.py -q
```
**Audit**: tag `phase-2`; `docs/CONNECTOME.md` with the value_counts and the final counts;
the npz size; those three sanity numbers.

---

## Phase 3 — LIF simulation (≈ 3 h)

- `brain/params.py`: `BrainParams` (values from CLAUDE.md, reference to Shiu et al. 2024).
- `brain/lif.py`: `simulate(odors: np.ndarray[n_trials, n_pn], seed) -> TrialResult`
  with fields `kc_counts (n_trials, n_kc) int16`, `mbon_counts`, `apl_counts`,
  `kc_active_fraction`, `spike_raster` (optional, for the TUI, only for the first N trials).
  Vectorised over trials; the synaptic current through one `csr @ dense` per step.
- Weight calibration (`syn_scale`): pick it so that for a typical odour (30 % of PNs active
  at 150 Hz) the active KC share is 5–10 % **with APL** and > 30 % without APL. The script
  `scripts/calibrate.py` prints the curve; the result → into `BrainParams` with a comment.
- Benchmark `scripts/bench.py` → `docs/BENCH.md` (trials/s at batch 64/256/1024).
- Tests: determinism (same seed → bit for bit), sparsity within bounds, two different odours →
  different KC patterns (Jaccard < 0.5), the same odour with a different seed → similar
  (Jaccard > 0.5).

**Check**
```
uv run pytest tests/test_lif.py -q
uv run python scripts/bench.py   # ≥ 200 trials/s at batch 256
```
**Audit**: tag `phase-3`; BENCH.md; the calibration curve; the Jaccard test.

---

## Phase 4 — Tokenizer and odour (≈ 3 h)

- `tokenizer/candidates.py`: `iter_candidates(source: str, opts) -> Iterator[Candidate]`
  on `tokenize.generate_tokens`. A candidate: (a) a STRING token, (b) a chain NAME(.NAME)*
  right before `(`. For each — a `Window` (12 before / 6 after, features from CLAUDE.md),
  the position (line:column), and for (b) the already-formed key name (`a.b_c` → `a-b-c`),
  for (a) the raw string without quotes. Also collect the kwargs of the call in which the
  candidate is the first argument (names + the `_path=` value), and for every kwarg its own
  `Window`. All on tokens, no ast. f-strings / concatenations → no candidates are created (as
  in the original — check with the reference that they are ignored there too).
- `odor/encoder.py`: `encode(window, opts) -> np.ndarray[n_pn]`. Token normalisation per
  CLAUDE.md; hashing of n-grams (1..3) with the relative position into `n_pn` buckets
  (blake2b salted with `ENCODER_VERSION`), value = the sum of weights, then `tanh`
  normalisation into [0,1]. `ENCODER_VERSION` is a constant, bumped on any change.
- Tests: the candidates from the fixtures agree with what `reference/` finds (the set of
  positions of positive candidates ⊂ the set of all candidates — that is recall ≡ 1 at the
  candidate level); the encoder is deterministic; different windows → different vectors.

**Check**
```
uv run pytest tests/test_tokenizer.py tests/test_odor.py -q
```
**Audit**: tag `phase-4`; the `iter_candidates` output for fixture no. 1 (a human-readable
dump); the distribution of active PNs per odour (a histogram in docs).

---

## Phase 5 — Dataset and dopamine (≈ 4 h) — the riskiest

- `scripts/make_dataset.py`: a grammar of random snippets (see CLAUDE.md), ≥ 20 000
  snippets, for each the candidates from Phase 4 and the label from `reference/`. Two tables:
  `keys.jsonl` (window, label is_key) and `kwargs.jsonl` (window, label is_placeable).
  Class balance ≈ 1:3, split 80/10/10, fixed seed. Store the odours already encoded (npz),
  so that train does not depend on the tokenizer.
- `dopamine/train.py`: run the odours through the brain (in batches), KC state → features
  (`[spiked, log1p(count)]`), train two linear readouts (delta rule, L2, early stopping on
  val). Save `data/mbon_weights.npz` with `brain_hash`, `encoder_version`, `metrics`.
- `dopamine/infer.py`: `Judge.is_key(states) -> (labels, margins)`, resniff logic when
  |margin| < θ (θ is in the npz as well).
- `docs/METRICS.md`: precision/recall/F1 on the test split for both tasks, the confusion
  matrix, the 10 worst examples (window text). Separately — the result on the golden fixtures
  (holdout).

Goals: P, R ≥ 0.995 (test), 100 % on the fixtures. If it does not work out, in this order:
1) check KC sparsity and that APL works; 2) enlarge the window/n-grams; 3) add
`log1p(count)` features if there were only binary ones; 4) resniff up to 5 trials. Do not add
rules to the code. If after that it is < 0.99 — stop, write to the reviewer with METRICS.md.

**Check**
```
uv run python scripts/make_dataset.py && uv run python scripts/train.py
uv run pytest tests/test_judge_on_fixtures.py -q
```
**Audit**: tag `phase-5`; METRICS.md; the weights size; the train time.

---

## Phase 6 — CLI, merging, TUI (≈ 4 h)

- `cli/`: a click group `ftl` with `extract`, `stub`, `check`, `config`. pyproject parsing
  (`--config`, upward search), merge CLI > toml > defaults; check the list of defaults
  against the original's `ftl config sample --command extract`.
- `cli/extract.py`: file walk (exclude globs as in the original: `__pycache__`, `venv`,
  `.venv`, `.git`, `.pytest_cache`), for every file — candidates → odours → brain
  (a batch over all candidates of the file or over `--fly-batch`) → Judge → `ftl/` writer.
  `--cache`: a cache keyed by (mtime, size, opts-hash, weights-hash) in
  `.ftl-extract-cache/extract-<version>-v<schema>.bin` — our own format, the name as in the
  original.
- `--fly-audit`: the reference in parallel; differences → stderr and exit 1.
- `tui/`: after the mock-up in CLAUDE.md; all numbers real; switched off without a tty and
  with `--fly-no-tui`.
- The statistics at the end — as in the original + the fly block (`neurons`, `trials`,
  `resniffs`, `trials/s`).

**Check**
```
uv run pytest -q                                   # everything, incl. golden through the fly
uv run python scripts/compare_with_reference.py    # every fixture: the diff is empty
uv run ftl extract tests/fixtures/projects/basic/app /tmp/out -l en --fly-audit
```
**Audit**: tag `phase-6`; a terminal recording (asciinema or gif) of one run with the TUI;
the `compare_with_reference.py` output; confirmation that `test_no_ast_in_hot_path` is still
green.

---

## Phase 7 — Real project, README, release (≈ 2 h)

- Run on the user's real project (an aiogram bot with `i18n`) with `--fly-audit`; the
  differences go to an issue list, with an analysis of why the fly erred (add such cases to
  the dataset grammar, retrain, repeat).
- README: what it is, how it works (one diagram PN→KC⇄APL→MBON), installation, how to
  download the connectome, an honest «Limitations» section (speed, stub/check), FlyWire
  citations (CC-BY 4.0: Dorkenwald et al. 2024; Schlegel et al. 2024; Shiu et al. 2024 for the
  model).
- `uv build`, check that the wheel contains `data/*.npz`, install into a clean venv, run a
  fixture.

**Check**
```
uv build && pip install dist/*.whl -t /tmp/clean && (cd /tmp/clean && python -m fly_ftl_extract --version)
```
**Audit**: tag `v0.1.0`; README; the report from the real project (number of keys,
differences, trials/s).

---

## What to send the reviewer after every phase

1. `git diff phase-(N-1)..phase-N --stat` and a link to / archive of the repo.
2. The output of the commands from the «Check» block (complete, not truncated).
3. The artefacts named under «Audit» for this phase.
4. The list of decisions that had to be taken against CLAUDE.md/PLAN.md, with the reason.

The reviewer answers with a prioritised list of remarks: **blocker** (the phase is not
accepted), **should** (fix before the next phase), **nit**.
