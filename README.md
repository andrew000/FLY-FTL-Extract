# fly-ftl-extract

Drop-in replacement for [`ftl-extract`](https://github.com/andrew000/FTL-Extract) 0.12.1 — the
same `ftl extract <code> <locales>`, with the same options and byte-for-byte the same output —
in which the decision «is this a Fluent key» is made not by a parser but by a simulated
fruit-fly mushroom body: LIF neurons on the real FlyWire FAFB v783 connectome (124 projection
neurons → 2597 Kenyon cells ⇄ 1 APL → 48 MBONs), a readout on the MBONs trained with
«dopamine» (delta rule). The tokenizer only proposes candidates (string literals and
attribute chains before `(`) and encodes their context into an odour; the fly decides which
of them is a key and which kwarg is placeable.
**Switch the fly off and no keys are found:** the hot path has neither `ast` nor rules about
`i18n.get`; the test `tests/test_no_ast_in_hot_path.py` checks that a plain `ftl extract`
loads neither the AST reference nor the audit.

## How it works

```
Python file ──tokenize──▶ candidate + window (6 tokens before · focus · 3 after)
                                   │
                                   ▼  feature hashing → 14 puffs × 124 PN (rate coding, tanh)
        ┌─────────────────────────────────────────────────────────────────────┐
        │   PN 124  ──ACh──▶  KC 2597  ──ACh──▶  MBON 48       mushroom body  │
        │  Poisson,          LIF, τm 20 ms,        LIF          of FlyWire    │
        │  ≤ 200 Hz         V_th −45 mV               ▲         FAFB v783,    │
        │                    │       ▲                │   right hemisphere    │
        │                    ▼       │ GABA ×0.5      │         syn_count ≥ 5 │
        │                   APL 1 ───┘                │                       │
        │               (feedback inhibition,         │                       │
        │             8–10 % of KCs active per puff)  │                       │
        │                                             │                       │
        │   DAN 165 — teaching signal only ──────┘      (current = 0 in the sim)│
        └─────────────────────────────────────────────────────────────────────┘
                                   │  KC spike counts per puff (14 × 2597)
                                   ▼
        readout  w · [KC spiked, log1p(spikes)] + b  →  margin
        |margin| < θ  →  «sniff again» (up to 5 extra trials with other seeds), vote = Σ margin
        margin > 0  →  key  /  kwarg placeable
```

- **Temporal coding.** The window = 14 puffs of 40 ms with no silence between them + 10 ms of
  silence (570 ms, 5700 steps of 0.1 ms): 6 puffs of context before the candidate, 5 puffs of
  focus (`[root][attr1][attr2][attr3][summary]` for a chain, `[<STR>][silence×3][summary]` for
  a string), 3 puffs after. The membrane is not reset between puffs — that is the memory of
  the preceding tokens. Every puff is a 124-PN vector: hashed slot features (normalised token,
  token + distance, type, bracket depth, bigram), 2 buckets per feature, ≈ 10 active PNs per
  puff.
- **Normalisation, not rules.** `--i18n-keys`, `-p`, `--ignore-attributes`, `--ignore-kwargs`
  affect only the tokens (`i18n` → `<I18N>`, `self` → `<PREFIX>`, `set_locale` → `<IGNORE>`,
  `when` → `<IGNORE_KW>`), so `-k LF` smells the same as `i18n`. The decision is the sign of
  the margin.
- **Resniff.** θ is chosen on val so that ≤ 10 % of windows get extra sniffs; trial seed =
  sha256(file bytes, candidate index, encoder version) — two runs give the same output.
- **In detail:** [CLAUDE.md](CLAUDE.md) (the contract), [docs/CONNECTOME.md](docs/CONNECTOME.md)
  (the subgraph), [docs/BENCH.md](docs/BENCH.md) (calibration, speed),
  [docs/ODOR.md](docs/ODOR.md) (the encoder), [docs/METRICS.md](docs/METRICS.md) (training),
  [docs/FORMAT.md](docs/FORMAT.md) (what exactly the original does and what is reproduced).

## Installation

**Python 3.14** is required (`requires-python = ">=3.14,<3.15"`). The package is not on PyPI;
it is installed from a local checkout or from the wheel (`uv build` → `dist/`):

```bash
uv tool install --python 3.14 /path/to/checkout/fly_ftl_extract                        # from a checkout, or
uv tool install --python 3.14 ./dist/fly_ftl_extract-0.1.0-py3-none-any.whl
pip install ./dist/fly_ftl_extract-0.1.0-py3-none-any.whl                            # in a 3.14 venv
```

The wheel contains the connectome (`fly_ftl_extract/data/mb_fafb783.npz`, 106 KB), its
metadata (`meta.json`) and the trained weights (`mbon_weights.npz`, 598 KB); runtime
dependencies — `numpy`, `scipy`, `rich`, `click`, `fluent.syntax`.

**Entry-point conflict.** The package installs the scripts `ftl` and `fly-ftl`. The original
`ftl-extract` also installs `ftl`; in one environment the last one installed wins, so do not
put both into one venv — keep them in separate `uv tool` environments or call ours as
`fly-ftl`.

## Usage

Everything as in the original 0.12.1 (the FTL-Extract README): `ftl extract CODE_PATH
LOCALES_PATH`, `-l/--language`, `-k/--i18n-keys`, `-K`, `-p/--i18n-keys-prefix`, `-e/-E`,
`-i/-I`, `--ignore-kwargs`, `--default-ftl-file`, `--comment-keys-mode {comment,warn}`,
`--line-endings`, `--dry-run`, `--cache`/`--cache-path`/`--clear-cache`,
`--allow-parse-errors`, `-v`, the global `--config`, the `[tool.ftl-extract.extract]` section
in `pyproject.toml` (CLI > pyproject > defaults), `ftl config sample`. Output, exit codes and
the `.ftl` tree — as with the Rust binary; after its `✅ Done` the block `[INFO  fly] Fly
statistics:` is printed (neurons, files, candidates, trials, resniffs, trials/s, brain time).

```bash
ftl extract app/bot app/bot/locales -l en -l uk          # drop-in
ftl extract app locales --fly-audit                       # run the AST reference alongside, differences → stderr, exit 1
ftl extract app locales -v --fly-no-tui                   # the margin of every window after ✅ Done
```

Our options (all prefixed `--fly-`):

| option | what it does |
|---|---|
| `--fly-audit` | the AST reference (`fly_ftl_extract.reference`) runs alongside; every difference is a `[WARN  fly::audit] …` line, exit 1 |
| `--fly-no-tui` | a plain log even in a terminal (without a tty the TUI switches itself off) |
| `--fly-workers N` | processes for the brain (spawn; default CPU−2, ≤ 16); below 128 windows always a single process |
| `--fly-batch N` | trials per brain call (256 in a single process, 64 in a worker, 16 under the TUI) |
| `--fly-trials N` | base trials per window before the resniff rule (the margins are summed) |
| `--fly-seed S` | a salt on every trial seed — «another nose»; 0 = the production seeds |

### TUI

In a terminal `ftl extract` draws a live panel (`rich`, ≤ 15 fps): a spike raster of the
Kenyon cells (one dot = one cell that spiked in the trial's summary puff, one row = one
trial), the PN/KC/APL/MBON activity of the last trial, an event log (ODOR → MBON margin →
KEY / NOT A KEY / placeable, RESNIFF) and counters. All numbers are real — from the
connectome, the spikes and the readout; the module has no `random`.

![TUI: ftl extract on the basic fixture](docs/tui_basic.svg)

## Accuracy

Dataset `grammar-5`: 20 000 synthetic snippets labelled by the AST reference (460 672
candidates, 147 876 kwargs; split 80/10/10 by snippet). The golden fixtures from `tests/` and
the real bot's code are not part of the dataset — they are the holdout.

| what | result | source |
|---|---|---|
| keys, test, with resniff (≤ 5 extra trials, 9.8 % of windows) | **P 0.9987 · R 1.0000** (tp 11554, fp 15, fn 0, tn 34354) | [METRICS.md §1](docs/METRICS.md) |
| keys, test, without resniff | P 0.9949 · R 0.9978 | same |
| kwargs (placeable / ignore), test | P 1.0000 · R 1.0000 | same |
| golden fixtures through the whole fly | 24 / 24 files match the reference | `tests/test_judge_on_fixtures.py` |
| the real `ftl 0.12.1` against ours on every fixture | **31 / 31** runs byte for byte (exit, stderr, tree), `config sample` too | [COMPARE.md](docs/COMPARE.md) |
| a real aiogram bot (280 files, 9 992 candidates, 538 keys) | **0 differences** from the reference (`--fly-audit`); the fly of the previous corpus `grammar-4` had 13 on the same code (all of them `L("…")` in dict values, constructor fields and decorator arguments), 15 in the first run | [REAL_PROJECT.md](docs/REAL_PROJECT.md) — the honest before/after story |
| the old weights (`grammar-4`) on the `grammar-5` test | keys R 0.870 (1503 misses) — the price of call positions that were not in the corpus | [METRICS.md §8](docs/METRICS.md) |

## Limitations

- One hemisphere (right: 2597 KCs vs 2580 in the left); the left one is not used.
- 15 GABAergic iPNs per side are excluded from the PNs (three have ≥ 5 synapses onto KCs and would break the rule «PN→KC only positive»).
- `syn_scale = 5.0` and `apl_scale = 0.5` are calibrated on our odours (8–10 % KCs per puff, APL inhibits ≥ 2×), not taken from Shiu et al.; the APL weight is scaled separately.
- A 40 ms puff, not 20 (the reviewer asked for 20; at 20 the per-puff KC code was not reproducible — Jaccard 0.34).
- Synapse threshold `syn_count ≥ 5` per neuron pair; DANs carry no current in the forward simulation.
- Speed ≈ 50 trials/s per process (one trial = 5700 LIF steps), 290–410 trials/s on 16 processes depending on machine load; a bot with ~10 000 candidates takes 24–34 s against 0.02 s for the Rust original.
- `ftl stub` and `ftl check` are not implemented (a message and exit 2).
- The `--cache` cache is written to the same file as in the original, but the format is ours and not compatible with the original.
- `-v` does not reproduce the original's debug lines (`globset`, `Saved …`); our margins come instead.
- `i18n.get("dotted.key.name")` and other invalid Fluent identifiers are written as is — the original does the same (the next run of either will fail reading the `.ftl`).
- The fly makes mistakes. On the `grammar-5` test split (with resniff) 15 false keys among 34 369 negatives and 0 misses among 11 554 positives; all 10 worst false keys are `self.get("…")` / `cls.get("…")` with `-p self -p cls` (margin up to +10.5; for the reference a prefix without an i18n name right after it is not a key). The fly learns only what is in the grammar: the first version (`grammar-4`) missed on the real bot every `L("…", _path=…)` in dict values and in constructor fields (`description=L(…)` on the line after a recognised `name=L(…)`, margin −1…−31), `LF("…")` as the first argument of the decorator `@router.message(` (−8…−32), and called a string in a tuple `("captcha_timeout_task", …)` without a call a key (+7.00). This was cured not with rules in the code but with the `grammar-5` corpus holding these positions and their twin negatives: on the same bot it became 0 differences, and the same 13 windows give +5.7…+40.7 ([REAL_PROJECT.md](docs/REAL_PROJECT.md)). The next unseen construct will likewise be a miss until it gets into the grammar.

## Citations and licences

- Connectome: **FlyWire** FAFB v783, Zenodo [10676866](https://zenodo.org/records/10676866), CC-BY 4.0.
  Dorkenwald S. et al. *Neuronal wiring diagram of an adult brain.* Nature 634, 124–138 (2024).
  Schlegel P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature 634, 139–152 (2024) — neuron annotations ([flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)).
- Neuron model and parameters: Shiu P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature 634, 210–219 (2024) ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).
- Inspiration (the mushroom body as a hash function): Dasgupta S., Stevens C. F., Navlakha S. *A neural algorithm for a fundamental computing problem.* Science 358, 793–796 (2017).
- Fluent: [python-fluent](https://github.com/projectfluent/python-fluent) (`fluent.syntax`, Apache 2.0) — parser and serializer, byte-for-byte compatible with fluent-rs on every golden file.
- The original: [FTL-Extract](https://github.com/andrew000/FTL-Extract) © andrew000, MIT. The output format, key order (`FxHashMap`), file walk and message texts are reproduced from the behaviour of the 0.12.1 binary.
- This package is MIT. The data `fly_ftl_extract/data/mb_fafb783.npz` is a derivative work of FlyWire (CC-BY 4.0).

## How to reproduce

Everything is deterministic (the seeds are fixed); the times are from this machine (train.py: 30 processes for the brain; the `ftl extract` pool: 16).

| step | command | what it does | time |
|---|---|---|---|
| 1 | download `proofread_connections_783.feather` (852 MB), `proofread_root_ids_783.npy`, `Supplemental_file1_neuron_annotations.tsv` into `.cache/flywire/` | raw FlyWire data | — |
| 2 | `uv run --extra connectome python scripts/build_connectome.py` | the mushroom-body subgraph → `data/mb_fafb783.npz` + `meta.json`, `docs/CONNECTOME.md` | ≈ 4 s (the feather is read through a memory map in batches, only the needed columns); a repeated run gives byte-for-byte the same npz, only the build date changes in `meta.json` |
| 3 | `uv run python scripts/make_dataset.py` | 20 000 snippets of the `grammar-5` grammar (14 grammar-4 mutation families + 10 call-position families), labelling by the AST reference, encoding → `data/dataset/` | 12 s generation + 212 s labelling and encoding |
| 4 | `uv run python scripts/train.py --lr 0.005 --epochs 100 --patience 10 --train-seeds 6 [--baseline-weights old.npz]` | the brain on every window (train × 6 seeds, val, test; state cache in `.cache/brain_states/`, ≈ 8 GB per corpus), delta rule, θ, fixtures → `data/mbon_weights.npz`, `docs/METRICS.md` | 12 260 s (3.4 h) without the state cache, 30 processes: brain keys 4466 s (2.3 M trials, ≈ 665 trials/s), kwargs 1310 s, readout keys 5506 s (86 epochs, early stopping), kwargs 223 s, fixtures 23 s; with the state cache — only the readout |
| 5 | `uv run pytest -q` | 297 tests (296 passed + 1 diagnostic xfail), among them golden through the fly and `--fly-audit` on the fixtures | ≈ 3 min |
| 6 | `uv run python scripts/compare_with_reference.py` | the real `ftl 0.12.1` (via `uv tool run`) against ours → `docs/COMPARE.md` | ≈ 45 s |
