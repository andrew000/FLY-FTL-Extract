# Real project: a private aiogram bot through the fly (Phase 7, item 1)

Run of 2026-09-17, `fly-ftl-extract` 0.1.0 (encoder `fly-odor-6`, brain `cd3a9c7616718a25`,
weights from `docs/METRICS.md`). The project is the owner's own aiogram bot with
`[tool.ftl-extract.extract]` in `pyproject.toml` (`code-path = "app/bot"`, languages en/uk/ru,
`i18n-keys-append = ["LF", "LazyProxy"]`, `ignore-attributes-append = ["core"]`,
`line-endings = "crlf"`, `cache = true`, `comment-junks = true`). A read-only run from the
project root:

```
ftl extract --dry-run --fly-no-tui --fly-audit -v --cache-path <scratch>
```

## Speed

| | our `ftl` (the fly) | the original `ftl 0.12.1` |
|---|---:|---:|
| .py files in the tree / without i18n names / sniffed | 291 / 107 / 184 | — / — / 99 with keys |
| tokenizer candidates | 10 163 (4260 chains, 874 strings, 1192 kwargs were judged; the rest are strings without a name or are not judged) | — |
| brain trials | 9 201 (of them 2 875 resniffs — 575 windows × 5) | — |
| trials/s | **410.7** (16 processes, batch 64) | — |
| brain time / whole `FTL Extraction` / wall | 22.4 s / 23.2 s / 24.3 s | 0.013 s / 0.026 s (+ 0.28 s start-up via `uv tool run`) |
| keys in code | 495 (+ 8 missed − 1 false = 502) | 502 |

The weights and the connectome are loaded once per worker (16 spawn processes; the ≈ 2–3 s
start-up is part of the wall time). On one process this would be ≈ 9 201 / 53 ≈ 3 min.

## Differences from the reference: 15 (`[WARN  fly::audit]`)

Teacher = `fly_ftl_extract.reference` (AST, the original's semantics, verified by golden).
Margin is the sum over trials (6 trials = the first was `|m| < θ = 5.59` and 5 resniffs were
added).

| # | file : line | code | fly (margin, trials) | teacher |
|---|---|---|---|---|
| 1 | `app/bot/main.py:208:10` | `("captcha_timeout_task", dispatcher_workflow_data.get("captcha_timeout_task")),` — a tuple element inside the tuple `background_tasks = (…)`, not a call argument | **key** `captcha_timeout_task`, +7.00 (6) | not a key (no call) — a false key; it is also the reason for `files with keys: 100 vs 99` |
| 2 | `handlers/cbs/inventory/simple_inventory/main.py:104:36` | `await cb.message.edit_text(i18n.inventory.deprecated(_path="inventory/general.ftl"))` after `if not state_data:` | not a key, −6.54 (1) | key `inventory-deprecated` — a miss; the same call on line 141 is recognised (+), so in the tree only the position shifts |
| 3 | `storages/psql/inventory/items/stackable_ids.py:346:22` | `StackableID.KUS: L("item-kus-aliases", _path="inventory/stackable.ftl"),` — a dict value in `STACKABLE_ITEM_ALIASES: Final[dict[…]] = {…}` | not a key, −19.19 (6) | key `item-kus-aliases` — a miss |
| 4 | `…/stackable_ids.py:347:24` | `StackableID.ARMOR: L("item-armor-aliases", …)` | not a key, −17.89 (6) | key — a miss |
| 5 | `…/stackable_ids.py:348:24` | `StackableID.KUKUS: L("item-kukus-aliases", …)` | not a key, −24.54 (6) | key — a miss |
| 6 | `…/stackable_ids.py:349:26` | `StackableID.KUSCOIN: L("item-kuscoin-aliases", …)` | not a key, −30.93 (6) | key — a miss |
| 7 | `…/stackable_ids.py:350:22` | `StackableID.VIP: L("item-vip-aliases", …)` | not a key, −27.80 (6) | key — a miss |
| 8 | `…/stackable_ids.py:177:21` | `description=L("resource-silver-description", _path="inventory/resources.ftl"),` — a field of the constructor `Resource(…)` in a dict value; the previous line `name=L("resource-silver-name", …)` is recognised (+27.08, 6) | not a key, −5.56 (6) | key `resource-silver-description` — a miss |
| 9 | `…/stackable_ids.py:257:21` | `description=L("resource-topaz-description", …)` | not a key, −1.94 (6) | key — a miss |
| 10 | `…/stackable_ids.py:289:21` | `description=L("resource-uranium-description", …)` | not a key, −1.03 (6) | key — a miss |
| 11 | `handlers/msgs/kus/kukus_handler.py:54:5` | `@router.message(` ⏎ `LF("item-kukus-name", _path="inventory/stackable.ftl"),` — the decorator's first argument | not a key, −18.67 (6) | key `item-kukus-name` — a miss; the key also occurs in `stackable_ids.py:111` (recognised, +6.04), so in the tree the position shifts |
| 12 | `handlers/msgs/kus/kus_handler.py:66:5` | `@router.message(` ⏎ `LF("item-kus-name", _path=…),` | not a key, −21.33 (6) | key `item-kus-name` — a miss; also in `stackable_ids.py:93` (+17.08) → position shift |
| 13 | `handlers/cmds/user_settings/gender.py:22:10` | `"m": L("settings-gender-male-btn", _path="cmds/user_settings.ftl"),` — a dict value in `CB_DATA_TO_GENDER = {…}` | not a key, −9.35 (6) | key — a miss; the chain `i18n.settings.gender.male.btn(…)` on line 75 is recognised → position shift |
| 14 | `…/gender.py:23:10` | `"f": L("settings-gender-female-btn", …),` | not a key, −14.42 (6) | key — a miss; likewise line 79 |
| 15 | statistics | — | files with keys 100 | 99 (a consequence of no. 1) |

13 of the 14 difference windows were «unsure» on the first trial (`|m| < θ`) and received 5
resniffs each; the sum stayed on the wrong side. The exception is no. 2
(`i18n.inventory.deprecated(…)`): −6.54 on the first trial, i.e. a confident error without
resniff. Closest to zero are nos. 9–10 (−1.9, −1.0): one extra puff of context could have
flipped them. The full log with the margins of all 6 326 windows is `bot_run.txt` in the
report to the reviewer (not committed to the repo: it is someone else's code).

## What the misses have in common

All 13 misses are a call `L("…")` / `LF("…")` (a name from `--i18n-keys`, i.e. `<I18N> ( <STR>`)
whose **left context is not `= `, `return`, the `(` of another call or the `,` of an
argument**, but:

- a dict value after `:` (nos. 3–7, 13–14);
- a keyword argument of a dataclass constructor on the line after an identical recognised one
  (nos. 8–10: `name=L(…)` — yes, `description=L(…)` — no; to the left in the window stands
  `<NL>`, `<NAME> = <I18N> ( <STR> , _path = <STR> ) , <NL>`);
- the first argument of the decorator `@router.message(` after a line break (nos. 11–12).

The `grammar-4` grammar has no such constructs (`make_dataset.py`: contexts
return/list/dict/arg/file start exist for **mutations**, but the positive `<I18N>(…)` calls
there stand as an expression statement, an argument, an assignment or a list element). The
false key no. 1 is a string after the `(` of a tuple without a call: the grammar has tuples
of strings, but not with this neighbourhood (`<STR> , <NAME> . get ( <STR> )`).

The differences **were not fixed**: by the reviewer's decision the retraining (grammar-5) and
the GPU question are theirs to decide. What is NOT a fly error: `comment-junks` in the
config — the original prints `[WARN  cli] comment-junks has no effect …` as the first line;
now so do we (commit `b183a1d`).

## After grammar-5 (the same day, updated bot code)

The reviewer's decision after the report above: the `grammar-5` corpus
(`scripts/make_dataset.py`, `GRAMMAR5_FAMILIES`; shares — METRICS.md §6c) — the call
`<I18N>(<STR>)` in positions grammar-4 never had (a dict value, a kwarg of another
constructor, a decorator argument, a sequence element, `return`/`yield`), and twin negatives
without an i18n call; retraining (`train.py --lr 0.005 --epochs 100 --patience 10
--train-seeds 6`, keys stopped at epoch 86). The bot's code is not part of the dataset — it is
the holdout. The grammar-5 test before/after (the old weights on the same test states) is
METRICS.md §8: keys recall 0.870 → 1.000 with resniff.

The bot changed during these hours (the owner is working: 538 keys instead of 502, `main.py`
rewritten — the tuple `("captcha_timeout_task", …)` is gone, 0 commented keys instead of 36),
so both runs below were made on **one and the same** state of today's code, with the same
read-only recipe (`--dry-run --fly-no-tui --fly-audit -v --cache-path <scratch>`).

| | grammar-4 weights (before) | grammar-5 weights (after) | the original `ftl 0.12.1` |
|---|---:|---:|---:|
| .py in the tree / without i18n names / sniffed | 280 / 104 / 176 | 280 / 104 / 176 | — / — / 100 with keys |
| tokenizer candidates | 9 992 | 9 992 | — |
| brain trials (of them resniff) | 9 453 (3 160) | 9 356 (3 045) | — |
| trials/s (16 processes, batch 64) | 285 | 290 | — |
| brain time / wall | 33.1 s / 35 s | 32.3 s / 34.3 s | 0.019 s / 0.079 s |
| keys in code | 530 | **538** | 538 |
| files with keys | 100 | 100 | 100 |
| **`--fly-audit` differences** | **13** | **0** | — |

The trials/s are lower than the morning's 411 (the machine is busy with the owner's other work); the
determinism does not depend on this.

### The same 13 windows: margin before → after

| # | file : line | code | before (grammar-4) | after (grammar-5) |
|---|---|---|---:|---:|
| 2 | `simple_inventory/main.py:104:36` | `i18n.inventory.deprecated(_path=…)` after `if not state_data:` | −6.54 (1 trial) | **+40.69** (6) |
| 3 | `stackable_ids.py:346` | `StackableID.KUS: L("item-kus-aliases", _path=…)` | −19.19 (6) | **+11.50** (1) |
| 4 | `stackable_ids.py:347` | `…ARMOR: L("item-armor-aliases", …)` | −17.89 (6) | **+6.90** (1) |
| 5 | `stackable_ids.py:348` | `…KUKUS: L("item-kukus-aliases", …)` | −24.54 (6) | **+7.14** (1) |
| 6 | `stackable_ids.py:349` | `…KUSCOIN: L("item-kuscoin-aliases", …)` | −30.93 (6) | **+5.99** (1) |
| 7 | `stackable_ids.py:350` | `…VIP: L("item-vip-aliases", …)` | −27.80 (6) | **+6.07** (1) |
| 8 | `stackable_ids.py:177` | `description=L("resource-silver-description", …)` | −5.56 (6) | **+5.72** (1) |
| 9 | `stackable_ids.py:257` | `description=L("resource-topaz-description", …)` | −1.94 (6) | **+7.72** (1) |
| 10 | `stackable_ids.py:289` | `description=L("resource-uranium-description", …)` | −1.03 (6) | **+36.89** (6) |
| 11 | `kukus_handler.py:57` | `@router.message(` ⏎ `LF("item-kukus-name", …)` | −32.31 (6) | **+22.14** (6) |
| 12 | `kus_handler.py:69` | `@router.message(` ⏎ `LF("item-kus-name", …)` | −8.16 (1) | **+25.88** (6) |
| 13 | `gender.py:28` | `"m": L("settings-gender-male-btn", …)` | −5.12 (6) | **+9.07** (1) |
| 14 | `gender.py:29` | `"f": L("settings-gender-female-btn", …)` | −18.47 (6) | **+9.46** (1) |

The numbers are those of the first run's table (the bot's lines shifted by a few positions).
Nos. 1 and 15 from the first run (the false key in a tuple and the file counter) cannot be
re-checked on today's code — the construct is gone; a twin of this construct is in the corpus
(`g5-neg-seq-str`), and `"f"` / `"m"` — string dict keys next to the keys — the fly, as
before, does not call keys (−11.08). Every `_path=` of a found key is ignored, as it should
be. The full logs with the margins of every window are `bot_run_g4_today.txt` and
`bot_run_g5.txt` in the report to the reviewer (not committed to the repo: it is someone
else's code).

Phase 7 gates after grammar-5: the bot's `--fly-audit --dry-run` — 0 differences ✓; fixtures
24/24 ✓; test grammar-5 keys P 0.9987 R 1.0000, kwargs P 1.0000 R 1.0000 ✓ (METRICS.md §1).
