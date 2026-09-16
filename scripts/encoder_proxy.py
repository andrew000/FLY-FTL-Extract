"""Encoder proxy: how separable is ``is_key`` on the odours *themselves*?

A logistic regression on the odour vectors (no brain) is an upper bound for what any
readout can get after the noisy mushroom body: information the hash destroyed cannot come
back.  It runs in a minute per encoder variant, so it is the tool for choosing encoder
levers *before* the brain is run (auditor's order after Phase 5: the brain is not run until
the proxy reaches F1 ≥ 0.996 on the test split).

Corpus and split are exactly those of ``scripts/make_dataset.py`` (same generator, seed,
balancing and 80/10/10 permutation, via ``balance_and_split``), so the proxy's test split
*is* the dataset's test split.  For the temporal encoder (``fly-odor-4``) the proxy features
are the concatenation of the slot vectors (``n_slots × n_pn``); for the ``fly-odor-3``
baseline they are its single 124-vector.  Results → ``docs/encoder_proxy.json`` (rendered
in ``docs/METRICS.md`` §6); the worst test errors of every variant → ``docs/proxy_errors/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.dopamine import Readout, Scores, dan_update, score
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor import encoder as enc
from fly_ftl_extract.tokenizer.candidates import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dataset import (
    DEFAULT_SEED,
    DEFAULT_SNIPPETS,
    GENERATOR_VERSION,
    Row,
    balance_and_split,
    generate_snippets,
    rows_of_snippet,
)

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "encoder_proxy.json"
ERRORS_DIR = REPO / "docs" / "proxy_errors"
ACTIVE = 0.1
EPOCHS = 40
PATIENCE = 6
LR = 0.5
L2 = 1e-5
BATCH = 512
WORST = 40


@dataclass(frozen=True)
class Variant:
    """One encoder setting to probe."""

    name: str
    params: enc.EncoderParams
    n_pn: int = enc.N_PN_DEFAULT
    salt: str = enc.ENCODER_VERSION
    temporal: bool = True
    """``True``: slot vectors concatenated (fly-odor-4); ``False``: fly-odor-3 n-grams."""


W1 = {"feature_weight": 1.0}
SLOTS = enc.EncoderParams(bigrams=False, hashes_per_feature=1, **W1)
VARIANTS = (
    Variant(
        "fly-odor-3: one vector, context 6/3, 3-grams",
        enc.EncoderParams(**W1),
        salt="fly-odor-3",
        temporal=False,
    ),
    Variant("slots 6/1/3, token+role features, 1 hash", SLOTS),
    Variant(
        "slots, token+role, 2 hashes per feature",
        enc.EncoderParams(bigrams=False, hashes_per_feature=2, **W1),
    ),
    Variant("slots + bigrams, 1 hash", enc.EncoderParams(bigrams=True, hashes_per_feature=1, **W1)),
    Variant(
        "fly-odor-4: slots + bigrams, 2 hashes, feature weight 1 (PN 0.76)",
        enc.EncoderParams(**W1),
        salt="fly-odor-4",
    ),
    Variant("slots, token+role, 1024 buckets (ceiling)", SLOTS, n_pn=1024),
    Variant(
        "slots + bigrams, 2 hashes, 1024 buckets (ceiling)", enc.EncoderParams(**W1), n_pn=1024
    ),
    Variant(
        "fly-odor-5: slots + bigrams, 2 hashes, feature weight 2 (PN 0.96)", enc.EncoderParams()
    ),
)


def encode_rows(rows: list[Row], variant: Variant) -> np.ndarray:
    """Proxy feature matrix of rows whose ``odor`` holds ``(window, options)``."""
    out = []
    for r in rows:
        window, options = r.odor
        assert isinstance(window, Window)
        assert isinstance(options, ExtractOptions)
        if variant.temporal:
            vec = enc.encode(window, options, variant.n_pn, variant.params, salt=variant.salt)
            out.append(vec.reshape(-1))
        else:
            out.append(
                enc.hash_features(
                    enc.ngram_features(window, options, variant.params),
                    variant.n_pn,
                    salt=variant.salt,
                )
            )
    return np.stack(out)


def probe(  # noqa: PLR0917
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[float, Scores, np.ndarray, Readout]:
    """Logistic regression (delta rule), early stopping on val F1; test scores, test margins
    and the readout of the best epoch."""
    rng = np.random.default_rng(0)
    readout = Readout(np.zeros(x_tr.shape[1], np.float32), 0.0, "binary")
    best_f1, best_w, best_b, since = -1.0, readout.w.copy(), 0.0, 0
    y = y_tr.astype(np.float32)
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s : s + BATCH]
            dan_update(readout, x_tr[idx], y[idx], lr=LR, l2=L2)
        f1 = score(readout.margin_from_features(x_va) > 0, y_va).f1
        if f1 > best_f1:
            best_f1, best_w, best_b, since = f1, readout.w.copy(), readout.b, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    best = Readout(best_w, best_b, "binary")
    margins = best.margin_from_features(x_te)
    return best_f1, score(margins > 0, y_te), margins, best


@dataclass
class Corpus:
    """The dataset's rows (odour = ``(window, options)``) with labels and split indices."""

    rows: list[Row]
    y: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_corpus(table: str, n_snippets: int, seed: int) -> Corpus:
    """Regenerate the corpus and reproduce ``make_dataset``'s balanced rows and split."""
    snippets, _ = generate_snippets(n_snippets, seed)

    def keep_windows(windows: list[Window], options: ExtractOptions) -> np.ndarray:
        arr = np.empty(len(windows), dtype=object)
        for i, w in enumerate(windows):
            arr[i] = (w, options)
        return arr

    key_rows: list[Row] = []
    kwarg_rows: list[Row] = []
    for s in snippets:
        result = rows_of_snippet(s, encode=keep_windows)
        if result is None:
            continue
        key_rows.extend(result[0])
        kwarg_rows.extend(result[1])
    key_rows, kwarg_rows, split_of = balance_and_split(key_rows, kwarg_rows, len(snippets), seed)
    rows = key_rows if table == "keys" else kwarg_rows
    split = np.array([split_of[r.snippet] for r in rows])
    y = np.array([r.label for r in rows])
    tr, va, te = (np.flatnonzero(split == k) for k in range(3))
    return Corpus(rows, y, tr, va, te)


def window_text(row: Row) -> str:
    window, options = row.odor
    return " ".join(t for t, _ in enc.normalize_window(window, options))


def worst_errors(rows: list[Row], margins: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    wrong = np.flatnonzero((margins > 0) != y)
    order = wrong[np.argsort(-np.abs(margins[wrong]))][:WORST]
    return [
        {
            "label": bool(y[i]),
            "margin": float(margins[i]),
            "snippet": rows[i].snippet,
            "key_name": rows[i].meta.get("key_name"),
            "window": window_text(rows[i]),
        }
        for i in order
    ]


def error_categories(rows: list[Row], margins: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Coarse buckets of the test errors: what stands right after / before the focus."""
    cats: Counter[str] = Counter()
    for i in np.flatnonzero((margins > 0) != y):
        window, options = rows[i].odor
        seq = enc.normalize_window(window, options)
        after = [t for t, p in seq if p >= len(window.focus)]
        before = [t for t, p in seq if p < 0]
        side = "FP" if margins[i] > 0 else "FN"
        cats[
            f"{side} after={' '.join(after[:2]) or '<eof>'} before={' '.join(before[-3:]) or '<bof>'}"
        ] += 1
    return dict(cats.most_common(15))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snippets", type=int, default=DEFAULT_SNIPPETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--table", choices=("keys", "kwargs"), default="keys")
    parser.add_argument("--only", type=int, nargs="*", help="indices of VARIANTS to run")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    corpus = load_corpus(args.table, args.snippets, args.seed)
    rows, y, tr, va, te = corpus.rows, corpus.y, corpus.train, corpus.val, corpus.test
    print(
        f"{GENERATOR_VERSION} seed {args.seed}: {args.table} rows {len(rows)} "
        f"({int(y.sum())} positive); train {len(tr)}, val {len(va)}, test {len(te)} "
        f"({time.perf_counter() - t0:.0f} s)",
        flush=True,
    )
    test_rows = [rows[i] for i in te]

    results: list[dict[str, object]] = []
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    chosen = args.only or range(len(VARIANTS))
    for vi in chosen:
        v = VARIANTS[vi]
        t1 = time.perf_counter()
        x = encode_rows(rows, v)
        t_encode = time.perf_counter() - t1
        val_f1, test, margins, _readout = probe(x[tr], y[tr], x[va], y[va], x[te], y[te])
        if v.temporal:
            puffs = x.reshape(len(rows), v.params.n_slots, v.n_pn)
            non_empty = puffs.sum(axis=2) > 0
            active = float((puffs > ACTIVE).sum(axis=2)[non_empty].mean())
        else:
            active = float((x > ACTIVE).sum(axis=1).mean())
        row = {
            "name": v.name,
            "temporal": v.temporal,
            "n_pn": v.n_pn,
            "dims": int(x.shape[1]),
            "salt": v.salt,
            "params": {
                k: getattr(v.params, k)
                for k in ("context_before", "context_after", "hashes_per_feature", "bigrams")
            },
            "active_pn_per_puff": active,
            "val_f1": val_f1,
            "test": test.as_dict(),
            "encode_s": t_encode,
            "seconds": time.perf_counter() - t1,
        }
        results.append(row)
        errors = {
            "variant": v.name,
            "categories": error_categories(test_rows, margins, y[te]),
            "worst": worst_errors(test_rows, margins, y[te]),
        }
        (ERRORS_DIR / f"{args.table}_{vi}.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(
            f"[{vi}] {v.name:50s} dims {x.shape[1]:5d}  active/puff {active:4.1f}  "
            f"val F1 {val_f1:.4f}  test P {test.precision:.4f} R {test.recall:.4f} "
            f"F1 {test.f1:.4f}  ({row['seconds']:.0f} s)",
            flush=True,
        )

    out_path = OUT_JSON if args.table == "keys" else OUT_JSON.with_name("encoder_proxy_kwargs.json")
    previous: dict[str, object] = {}
    if out_path.exists() and args.only:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
    previous_rows = previous.get("rows", [])
    assert isinstance(previous_rows, list)
    merged: dict[str, dict[str, object]] = {str(r["name"]): r for r in previous_rows}
    for r in results:
        merged[str(r["name"])] = r
    out_path.write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "encoder_version": enc.ENCODER_VERSION,
                "seed": args.seed,
                "snippets": args.snippets,
                "table": args.table,
                "rows_total": len(rows),
                "train_rows": len(tr),
                "val_rows": len(va),
                "test_rows": len(te),
                "test_positive": int(y[te].sum()),
                "probe": {
                    "lr": LR,
                    "l2": L2,
                    "batch": BATCH,
                    "epochs": EPOCHS,
                    "patience": PATIENCE,
                },
                "rows": list(merged.values()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {out_path} ({time.perf_counter() - t0:.0f} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
