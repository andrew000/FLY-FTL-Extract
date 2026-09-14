"""Encoder proxy: how separable is ``is_key`` on the odours *themselves*?

A logistic regression on the 124-dim odour vectors (no brain) is an upper bound for what
any readout can get after the noisy mushroom body: information the hash destroyed cannot
come back.  It runs in seconds per encoder variant, so it is the tool for choosing encoder
levers (window, n-gram density, bag n-grams, top-k PN) and for showing what the number of
PN buckets does.  Results → ``docs/encoder_proxy.json`` (rendered in ``docs/METRICS.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.dopamine import Readout, dan_update, score
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor import encoder as enc
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.reference.labels import label_candidates
from fly_ftl_extract.tokenizer.candidates import Window, iter_candidates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dataset import generate_snippets

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "encoder_proxy.json"
NEGATIVE_RATIO = 3
VAL_EVERY = 5  # snippet id % 5 == 0 -> validation
ACTIVE = 0.1


@dataclass(frozen=True)
class Variant:
    """One encoder setting to probe."""

    name: str
    params: enc.EncoderParams
    n_pn: int = enc.N_PN_DEFAULT
    salt: str = enc.ENCODER_VERSION
    top_k: int = 0


VARIANTS = (
    Variant(
        "fly-odor-1: context 12/6, 3-grams, bag 0.5",
        enc.EncoderParams(context_before=12, context_after=6, bag_weight=0.5),
    ),
    Variant(
        "12/6 without bag", enc.EncoderParams(context_before=12, context_after=6, bag_weight=0.0)
    ),
    Variant(
        "12/6, 2-grams, without bag",
        enc.EncoderParams(context_before=12, context_after=6, max_ngram=2, bag_weight=0.0),
    ),
    Variant("6/3, bag 0.5", enc.EncoderParams(context_before=6, context_after=3, bag_weight=0.5)),
    Variant("fly-odor-2: context 6/3, without bag", enc.EncoderParams()),
    Variant("6/3, tau 2, without bag", enc.EncoderParams(distance_tau=2.0)),
    Variant("4/2, without bag", enc.EncoderParams(context_before=4, context_after=2)),
    Variant(
        "3/2, 2-grams, without bag",
        enc.EncoderParams(context_before=3, context_after=2, max_ngram=2),
    ),
    Variant(
        "fly-odor-1 + top-40 PN",
        enc.EncoderParams(context_before=12, context_after=6, bag_weight=0.5),
        top_k=40,
    ),
    Variant("fly-odor-2 + top-40 PN", enc.EncoderParams(), top_k=40),
    Variant("fly-odor-2, salt salt-a", enc.EncoderParams(), salt="salt-a"),
    Variant("fly-odor-2, salt salt-b", enc.EncoderParams(), salt="salt-b"),
    Variant("fly-odor-2, 160 buckets", enc.EncoderParams(), n_pn=160),
    Variant("fly-odor-2, 256 buckets", enc.EncoderParams(), n_pn=256),
    Variant("fly-odor-2, 1024 buckets", enc.EncoderParams(), n_pn=1024),
    Variant(
        "fly-odor-1, 1024 buckets",
        enc.EncoderParams(context_before=12, context_after=6, bag_weight=0.5),
        n_pn=1024,
    ),
)


def encode(window: Window, options: ExtractOptions, variant: Variant) -> np.ndarray:
    vec = np.zeros(variant.n_pn)
    for feature, weight in enc.features(window, options, variant.params):
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8, salt=variant.salt.encode("ascii")[:16]
        ).digest()
        vec[int.from_bytes(digest, "little") % variant.n_pn] += weight
    out = np.tanh(vec).astype(np.float32)
    if variant.top_k and variant.top_k < variant.n_pn:
        drop = np.argpartition(-out, variant.top_k)[variant.top_k :]
        out[drop] = 0.0
    return out


def probe(
    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, epochs: int = 40
) -> float:
    """Best validation F1 of a logistic regression trained with the delta rule."""
    rng = np.random.default_rng(0)
    readout = Readout(np.zeros(x_tr.shape[1], np.float32), 0.0, "binary")
    best = 0.0
    y = y_tr.astype(np.float32)
    for _ in range(epochs):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), 512):
            idx = order[s : s + 512]
            dan_update(readout, x_tr[idx], y[idx], lr=0.5, l2=1e-5)
        best = max(best, score(readout.margin_from_features(x_va) > 0, y_va).f1)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snippets", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args(argv)
    snippets, _ = generate_snippets(args.snippets, args.seed)
    windows: list[tuple[Window, ExtractOptions]] = []
    labels: list[bool] = []
    sids: list[int] = []
    for s in snippets:
        options = s.config.options()
        cands = list(iter_candidates(s.source, options))
        lab, _ = label_candidates(cands, key_occurrences("s.py", s.source, options))
        windows.extend((c.window, options) for c in cands)
        labels.extend(lab)
        sids.extend([s.id] * len(cands))
    y = np.array(labels)
    sid = np.array(sids)
    rng = np.random.default_rng(1)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    keep = np.sort(
        np.concatenate(
            [pos, rng.choice(neg, min(len(neg), NEGATIVE_RATIO * len(pos)), replace=False)]
        )
    )
    val = (sid[keep] % VAL_EVERY) == 0
    yk = y[keep]
    print(
        f"{len(windows)} candidates, {len(pos)} positive; probe rows {len(keep)}, val {int(val.sum())}",
        flush=True,
    )
    rows = []
    for v in VARIANTS:
        t0 = time.perf_counter()
        x = np.stack([encode(windows[i][0], windows[i][1], v) for i in keep])
        f1 = probe(x[~val], yk[~val], x[val], yk[val])
        rows.append(
            {
                "name": v.name,
                "n_pn": v.n_pn,
                "top_k": v.top_k,
                "salt": v.salt,
                "params": {
                    k: getattr(v.params, k)
                    for k in (
                        "max_ngram",
                        "distance_tau",
                        "bag_weight",
                        "context_before",
                        "context_after",
                    )
                },
                "active_pn": float((x > ACTIVE).sum(axis=1).mean()),
                "mass": float(x.sum(axis=1).mean()),
                "val_f1": f1,
            }
        )
        print(
            f"{v.name:44s} n_pn {v.n_pn:4d}  active {rows[-1]['active_pn']:5.1f}  val F1 {f1:.4f}  ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    OUT_JSON.write_text(
        json.dumps(
            {
                "snippets": args.snippets,
                "seed": args.seed,
                "probe_rows": len(keep),
                "val_rows": int(val.sum()),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
