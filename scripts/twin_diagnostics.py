"""Twin diagnostics for the one fixture candidate the attempt-10 fly gets wrong.

Auditor's request after attempt 10.  The candidate is ``ignore_attrs/app/main.py`` line 6,
``i18n.core.internal()`` (``-I core``: ``core`` is a first-level ignore attribute, so the
teacher says *not a key*); its twin is the same file with line 6 replaced by
``i18n.nested.internal()`` (a key).  Two questions, weights untouched:

1. **Proxy.**  The linear probe of ``scripts/encoder_proxy.py`` (fly-odor-5 odours of the
   grammar-4 train split) — what margins does it give the two windows?  Expected: clearly
   separated.
2. **Twins in the brain.**  The same two windows, encoded and run through
   ``Brain.simulate_sequence`` with the *same* seed: (a) how many PN buckets differ per
   slot, (b) Jaccard of the Kenyon-cell patterns in the focus puff and over the whole
   trial, (c) the fly's margin (``data/mbon_weights.npz``) for both.  Expected: 2-4 buckets
   in one puff, focus-puff Jaccard > 0.85, nearly equal margins — the KC code does not
   see ``<IGNORE>`` inside the focus.

Results → ``docs/twin_diagnostics.json`` (rendered as METRICS.md §2b).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, load
from fly_ftl_extract.dopamine import weights as weights_module
from fly_ftl_extract.dopamine.seed import sniff_seed, trial_seed
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor.encoder import DEFAULT_ENCODER, ENCODER_VERSION, encode, normalize_window
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.tokenizer.candidates import Candidate, iter_candidates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixtures import fixture_files
from encoder_proxy import VARIANTS, encode_rows, load_corpus, probe
from make_dataset import DEFAULT_SEED, DEFAULT_SNIPPETS

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "twin_diagnostics.json"
FIXTURE, FILE, LINE = "ignore_attrs", "main.py", 6
TWIN_LINE = "    i18n.nested.internal()"
EXTRA_SNIFFS = 2
ODOUR_EPS = 1e-6


def candidate_at(source: str, options: ExtractOptions, line: int) -> tuple[int, Candidate]:
    for i, c in enumerate(iter_candidates(source, options)):
        if c.line == line and c.kind == "chain":
            return i, c
    msg = f"no chain candidate on line {line}"
    raise LookupError(msg)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a > 0, b > 0
    union = int((x | y).sum())
    return float((x & y).sum() / union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--proxy-only", action="store_true", help="skip the brain part (no matching weights yet)"
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    f = next(f for f in fixture_files(FIXTURE) if f.path.endswith(FILE))
    assert f.source is not None
    lines = f.source.splitlines()
    original_line = lines[LINE - 1]
    twin_lines = list(lines)
    twin_lines[LINE - 1] = TWIN_LINE
    twin_source = "\n".join(twin_lines) + "\n"
    sources = {"original": f.source, "twin": twin_source}
    teacher = {
        name: [
            k.key
            for k in key_occurrences(f.path, src, f.options)
            if k.source_location and k.source_location.line == LINE
        ]
        for name, src in sources.items()
    }
    cands = {name: candidate_at(src, f.options, LINE) for name, src in sources.items()}
    windows = {name: c.window for name, (_, c) in cands.items()}
    texts = {
        name: " ".join(t for t, _ in normalize_window(w, f.options)) for name, w in windows.items()
    }
    puffs = {name: encode(w, f.options) for name, w in windows.items()}
    print(
        f"original: {original_line.strip()!r} teacher {teacher['original']} window {texts['original']}"
    )
    print(f"twin:     {TWIN_LINE.strip()!r} teacher {teacher['twin']} window {texts['twin']}")

    # ---- 1. proxy ---------------------------------------------------------------------
    variant = next(v for v in VARIANTS if v.name.startswith(ENCODER_VERSION))
    corpus = load_corpus("keys", DEFAULT_SNIPPETS, DEFAULT_SEED)
    x = encode_rows(corpus.rows, variant)
    y = corpus.y
    val_f1, test_scores, _margins, readout = probe(
        x[corpus.train],
        y[corpus.train],
        x[corpus.val],
        y[corpus.val],
        x[corpus.test],
        y[corpus.test],
    )
    proxy_margin = {
        name: float(readout.margin_from_features(p.reshape(1, -1))[0]) for name, p in puffs.items()
    }
    print(
        f"proxy ({variant.name}): val F1 {val_f1:.4f}, test F1 {test_scores.f1:.4f}; "
        f"margin original {proxy_margin['original']:+.3f}, twin {proxy_margin['twin']:+.3f}",
        flush=True,
    )

    # ---- 2. twins in the brain ----------------------------------------------------------
    a, b = puffs["original"], puffs["twin"]
    differing = [int((np.abs(a[k] - b[k]) > ODOUR_EPS).sum()) for k in range(a.shape[0])]
    active = [(int((a[k] > 0).sum()), int((b[k] > 0).sum())) for k in range(a.shape[0])]
    focus = DEFAULT_ENCODER.context_before
    print(f"differing PN buckets per slot: {differing}")
    if args.proxy_only:
        OUT_JSON.with_name(f"twin_proxy_{ENCODER_VERSION}.json").write_text(
            json.dumps(
                {
                    "encoder_version": ENCODER_VERSION,
                    "proxy": {
                        "variant": variant.name,
                        "val_f1": val_f1,
                        "test_f1": test_scores.f1,
                        "test_precision": test_scores.precision,
                        "test_recall": test_scores.recall,
                        "margin_original": proxy_margin["original"],
                        "margin_twin": proxy_margin["twin"],
                    },
                    "differing_buckets_per_slot": differing,
                    "active_buckets_per_slot": active,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 0
    brain = Brain(load(), DEFAULT_PARAMS)
    weights = weights_module.load(brain_hash=brain.hash, encoder_version=ENCODER_VERSION)
    content = f.source.encode("utf-8")
    base_seed = trial_seed(content, cands["original"][0], ENCODER_VERSION)
    seeds = [sniff_seed(base_seed, k) for k in range(1 + EXTRA_SNIFFS)]
    per_seed = []
    for k, sd in enumerate(seeds):
        res = brain.simulate_sequence(np.stack([a, b]), np.array([sd, sd], dtype=np.uint64))
        kc = res.kc_counts
        margins = weights.key.margin(kc)
        per_seed.append(
            {
                "sniff": k,
                "seed": int(sd),
                "jaccard_focus_puff": jaccard(kc[0, focus], kc[1, focus]),
                "jaccard_trial": jaccard(kc[0], kc[1]),
                "jaccard_per_puff": [jaccard(kc[0, s], kc[1, s]) for s in range(kc.shape[1])],
                "kc_active_focus": [int((kc[0, focus] > 0).sum()), int((kc[1, focus] > 0).sum())],
                "kc_differing_focus": int(((kc[0, focus] > 0) != (kc[1, focus] > 0)).sum()),
                "margin_original": float(margins[0]),
                "margin_twin": float(margins[1]),
            }
        )
        r = per_seed[-1]
        print(
            f"sniff {k}: focus-puff Jaccard {r['jaccard_focus_puff']:.3f} (KC differing "
            f"{r['kc_differing_focus']}), trial Jaccard {r['jaccard_trial']:.3f}; fly margin "
            f"original {r['margin_original']:+.2f}, twin {r['margin_twin']:+.2f}",
            flush=True,
        )
    same_odor_ref = brain.simulate_sequence(
        np.stack([a, a]), np.array([seeds[0], seeds[1]], dtype=np.uint64)
    ).kc_counts
    noise_focus = jaccard(same_odor_ref[0, focus], same_odor_ref[1, focus])
    noise_trial = jaccard(same_odor_ref[0], same_odor_ref[1])
    print(
        f"reference: same odour, two seeds — focus-puff Jaccard {noise_focus:.3f}, trial {noise_trial:.3f}"
    )

    payload = {
        "fixture": f"{FIXTURE}/{f.path}",
        "line": LINE,
        "original": {
            "source": original_line.strip(),
            "teacher_keys": teacher["original"],
            "window": texts["original"],
        },
        "twin": {
            "source": TWIN_LINE.strip(),
            "teacher_keys": teacher["twin"],
            "window": texts["twin"],
        },
        "encoder_version": ENCODER_VERSION,
        "proxy": {
            "variant": variant.name,
            "val_f1": val_f1,
            "test_f1": test_scores.f1,
            "margin_original": proxy_margin["original"],
            "margin_twin": proxy_margin["twin"],
        },
        "odour": {
            "differing_buckets_per_slot": differing,
            "active_buckets_per_slot": active,
            "focus_slot": focus,
        },
        "brain": {
            "brain_hash": brain.hash,
            "theta_key": weights.theta_key,
            "per_seed": per_seed,
            "same_odour_two_seeds_focus_jaccard": noise_focus,
            "same_odour_two_seeds_trial_jaccard": noise_trial,
        },
        "seconds": time.perf_counter() - t0,
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
