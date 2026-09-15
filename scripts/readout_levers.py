"""Lever harness: which brain / encoder lever raises the readout's validation F1?

After attempts 7-8 the temporal-code readout plateaued at val F1 0.96 because the
per-puff Kenyon-cell code is noisy (docs/METRICS.md §7).  This runs a *small* version of
``scripts/train.py`` — the first ``--n-train`` keys train rows × ``seeds`` sniffs, the first
``--n-val`` val rows, the brain in the given setting, the CSR readout with lr 0.005 — for
several settings and prints one ``RESULT`` line each.  Absolute numbers are lower than the
full run (less data); the *ranking* of the settings is what this is for.  ``drive``
multiplies the stored odour values (a stand-in for a larger encoder ``feature_weight``).
Results go to ``docs/lever_harness.json`` (rendered in METRICS.md §7).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fly_ftl_extract.brain import Brain, BrainParams, load
from fly_ftl_extract.dopamine import SparseStates, TrainConfig, score, train_readout
from fly_ftl_extract.odor.encoder import ENCODER_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DATASET_DIR, simulate_all

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "docs" / "lever_harness.json"

SETTINGS: dict[str, tuple[BrainParams, float, int]] = {
    "base_p20_a0.3_s10_x3": (BrainParams(syn_scale=10.0, apl_scale=0.3, puff_ms=20.0), 1.0, 3),
    "p30_a0.4_s10_x3": (BrainParams(syn_scale=10.0, apl_scale=0.4, puff_ms=30.0), 1.0, 3),
    "p30_drive1.27_a0.5_s10_x3": (
        BrainParams(syn_scale=10.0, apl_scale=0.5, puff_ms=30.0),
        1.27,
        3,
    ),
    "p20_a0.3_s10_x6": (BrainParams(syn_scale=10.0, apl_scale=0.3, puff_ms=20.0), 1.0, 6),
    "p40_a0.5_s10_x3": (BrainParams(syn_scale=10.0, apl_scale=0.5, puff_ms=40.0), 1.0, 3),
}


def run(  # noqa: PLR0917
    name: str,
    params: BrainParams,
    drive: float,
    seeds: int,
    n_train: int,
    n_val: int,
    epochs: int = 40,
    patience: int = 6,
) -> dict[str, object]:
    z = np.load(DATASET_DIR / "keys.npz")
    odors, label, split = z["odors"], z["label"], z["split"]
    tr = np.flatnonzero(split == 0)[:n_train]
    va = np.flatnonzero(split == 1)[:n_val]
    x_tr = np.minimum(odors[tr] * drive, 1.0).astype(np.float32)
    x_va = np.minimum(odors[va] * drive, 1.0).astype(np.float32)
    brain = Brain(load(), params)
    t0 = time.perf_counter()
    counts_tr = np.concatenate(
        [simulate_all(brain, x_tr, 100 + s, f"{name} seed {s}", batch=64) for s in range(seeds)]
    )
    counts_va = simulate_all(brain, x_va, 7, f"{name} val", batch=64)
    t_brain = time.perf_counter() - t0
    act = x_tr.any(axis=2)
    kc = float((counts_tr[: len(tr)] > 0).mean(axis=2)[act].mean())
    y_tr = np.tile(label[tr], seeds)
    t1 = time.perf_counter()
    readout, log = train_readout(
        SparseStates(counts_tr),
        y_tr,
        counts_va,
        label[va],
        mode="both",
        config=TrainConfig(lr=0.005, max_epochs=epochs, patience=patience),
    )
    # a second val seed: how much does averaging two sniffs add?
    counts_vb = simulate_all(brain, x_va, 8, f"{name} val b", batch=64)
    m_a, m_b = readout.margin_batched(counts_va), readout.margin_batched(counts_vb)
    f1_2 = score((m_a + m_b) > 0, label[va]).f1
    row: dict[str, object] = {
        "name": name,
        "puff_ms": params.puff_ms,
        "syn_scale": params.syn_scale,
        "apl_scale": params.apl_scale,
        "drive": drive,
        "seeds": seeds,
        "n_train": len(tr),
        "n_val": len(va),
        "kc_active_per_puff": kc,
        "val_f1": log.best_val_f1,
        "best_epoch": log.best_epoch,
        "epochs": len(log.epochs),
        "val_f1_two_sniffs": f1_2,
        "brain_s": t_brain,
        "train_s": time.perf_counter() - t1,
    }
    print(
        f"RESULT {name}: kc/puff {kc:.3f} best val F1 {log.best_val_f1:.4f} at epoch "
        f"{log.best_epoch} ({len(log.epochs)} epochs) | 2-sniff avg F1 {f1_2:.4f} | "
        f"brain {t_brain:.0f} s train {row['train_s']:.0f} s",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=10000)
    ap.add_argument("--which", nargs="*", default=None, help="setting names to run")
    args = ap.parse_args()
    rows: dict[str, dict[str, object]] = {}
    if OUT_JSON.exists():
        previous = json.loads(OUT_JSON.read_text(encoding="utf-8"))["rows"]
        rows = {str(r["name"]): r for r in previous}
    for name, (params, drive, seeds) in SETTINGS.items():
        if args.which and name not in args.which:
            continue
        rows[name] = run(name, params, drive, seeds, args.n_train, args.n_val)
        OUT_JSON.write_text(
            json.dumps({"encoder_version": ENCODER_VERSION, "rows": list(rows.values())}, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
