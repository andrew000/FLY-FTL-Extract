"""Build ``fly_ftl_extract/data/mb_fafb783.npz`` (+ ``meta.json``, ``docs/CONNECTOME.md``).

Inputs (downloaded by hand into ``.cache/flywire/``, see the error text below):

* ``proofread_connections_783.feather`` — Zenodo 10676866, "FlyWire Whole-brain Connectome
  Connectivity Data", CC-BY 4.0 (Dorkenwald et al. 2024).  Columns: ``pre_pt_root_id``,
  ``post_pt_root_id``, ``neuropil``, ``syn_count`` and per-edge neurotransmitter scores
  ``gaba_avg ach_avg glut_avg oct_avg ser_avg da_avg``.  There is **no** ``nt_type`` column.
* ``Supplemental_file1_neuron_annotations.tsv`` — flyconnectome/flywire_annotations
  (Schlegel et al. 2024).  Columns used: ``root_id``, ``side``, ``super_class``,
  ``cell_class``, ``cell_sub_class``, ``cell_type``, ``top_nt``, ``known_nt``.
* ``proofread_root_ids_783.npy`` — the set of proofread neurons of the same release.

Selection (one hemisphere, the one with more Kenyon cells):

* PN   = ``cell_class == "ALPN"`` and ``cell_sub_class == "uniglomerular"`` (the glomerulus is
  taken from ``cell_type``, e.g. ``DA1_lPN`` → ``DA1``);
* KC   = ``cell_class == "Kenyon_Cell"`` (every sub class);
* APL  = ``cell_type == "APL"`` (``cell_class == "MBIN"``);
* MBON = ``cell_class == "MBON"``;
* DAN  = ``cell_class == "DAN"``.

Edges between selected neurons with ``syn_count >= 5``.  The sign comes from the
*presynaptic neuron's* transmitter: ``known_nt`` (first classical transmitter) when present,
otherwise ``top_nt``.  ``top_nt`` alone would be wrong: the predictor labels every Kenyon cell
``dopamine`` while ``known_nt`` says ``acetylcholine``.  Signs: ACh → +1, GABA and glutamate
→ −1, dopamine / octopamine / serotonin → 0 (DANs are a teaching signal, not a current).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "flywire"
FEATHER = CACHE / "proofread_connections_783.feather"
ANNOTATIONS = CACHE / "Supplemental_file1_neuron_annotations.tsv"
PROOFREAD_IDS = CACHE / "proofread_root_ids_783.npy"
OUT_NPZ = REPO / "fly_ftl_extract" / "data" / "mb_fafb783.npz"
OUT_META = REPO / "fly_ftl_extract" / "data" / "meta.json"
OUT_DOC = REPO / "docs" / "CONNECTOME.md"

SYN_THRESHOLD = 5
SIGN_BY_NT: dict[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
}
CLASSICAL_NT = frozenset(SIGN_BY_NT) | {"histamine", "tyramine"}
EDGE_NT_COLUMNS = {
    "ach_avg": "acetylcholine",
    "gaba_avg": "gaba",
    "glut_avg": "glutamate",
    "da_avg": "dopamine",
    "oct_avg": "octopamine",
    "ser_avg": "serotonin",
}
GLOMERULUS_RE = re.compile(r"^([A-Za-z0-9+]+)_[a-z0-9]*PN")

DOWNLOAD_HELP = f"""FlyWire source files are missing. Download them by hand into {CACHE}:

  1. https://zenodo.org/records/10676866  ->  proofread_connections_783.feather (~852 MB)
     and proofread_root_ids_783.npy   (CC-BY 4.0, Dorkenwald et al. 2024)
  2. https://github.com/flyconnectome/flywire_annotations/blob/main/supplemental_files/
     Supplemental_file1_neuron_annotations.tsv   (Schlegel et al. 2024)

Then run:  uv run python scripts/build_connectome.py
"""


def primary_transmitter(known_nt: object, top_nt: object) -> tuple[str, str]:
    """``(transmitter, source)``: first classical transmitter of ``known_nt``, else ``top_nt``."""
    if isinstance(known_nt, str):
        for part in re.split(r"[;,]", known_nt):
            token = part.strip().lower()
            if token.endswith("-negative"):
                continue
            if token in CLASSICAL_NT:
                return token, "known_nt"
    if isinstance(top_nt, str) and top_nt:
        return top_nt.lower(), "top_nt"
    return "unknown", "none"


def value_counts_md(series: pd.Series, title: str, limit: int | None = None) -> str:
    counts = series.value_counts(dropna=False)
    shown = counts if limit is None else counts.head(limit)
    lines = [f"### {title}", "", "| value | count |", "|---|---:|"]
    lines.extend(f"| `{k}` | {v} |" for k, v in shown.items())
    if limit is not None and len(counts) > limit:
        lines.append(f"| … ({len(counts) - limit} more) | |")
    return "\n".join(lines) + "\n"


def select_neurons(ann: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, int]]:
    kc_by_side = ann[ann.cell_class == "Kenyon_Cell"].side.value_counts().to_dict()
    side = max(("left", "right"), key=lambda s: kc_by_side.get(s, 0))
    on_side = ann[ann.side == side]
    uni_pn = on_side[(on_side.cell_class == "ALPN") & (on_side.cell_sub_class == "uniglomerular")]
    # 15 uniglomerular PNs per side are GABAergic (mlALT iPNs); three of them make >= 5 synapses
    # onto KCs, which would put negative PN->KC weights into the calyx. The excitatory calyx
    # input is cholinergic, so PN = uniglomerular ALPN whose transmitter is acetylcholine.
    pn_nt = [
        primary_transmitter(k, t)[0] for k, t in zip(uni_pn.known_nt, uni_pn.top_nt, strict=True)
    ]
    groups = {
        "PN": uni_pn[np.array(pn_nt) == "acetylcholine"],
        "KC": on_side[on_side.cell_class == "Kenyon_Cell"],
        "APL": on_side[on_side.cell_type == "APL"],
        "MBON": on_side[on_side.cell_class == "MBON"],
        "DAN": on_side[on_side.cell_class == "DAN"],
    }
    frames = []
    for group, frame in groups.items():
        f = frame.copy()
        f["group"] = group
        frames.append(f)
    selected = pd.concat(frames, ignore_index=True)
    if selected.root_id.duplicated().any():
        msg = "a neuron fell into two groups"
        raise SystemExit(msg)
    selected.attrs["uniglomerular_pn_total"] = len(uni_pn)
    selected.attrs["uniglomerular_pn_excluded_by_nt"] = int(
        (np.array(pn_nt) != "acetylcholine").sum()
    )
    return selected, side, kc_by_side


def read_edges(root_ids: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Edges of the feather file whose both ends are in ``root_ids`` (batch-wise, low memory)."""
    wanted = pa.array(root_ids.astype(np.int64))
    columns = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *EDGE_NT_COLUMNS]
    reader = pa.ipc.open_file(pa.memory_map(str(FEATHER)))
    parts = []
    total = 0
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        total += batch.num_rows
        mask = pc.and_(
            pc.is_in(batch.column("pre_pt_root_id"), value_set=wanted),
            pc.is_in(batch.column("post_pt_root_id"), value_set=wanted),
        )
        kept = batch.filter(mask).select(columns)
        if kept.num_rows:
            parts.append(kept.to_pandas())
    edges = pd.concat(parts, ignore_index=True)
    return edges, total


def main() -> int:
    for path in (FEATHER, ANNOTATIONS, PROOFREAD_IDS):
        if not path.exists():
            print(DOWNLOAD_HELP, file=sys.stderr)
            return 2

    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    proofread = np.load(PROOFREAD_IDS).astype(np.uint64)
    selected, side, kc_by_side = select_neurons(ann)
    selected = selected.sort_values(
        ["group", "root_id"],
        key=lambda s: (
            s.map({"PN": 0, "KC": 1, "APL": 2, "MBON": 3, "DAN": 4}) if s.name == "group" else s
        ),
    ).reset_index(drop=True)

    nt_info = [
        primary_transmitter(k, t) for k, t in zip(selected.known_nt, selected.top_nt, strict=True)
    ]
    selected["nt"] = [n for n, _ in nt_info]
    selected["nt_source"] = [s for _, s in nt_info]
    selected["sign"] = [SIGN_BY_NT.get(n, 0) for n in selected.nt]
    selected["nt_known_to_table"] = selected.nt.isin(SIGN_BY_NT)
    selected["glomerulus"] = [
        (GLOMERULUS_RE.match(str(t)).group(1) if g == "PN" and GLOMERULUS_RE.match(str(t)) else "")
        for g, t in zip(selected.group, selected.cell_type, strict=True)
    ]
    root_ids = selected.root_id.to_numpy().astype(np.uint64)
    missing = np.setdiff1d(root_ids, proofread)
    if len(missing):
        msg = f"{len(missing)} selected root ids are not in proofread_root_ids_783.npy"
        raise SystemExit(msg)

    edges, total_rows = read_edges(root_ids)
    n_between = len(edges)
    edges = edges[edges.syn_count >= SYN_THRESHOLD].reset_index(drop=True)
    n_dropped = n_between - len(edges)

    index = pd.Series(np.arange(len(selected)), index=selected.root_id.to_numpy().astype(np.int64))
    pre = index[edges.pre_pt_root_id.to_numpy()].to_numpy()
    post = index[edges.post_pt_root_id.to_numpy()].to_numpy()
    sign = selected.sign.to_numpy()[pre]
    syn = edges.syn_count.to_numpy().astype(np.int32)
    n = len(selected)
    w = sp.coo_matrix((sign * syn, (pre, post)), shape=(n, n)).tocsr()
    w.sum_duplicates()
    syn_csr = sp.coo_matrix((syn, (pre, post)), shape=(n, n)).tocsr()
    syn_csr.sum_duplicates()
    if (w.indptr != syn_csr.indptr).any() or (w.indices != syn_csr.indices).any():
        msg = "sparse layouts diverged"
        raise SystemExit(msg)

    # per-edge NT argmax vs the presynaptic neuron label (documentation only)
    edge_scores = edges[list(EDGE_NT_COLUMNS)].to_numpy()
    edge_argmax = np.array(list(EDGE_NT_COLUMNS.values()))[edge_scores.argmax(axis=1)]
    pre_nt = selected.nt.to_numpy()[pre]
    agree = int((edge_argmax == pre_nt).sum())
    unknown_pre = int((~selected.nt_known_to_table.to_numpy()[pre]).sum())

    groups = selected.group.to_numpy()
    idx = {
        g: np.flatnonzero(groups == g).astype(np.int32) for g in ("PN", "KC", "APL", "MBON", "DAN")
    }
    kc, pn, apl = idx["KC"], idx["PN"], idx["APL"]
    pn_kc = syn_csr[pn][:, kc]
    pn_inputs_per_kc = np.asarray((pn_kc > 0).sum(axis=0)).ravel()
    apl_kc = w[apl][:, kc]
    kc_apl = w[kc][:, apl]
    stats = {
        "pn_to_kc_edges": int(pn_kc.nnz),
        "pn_to_kc_min_weight": int(w[pn][:, kc].data.min()) if pn_kc.nnz else 0,
        "mean_pn_inputs_per_kc": float(pn_inputs_per_kc.mean()),
        "median_pn_inputs_per_kc": float(np.median(pn_inputs_per_kc)),
        "kc_without_pn_input": int((pn_inputs_per_kc == 0).sum()),
        "apl_to_kc_edges": int(apl_kc.nnz),
        "apl_to_kc_max_weight": int(apl_kc.data.max()) if apl_kc.nnz else 0,
        "kc_to_apl_edges": int(kc_apl.nnz),
        "kc_to_mbon_edges": int(w[kc][:, idx["MBON"]].nnz),
        "gaba_pn_count": int(((selected.group == "PN") & (selected.nt == "gaba")).sum()),
        "gaba_pn_to_kc_edges": int(
            syn_csr[np.flatnonzero((groups == "PN") & (selected.nt.to_numpy() == "gaba"))][
                :, kc
            ].nnz
        ),
    }

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        indptr=w.indptr.astype(np.int64),
        indices=w.indices.astype(np.int32),
        data=w.data.astype(np.int32),
        syn_count=syn_csr.data.astype(np.int32),
        root_id=root_ids,
        cell_class=selected.cell_class.fillna("").to_numpy().astype(str),
        cell_sub_class=selected.cell_sub_class.fillna("").to_numpy().astype(str),
        cell_type=selected.cell_type.fillna("").to_numpy().astype(str),
        side=selected.side.to_numpy().astype(str),
        glomerulus=selected.glomerulus.to_numpy().astype(str),
        nt=selected.nt.to_numpy().astype(str),
        nt_source=selected.nt_source.to_numpy().astype(str),
        sign=selected.sign.to_numpy().astype(np.int8),
        pn_idx=idx["PN"],
        kc_idx=idx["KC"],
        apl_idx=idx["APL"],
        mbon_idx=idx["MBON"],
        dan_idx=idx["DAN"],
    )
    npz_sha = hashlib.sha256(OUT_NPZ.read_bytes()).hexdigest()
    counts = {g: len(idx[g]) for g in idx}
    nt_table = selected.groupby(["group", "nt", "nt_source"]).size().reset_index(name="n")
    meta = {
        "dataset": "FlyWire FAFB v783",
        "side": side,
        "kc_by_side": {k: int(v) for k, v in kc_by_side.items()},
        "neurons": counts,
        "n_neurons": int(n),
        "uniglomerular_pn_total": selected.attrs["uniglomerular_pn_total"],
        "uniglomerular_pn_excluded_by_nt": selected.attrs["uniglomerular_pn_excluded_by_nt"],
        "edges": int(w.nnz),
        "syn_threshold": SYN_THRESHOLD,
        "edges_between_selected_before_threshold": int(n_between),
        "edges_dropped_by_threshold": int(n_dropped),
        "edges_with_pre_nt_outside_sign_table": unknown_pre,
        "edge_argmax_agrees_with_pre_neuron_nt": agree,
        "feather_rows_total": int(total_rows),
        "sign_by_nt": SIGN_BY_NT,
        "sanity": stats,
        "sha256_npz": npz_sha,
        "sha256_sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ANNOTATIONS, PROOFREAD_IDS)
        },
        "built": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license": "CC-BY 4.0 (FlyWire connectome data and annotations)",
        "citations": [
            "Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult brain. Nature 634, 124–138.",
            "Schlegel P. et al. (2024) Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634, 139–152.",
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

    write_doc(
        ann=ann,
        side=side,
        kc_by_side=kc_by_side,
        counts=counts,
        meta=meta,
        nt_table=nt_table,
        edges=edges,
    )
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "side",
                    "kc_by_side",
                    "neurons",
                    "edges",
                    "edges_dropped_by_threshold",
                    "sanity",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1e6:.2f} MB), sha256 {npz_sha[:16]}…")
    return 0


def write_doc(
    *,
    ann: pd.DataFrame,
    side: str,
    kc_by_side: dict[str, int],
    counts: dict[str, int],
    meta: dict[str, object],
    nt_table: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    mb_like = ann[
        ann.cell_class.isin(["Kenyon_Cell", "MBON", "DAN", "MBIN"])
        | ann.cell_type.fillna("").str.contains(r"KC|MBON|APL|DAN|PAM|PPL", regex=True)
    ]
    mb_table = (
        mb_like.groupby(["cell_class", "cell_sub_class", "cell_type"], dropna=False)
        .side.value_counts()
        .unstack(fill_value=0)
    )
    alpn = (
        ann[ann.cell_class == "ALPN"]
        .groupby("cell_sub_class", dropna=False)
        .side.value_counts()
        .unstack(fill_value=0)
    )
    nt_lines = ["| group | nt | source | n |", "|---|---|---|---:|"]
    nt_lines.extend(
        f"| {r.group} | {r.nt} | {r.nt_source} | {r.n} |" for r in nt_table.itertuples()
    )
    neuropils = edges.neuropil.value_counts()
    np_lines = ["| neuropil | edges |", "|---|---:|"]
    np_lines.extend(f"| {k} | {v} |" for k, v in neuropils.items())
    sanity = meta["sanity"]
    assert isinstance(sanity, dict)

    doc = f"""# The mushroom-body connectome: what the data holds and what was selected

Generated by `scripts/build_connectome.py` ({meta["built"]}). Sources: FlyWire FAFB v783
(Zenodo 10676866, CC-BY 4.0; Dorkenwald et al. 2024) and the flyconnectome annotations
(Schlegel et al. 2024). Nothing below is invented: every table is a `value_counts()` over the
TSV.

## 1. Schema of the sources (as is)

`proofread_connections_783.feather`: {meta["feather_rows_total"]} rows, columns
`pre_pt_root_id:int64, post_pt_root_id:int64, neuropil:string, syn_count:int64,
gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg: double`. **There is no `nt_type`
column** — CLAUDE.md was wrong; there are per-edge scores of six transmitters.

`Supplemental_file1_neuron_annotations.tsv`: {len(ann)} rows, {len(ann.columns)} columns:
`{", ".join(ann.columns)}`. The identifier is `root_id` (not `root_783`).

`proofread_root_ids_783.npy`: {len(np.load(PROOFREAD_IDS))} ids.

## 2. Distributions in the annotations

{value_counts_md(ann.super_class, "super_class")}
{value_counts_md(ann.cell_class, "cell_class")}
{value_counts_md(ann.cell_sub_class, "cell_sub_class (first 40)", limit=40)}
{value_counts_md(ann.side, "side")}
{value_counts_md(ann.flow, "flow")}
{value_counts_md(ann.top_nt, "top_nt (predicted transmitter)")}

### cell_type containing KC / MBON / APL / DAN / PAM / PPL (+ the classes Kenyon_Cell, MBON, DAN, MBIN), by side

```
{mb_table.to_string()}
```

### ALPN by cell_sub_class

```
{alpn.to_string()}
```

Uniglomerular PNs: `cell_type` encodes the glomerulus (`DA1_lPN`, `DM2_lPN`, `VA1v_adPN` …),
there is no separate glomerulus column in the TSV — the glomerulus is taken from `cell_type`
with the regex `^([A-Za-z0-9+]+)_[a-z0-9]*PN` and stored in the npz as `glomerulus`.

## 3. Selection

- Hemisphere: KCs by side — left {kc_by_side.get("left", 0)}, right {kc_by_side.get("right", 0)} → **{side}**.
- APL: the `MBIN` class holds `APL` and `DPM`, one per side → exactly 1 APL in the chosen hemisphere.
- Groups (only `side == "{side}"`):

| group | rule | n |
|---|---|---:|
| PN | `cell_class == ALPN`, `cell_sub_class == uniglomerular`, transmitter acetylcholine | {counts["PN"]} |
| KC | `cell_class == Kenyon_Cell` (all subclasses) | {counts["KC"]} |
| APL | `cell_type == APL` | {counts["APL"]} |
| MBON | `cell_class == MBON` | {counts["MBON"]} |
| DAN | `cell_class == DAN` | {counts["DAN"]} |
| total | | {meta["n_neurons"]} |

There are {meta["uniglomerular_pn_total"]} uniglomerular ALPNs on this side; {meta["uniglomerular_pn_excluded_by_nt"]} of them are
GABAergic (iPNs of the mediolateral tract) and **excluded**: in the first run without this
filter three such PNs had ≥ {SYN_THRESHOLD} synapses onto KCs and gave negative PN→KC weights,
contradicting the sanity rule «PN→KC only positive» (the excitatory input to the calyx is
cholinergic). This is a deviation from PLAN («uniglomerular ALPN» without a transmitter
qualifier).

Every selected `root_id` is present in `proofread_root_ids_783.npy` (checked by the script and a test).

## 4. Transmitters and sign

`top_nt` (the predictor) gives **`dopamine` for all {counts["KC"]} KCs**, whereas `known_nt`
(literature) gives `acetylcholine`. Hence the sign is taken as follows: the first classical
transmitter from `known_nt` (the `*-negative` tokens and co-transmitters such as `sNPF`,
`nitric oxide` are skipped), and if `known_nt` is empty — `top_nt`. The sign table:
ACh → +1, GABA → −1, glutamate → −1, dopamine/octopamine/serotonin → 0.

{chr(10).join(nt_lines)}

Edges whose presynaptic transmitter is outside the sign table: {meta["edges_with_pre_nt_outside_sign_table"]}.
The per-edge argmax of the six `*_avg` agrees with the presynaptic neuron's label in
{meta["edge_argmax_agrees_with_pre_neuron_nt"]} of {meta["edges"]} edges (the disagreements are mostly
KCs that the predictor considers dopaminergic).

## 5. Edges

- Between the selected neurons in the feather: {meta["edges_between_selected_before_threshold"]} edges;
  the threshold `syn_count >= {SYN_THRESHOLD}` dropped {meta["edges_dropped_by_threshold"]}; **{meta["edges"]}** remain.
- The matrix `W[pre, post] = sign(pre) · syn_count`, CSR (`indptr`, `indices`, `data`), plus `syn_count`.

{chr(10).join(np_lines)}

## 6. Sanity numbers

| quantity | value |
|---|---:|
| PN→KC edges | {sanity["pn_to_kc_edges"]} |
| minimum PN→KC weight (must be > 0) | {sanity["pn_to_kc_min_weight"]} |
| GABAergic uniglomerular PNs / their edges onto KCs | {sanity["gaba_pn_count"]} / {sanity["gaba_pn_to_kc_edges"]} |
| mean number of PN inputs per KC | {sanity["mean_pn_inputs_per_kc"]:.2f} |
| median PN inputs per KC | {sanity["median_pn_inputs_per_kc"]:.0f} |
| KCs without any PN input (≥ {SYN_THRESHOLD} synapses) | {sanity["kc_without_pn_input"]} |
| APL→KC edges / maximum weight (must be < 0) | {sanity["apl_to_kc_edges"]} / {sanity["apl_to_kc_max_weight"]} |
| KC→APL edges | {sanity["kc_to_apl_edges"]} |
| KC→MBON edges | {sanity["kc_to_mbon_edges"]} |

## 7. Artefact

`fly_ftl_extract/data/mb_fafb783.npz` ({OUT_NPZ.stat().st_size / 1e6:.2f} MB), sha256 `{meta["sha256_npz"]}`;
`fly_ftl_extract/data/meta.json` — every number above, the licence and the citations.
"""
    OUT_DOC.write_text(doc, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
