"""Sanity checks on the packaged mushroom-body connectome (data/mb_fafb783.npz)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_ftl_extract.brain import connectome as cx

REPO = Path(__file__).resolve().parent.parent
PROOFREAD_IDS = REPO / ".cache" / "flywire" / "proofread_root_ids_783.npy"

MIN_PN_INPUTS_PER_KC = 3
MAX_PN_INPUTS_PER_KC = 12


@pytest.fixture(scope="module")
def brain() -> cx.Connectome:
    if not cx.data_path(cx.NPZ_NAME).exists():
        pytest.skip("data/mb_fafb783.npz not built (run scripts/build_connectome.py)")
    return cx.load()


def test_missing_file_gives_instructions(tmp_path: Path) -> None:
    with pytest.raises(cx.ConnectomeMissingError, match="build_connectome"):
        cx.load(tmp_path / "nope.npz")


def test_group_sizes(brain: cx.Connectome) -> None:
    assert len(brain.kc_idx) >= cx.MIN_KC
    assert len(brain.pn_idx) >= cx.MIN_PN
    assert len(brain.apl_idx) == cx.EXPECTED_APL
    assert len(brain.mbon_idx) >= cx.MIN_MBON
    assert len(brain.dan_idx) > 0
    groups = np.concatenate(
        [brain.pn_idx, brain.kc_idx, brain.apl_idx, brain.mbon_idx, brain.dan_idx]
    )
    assert len(np.unique(groups)) == brain.n_neurons == len(groups)
    assert brain.hemisphere in ("left", "right")


def test_pn_to_kc_edges_are_excitatory(brain: cx.Connectome) -> None:
    block = brain.weights[brain.pn_idx][:, brain.kc_idx]
    assert block.nnz > 0
    assert block.data.min() > 0


def test_apl_to_kc_edges_are_inhibitory(brain: cx.Connectome) -> None:
    block = brain.weights[brain.apl_idx][:, brain.kc_idx]
    assert block.nnz > 0
    assert block.data.max() < 0


def test_mean_pn_inputs_per_kc_is_biological(brain: cx.Connectome) -> None:
    block = brain.syn_count[brain.pn_idx][:, brain.kc_idx]
    inputs = np.asarray((block > 0).sum(axis=0)).ravel()
    assert MIN_PN_INPUTS_PER_KC <= inputs.mean() <= MAX_PN_INPUTS_PER_KC


def test_edge_sign_matches_presynaptic_transmitter(brain: cx.Connectome) -> None:
    coo = brain.weights.tocoo()
    expected = brain.sign[coo.row] * brain.syn_count.tocoo().data
    assert np.array_equal(coo.data, expected)
    assert (brain.syn_count.data >= 5).all()
    assert all(brain.nt[brain.kc_idx] == "acetylcholine")
    assert all(brain.nt[brain.apl_idx] == "gaba")
    assert (brain.sign[brain.dan_idx] == 0).all()


def test_weights_have_no_explicit_zeros_and_dan_edges_are_separate(brain: cx.Connectome) -> None:
    assert (brain.weights.data != 0).all()
    assert brain.weights[brain.dan_idx].nnz == 0
    assert brain.syn_count[brain.dan_idx].nnz == 0
    dan = brain.dan_edges.tocoo()
    assert brain.n_dan_edges > 0
    assert np.isin(dan.row, brain.dan_idx).all()
    assert (dan.data >= 5).all()
    assert brain.dan_edges[brain.dan_idx][:, brain.kc_idx].nnz > 0
    # the two edge sets are disjoint and together cover every thresholded pair
    overlap = brain.syn_count.multiply(brain.dan_edges)
    assert overlap.nnz == 0


def test_kc_to_kc_edges_are_counted(brain: cx.Connectome) -> None:
    meta = json.loads(cx.data_path(cx.META_NAME).read_text(encoding="utf-8"))
    block = brain.weights[brain.kc_idx][:, brain.kc_idx]
    assert block.nnz == meta["sanity"]["kc_to_kc_edges"]
    assert block.data.min() > 0


def test_pn_glomeruli_are_recorded(brain: cx.Connectome) -> None:
    glomeruli = brain.glomerulus[brain.pn_idx]
    assert (glomeruli != "").all()
    assert len(set(glomeruli.tolist())) >= 40
    assert (brain.glomerulus[brain.kc_idx] == "").all()


def test_meta_matches_file(brain: cx.Connectome) -> None:
    meta = json.loads(cx.data_path(cx.META_NAME).read_text(encoding="utf-8"))
    assert meta["sha256_npz"] == brain.sha256
    assert meta["n_neurons"] == brain.n_neurons
    assert meta["edges"] == brain.n_edges
    assert meta["dan_edges"] == brain.n_dan_edges
    assert meta["sanity"]["dan_to_kc_edges"] == brain.dan_edges[brain.dan_idx][:, brain.kc_idx].nnz
    assert "CC-BY 4.0" in meta["license"]
    assert any("Dorkenwald" in c for c in meta["citations"])
    assert any("Schlegel" in c for c in meta["citations"])


def test_all_root_ids_are_proofread(brain: cx.Connectome) -> None:
    if not PROOFREAD_IDS.exists():
        pytest.skip("proofread_root_ids_783.npy not downloaded (.cache/flywire)")
    proofread = np.load(PROOFREAD_IDS).astype(np.uint64)
    assert np.isin(brain.root_id, proofread).all()
    assert len(np.unique(brain.root_id)) == brain.n_neurons
