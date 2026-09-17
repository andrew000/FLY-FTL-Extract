"""The brain-state cache must never hand back a half-written file."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _statecache import (  # noqa: E402
    CacheIntegrityError,
    adopt,
    load_states,
    save_states,
    sidecar_path,
)


def counts_array() -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((50, 4, 30)) < 0.2).astype(np.uint8) * 2


def test_roundtrip_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "keys_train_seed_101_abc.npz"
    counts = counts_array()
    save_states(path, counts)
    assert path.exists()
    assert sidecar_path(path).exists()
    assert not list(tmp_path.glob("*.tmp*"))
    back = load_states(path)
    assert back is not None
    assert np.array_equal(back, counts)
    assert load_states(tmp_path / "missing.npz") is None


def test_truncated_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "keys_train_seed_106_abc.npz"
    save_states(path, counts_array())
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])  # the power went out here
    with pytest.raises(CacheIntegrityError, match="does not match"):
        load_states(path)


def test_file_without_checksum_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "keys_val_abc.npz"
    np.savez_compressed(path, counts=counts_array())
    with pytest.raises(CacheIntegrityError, match=r"no .sha256 checksum"):
        load_states(path)


def test_adopt_blesses_a_complete_legacy_file_and_rejects_a_stump(tmp_path: Path) -> None:
    good = tmp_path / "legacy_good.npz"
    counts = counts_array()
    np.savez_compressed(good, counts=counts)
    info = adopt(good, expected_rows=50)
    assert info["rows"] == 50
    assert load_states(good) is not None
    with pytest.raises(CacheIntegrityError, match="rows"):
        adopt(good, expected_rows=51)
    stump = tmp_path / "legacy_stump.npz"
    stump.write_bytes(good.read_bytes()[:1000])
    with pytest.raises(CacheIntegrityError, match="zip"):
        adopt(stump)
    zero_tail = tmp_path / "legacy_zero_tail.npz"
    tail = counts.copy()
    tail[-1] = 0
    np.savez_compressed(zero_tail, counts=tail)
    with pytest.raises(CacheIntegrityError, match="last row"):
        adopt(zero_tail)
