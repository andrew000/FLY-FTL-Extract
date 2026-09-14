"""The FxHashMap/SwissTable order model must reproduce the real binary's key order.

``tests/fixtures/rustorder_cases.json`` holds 30 runs of ``ftl extract`` 0.12.1 on synthetic
projects (1–120 keys, 1–30 files, duplicated keys): for each, the Python files with their
keys in source order and the order in which the binary wrote them to ``_default.ftl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_ftl_extract.ftl import rustorder
from fly_ftl_extract.ftl.rustorder import RustMap, fx_hash_str, simulate_new_key_order

CASES = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "rustorder_cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("name", sorted(CASES))
def test_reference_key_order_is_reproduced(name: str) -> None:
    case = CASES[name]
    assert simulate_new_key_order(case["files"]) == case["order"]


def test_fx_hash_matches_rustc_hash_test_vectors() -> None:
    # `hash_one(HashBytes(b"uwu"))` in rustc-hash's own tests hashes the bytes without the
    # 0xff suffix that `str` adds; reproduce by hashing through the same primitive.
    def bytes_only(data: bytes) -> int:
        h = rustorder._fx_add(0, rustorder._hash_bytes(data))
        return ((h << rustorder._ROTATE) | (h >> (64 - rustorder._ROTATE))) & rustorder._MASK64

    assert bytes_only(b"") == 17606491139363777937
    assert bytes_only(b"uwu") == 7168164714682931527
    assert bytes_only(b"These are some bytes for testing rustc_hash.") == 2349210501944688211
    assert fx_hash_str("a") != fx_hash_str("b")


def test_rustmap_basic_semantics() -> None:
    table: RustMap[int] = RustMap()
    for i in range(40):
        assert table.insert(f"k{i}", i) is None
    assert len(table) == 40
    assert table.insert("k3", 99) == 3
    assert table["k3"] == 99
    assert set(table.keys()) == {f"k{i}" for i in range(40)}
    assert table.remove("k5") == 5
    assert "k5" not in table
    assert len(table) == 39
    assert table.remove("k5") is None
    assert table.debug_format().startswith("{")
