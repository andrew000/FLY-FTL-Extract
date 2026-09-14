"""Reproduce the iteration order of the original's ``FxHashMap<String, _>``.

The Rust binary keeps Fluent keys in ``rustc_hash::FxHashMap`` (``std::HashMap`` backed by
hashbrown's SwissTable) and writes new keys in *iteration order* of those maps.  To be
byte-for-byte compatible we simulate exactly that:

* :func:`fx_hash_str` — ``rustc-hash`` 2.1 ``FxHasher`` for a ``str`` key (``write(bytes)``
  followed by the ``0xff`` prefix-free suffix that ``Hash for str`` appends on stable);
* :class:`RustMap` — the parts of hashbrown 0.16 that determine bucket positions: 16-wide
  groups, ``h1 = hash & mask``, first free slot from there (linear within the group,
  triangular probing across groups), growth 4 → 8 → 16 → … buckets at 3, 7, 14, … items,
  re-insertion in bucket order on resize, tombstones on removal;
* :func:`rayon_tree` — the shape of ``par_iter().fold().reduce()`` over the sorted file
  list: every file is its own leaf, the list is split at ``len // 2``.

Validated against 30 runs of the real binary (``tests/fixtures/rustorder_cases.json``,
``tests/test_rustorder.py``): 30/30 identical.  See ``docs/FORMAT.md`` §7.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from fly_ftl_extract.files import path_sort_key

_MASK64 = (1 << 64) - 1
_FX_K = 0xF1357AEA2E62A9C5
_SEED1 = 0x243F6A8885A308D3
_SEED2 = 0x13198A2E03707344
_PREVENT_TRIVIAL_ZERO_COLLAPSE = 0xA4093822299F31D0
_ROTATE = 26
_STR_SUFFIX = 0xFF
_GROUP_WIDTH = 16
_SMALL_TABLE_LIMIT = 15
_MIN_CAP = 3
_LOAD_NUM = 7
_LOAD_DEN = 8


def _multiply_mix(x: int, y: int) -> int:
    full = x * y
    return ((full & _MASK64) ^ (full >> 64)) & _MASK64


def _hash_bytes(data: bytes) -> int:
    n = len(data)
    s0, s1 = _SEED1, _SEED2
    if n <= 2 * 8:
        if n >= 8:  # noqa: PLR2004
            s0 ^= int.from_bytes(data[0:8], "little")
            s1 ^= int.from_bytes(data[n - 8 :], "little")
        elif n >= 4:  # noqa: PLR2004
            s0 ^= int.from_bytes(data[0:4], "little")
            s1 ^= int.from_bytes(data[n - 4 :], "little")
        elif n > 0:
            s0 ^= data[0]
            s1 ^= (data[n - 1] << 8) | data[n // 2]
    else:
        off = 0
        while off < n - 16:
            x = int.from_bytes(data[off : off + 8], "little")
            y = int.from_bytes(data[off + 8 : off + 16], "little")
            t = _multiply_mix(s0 ^ x, _PREVENT_TRIVIAL_ZERO_COLLAPSE ^ y)
            s0, s1 = s1, t
            off += 16
        suffix = data[n - 16 :]
        s0 ^= int.from_bytes(suffix[0:8], "little")
        s1 ^= int.from_bytes(suffix[8:16], "little")
    return _multiply_mix(s0, s1) ^ n


def _fx_add(h: int, word: int) -> int:
    return ((h + word) * _FX_K) & _MASK64


def fx_hash_str(key: str) -> int:
    """``FxBuildHasher.hash_one(&String)`` on a 64-bit target."""
    h = _fx_add(0, _hash_bytes(key.encode("utf-8")))
    h = _fx_add(h, _STR_SUFFIX)
    return ((h << _ROTATE) | (h >> (64 - _ROTATE))) & _MASK64


def capacity_to_buckets(cap: int) -> int:
    """Hashbrown ``capacity_to_buckets`` for element sizes above 3 bytes."""
    if cap < _SMALL_TABLE_LIMIT:
        cap = max(_MIN_CAP, cap)
        if cap < 4:  # noqa: PLR2004
            return 4
        if cap < 8:  # noqa: PLR2004
            return 8
        return 16
    adjusted = cap * _LOAD_DEN // _LOAD_NUM
    return 1 << (adjusted - 1).bit_length()


def bucket_mask_to_capacity(mask: int) -> int:
    """Hashbrown ``bucket_mask_to_capacity``."""
    if mask < 8:  # noqa: PLR2004
        return mask
    return ((mask + 1) // 8) * 7


@dataclass
class _Slot:
    key: str
    hash: int


class RustMap[V]:
    """Insertion/iteration-order model of ``HashMap<String, V, FxBuildHasher>``.

    Only string keys are supported (that is all the original hashes for ordering).
    """

    _EMPTY = 0
    _DELETED = 1
    _FULL = 2

    def __init__(self, capacity: int = 0) -> None:
        self._buckets = 0
        self._ctrl: list[int] = []
        self._slots: dict[int, _Slot] = {}
        self._values: dict[str, V] = {}
        self._index: dict[str, int] = {}
        self._items = 0
        self._growth_left = 0
        if capacity > 0:
            self._allocate(capacity_to_buckets(capacity))

    # -- table internals --------------------------------------------------------------
    def _allocate(self, buckets: int) -> None:
        self._buckets = buckets
        self._ctrl = [self._EMPTY] * buckets
        self._slots = {}
        self._index = {}
        self._growth_left = bucket_mask_to_capacity(buckets - 1)

    def _ctrl_at(self, pos: int) -> int:
        """Control byte as hashbrown reads it, including the mirrored trailing group."""
        if pos < self._buckets:
            return self._ctrl[pos]
        mirrored = pos - self._buckets
        if mirrored < self._buckets:
            return self._ctrl[mirrored]
        return self._EMPTY

    def _find_insert_slot(self, h: int) -> int:
        mask = self._buckets - 1
        pos = h & mask
        stride = 0
        while True:
            for bit in range(_GROUP_WIDTH):
                if self._ctrl_at(pos + bit) != self._FULL:
                    index = (pos + bit) & mask
                    if self._ctrl[index] == self._FULL:  # fix_insert_slot (tables < 16)
                        for j in range(_GROUP_WIDTH):
                            if self._ctrl_at(j) != self._FULL:
                                return j
                        msg = "hashbrown invariant violated: no free slot in group 0"
                        raise AssertionError(msg)
                    return index
            stride += _GROUP_WIDTH
            pos = (pos + stride) & mask

    def _full_indices(self) -> list[int]:
        return [i for i in range(self._buckets) if self._ctrl[i] == self._FULL]

    def _resize(self, capacity: int) -> None:
        old = [self._slots[i] for i in self._full_indices()]
        self._allocate(capacity_to_buckets(capacity))
        for slot in old:
            index = self._find_insert_slot(slot.hash)
            self._ctrl[index] = self._FULL
            self._slots[index] = slot
            self._index[slot.key] = index
        self._growth_left -= len(old)

    def reserve(self, additional: int) -> None:
        """``HashMap::reserve``: grow when ``additional`` does not fit."""
        if additional <= self._growth_left:
            return
        new_items = self._items + additional
        full_capacity = bucket_mask_to_capacity(self._buckets - 1) if self._buckets else 0
        if new_items <= full_capacity // 2:
            msg = "rehash_in_place is not modelled (never reached by ftl extract)"
            raise NotImplementedError(msg)
        self._resize(max(new_items, full_capacity + 1))

    def _place(self, key: str, h: int, value: V) -> None:
        if self._buckets == 0:
            self.reserve(1)
        index = self._find_insert_slot(h)
        if self._growth_left == 0 and self._ctrl[index] == self._EMPTY:
            self.reserve(1)
            index = self._find_insert_slot(h)
        if self._ctrl[index] == self._EMPTY:
            self._growth_left -= 1
        self._ctrl[index] = self._FULL
        self._slots[index] = _Slot(key, h)
        self._index[key] = index
        self._values[key] = value
        self._items += 1

    # -- public API ---------------------------------------------------------------------
    def insert(self, key: str, value: V) -> V | None:
        """``HashMap::insert``: reserves first (even for an existing key), returns the old value."""
        self.reserve(1)
        if key in self._values:
            old = self._values[key]
            self._values[key] = value
            return old
        self._place(key, fx_hash_str(key), value)
        return None

    def replace(self, key: str, value: V) -> V:
        """``*map.get_mut(key) = value`` / ``OccupiedEntry::insert``: no reserve, slot unchanged."""
        old = self._values[key]
        self._values[key] = value
        return old

    def entry_insert(self, key: str, value: V) -> bool:
        """``map.entry(key).or_insert(value)``: no reserve when the key exists; True if inserted."""
        if key in self._values:
            return False
        self._place(key, fx_hash_str(key), value)
        return True

    def remove(self, key: str) -> V | None:
        """``HashMap::remove`` (``erase``: EMPTY if a whole group is free, else DELETED)."""
        if key not in self._values:
            return None
        index = self._index.pop(key)
        mask = self._buckets - 1
        before = (index - _GROUP_WIDTH) & mask
        empty_before = [self._ctrl_at(before + i) == self._EMPTY for i in range(_GROUP_WIDTH)]
        empty_after = [self._ctrl_at(index + i) == self._EMPTY for i in range(_GROUP_WIDTH)]
        leading = 0
        for flag in reversed(empty_before):
            if flag:
                break
            leading += 1
        trailing = 0
        for flag in empty_after:
            if flag:
                break
            trailing += 1
        if leading + trailing >= _GROUP_WIDTH:
            self._ctrl[index] = self._DELETED
        else:
            self._ctrl[index] = self._EMPTY
            self._growth_left += 1
        del self._slots[index]
        self._items -= 1
        return self._values.pop(key)

    def get(self, key: str) -> V | None:
        """Value for ``key`` or ``None``."""
        return self._values.get(key)

    def __getitem__(self, key: str) -> V:
        return self._values[key]

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return self._items

    def __bool__(self) -> bool:
        return self._items > 0

    def keys(self) -> list[str]:
        """Keys in Rust iteration order (ascending bucket index)."""
        return [self._slots[i].key for i in self._full_indices()]

    def values(self) -> list[V]:
        """Values in Rust iteration order."""
        return [self._values[k] for k in self.keys()]

    def items(self) -> list[tuple[str, V]]:
        """``(key, value)`` pairs in Rust iteration order."""
        return [(k, self._values[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def debug_format(self, fmt: Callable[[V], str] = str) -> str:
        """``{:?}`` of the map as the original prints statistics: ``{"en": 1, "uk": 0}``."""
        return "{" + ", ".join(f'"{k}": {fmt(v)}' for k, v in self.items()) + "}"


def rayon_tree[T, R](
    leaves: Sequence[T],
    fold: Callable[[T], R],
    reduce: Callable[[R, R], R],
) -> R:
    """Combine ``leaves`` the way rayon's ``par_iter().fold().reduce()`` does.

    Each element becomes its own leaf, the sequence is split at ``len // 2`` recursively
    and the two halves are combined with ``reduce(left, right)``.  Validated empirically
    (the result did not depend on ``RAYON_NUM_THREADS`` 1/2/4/8 for up to 30 files).
    """
    if not leaves:
        msg = "rayon_tree needs at least one leaf"
        raise ValueError(msg)

    def helper(items: Sequence[T]) -> R:
        if len(items) > 1:
            mid = len(items) // 2
            return reduce(helper(items[:mid]), helper(items[mid:]))
        return fold(items[0])

    return helper(leaves)


def simulate_new_key_order(files: dict[str, list[str]]) -> list[str]:
    """Order in which the original writes new keys (``files``: name → keys in source order).

    A compact end-to-end model used by ``tests/test_rustorder.py``; the real pipeline
    (:mod:`fly_ftl_extract.ftl.merge` / :mod:`fly_ftl_extract.ftl.process`) drives the same
    :class:`RustMap` step by step.
    """
    names = sorted(files, key=path_sort_key)

    def file_map(name: str) -> RustMap[None]:
        table: RustMap[None] = RustMap()
        for key in files[name]:
            table.entry_insert(key, None)
        return table

    def fold(name: str) -> RustMap[None]:
        acc: RustMap[None] = RustMap()
        for key in file_map(name):
            acc.entry_insert(key, None)
        return acc

    def reduce(a: RustMap[None], b: RustMap[None]) -> RustMap[None]:
        target, source = (a, b) if len(a) > len(b) else (b, a)
        for key in source:
            target.entry_insert(key, None)
        return target

    in_code = rayon_tree(names, fold, reduce)
    keys_to_add: RustMap[None] = RustMap()
    for key in in_code:
        keys_to_add.insert(key, None)
    return keys_to_add.keys()
