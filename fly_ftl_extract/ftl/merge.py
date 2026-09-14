"""Combine per-file extraction results into the code key map, like ``code_extractor.rs``.

Every extractor (the ast reference now, the fly later) produces, per Python file, the keys
in first-occurrence order plus diagnostics.  This module replays the original's parallel
fold/reduce over those files with :class:`~fly_ftl_extract.ftl.rustorder.RustMap`, so the
resulting iteration order — and therefore the order of new keys in the ``.ftl`` files — is
the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fly_ftl_extract.ftl.model import Diagnostic, FluentKey, merge_key_occurrence
from fly_ftl_extract.ftl.rustorder import RustMap, rayon_tree


@dataclass
class FileExtraction:
    """Keys and diagnostics of one Python file, keys in the matcher's map order."""

    path: str
    keys: RustMap[FluentKey] = field(default_factory=RustMap)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def add(self, key: FluentKey) -> None:
        """``I18nMatcher::add_fluent_key``: merge into an existing occurrence or insert."""
        kept = self.keys.get(key.key)
        if kept is None:
            self.keys.insert(key.key, key)
            return
        conflict = merge_key_occurrence(kept, key)
        if conflict is not None:
            self.diagnostics.append(conflict)


@dataclass
class CodeExtraction:
    """All keys of a code tree, in the original's map iteration order."""

    keys: RustMap[FluentKey]
    diagnostics: list[Diagnostic]
    py_files_count: int
    py_files_with_keys: int


def _merge_into(target: RustMap[FluentKey], diagnostics: list[Diagnostic], key: FluentKey) -> None:
    kept = target.get(key.key)
    if kept is None:
        target.entry_insert(key.key, key)
        return
    conflict = merge_key_occurrence(kept, key)
    if conflict is not None:
        diagnostics.append(conflict)


_Acc = tuple[RustMap[FluentKey], list[Diagnostic]]


def merge_extractions(files: list[FileExtraction], py_files_count: int) -> CodeExtraction:
    """Replay ``extract_fluent_keys``: fold each file into an accumulator, reduce pairwise."""
    with_keys = sum(1 for f in files if len(f.keys) > 0)
    if not files:
        return CodeExtraction(RustMap(), [], py_files_count, 0)

    def fold(file: FileExtraction) -> _Acc:
        acc: RustMap[FluentKey] = RustMap()
        diagnostics = list(file.diagnostics)
        for key in file.keys.values():
            _merge_into(acc, diagnostics, key)
        return acc, diagnostics

    def reduce(a: _Acc, b: _Acc) -> _Acc:
        (target, target_diag), (source, source_diag) = (a, b) if len(a[0]) > len(b[0]) else (b, a)
        for key in source.values():
            _merge_into(target, target_diag, key)
        target_diag.extend(source_diag)
        return target, target_diag

    keys, diagnostics = rayon_tree(files, fold, reduce)
    diagnostics.sort(key=Diagnostic.sort_key)
    return CodeExtraction(keys, diagnostics, py_files_count, with_keys)
