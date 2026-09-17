"""``--cache`` / ``--cache-path`` / ``--clear-cache``: the fly's verdicts memoised per file.

Same place and name pattern as the original (``.ftl-extract-cache/extract-<version>-v<schema>.bin``
in the working directory, or under ``--cache-path``; a path ending in ``.bin`` is the file
itself), own format: a magic header and a zlib-compressed JSON document.

An entry is valid while nothing that could change a single spike has changed:

* the file — ``st_mtime_ns`` and ``st_size``;
* the odour — every option that drives the token normalisation (``--i18n-keys``, ``-p``,
  ``--ignore-attributes``, ``--ignore-kwargs``) and :data:`ENCODER_VERSION`;
* the fly — brain hash, the weights themselves (sha256 of both readouts and θ), the
  resniff limit, ``--fly-trials`` and ``--fly-seed``.

A cached file stores the key *occurrences* the fly found (key, placeable kwargs, ``_path``
literal, call site, ``**kwargs`` flag); on a hit they are replayed through
:meth:`FileExtraction.add`, so the per-file merge and its diagnostics are rebuilt exactly.
``--default-ftl-file`` is applied at replay time and therefore not part of the fingerprint.
Damaged or foreign cache files are ignored (a cache is a shortcut, never an error).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import zlib
from dataclasses import dataclass
from typing import Any

from fly_ftl_extract import __version__
from fly_ftl_extract.dopamine.judge import Judge, JudgedFile
from fly_ftl_extract.ftl.merge import FileExtraction
from fly_ftl_extract.ftl.model import (
    CodeLocation,
    ExtractOptions,
    FluentKey,
    code_ftl_path,
    code_message,
)
from fly_ftl_extract.odor.encoder import ENCODER_VERSION

CACHE_SCHEMA = 1
DEFAULT_CACHE_DIR = ".ftl-extract-cache"
MAGIC = b"FLYFTLC\x01"


def cache_file(options: ExtractOptions) -> str:
    """Where the cache lives for these options (``--cache-path`` or the default directory)."""
    base = options.cache_path or DEFAULT_CACHE_DIR
    if base.endswith(".bin") or os.path.isfile(base):
        return base
    return os.path.join(base, f"extract-{__version__}-v{CACHE_SCHEMA}.bin")


def fly_fingerprint(options: ExtractOptions, judge: Judge) -> str:
    """Everything except the file itself that decides the verdicts."""
    w = judge.weights
    weights_digest = hashlib.sha256()
    weights_digest.update(w.key.w.astype("<f4").tobytes())
    weights_digest.update(w.kwarg.w.astype("<f4").tobytes())
    payload = {
        "encoder": ENCODER_VERSION,
        "brain": w.brain_hash,
        "weights": weights_digest.hexdigest(),
        "bias": [w.key.b, w.kwarg.b],
        "theta": [w.theta_key, w.theta_kwarg],
        "max_resniff": judge.max_resniff,
        "base_trials": judge.base_trials,
        "seed_salt": judge.seed_salt,
        "i18n_keys": sorted(options.i18n_keys),
        "i18n_keys_prefix": sorted(options.i18n_keys_prefix),
        "ignore_attributes": sorted(options.ignore_attributes),
        "ignore_kwargs": sorted(options.ignore_kwargs),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def file_fingerprint(path: str, fly: str) -> str | None:
    """``sha256(mtime_ns, size, fly fingerprint)`` of ``path``; ``None`` if it cannot be stat-ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    payload = f"{st.st_mtime_ns}:{st.st_size}:{fly}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class CachedMeta:
    """Counters a hit contributes to the fly statistics."""

    candidates: int
    keys: int


class ExtractCache:
    """The per-file memo behind ``--cache``."""

    def __init__(self, path: str, fly: str) -> None:
        self.path = path
        self.fly = fly
        self.entries: dict[str, dict[str, Any]] = {}
        self.dirty = False

    # ------------------------------------------------------------------------ storage

    @classmethod
    def open(cls, options: ExtractOptions, judge: Judge) -> ExtractCache:
        """Load (or start) the cache for ``options``; ``--clear-cache`` deletes it first."""
        path = cache_file(options)
        cache = cls(path, fly_fingerprint(options, judge))
        if options.clear_cache:
            with contextlib.suppress(OSError):
                os.remove(path)
            cache.dirty = True
            return cache
        cache._load()
        return cache

    def _load(self) -> None:
        try:
            with open(self.path, "rb") as fh:
                raw = fh.read()
        except OSError:
            return
        if not raw.startswith(MAGIC):
            return
        try:
            doc = json.loads(zlib.decompress(raw[len(MAGIC) :]).decode("utf-8"))
        except zlib.error, UnicodeDecodeError, json.JSONDecodeError:
            return
        if not isinstance(doc, dict) or doc.get("schema") != CACHE_SCHEMA:
            return
        entries = doc.get("entries")
        if isinstance(entries, dict):
            self.entries = {
                str(k): v for k, v in entries.items() if isinstance(v, dict) and "fp" in v
            }

    def save(self) -> None:
        """Write atomically (tmp + ``os.replace``); creates the directory."""
        if not self.dirty:
            return
        doc = {"schema": CACHE_SCHEMA, "version": __version__, "entries": self.entries}
        data = MAGIC + zlib.compress(json.dumps(doc, sort_keys=True).encode("utf-8"), 6)
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, self.path)
        self.dirty = False

    # ------------------------------------------------------------------------- lookups

    def lookup(
        self, path: str, options: ExtractOptions
    ) -> tuple[FileExtraction, CachedMeta] | None:
        """The stored extraction of ``path`` when the file and the fly are unchanged."""
        entry = self.entries.get(path)
        if entry is None:
            return None
        fp = file_fingerprint(path, self.fly)
        if fp is None or entry.get("fp") != fp:
            return None
        try:
            return self._replay(path, entry, options), CachedMeta(
                int(entry["candidates"]), int(entry["keys"])
            )
        except KeyError, TypeError, ValueError:
            return None

    @staticmethod
    def _replay(path: str, entry: dict[str, Any], options: ExtractOptions) -> FileExtraction:
        result = FileExtraction(path)
        for key, placeable, raw_path, line, column, star in entry["occurrences"]:
            location = CodeLocation(path, int(line), int(column))
            result.add(
                FluentKey(
                    str(key),
                    code_message(str(key), [str(p) for p in placeable]),
                    code_ftl_path(raw_path, options.default_ftl_file),
                    source_location=location,
                    kwargs_unknown=location if star else None,
                )
            )
        return result

    def store(self, path: str, judged: JudgedFile) -> None:
        """Remember what the fly found in ``path`` (as it is on disk right now)."""
        fp = file_fingerprint(path, self.fly)
        if fp is None:
            return
        self.entries[path] = {
            "fp": fp,
            "candidates": len(judged.candidates),
            "keys": len(judged.keys),
            "occurrences": [
                [
                    k.key_name,
                    list(k.placeable),
                    k.path_value,
                    k.call_position[0],
                    k.call_position[1],
                    k.candidate.kwargs_unknown,
                ]
                for k in judged.keys
            ],
        }
        self.dirty = True
