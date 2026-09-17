"""Crash-safe cache of brain states (``.cache/brain_states/*.npz``).

The KC states of a training seed take 10 minutes of 30 processes and a gigabyte on disk,
so ``scripts/train.py`` caches them.  A power loss during ``np.savez_compressed`` once left
a 47 MB stump that ``np.load`` would have rejected only by luck (a zip that is *almost*
complete loads and silently yields fewer rows).  Every file is therefore

* written to ``<name>.tmp.npz`` and moved into place with ``os.replace`` (atomic on NTFS
  and POSIX), so a crash leaves a ``.tmp`` and never a half-written ``.npz``;
* accompanied by ``<name>.npz.sha256`` holding the digest of the final bytes, written
  *before* the rename, so that a file without a checksum is by construction incomplete;
* verified on load — digest mismatch, missing checksum or a truncated archive raise
  :class:`CacheIntegrityError` with the fix (delete the file) instead of returning data.

``adopt`` blesses a file written before this module existed after checking it the
expensive way (zip test, expected row count, non-empty last row).
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np

SIDECAR_SUFFIX = ".sha256"
TMP_SUFFIX = ".tmp"
_CHUNK = 1 << 24


class CacheIntegrityError(RuntimeError):
    """A cached state file is incomplete, tampered with or unverifiable."""


def sidecar_path(path: Path) -> Path:
    """``x.npz`` → ``x.npz.sha256``."""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def save_states(path: Path, counts: np.ndarray) -> None:
    """Write ``counts`` atomically with a checksum sidecar."""
    # numpy appends ".npz" to a name that lacks it, so the temporary keeps the suffix
    tmp = path.with_name(path.stem + TMP_SUFFIX + path.suffix)
    side_tmp = sidecar_path(path).with_name(sidecar_path(path).name + TMP_SUFFIX)
    np.savez_compressed(tmp, counts=counts)
    digest = file_sha256(tmp)
    side_tmp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(side_tmp, sidecar_path(path))
    os.replace(tmp, path)


def load_states(path: Path) -> np.ndarray | None:
    """The cached ``counts`` (``None`` when not cached); raises on a damaged file."""
    if not path.exists():
        return None
    side = sidecar_path(path)
    if not side.exists():
        msg = (
            f"{path} has no {SIDECAR_SUFFIX} checksum: it was not written to completion "
            f"(or predates the checksummed cache). Delete it and rerun."
        )
        raise CacheIntegrityError(msg)
    recorded = side.read_text(encoding="ascii").split()[0]
    actual = file_sha256(path)
    if recorded != actual:
        msg = f"{path}: sha256 {actual[:16]}… does not match the recorded {recorded[:16]}…. Delete it and rerun."
        raise CacheIntegrityError(msg)
    try:
        with np.load(path) as z:
            return np.asarray(z["counts"])
    except (zipfile.BadZipFile, ValueError, KeyError, OSError) as err:
        msg = f"{path} cannot be read although its checksum matches ({err}). Delete it and rerun."
        raise CacheIntegrityError(msg) from err


def adopt(path: Path, expected_rows: int | None = None) -> dict[str, object]:
    """Verify a legacy file the expensive way and write its checksum sidecar.

    Raises :class:`CacheIntegrityError` when the archive is damaged, the row count differs
    from ``expected_rows`` or the last row is all zero (a truncated block).
    """
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except zipfile.BadZipFile as err:
        msg = f"{path}: not a complete zip archive ({err})"
        raise CacheIntegrityError(msg) from err
    if bad is not None:
        msg = f"{path}: corrupt member {bad}"
        raise CacheIntegrityError(msg)
    with np.load(path) as z:
        counts = np.asarray(z["counts"])
    if expected_rows is not None and counts.shape[0] != expected_rows:
        msg = f"{path}: {counts.shape[0]} rows, expected {expected_rows}"
        raise CacheIntegrityError(msg)
    if not counts[-1].any():
        msg = f"{path}: the last row is all zero — a truncated block"
        raise CacheIntegrityError(msg)
    digest = file_sha256(path)
    side = sidecar_path(path)
    side_tmp = side.with_name(side.name + TMP_SUFFIX)
    side_tmp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(side_tmp, side)
    return {
        "path": str(path),
        "rows": int(counts.shape[0]),
        "shape": list(counts.shape),
        "last_row_nnz": int(np.count_nonzero(counts[-1])),
        "sha256": digest,
    }
