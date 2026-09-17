"""Process-pool side of ``ftl extract``: one brain and one set of weights per worker.

Windows has no ``fork``; the pool uses the ``spawn`` context, so every worker imports this
module afresh and builds its :class:`~fly_ftl_extract.dopamine.Judge` once in
:func:`init_worker`.  A job is a *chunk* of whole files (about one brain batch of windows):
the candidates of a file are judged together (their seeds come from the file bytes, the
resniff rule and the keyword-argument round are per file), and several files per chunk
keep the four brain rounds full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fly_ftl_extract.dopamine.judge import Judge, JudgedFile
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.tokenizer.candidates import Candidate


@dataclass
class FileJob:
    """One Python file that is ready for the fly: tokenized, syntax checked."""

    index: int
    """Position in the walker's file list (the merge order must be the walker's)."""
    path: str
    """Display path, as the walker built it."""
    content: bytes
    """Raw bytes — part of every trial seed."""
    candidates: list[Candidate]

    @property
    def windows(self) -> int:
        """Upper bound of windows the fly may sniff: nameable candidates + their kwargs."""
        return sum(1 + len(c.kwargs) for c in self.candidates if c.key_name is not None)


def chunk_jobs(jobs: list[FileJob], target_windows: int) -> list[list[FileJob]]:
    """Group consecutive jobs until a chunk holds about ``target_windows`` windows."""
    chunks: list[list[FileJob]] = []
    current: list[FileJob] = []
    size = 0
    for job in jobs:
        if current and size + job.windows > target_windows:
            chunks.append(current)
            current, size = [], 0
        current.append(job)
        size += job.windows
    if current:
        chunks.append(current)
    return chunks


_JUDGE: Judge | None = None


def init_worker(options: ExtractOptions, judge_kwargs: dict[str, Any]) -> None:
    """Pool initializer: load connectome + weights once for this process."""
    global _JUDGE  # noqa: PLW0603 — per-process singleton is the point
    _JUDGE = Judge(options, **judge_kwargs)


def judge_chunk(chunk: list[FileJob]) -> list[tuple[int, JudgedFile]]:
    """Judge a chunk of files in the worker; returns ``(job.index, judged)`` pairs."""
    if _JUDGE is None:
        msg = "worker not initialised (init_worker was not run)"
        raise RuntimeError(msg)
    judged = _JUDGE.judge_many([(job.candidates, job.content) for job in chunk])
    return [(job.index, result) for job, result in zip(chunk, judged, strict=True)]
