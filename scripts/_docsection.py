"""Replace one marked section of a Markdown file (used by calibrate.py and bench.py)."""

from __future__ import annotations

from pathlib import Path


def update_section(path: Path, name: str, body: str, skeleton: str) -> None:
    """Write ``body`` between ``<!-- name:start -->`` / ``<!-- name:end -->`` in ``path``.

    Creates the file from ``skeleton`` when it does not exist; appends the markers when
    they are missing.  Everything outside the markers is left untouched.
    """
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    text = path.read_text(encoding="utf-8") if path.exists() else skeleton
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        head = text[: text.index(start)]
        tail = text[text.index(end) + len(end) :]
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
