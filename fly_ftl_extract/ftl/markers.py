"""``# ftl-extract: ignore ...`` markers in the comment attached to a stored message.

Port of ``common/src/ignore_marker.rs``: every line of the attached comment is parsed
(trimmed, ASCII-lowercased) and the results are merged.

* a line that is exactly ``ignore`` is a legacy alias of ``ignore untranslated``;
* otherwise the line must contain ``ftl-extract:``; what follows is the directive;
* ``ignore-untranslated`` is a legacy alias of ``ignore untranslated``;
* the directive must be ``ignore`` alone or followed by whitespace and names separated by
  commas or whitespace (``ignored`` / ``ignores`` are not markers);
* ``all`` ignores every check; ``ignore`` with no names means ``all``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fluent.syntax import ast as fl

_PREFIX = "ftl-extract:"


@dataclass
class IgnoreMarker:
    """Which checks a message opts out of."""

    all: bool = False
    names: set[str] = field(default_factory=set)

    def ignores(self, check: str) -> bool:
        """Whether the marker opts out of ``check`` (``stale``, ``kwargs``, ``untranslated`` …)."""
        return self.all or check in self.names

    def _merge(self, other: IgnoreMarker) -> None:
        self.all |= other.all
        self.names |= other.names

    def _is_empty(self) -> bool:
        return not self.all and not self.names

    @classmethod
    def parse(cls, comment: fl.BaseComment | None) -> IgnoreMarker | None:
        """The marker in ``comment``, or ``None`` when absent or not a marker."""
        if comment is None:
            return None
        marker = cls()
        for line in (comment.content or "").split("\n"):
            parsed = _parse_line(line)
            if parsed is not None:
                marker._merge(parsed)
        return None if marker._is_empty() else marker


def _lower_ascii(text: str) -> str:
    return "".join(c.lower() if "A" <= c <= "Z" else c for c in text)


def _parse_line(line: str) -> IgnoreMarker | None:
    normalized = _lower_ascii(line.strip())
    if normalized == "ignore":
        return IgnoreMarker(names={"untranslated"})
    position = normalized.find(_PREFIX)
    if position == -1:
        return None
    directive = normalized[position + len(_PREFIX) :].strip()
    if directive == "ignore-untranslated":
        return IgnoreMarker(names={"untranslated"})
    if not directive.startswith("ignore"):
        return None
    names = directive[len("ignore") :]
    if names and not names[0].isspace():
        return None
    marker = IgnoreMarker()
    any_name = False
    for name in names.replace(",", " ").split():
        any_name = True
        if name == "all":
            marker.all = True
        else:
            marker.names.add(name)
    if not any_name:
        marker.all = True
    return marker


def marker_ignores(comment: fl.BaseComment | None, check: str) -> bool:
    """Whether the attached ``comment`` carries a marker covering ``check``."""
    marker = IgnoreMarker.parse(comment)
    return marker is not None and marker.ignores(check)
