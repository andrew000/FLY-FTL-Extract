"""Key model shared by the reference extractor, the fly and the ``.ftl`` writer.

Mirrors ``matcher.rs`` / ``diagnostics.rs`` of the original: a :class:`FluentKey` is one
Fluent entry (message, term or comment) bound to a ``.ftl`` path; keys found in code carry
their call site and the sorted keyword arguments that become ``{ $placeables }``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePath
from typing import Literal

from fluent.syntax import ast as fl

from fly_ftl_extract.files import path_sort_key

FluentEntry = fl.Message | fl.Term | fl.Comment | fl.GroupComment | fl.ResourceComment

POSITION_MAX = sys.maxsize
"""``usize::MAX``: the position of a key that comes from code, sorted after stored entries."""

PATH_KWARG = "_path"
GET_ATTR = "get"


@dataclass(frozen=True)
class CodeLocation:
    """``path:line:column`` of a call, column counted in characters from 1."""

    path: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}"

    def sort_key(self) -> tuple[tuple[str, ...], int, int]:
        """Rust ``Ord``: path component-wise, then line, then column."""
        return (path_sort_key(self.path), self.line, self.column)

    def __lt__(self, other: CodeLocation) -> bool:
        return self.sort_key() < other.sort_key()


def _min_location(a: CodeLocation | None, b: CodeLocation | None) -> CodeLocation | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


class DiagnosticKind(StrEnum):
    """``ExtractionDiagnosticKind`` with its ``as_str`` spelling."""

    KEY_PATH_CONFLICT = "key-path-conflict"
    KEY_MESSAGE_CONFLICT = "key-message-conflict"
    KEY_TYPE_CONFLICT = "key-type-conflict"
    PARSE_ERROR = "parse-error"
    READ_ERROR = "read-error"
    INVALID_UTF8 = "invalid-utf8"

    @property
    def is_file_error(self) -> bool:
        """Whole-file problems, as opposed to conflicts between valid keys."""
        return self in (self.PARSE_ERROR, self.READ_ERROR, self.INVALID_UTF8)


@dataclass
class Diagnostic:
    """One extraction problem, printed as ``[kind] message (loc, loc)``."""

    kind: DiagnosticKind
    key: str | None
    message: str
    locations: list[CodeLocation]

    @property
    def is_file_error(self) -> bool:
        """See :attr:`DiagnosticKind.is_file_error`."""
        return self.kind.is_file_error

    def __str__(self) -> str:
        text = f"[{self.kind}] {self.message}"
        if self.locations:
            text += " (" + ", ".join(str(loc) for loc in self.locations) + ")"
        return text

    def sort_key(self) -> tuple[tuple[int, str], str, str, list[tuple[tuple[str, ...], int, int]]]:
        """Order used by the original before printing (``None`` key sorts first)."""
        key = (0, "") if self.key is None else (1, self.key)
        return (key, self.message, str(self.kind), [loc.sort_key() for loc in self.locations])


def file_diagnostic(
    kind: DiagnosticKind, path: str, message: str, line: int, column: int
) -> Diagnostic:
    """A diagnostic about a whole file (read / decode / parse error)."""
    return Diagnostic(kind, None, message, [CodeLocation(path, line, column)])


def normalize_ftl_path(raw: str) -> tuple[str, ...]:
    """Component tuple used for path equality, like Rust ``PathBuf`` ``==`` / ``Hash``."""
    return PurePath(raw).parts


def code_ftl_path(raw_path: str | None, default_ftl_file: str) -> str:
    """The ``.ftl`` path a call writes to, as the original builds and displays it."""
    if not raw_path:
        return default_ftl_file
    if PurePath(raw_path).suffix == "":
        return os.path.join(raw_path, default_ftl_file)
    return raw_path


@dataclass
class FluentKey:
    """A Fluent entry bound to a locale file (``FluentKey`` of ``matcher.rs``)."""

    key: str
    entry: FluentEntry
    path: str
    """Display path relative to the locale dir: ``_path`` literal for code keys (plus the
    default file name when it had no extension), ``os.sep``-joined for stored keys."""
    locale: str | None = None
    position: int = POSITION_MAX
    source_location: CodeLocation | None = None
    kwargs_unknown: CodeLocation | None = None
    """First call site that passed ``**kwargs`` among all occurrences."""
    depends_on_keys: set[str] = field(default_factory=set)

    @property
    def path_key(self) -> tuple[str, ...]:
        """Component-wise identity of :attr:`path`."""
        return normalize_ftl_path(self.path)

    def kept_call_has_double_star(self) -> bool:
        """Whether the occurrence this key's variables come from passed ``**kwargs``."""
        return self.kwargs_unknown is not None and self.kwargs_unknown == self.source_location

    def is_message(self) -> bool:
        """Whether the entry is a message (terms and comments are not)."""
        return isinstance(self.entry, fl.Message)

    def attached_comment(self) -> fl.BaseComment | None:
        """The comment the Fluent parser attached to the stored message, if any."""
        if isinstance(self.entry, fl.Message | fl.Term):
            return self.entry.comment
        return None


def code_message(key: str, kwargs: list[str]) -> fl.Message:
    """Placeholder for a key found in code: the key text plus ``{ $kw }`` per kwarg, sorted."""
    elements: list[fl.TextElement | fl.Placeable] = [fl.TextElement(key)]
    elements.extend(
        fl.Placeable(fl.VariableReference(fl.Identifier(name))) for name in sorted(kwargs)
    )
    return fl.Message(fl.Identifier(key), fl.Pattern(elements))


def kwargs_from_key(key: FluentKey) -> list[str]:
    """Variables of a code key's placeholder (top-level ``{ $name }`` of the message value)."""
    entry = key.entry
    if not isinstance(entry, fl.Message) or entry.value is None:
        return []
    return [
        element.expression.id.name
        for element in entry.value.elements
        if isinstance(element, fl.Placeable)
        and isinstance(element.expression, fl.VariableReference)
    ]


def _messages_equal(a: fl.Message, b: fl.Message) -> bool:
    return a.equals(b)


def order_conflict_sides(a: FluentKey, b: FluentKey) -> tuple[FluentKey, FluentKey]:
    """The two occurrences ordered by call site."""
    if (
        a.source_location is not None
        and b.source_location is not None
        and b.source_location < a.source_location
    ):
        return b, a
    if a.source_location is None and b.source_location is not None:
        return b, a
    return a, b


def _conflict(
    kind: DiagnosticKind, existing: FluentKey, new: FluentKey, message: str
) -> Diagnostic:
    first, second = order_conflict_sides(existing, new)
    locations = [loc for loc in (first.source_location, second.source_location) if loc is not None]
    return Diagnostic(kind, new.key, message, locations)


def _describe_kwargs(key: FluentKey) -> str:
    kwargs = kwargs_from_key(key)
    return ", ".join(kwargs) if kwargs else "no keyword arguments"


def merge_key_occurrence(kept: FluentKey, other: FluentKey) -> Diagnostic | None:
    """Merge another occurrence of ``kept.key`` into ``kept``; return the conflict, if any.

    Same rules as the original: different ``_path`` is always a conflict; only calls without
    ``**kwargs`` are compared; a call without ``**`` beats one with it, earlier call site
    wins between equals; ``kwargs_unknown`` remembers the first ``**`` call site.
    """
    if kept.path_key != other.path_key:
        first, second = order_conflict_sides(kept, other)
        return _conflict(
            DiagnosticKind.KEY_PATH_CONFLICT,
            kept,
            other,
            f"Fluent key {first.key} has different paths: {first.path} and {second.path}",
        )

    kept_star = kept.kept_call_has_double_star()
    other_star = other.kept_call_has_double_star()

    conflict: Diagnostic | None = None
    if not (kept_star or other_star):
        first, second = order_conflict_sides(kept, other)
        if isinstance(kept.entry, fl.Message) and isinstance(other.entry, fl.Message):
            if not _messages_equal(kept.entry, other.entry):
                conflict = _conflict(
                    DiagnosticKind.KEY_MESSAGE_CONFLICT,
                    kept,
                    other,
                    f"Fluent key {first.key} is used with different keyword arguments: "
                    f"{_describe_kwargs(first)} and {_describe_kwargs(second)}",
                )
        else:
            conflict = _conflict(
                DiagnosticKind.KEY_TYPE_CONFLICT,
                kept,
                other,
                f"Fluent key {first.key} is not a Message in one of the entries.",
            )

    first_double_star = _min_location(kept.kwargs_unknown, other.kwargs_unknown)

    if kept_star and not other_star:
        other_wins = True
    elif not kept_star and other_star:
        other_wins = False
    else:
        other_wins = (
            other.source_location is not None
            and kept.source_location is not None
            and other.source_location < kept.source_location
        )
    if other_wins and conflict is None:
        kept.key = other.key
        kept.entry = other.entry
        kept.path = other.path
        kept.locale = other.locale
        kept.position = other.position
        kept.source_location = other.source_location
        kept.depends_on_keys = other.depends_on_keys
    kept.kwargs_unknown = first_double_star
    return conflict


CommentKeysMode = Literal["comment", "warn"]
LineEndings = Literal["default", "lf", "cr", "crlf"]

DEFAULT_I18N_KEYS: frozenset[str] = frozenset({"i18n", "L", "LazyProxy", "LazyFilter"})
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {"**/.venv/**", "**/venv/**", "**/.git/**", "**/__pycache__/**", "**/.pytest_cache/**"}
)
DEFAULT_IGNORE_ATTRIBUTES: frozenset[str] = frozenset(
    {"set_locale", "use_locale", "use_context", "set_context"}
)
DEFAULT_IGNORE_KWARGS: frozenset[str] = frozenset()
DEFAULT_FTL_FILENAME = "_default.ftl"
DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class ExtractOptions:
    """Fully resolved options of one ``ftl extract`` run (CLI > pyproject > defaults)."""

    code_path: str
    locales_path: str
    languages: tuple[str, ...] = (DEFAULT_LANGUAGE,)
    i18n_keys: frozenset[str] = DEFAULT_I18N_KEYS
    i18n_keys_prefix: frozenset[str] = frozenset()
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS
    ignore_attributes: frozenset[str] = DEFAULT_IGNORE_ATTRIBUTES
    ignore_kwargs: frozenset[str] = DEFAULT_IGNORE_KWARGS
    default_ftl_file: str = DEFAULT_FTL_FILENAME
    comment_keys_mode: CommentKeysMode = "comment"
    line_endings: LineEndings = "default"
    dry_run: bool = False
    allow_parse_errors: bool = False
    cache: bool = False
    cache_path: str | None = None
    clear_cache: bool = False
