"""Import the stored ``.ftl`` files of one locale (port of ``ftl_importer.rs``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from fluent.syntax import FluentParser
from fluent.syntax import ast as fl

from fly_ftl_extract.files import find_ftl_files, path_sort_key
from fly_ftl_extract.ftl.model import FluentKey
from fly_ftl_extract.ftl.rustorder import RustMap, rayon_tree


class ExtractionError(Exception):
    """A fatal problem; the message is printed as ``Error during extraction: <message>``."""


@dataclass
class DuplicateKey:
    """A message or term one locale defines more than once."""

    locale: str
    key: str
    definitions: list[str]

    def __str__(self) -> str:
        head = f"Fluent key {self.key} is defined more than once in locale {self.locale}: "
        if not self.definitions:
            return head
        if len(self.definitions) == 1:
            return head + self.definitions[0]
        *rest, last = self.definitions
        return head + ", ".join(rest) + " and " + last


@dataclass
class LocaleImport:
    """Messages, terms and standalone comments of a locale, keyed by name."""

    messages: RustMap[FluentKey] = field(default_factory=RustMap)
    terms: RustMap[FluentKey] = field(default_factory=RustMap)
    misc: list[FluentKey] = field(default_factory=list)
    duplicates: dict[str, set[str]] = field(default_factory=dict)
    files_count: int = 0


def _ftl_syntax_error(path: str, resource: fl.Resource) -> str:
    junk = next(e for e in resource.body if isinstance(e, fl.Junk))
    annotation = junk.annotations[0] if junk.annotations else None
    what = annotation.message if annotation is not None else "syntax error"
    more = sum(1 for e in resource.body if isinstance(e, fl.Junk)) - 1
    suffix = f" (and {more} more)" if more > 0 else ""
    return f"Failed to parse FTL file {path}:1:1: {what}{suffix}"


def import_file(full_path: str, rel_path: str, locale: str) -> LocaleImport:
    """Parse one ``.ftl`` file; the maps are sized like the original (``with_capacity``)."""
    try:
        with open(full_path, encoding="utf-8", newline="") as fh:
            content = fh.read()
    except OSError as err:
        raise ExtractionError(f"Failed to read FTL file: {full_path}") from err
    resource = FluentParser(with_spans=False).parse(content)
    if any(isinstance(e, fl.Junk) for e in resource.body):
        raise ExtractionError(_ftl_syntax_error(full_path, resource))

    result = LocaleImport(messages=RustMap(capacity=len(resource.body)))
    for position, entry in enumerate(resource.body):
        if isinstance(entry, fl.Message):
            name = entry.id.name
            key = FluentKey(name, entry, rel_path, locale, position)
            if result.messages.insert(name, key) is not None:
                result.duplicates.setdefault(name, set()).add(rel_path)
        elif isinstance(entry, fl.Term):
            name = entry.id.name
            key = FluentKey(name, entry, rel_path, locale, position)
            if result.terms.insert(name, key) is not None:
                result.duplicates.setdefault(f"-{name}", set()).add(rel_path)
        elif isinstance(entry, fl.Comment | fl.GroupComment | fl.ResourceComment):
            result.misc.append(FluentKey("", entry, rel_path, locale, position))
        else:
            raise ExtractionError(f"Unsupported entry in {full_path}: {entry!r}")
    return result


def _merge_map(
    a: RustMap[FluentKey], b: RustMap[FluentKey], duplicates: dict[str, set[str]], *, term: bool
) -> RustMap[FluentKey]:
    if len(a) == 0:
        return b
    a.reserve(len(b))
    for name, key in b.items():
        existing = a.get(name)
        if existing is not None:
            dup_name = f"-{name}" if term else name
            files = duplicates.setdefault(dup_name, set())
            files.add(existing.path)
            files.add(key.path)
            a.insert(name, key)
        else:
            a.entry_insert(name, key)
    return a


def _merge_imports(a: LocaleImport, b: LocaleImport) -> LocaleImport:
    duplicates = a.duplicates
    for name, files in b.duplicates.items():
        duplicates.setdefault(name, set()).update(files)
    return LocaleImport(
        messages=_merge_map(a.messages, b.messages, duplicates, term=False),
        terms=_merge_map(a.terms, b.terms, duplicates, term=True),
        misc=a.misc + b.misc,
        duplicates=duplicates,
    )


def _definition_lines(content: str, key: str) -> list[int]:
    lines: list[int] = []
    for index, line in enumerate(content.splitlines()):
        if line.startswith(key) and line[len(key) :].lstrip().startswith("="):
            lines.append(index + 1)
    return lines


def _duplicate_keys(
    locale_dir: str, locale: str, duplicates: dict[str, set[str]]
) -> list[DuplicateKey]:
    result: list[DuplicateKey] = []
    for key in sorted(duplicates):
        definitions: list[str] = []
        for file in sorted(duplicates[key], key=path_sort_key):
            display = os.path.join(locale, file)
            try:
                with open(os.path.join(locale_dir, file), encoding="utf-8", newline="") as fh:
                    content = fh.read()
            except OSError:
                content = ""
            lines = _definition_lines(content, key)
            if lines:
                definitions.extend(f"{display}:{line}" for line in lines)
            else:
                definitions.append(display)
        result.append(DuplicateKey(locale, key, definitions))
    return result


def import_locale(locales_path: str, locale: str) -> tuple[LocaleImport, list[DuplicateKey]]:
    """Import every ``.ftl`` file of ``locale`` (merged with the same tree as the original)."""
    locale_dir = os.path.join(locales_path, locale)
    files = find_ftl_files(locale_dir)
    if not files:
        return LocaleImport(), []
    imported = rayon_tree(
        files,
        lambda rel: import_file(os.path.join(locale_dir, rel), rel, locale),
        _merge_imports,
    )
    imported.files_count = len(files)
    return imported, _duplicate_keys(locale_dir, locale, imported.duplicates)
