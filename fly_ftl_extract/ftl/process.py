"""Merge code keys into one locale and write its files (port of ``ftl_extractor.rs``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from fluent.syntax import FluentSerializer
from fluent.syntax import ast as fl

from fly_ftl_extract.ftl.importer import ExtractionError, LocaleImport
from fly_ftl_extract.ftl.markers import marker_ignores
from fly_ftl_extract.ftl.model import ExtractOptions, FluentKey, LineEndings
from fly_ftl_extract.ftl.rustorder import RustMap
from fly_ftl_extract.ftl.variables import message_variables, term_variables


@dataclass
class LocaleStatistics:
    """Per-locale counters of the original's ``ExtractionStatistics``."""

    files_count: int = 0
    stored_keys: int = 0
    updated: int = 0
    added: int = 0
    commented: int = 0


@dataclass
class LocaleResult:
    """What processing one locale produced."""

    statistics: LocaleStatistics
    warnings: list[str] = field(default_factory=list)
    """``warn`` mode messages, in the original's (hash) order."""


class _Entries:
    def __init__(self, messages: RustMap[FluentKey], terms: RustMap[FluentKey]) -> None:
        self._messages = messages
        self._terms = terms

    def message(self, name: str) -> fl.Message | None:
        key = self._messages.get(name)
        if key is not None and isinstance(key.entry, fl.Message):
            return key.entry
        return None

    def term(self, name: str) -> fl.Term | None:
        key = self._terms.get(name)
        if key is not None and isinstance(key.entry, fl.Term):
            return key.entry
        return None


def extract_kwargs(
    key: FluentKey,
    terms: RustMap[FluentKey],
    all_keys: RustMap[FluentKey],
    depend_keys: set[str],
) -> set[str]:
    """Variables ``key`` needs from code; records referenced messages in ``depend_keys``."""
    entries = _Entries(all_keys, terms)
    if isinstance(key.entry, fl.Message):
        collected = message_variables(entries, key.entry)
    elif isinstance(key.entry, fl.Term):
        collected = term_variables(entries, key.entry)
    else:
        return set()
    depend_keys.update(collected.referenced_messages)
    if collected.unknown_references:
        unknown = collected.unknown_references[0]
        kind = "Term" if isinstance(key.entry, fl.Term) else "Message"
        if unknown.kind == "message":
            raise ExtractionError(
                f"{kind} `{key.key}` in {key.path} references unknown message `{unknown.name}`"
            )
        raise ExtractionError(
            f"{kind} `{key.key}` in {key.path} references unknown term `-{unknown.name}`"
        )
    return collected.variables


_serializer = FluentSerializer(with_junk=False)


def comment_ftl_key(key: FluentKey) -> None:
    """Turn the entry into a comment holding its serialized text, one comment line per line."""
    if isinstance(key.entry, fl.Comment):
        return
    if isinstance(key.entry, fl.Message | fl.Term):
        text = _serializer.serialize_entry(key.entry)
    else:
        text = ""
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    # fluent-rs writes a whitespace-only line as a bare ``#``; python-fluent only an empty one.
    lines = [line if line.strip() else "" for line in lines]
    key.entry = fl.Comment("\n".join(lines))


def generate_ftl(keys: list[FluentKey]) -> str:
    """Serialize ``keys`` sorted by ``(position, input index)``."""
    ordered = sorted(enumerate(keys), key=lambda item: (item[1].position, item[0]))
    resource = fl.Resource([key.entry for _, key in ordered])
    return _serializer.serialize(resource)


def normalize_line_endings(text: str, line_endings: LineEndings) -> str:
    """Apply ``--line-endings`` to the serialized file."""
    if line_endings == "lf":
        return text.replace("\r\n", "\n").replace("\r", "\n")
    if line_endings == "cr":
        return text.replace("\r\n", "\r").replace("\n", "\r")
    if line_endings == "crlf":
        return text.replace("\r", "").replace("\n", "\r\n")
    return text


def _write(path: str, content: str, line_endings: LineEndings) -> None:
    data = normalize_line_endings(content, line_endings).encode("utf-8")
    try:
        with open(path, "rb") as fh:
            if fh.read() == data:
                return
    except OSError:
        pass
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _group_by_path(keys: list[FluentKey]) -> dict[tuple[str, ...], tuple[str, list[FluentKey]]]:
    groups: dict[tuple[str, ...], tuple[str, list[FluentKey]]] = {}
    for key in keys:
        groups.setdefault(key.path_key, (key.path, []))[1].append(key)
    return groups


def process_locale(
    lang: str,
    stored: LocaleImport,
    in_code: RustMap[FluentKey],
    options: ExtractOptions,
) -> LocaleResult:
    """``process_language``: decide what to keep, comment, add; then write every touched file."""
    stats = LocaleStatistics(files_count=stored.files_count)
    result = LocaleResult(stats)
    stored_keys = stored.messages
    terms = stored.terms
    lang_dir = os.path.join(options.locales_path, lang)

    keys_to_comment: RustMap[FluentKey] = RustMap()
    keys_to_add: RustMap[FluentKey] = RustMap()

    # Code keys vs stored keys: path mismatch and new keys.
    for name, fluent_key in in_code.items():
        stored_key = stored_keys.get(name)
        if stored_key is not None:
            if fluent_key.path_key != stored_key.path_key:
                old_key = stored_keys.remove(name)
                if old_key is not None:
                    keys_to_comment.insert(name, old_key)
                keys_to_add.insert(name, fluent_key)
                stats.commented += 1
                stats.updated += 1
        else:
            keys_to_add.insert(name, fluent_key)
            stats.added += 1

    # Compare kwargs.
    depend_keys: set[str] = set()
    mismatches: list[tuple[str, FluentKey]] = []
    for name, fluent_key in in_code.items():
        stored_key = stored_keys.get(name)
        if stored_key is None:
            continue
        stored_args = extract_kwargs(stored_key, terms, stored_keys, depend_keys)
        if fluent_key.kwargs_unknown is not None:
            continue
        code_args = extract_kwargs(fluent_key, terms, in_code, depend_keys)
        if code_args != stored_args:
            if marker_ignores(stored_key.attached_comment(), "kwargs"):
                continue
            mismatches.append((name, fluent_key))
    for name, fluent_key in mismatches:
        stored_key = stored_keys.remove(name)
        if stored_key is not None:
            keys_to_comment.insert(name, stored_key)
            keys_to_add.insert(name, fluent_key)
            stats.commented += 1
            stats.updated += 1

    # Stored keys the code never calls but which opted out of the stale check.
    kept_by_marker = sorted(
        name
        for name, stored_key in stored_keys.items()
        if name not in in_code and marker_ignores(stored_key.attached_comment(), "stale")
    )
    for name in kept_by_marker:
        extract_kwargs(stored_keys[name], terms, stored_keys, depend_keys)
    kept_set = set(kept_by_marker)

    # Obsolete keys (stored but not in code).
    for name, value in stored_keys.items():
        if name in in_code or name in depend_keys or name in kept_set:
            continue
        keys_to_comment.insert(name, value)
        stats.commented += 1
        stored_keys.remove(name)

    # Comment or warn.
    if options.comment_keys_mode == "comment":
        for fluent_key in keys_to_comment.values():
            comment_ftl_key(fluent_key)
    else:
        for fluent_key in keys_to_comment.values():
            keys_to_add.remove(fluent_key.key)
            file_display = os.path.join(lang_dir, fluent_key.path)
            result.warnings.append(
                f"Key `{fluent_key.key}` in `{file_display}` is not in code "
                "(kwargs mismatch or missing)."
            )

    # Merge buckets and write.
    buckets = _group_by_path(stored_keys.values())
    for group in (keys_to_add.values(), keys_to_comment.values(), terms.values()):
        for path_key, (display, keys) in _group_by_path(group).items():
            buckets.setdefault(path_key, (display, []))[1].extend(keys)
    misc_by_path = _group_by_path(stored.misc)
    for path_key, (_, keys) in buckets.items():
        if path_key in misc_by_path:
            keys.extend(misc_by_path[path_key][1])

    for display, keys in buckets.values():
        full_path = os.path.join(lang_dir, display)
        content = generate_ftl(keys)
        if not options.dry_run:
            try:
                _write(full_path, content, options.line_endings)
            except OSError as err:
                raise ExtractionError(
                    f"Failed to write 1 .ftl file for locale `{lang}`:\n  - {full_path}: {err}"
                ) from err
        stats.stored_keys += sum(1 for key in keys if key.is_message())

    return result
