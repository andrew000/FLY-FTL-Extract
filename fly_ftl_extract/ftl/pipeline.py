"""Everything ``ftl extract`` does after the keys are known: abort, import, merge, write, log.

The extractor (ast reference or the fly) hands over a :class:`CodeExtraction`; this module
produces the same side effects and the same log lines as the original's ``extract()`` +
``main.rs`` statistics block.  Log lines are returned, not printed, so the CLI can route
them (plain stderr like the original, or the TUI event log).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fly_ftl_extract.ftl.importer import ExtractionError, LocaleImport, import_locale
from fly_ftl_extract.ftl.merge import CodeExtraction
from fly_ftl_extract.ftl.model import Diagnostic, ExtractOptions
from fly_ftl_extract.ftl.process import LocaleStatistics, process_locale
from fly_ftl_extract.ftl.rustorder import RustMap

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG_ERROR = 2

_LEVEL_WIDTH = 5
_CONTINUATION_INDENT = "    "


@dataclass(frozen=True)
class LogLine:
    """One env_logger record: ``[LEVEL target] message`` (multi-line messages indented by 4)."""

    level: str
    target: str
    message: str

    def render(self) -> str:
        """Text exactly as the original's env_logger prints it."""
        first, *rest = self.message.split("\n")
        lines = [f"[{self.level:<{_LEVEL_WIDTH}} {self.target}] {first}"]
        lines.extend(_CONTINUATION_INDENT + line for line in rest)
        return "\n".join(lines)


@dataclass
class ExtractOutcome:
    """Result of a run: exit code, log lines and (on success) statistics."""

    exit_code: int
    logs: list[LogLine] = field(default_factory=list)
    py_files_count: int = 0
    keys_in_code: int = 0
    per_locale: RustMap[LocaleStatistics] = field(default_factory=RustMap)

    def render_logs(self) -> str:
        """All log lines joined, one per line."""
        return "".join(line.render() + "\n" for line in self.logs)


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _check_diagnostics(
    diagnostics: list[Diagnostic], *, allow_parse_errors: bool, logs: list[LogLine]
) -> str | None:
    """Return the ``Extraction aborted`` message, or ``None`` when the run may continue."""
    if not diagnostics:
        return None
    file_errors = [d for d in diagnostics if d.is_file_error]
    conflicts = [d for d in diagnostics if not d.is_file_error]
    if allow_parse_errors:
        logs.extend(
            LogLine("WARN", "extractor::ftl", f"Skipping Python file: {d}") for d in file_errors
        )
        failing = conflicts
    else:
        failing = diagnostics
    if not failing:
        return None
    message = (
        f"Extraction aborted: {len(failing)} {_plural(len(failing), 'problem')} found in Python "
        "sources, no .ftl files were written."
    )
    message += "".join(f"\n  - {d}" for d in failing)
    if not allow_parse_errors and file_errors:
        message += (
            "\nPass --allow-parse-errors to skip unreadable or unparseable files and continue."
        )
    return message


def _import_languages(options: ExtractOptions) -> dict[str, LocaleImport]:
    imports: dict[str, LocaleImport] = {}
    duplicates = []
    for lang in options.languages:
        imported, dups = import_locale(options.locales_path, lang)
        imports[lang] = imported
        duplicates.extend(dups)
    if duplicates:
        message = (
            f"Extraction aborted: {len(duplicates)} {_plural(len(duplicates), 'problem')} found in "
            ".ftl files, no .ftl files were written."
        )
        message += "".join(f"\n  - {d}" for d in duplicates)
        raise ExtractionError(message)
    return imports


def run_extract(
    extraction: CodeExtraction, options: ExtractOptions, *, extraction_seconds: float | None = None
) -> ExtractOutcome:
    """Run the post-extraction pipeline and return logs, exit code and statistics.

    ``extraction_seconds`` is what the ``FTL Extraction completed in …`` line reports (the
    time the caller spent producing ``extraction``); without it the line shows ~0 s.
    """
    outcome = ExtractOutcome(EXIT_OK)
    logs = outcome.logs
    logs.append(LogLine("INFO", "cli", f"Code path: {options.code_path}"))
    logs.append(LogLine("INFO", "cli", f"Locales path: {options.locales_path}"))
    started = time.perf_counter()

    per_locale: RustMap[LocaleStatistics] = RustMap()
    for lang in options.languages:
        per_locale.insert(lang, LocaleStatistics())
    outcome.per_locale = per_locale
    outcome.py_files_count = extraction.py_files_with_keys
    elapsed = (
        extraction_seconds if extraction_seconds is not None else time.perf_counter() - started
    )
    logs.append(LogLine("INFO", "extractor::ftl", f"FTL Extraction completed in {elapsed:.3f}s."))

    abort = _check_diagnostics(
        extraction.diagnostics, allow_parse_errors=options.allow_parse_errors, logs=logs
    )
    if abort is not None:
        logs.append(LogLine("ERROR", "cli", f"Error during extraction: {abort}"))
        outcome.exit_code = EXIT_ERROR
        return outcome

    outcome.keys_in_code = len(extraction.keys)
    started = time.perf_counter()
    try:
        imports = _import_languages(options)
        for lang in options.languages:
            result = process_locale(lang, imports[lang], extraction.keys, options)
            logs.extend(LogLine("WARN", "extractor::ftl", w) for w in result.warnings)
            per_locale[lang].merge(result.statistics)
    except ExtractionError as err:
        logs.append(LogLine("ERROR", "cli", f"Error during extraction: {err}"))
        outcome.exit_code = EXIT_ERROR
        return outcome
    logs.append(
        LogLine(
            "INFO",
            "extractor::ftl",
            f"FTL Processing completed in {time.perf_counter() - started:.3f}s.",
        )
    )

    def fmt(attr: str) -> str:
        return per_locale.debug_format(lambda s: str(getattr(s, attr)))

    logs.append(LogLine("INFO", "cli", "Extraction statistics:"))
    logs.append(LogLine("INFO", "cli", f"  - Py files count: {outcome.py_files_count}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL files count: {fmt('files_count')}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL keys in code: {outcome.keys_in_code}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL keys stored: {fmt('stored_keys')}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL keys updated: {fmt('updated')}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL keys added: {fmt('added')}"))
    logs.append(LogLine("INFO", "cli", f"  - FTL keys commented: {fmt('commented')}"))
    return outcome


def done_line(elapsed: float) -> LogLine:
    """The final ``✅ Done in …s.`` line printed by ``main.rs``."""
    return LogLine("INFO", "cli", f"✅ Done in {elapsed:.3f}s.")
