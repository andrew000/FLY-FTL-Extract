r"""File discovery that mirrors the ``ignore``-crate walk of the original ``ftl`` binary.

Observed rules (see ``docs/FORMAT.md`` §6):

* only files with the wanted extension are yielded;
* hidden entries (name starts with ``.``) are skipped unless an ignore rule whitelists them;
* ``.gitignore`` files *inside* the walked directory are honored (and ``.ignore`` for locale
  directories), rules from parent directories are not;
* exclude globs (``--exclude-dirs``) have the highest precedence; a glob ending in ``/**``
  also excludes the directory itself;
* the result is sorted the way Rust sorts ``PathBuf`` — component by component.

``os.path`` is used deliberately: the original prints paths exactly as it joined them
(``app\\a.py`` on Windows, ``<config dir>\\src/bot`` for values from ``pyproject.toml``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_DEFAULT_IGNORE_FILES: tuple[str, ...] = (".gitignore",)
_LOCALE_IGNORE_FILES: tuple[str, ...] = (".ignore", ".gitignore")


def _glob_to_regex(pattern: str) -> str:
    """Translate a gitignore-style glob into a regex over a ``/``-separated relative path."""
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**", i):
                if (i == 0 or pattern[i - 1] == "/") and pattern.startswith("/", i + 2):
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
                continue
            cls = pattern[i + 1 : j]
            if cls.startswith("!"):
                cls = "^" + cls[1:]
            out.append("[" + cls.replace("\\", "\\\\") + "]")
            i = j + 1
        elif c == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class _Rule:
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


@dataclass
class IgnoreRules:
    """Ordered gitignore rules; the last matching rule decides."""

    rules: list[_Rule] = field(default_factory=list)

    @classmethod
    def from_lines(cls, lines: list[str]) -> IgnoreRules:
        """Parse gitignore lines (comments, blanks, ``!`` negation, trailing ``/``)."""
        rules: list[_Rule] = []
        for raw in lines:
            line = raw.rstrip("\r\n")
            if not line.endswith("\\ "):
                line = line.rstrip(" ")
            if not line or line.startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:]
            if line.startswith("\\") and line[1:2] in ("#", "!"):
                line = line[1:]
            dir_only = line.endswith("/")
            if dir_only:
                line = line.rstrip("/")
            line = line.removeprefix("./")
            anchored = "/" in line
            line = line.lstrip("/")
            body = _glob_to_regex(line)
            regex = re.compile(f"^{body}$" if anchored else f"(?:^|.*/){body}$")
            rules.append(_Rule(regex, negate, dir_only))
        return cls(rules)

    def matched(self, rel_path: str, *, is_dir: bool) -> bool | None:
        """``True`` = ignored, ``False`` = whitelisted, ``None`` = no rule matched."""
        result: bool | None = None
        for rule in self.rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.match(rel_path):
                result = not rule.negate
        return result


def exclude_rules(exclude_globs: set[str] | frozenset[str]) -> IgnoreRules:
    """Rules equivalent to ``build_exclude_matcher`` of the original."""
    lines: list[str] = []
    for glob in sorted(exclude_globs):
        lines.append(glob)
        if glob.endswith("/**"):
            lines.append(glob[:-3])
    return IgnoreRules.from_lines(lines)


def _components(path: str) -> tuple[str, ...]:
    return tuple(p for p in re.split(r"[\\/]+", path) if p)


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _read_ignore_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def _walk(
    root: str,
    *,
    extension: str,
    excludes: IgnoreRules,
    ignore_files: tuple[str, ...],
) -> list[str]:
    """Return relative paths (``os.sep``-joined) of matching files below ``root``."""
    found: list[str] = []

    def decide(rel: str, name: str, *, is_dir: bool, stack: list[tuple[str, IgnoreRules]]) -> bool:
        """Skip? Excludes first, then ignore files (deepest first), then hidden."""
        if excludes.matched(rel, is_dir=is_dir) is True:
            return True
        for base, rules in reversed(stack):
            sub = rel[len(base) :].lstrip("/") if base else rel
            verdict = rules.matched(sub, is_dir=is_dir)
            if verdict is True:
                return True
            if verdict is False:
                return False
        return _is_hidden(name)

    def visit(directory: str, rel_dir: str, stack: list[tuple[str, IgnoreRules]]) -> None:
        lines: list[str] = []
        for ignore_name in ignore_files:
            lines.extend(_read_ignore_file(os.path.join(directory, ignore_name)))
        here = [*stack, (rel_dir, IgnoreRules.from_lines(lines))] if lines else stack
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            is_dir = entry.is_dir(follow_symlinks=True)
            if decide(rel, entry.name, is_dir=is_dir, stack=here):
                continue
            if is_dir:
                visit(entry.path, rel, here)
            elif entry.is_file() and entry.name.endswith(extension):
                found.append(rel.replace("/", os.sep))

    visit(root, "", [])
    return found


def find_py_files(code_path: str, exclude_globs: set[str] | frozenset[str]) -> list[str]:
    """Python files below (or exactly at) ``code_path``, as display paths, in Rust sort order."""
    excludes = exclude_rules(exclude_globs)
    if os.path.isdir(code_path):
        rels = _walk(
            code_path, extension=".py", excludes=excludes, ignore_files=_DEFAULT_IGNORE_FILES
        )
        paths = [os.path.join(code_path, rel) for rel in rels]
    elif (
        os.path.isfile(code_path)
        and code_path.endswith(".py")
        and excludes.matched(code_path.replace(os.sep, "/"), is_dir=False) is not True
    ):
        paths = [code_path]
    else:
        paths = []
    return sorted(paths, key=_components)


def find_ftl_files(locale_dir: str) -> list[str]:
    """``.ftl`` files below ``locale_dir`` relative to it (``os.sep``), Rust sort order."""
    if not os.path.isdir(locale_dir):
        return []
    rels = _walk(
        locale_dir, extension=".ftl", excludes=IgnoreRules(), ignore_files=_LOCALE_IGNORE_FILES
    )
    return sorted(rels, key=_components)


def mentions_any_name(data: bytes, names: set[str] | frozenset[str]) -> bool:
    """The original's pre-filter: a file is parsed only if its bytes contain one of ``names``.

    ``names`` are the ``--i18n-keys`` and ``-p`` prefixes.  This is a *walk* rule, not a
    key decision (CLAUDE.md rule 2c): a file that never mentions ``i18n`` is skipped
    entirely, so e.g. a syntax error in it is never reported.
    """
    return any(name.encode("utf-8") in data for name in names)


def path_sort_key(path: str) -> tuple[str, ...]:
    """Sort key reproducing Rust ``PathBuf`` ordering (component-wise)."""
    return _components(path)
