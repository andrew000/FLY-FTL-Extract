"""The ast-based *teacher*: reproduces what ``ftl extract`` 0.12.1 finds in Python code.

Used only to label the training corpus, to check the fly (``--fly-audit``) and in tests.
Never imported from the extraction hot path (CLAUDE.md rule 2).

Semantics follow ``matcher.rs`` of the original, verified against ``tests/golden``:

* a call is a candidate when its callee is a bare name from ``--i18n-keys`` or an
  attribute chain whose root is such a name (or a ``-p`` prefix followed by such a name);
* ``name("k")`` / ``i18n.get("k")``: the first positional argument must be a string literal;
* ``i18n.a.b_c()``: the attributes joined with ``-`` (underscores kept) unless the first
  attribute after the root is in ``--ignore-attributes``;
* keyword arguments become ``{ $placeables }`` (sorted), ``_path=`` selects the file,
  ``**kwargs`` marks the key's variables as unverifiable.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator

from fly_ftl_extract.files import find_py_files, mentions_any_name
from fly_ftl_extract.ftl.merge import CodeExtraction, FileExtraction, merge_extractions
from fly_ftl_extract.ftl.model import (
    GET_ATTR,
    PATH_KWARG,
    CodeLocation,
    DiagnosticKind,
    ExtractOptions,
    FluentKey,
    code_ftl_path,
    code_message,
    file_diagnostic,
)
from fly_ftl_extract.ftl.pyerrors import (
    column_from_byte_offset,
    ruff_style_syntax_error,
    utf8_error_message,
)


def _calls_in_source_order(tree: ast.AST) -> Iterator[ast.Call]:
    """All ``Call`` nodes in pre-order of the source text (outer call before its arguments)."""
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls.sort(
        key=lambda c: (c.lineno, c.col_offset, -(c.end_lineno or 0), -(c.end_col_offset or 0))
    )
    return iter(calls)


def _first_positional_literal(call: ast.Call) -> str | None:
    """``find_positional(0)``: first positional argument before any ``*args``, if a str literal."""
    if not call.args or isinstance(call.args[0], ast.Starred):
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


class _Matcher:
    def __init__(self, path: str, source: str, options: ExtractOptions) -> None:
        self.path = path
        self.lines = source.splitlines(keepends=True)
        self.options = options
        self.result = FileExtraction(path)

    def location(self, call: ast.Call) -> CodeLocation:
        line_text = self.lines[call.lineno - 1] if call.lineno - 1 < len(self.lines) else ""
        return CodeLocation(
            self.path, call.lineno, column_from_byte_offset(line_text, call.col_offset)
        )

    def visit(self, tree: ast.AST) -> None:
        for call in _calls_in_source_order(tree):
            func = call.func
            if isinstance(func, ast.Attribute):
                self.process_attribute_call(call, func)
            elif isinstance(func, ast.Name) and func.id in self.options.i18n_keys:
                self.process_name_call(call)

    def process_attribute_call(self, call: ast.Call, func: ast.Attribute) -> None:
        attrs: list[str] = []
        current: ast.expr = func
        while isinstance(current, ast.Attribute):
            attrs.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return
        root = current.id
        if root in self.options.i18n_keys:
            self.process_i18n_key_call(call, attrs)
        elif (
            root in self.options.i18n_keys_prefix and attrs and attrs[-1] in self.options.i18n_keys
        ):
            attrs.pop()
            self.process_i18n_key_call(call, attrs)

    def process_i18n_key_call(self, call: ast.Call, attrs: list[str]) -> None:
        if not attrs or (len(attrs) == 1 and attrs[0] == GET_ATTR):
            self.process_name_call(call)
        else:
            if attrs[-1] in self.options.ignore_attributes:
                return
            self.add_key(call, "-".join(reversed(attrs)))

    def process_name_call(self, call: ast.Call) -> None:
        literal = _first_positional_literal(call)
        if literal is not None:
            self.add_key(call, literal)

    def add_key(self, call: ast.Call, key: str) -> None:
        raw_path: str | None = None
        kwargs: list[str] = []
        kwargs_unknown: CodeLocation | None = None
        for keyword in call.keywords:
            if keyword.arg is None:
                if kwargs_unknown is None:
                    kwargs_unknown = self.location(call)
                continue
            if keyword.arg == PATH_KWARG:
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                    raw_path = value.value
            elif keyword.arg not in self.options.ignore_kwargs:
                kwargs.append(keyword.arg)
        fluent_key = FluentKey(
            key,
            code_message(key, kwargs),
            code_ftl_path(raw_path, self.options.default_ftl_file),
            source_location=self.location(call),
            kwargs_unknown=kwargs_unknown,
        )
        self.result.add(fluent_key)


def extract_file(path: str, options: ExtractOptions) -> FileExtraction:
    """Keys and diagnostics of one Python file (display ``path`` as produced by the walker)."""
    result = FileExtraction(path)
    try:
        if os.path.getsize(path) == 0:
            return result
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as err:
        result.diagnostics.append(
            file_diagnostic(
                DiagnosticKind.READ_ERROR, path, f"Failed to read Python file: {err}", 1, 1
            )
        )
        return result
    if not mentions_any_name(data, options.i18n_keys | options.i18n_keys_prefix):
        return result
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as err:
        message, line, column = utf8_error_message(data, err)
        result.diagnostics.append(
            file_diagnostic(DiagnosticKind.INVALID_UTF8, path, message, line, column)
        )
        return result
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        message, line, column = ruff_style_syntax_error(err.msg, err.lineno, err.offset, source)
        result.diagnostics.append(
            file_diagnostic(
                DiagnosticKind.PARSE_ERROR,
                path,
                f"Failed to parse Python file: {message}",
                line,
                column,
            )
        )
        return result
    matcher = _Matcher(path, source, options)
    matcher.visit(tree)
    return matcher.result


def extract_code(options: ExtractOptions) -> CodeExtraction:
    """Walk ``code_path`` and extract every file, merged like the original."""
    paths = find_py_files(options.code_path, options.exclude_dirs)
    files = [extract_file(path, options) for path in paths]
    return merge_extractions(files, len(paths))
