"""Positions and messages of Python-file problems, formatted like the original (ast-free).

The original reports parse errors with ruff's parser text and a 1-based ``line:column``
where the column counts *characters*.  Python's ``SyntaxError`` (and ``tokenize`` errors in
the fly's path) have different texts; the classes below are mapped (each verified by a
golden fixture), everything else is passed through as an admitted approximation
(see ``docs/FORMAT.md`` §8).

| Python                                    | ruff (original)                         |
|-------------------------------------------|-----------------------------------------|
| ``'(' was never closed`` / EOF in stmt    | ``unexpected EOF while parsing`` at EOF |
| ``unexpected indent``                     | ``Unexpected indentation`` at line:1    |
| ``expected an indented block after X …``  | ``Expected an indented block after X``  |
| ``invalid syntax`` at a non-token char    | ``Got unexpected token $``              |
| ``invalid syntax`` after ``def``/``class`` | ``Expected an identifier``             |
"""

from __future__ import annotations

import re

RUFF_UNEXPECTED_EOF = "unexpected EOF while parsing"
RUFF_UNEXPECTED_INDENT = "Unexpected indentation"
RUFF_EXPECTED_IDENTIFIER = "Expected an identifier"

_EOF_HINTS = ("was never closed", "unexpected EOF", "EOF in multi-line")
_EXPECTED_BLOCK = re.compile(r"^expected an indented block after (.+?)(?: on line \d+)?$")
_NEEDS_IDENTIFIER = re.compile(r"(?:^|\s)(?:def|class)\s+$")
_TOKEN_START = re.compile(r"[\w\s\"'#()\[\]{}.,:;@=+\-*/%<>&|^~\\]", re.UNICODE)


def eof_location(source: str) -> tuple[int, int]:
    """1-based ``(line, column)`` of the end of ``source``, as ruff reports EOF errors."""
    line = source.count("\n") + 1
    last_newline = source.rfind("\n")
    column = len(source) - (last_newline + 1) + 1
    return line, column


def column_from_byte_offset(line_text: str, byte_offset: int) -> int:
    """1-based character column for a UTF-8 ``byte_offset`` into ``line_text``."""
    return len(line_text.encode("utf-8")[:byte_offset].decode("utf-8", errors="ignore")) + 1


def _line_text(source: str, line: int) -> str:
    lines = source.splitlines()
    return lines[line - 1] if 0 < line <= len(lines) else ""


def ruff_style_syntax_error(
    message: str, line: int | None, column: int | None, source: str
) -> tuple[str, int, int]:
    """Map a Python syntax error to ``(ruff message, line, column)``; see the module table."""
    if any(hint in message for hint in _EOF_HINTS):
        eof_line, eof_column = eof_location(source)
        return RUFF_UNEXPECTED_EOF, eof_line, eof_column
    line = line or 1
    column = column or 1
    if message == "unexpected indent":
        return RUFF_UNEXPECTED_INDENT, line, 1
    block = _EXPECTED_BLOCK.match(message)
    if block is not None:
        return f"Expected an indented block after {block.group(1)}", line, 1
    if message == "invalid syntax":
        text = _line_text(source, line)
        before = text[: column - 1]
        char = text[column - 1 : column]
        if char and not _TOKEN_START.match(char):
            return f"Got unexpected token {char}", line, column
        if _NEEDS_IDENTIFIER.search(before):
            return RUFF_EXPECTED_IDENTIFIER, line, column
    return message, line, column


def utf8_error_message(data: bytes, error: UnicodeDecodeError) -> tuple[str, int, int]:
    """Rust ``Utf8Error`` display and the ``line:column`` of the first bad byte."""
    valid = data[: error.start].decode("utf-8", errors="ignore")
    line, column = eof_location(valid)
    length = error.end - error.start
    if error.end >= len(data) and "unexpected end" in error.reason:
        text = f"incomplete utf-8 byte sequence from index {error.start}"
    else:
        text = f"invalid utf-8 sequence of {length} bytes from index {error.start}"
    return f"Python file is not valid UTF-8: {text}", line, column
