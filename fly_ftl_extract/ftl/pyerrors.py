"""Positions and messages of Python-file problems, formatted like the original (ast-free).

The original reports parse errors with ruff's parser text and a 1-based ``line:column``
where the column counts *characters*.  Python's ``SyntaxError`` (and ``tokenize`` errors in
the fly's path) have different texts and byte columns; the helpers here map what can be
mapped and leave the rest as an admitted approximation (see ``docs/FORMAT.md`` §8).
"""

from __future__ import annotations

RUFF_UNEXPECTED_EOF = "unexpected EOF while parsing"

_EOF_HINTS = ("was never closed", "unexpected EOF", "EOF in multi-line")


def eof_location(source: str) -> tuple[int, int]:
    """1-based ``(line, column)`` of the end of ``source``, as ruff reports EOF errors."""
    line = source.count("\n") + 1
    last_newline = source.rfind("\n")
    column = len(source) - (last_newline + 1) + 1
    return line, column


def column_from_byte_offset(line_text: str, byte_offset: int) -> int:
    """1-based character column for a UTF-8 ``byte_offset`` into ``line_text``."""
    return len(line_text.encode("utf-8")[:byte_offset].decode("utf-8", errors="ignore")) + 1


def ruff_style_syntax_error(
    message: str, line: int | None, column: int | None, source: str
) -> tuple[str, int, int]:
    """Map a Python syntax error to ``(ruff message, line, column)``.

    Unclosed brackets at end of file are reported by ruff as ``unexpected EOF while
    parsing`` at the EOF position; every other message is passed through as-is.
    """
    if any(hint in message for hint in _EOF_HINTS):
        eof_line, eof_column = eof_location(source)
        return RUFF_UNEXPECTED_EOF, eof_line, eof_column
    return message, line or 1, column or 1


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
