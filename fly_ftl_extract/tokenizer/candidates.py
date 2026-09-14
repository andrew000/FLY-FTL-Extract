"""Candidates for Fluent keys, found with ``tokenize`` alone (no ``ast``, CLAUDE.md rule 2).

A *candidate* is something the fly will smell and judge:

* ``string``: a string literal that stands as a positional argument — preceded by ``(``
  or ``,`` (``i18n.get("key")``, ``["a", "b"]``);
* ``chain``: a name or attribute chain ``NAME(.NAME)*`` immediately before ``(``
  (``i18n.balance.info(...)``, ``print(...)``).

Nothing here decides whether a candidate *is* a key.  What the tokenizer does decide is
purely lexical: bracket depth by counting, which call a candidate belongs to, the keyword
arguments of that call (each with its own :class:`Window`), the literal value of a string,
and the key a chain *would* produce (``some.key_1`` → ``some-key_1``: the leading
``-p`` prefix and ``--i18n-keys`` name are dropped, the rest joined with ``-``).

f-strings and t-strings arrive as ``FSTRING_*`` / ``TSTRING_*`` tokens, never as
``STRING``, so they cannot become candidates — the original ignores them too (fixture
``basic``, golden has no ``dynamic-…`` key).  Adjacent string literals (implicit
concatenation, one constant for the original) are merged into one candidate.  Bytes
literals are not candidates: the original only accepts ``str`` constants.
"""

from __future__ import annotations

import io
import keyword
import re
import tokenize
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, NamedTuple

from fly_ftl_extract.ftl.model import GET_ATTR, PATH_KWARG, ExtractOptions

WINDOW_BEFORE = 12
"""Tokens of context before the candidate (CLAUDE.md, «Odour»)."""
WINDOW_AFTER = 6
"""Tokens of context after the candidate."""

_SKIPPED = frozenset(
    {tokenize.NL, tokenize.COMMENT, tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT}
)
_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(_OPENERS.values())

FocusKind = Literal["string", "chain", "kwarg"]


class CandidateError(ValueError):
    """The file could not be tokenized (unterminated string/bracket, bad indentation…)."""


class Tok(NamedTuple):
    """One token as the encoder sees it: ``tokenize`` type name and source text."""

    type: str
    text: str


@dataclass(frozen=True)
class Window:
    """Context window of one focus (a candidate or a keyword argument)."""

    before: tuple[Tok, ...]
    focus: tuple[Tok, ...]
    after: tuple[Tok, ...]
    focus_kind: FocusKind
    depth: int
    """Bracket depth at the focus (0 = top level of a statement)."""
    first_positional: bool
    """The focus is the first positional argument of a call."""
    in_kwargs: bool
    """The focus lies inside the value of a keyword argument (or *is* the keyword)."""

    def tokens(self) -> tuple[Tok, ...]:
        """``before + focus + after`` in source order."""
        return self.before + self.focus + self.after


@dataclass(frozen=True)
class Kwarg:
    """One ``name=value`` of the call a candidate belongs to."""

    name: str
    line: int
    column: int
    window: Window
    path_value: str | None = None
    """Literal value when ``name == "_path"`` and the value is a single string literal."""


@dataclass(frozen=True)
class Candidate:
    """Something that might be a Fluent key."""

    kind: Literal["string", "chain"]
    line: int
    column: int
    """Start of the focus itself (1-based, characters), like the original's locations."""
    call_line: int | None
    call_column: int | None
    """Start of the callee chain of the call this candidate belongs to (the original
    reports keys at the call position): the chain itself, or the call whose first
    positional argument is this string; ``None`` when the string is not a call argument."""
    key_name: str | None
    """The key this candidate would produce: the literal value of a string, the joined
    attributes of a chain (``None`` for a bare ``name(`` or ``name.get(`` — there the key
    comes from the string argument)."""
    text: str
    """Raw source text of the focus."""
    window: Window
    kwargs: tuple[Kwarg, ...]
    kwargs_unknown: bool
    """The call passes ``**something``."""
    first_positional_is_string: bool
    """Chain candidates: the call's first positional argument is a string literal."""

    @property
    def position(self) -> tuple[int, int]:
        """``(line, column)`` of the focus."""
        return (self.line, self.column)

    @property
    def call_position(self) -> tuple[int, int] | None:
        """``(line, column)`` of the call, see :attr:`call_line`."""
        if self.call_line is None or self.call_column is None:
            return None
        return (self.call_line, self.call_column)


# --------------------------------------------------------------------------- literals

_STRING_RE = re.compile(r"^([A-Za-z]*)('''|\"\"\"|'|\")(.*)\2$", re.DOTALL)
_ESCAPE_RE = re.compile(
    r"\\(\r\n|\n|\r|N\{[^}]*\}|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[0-7]{1,3}|.)",
    re.DOTALL,
)
_SIMPLE_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


def _unescape(match: re.Match[str]) -> str:
    seq = match.group(1)
    if seq in ("\r\n", "\n", "\r"):
        return ""
    if seq[0] == "N":
        try:
            return unicodedata.lookup(seq[2:-1])
        except KeyError:
            return "\\" + seq
    if seq[0] == "x" or (seq[0] in "uU" and len(seq) > 1):
        return chr(int(seq[1:], 16))
    if seq[0] in "01234567":
        return chr(int(seq, 8))
    return _SIMPLE_ESCAPES.get(seq, "\\" + seq)


def string_literal_value(text: str) -> tuple[str, bool]:
    """``(value, is_bytes)`` of one ``STRING`` token, decoded without ``ast``."""
    match = _STRING_RE.match(text)
    if match is None:
        msg = f"not a string literal: {text!r}"
        raise ValueError(msg)
    prefix, _, body = match.groups()
    prefix = prefix.lower()
    is_bytes = "b" in prefix
    if "r" not in prefix:
        body = _ESCAPE_RE.sub(_unescape, body)
    return body, is_bytes


# --------------------------------------------------------------------------- tokens


def _significant(source: str) -> list[tokenize.TokenInfo]:
    try:
        return [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in _SKIPPED
        ]
    except (tokenize.TokenError, SyntaxError) as err:  # IndentationError is a SyntaxError
        raise CandidateError(str(err)) from err


def _depths(toks: list[tokenize.TokenInfo]) -> list[int]:
    """Bracket depth *before* each token, by counting."""
    depths = []
    depth = 0
    for tok in toks:
        depths.append(depth)
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth = max(depth - 1, 0)
    return depths


def _to_tok(tok: tokenize.TokenInfo) -> Tok:
    return Tok(tokenize.tok_name[tok.type], tok.string)


def _window(
    toks: list[tokenize.TokenInfo],
    depths: list[int],
    start: int,
    end: int,
    kind: FocusKind,
    *,
    first_positional: bool,
    in_kwargs: bool,
) -> Window:
    return Window(
        before=tuple(_to_tok(t) for t in toks[max(0, start - WINDOW_BEFORE) : start]),
        focus=tuple(_to_tok(t) for t in toks[start:end]),
        after=tuple(_to_tok(t) for t in toks[end : end + WINDOW_AFTER]),
        focus_kind=kind,
        depth=depths[start],
        first_positional=first_positional,
        in_kwargs=in_kwargs,
    )


def _is_op(tok: tokenize.TokenInfo, text: str) -> bool:
    return tok.type == tokenize.OP and tok.string == text


def _is_name(tok: tokenize.TokenInfo) -> bool:
    return tok.type == tokenize.NAME and not (keyword.iskeyword(tok.string))


def _chain_before(toks: list[tokenize.TokenInfo], open_index: int) -> tuple[int, int] | None:
    """``(start, end)`` of the ``NAME(.NAME)*`` chain right before ``toks[open_index]``."""
    j = open_index - 1
    if j < 0 or not _is_name(toks[j]):
        return None
    start = j
    while start - 2 >= 0 and _is_op(toks[start - 1], ".") and _is_name(toks[start - 2]):
        start -= 2
    return start, open_index


def _matching_close(toks: list[tokenize.TokenInfo], open_index: int) -> int | None:
    depth = 0
    for i in range(open_index, len(toks)):
        tok = toks[i]
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth -= 1
                if depth == 0:
                    return i
    return None


def _enclosing_open(toks: list[tokenize.TokenInfo], index: int) -> int | None:
    """Index of the unmatched opener before ``toks[index]``."""
    depth = 0
    for i in range(index - 1, -1, -1):
        tok = toks[i]
        if tok.type == tokenize.OP:
            if tok.string in _CLOSERS:
                depth += 1
            elif tok.string in _OPENERS:
                if depth == 0:
                    return i
                depth -= 1
    return None


def _in_kwargs(toks: list[tokenize.TokenInfo], index: int) -> bool:
    """Whether ``toks[index]`` lies inside the value of ``name=`` at any enclosing level."""
    i = index
    while True:
        opener = _enclosing_open(toks, i)
        if opener is None:
            return False
        # scan back from i to the opener at this level, looking for a top-level "=" after
        # the last top-level ","
        depth = 0
        for j in range(i - 1, opener, -1):
            tok = toks[j]
            if tok.type != tokenize.OP:
                continue
            if tok.string in _CLOSERS:
                depth += 1
            elif tok.string in _OPENERS:
                depth -= 1
            elif depth == 0 and tok.string == ",":
                break
            elif depth == 0 and tok.string == "=":
                return True
        i = opener


def _split_args(
    toks: list[tokenize.TokenInfo], open_index: int, close_index: int
) -> list[tuple[int, int]]:
    """``(start, end)`` token ranges of the top-level, comma-separated arguments."""
    args: list[tuple[int, int]] = []
    depth = 0
    start = open_index + 1
    for i in range(open_index + 1, close_index):
        tok = toks[i]
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
        elif tok.string == "," and depth == 0:
            if i > start:
                args.append((start, i))
            start = i + 1
    if close_index > start:
        args.append((start, close_index))
    return args


def _string_group(toks: list[tokenize.TokenInfo], start: int) -> int:
    """End index of the run of adjacent ``STRING`` tokens starting at ``start``."""
    end = start
    while end < len(toks) and toks[end].type == tokenize.STRING:
        end += 1
    return end


def _group_value(toks: list[tokenize.TokenInfo], start: int, end: int) -> tuple[str, bool]:
    parts = [string_literal_value(t.string) for t in toks[start:end]]
    return "".join(v for v, _ in parts), any(b for _, b in parts)


def chain_key_name(names: list[str], options: ExtractOptions) -> str | None:
    """The key a callee chain would produce, mirroring the original's ``-`` joining.

    Drops a leading ``-p`` prefix name followed by an ``--i18n-keys`` name, or a leading
    ``--i18n-keys`` name, otherwise the first name (it is the object the attributes hang
    on).  ``None`` for a bare call or ``x.get(`` — the key is then the string argument.
    """
    if (
        len(names) >= _PREFIX_AND_ROOT
        and names[0] in options.i18n_keys_prefix
        and names[1] in options.i18n_keys
    ):
        attrs = names[2:]
    else:
        attrs = names[1:]
    if not attrs or attrs == [GET_ATTR]:
        return None
    return "-".join(attrs)


@dataclass(frozen=True)
class _Call:
    chain_start: int
    open_index: int
    close_index: int
    kwargs: tuple[Kwarg, ...]
    kwargs_unknown: bool
    first_positional: tuple[int, int] | None
    """Token range of the first positional argument when it is a string group."""


def _analyse_call(
    toks: list[tokenize.TokenInfo], depths: list[int], chain_start: int, open_index: int
) -> _Call | None:
    close_index = _matching_close(toks, open_index)
    if close_index is None:
        return None
    kwargs: list[Kwarg] = []
    kwargs_unknown = False
    first_positional: tuple[int, int] | None = None
    seen_positional = False
    for start, end in _split_args(toks, open_index, close_index):
        first = toks[start]
        if _is_op(first, "**"):
            kwargs_unknown = True
            continue
        if end - start >= _NAME_EQ and first.type == tokenize.NAME and _is_op(toks[start + 1], "="):
            path_value = None
            if (
                first.string == PATH_KWARG
                and end - start == _NAME_EQ_VALUE
                and toks[start + 2].type == tokenize.STRING
            ):
                value, is_bytes = string_literal_value(toks[start + 2].string)
                if not is_bytes:
                    path_value = value
            kwargs.append(
                Kwarg(
                    name=first.string,
                    line=first.start[0],
                    column=first.start[1] + 1,
                    window=_window(
                        toks,
                        depths,
                        start,
                        start + 1,
                        "kwarg",
                        first_positional=False,
                        in_kwargs=True,
                    ),
                    path_value=path_value,
                )
            )
            continue
        if not seen_positional:
            seen_positional = True
            if not _is_op(first, "*") and first.type == tokenize.STRING:
                group_end = _string_group(toks, start)
                if group_end == end:
                    first_positional = (start, end)
    return _Call(
        chain_start, open_index, close_index, tuple(kwargs), kwargs_unknown, first_positional
    )


_NAME_EQ_VALUE = 3
_NAME_EQ = 2
_PREFIX_AND_ROOT = 2


def iter_candidates(source: str, options: ExtractOptions) -> Iterator[Candidate]:
    """Every candidate of one Python source, in source order.

    Raises :class:`CandidateError` when the source cannot be tokenized.
    """
    toks = _significant(source)
    depths = _depths(toks)
    calls: dict[int, _Call] = {}  # open_index -> call
    for i, tok in enumerate(toks):
        if _is_op(tok, "("):
            chain = _chain_before(toks, i)
            if chain is not None:
                call = _analyse_call(toks, depths, chain[0], i)
                if call is not None:
                    calls[i] = call

    i = 0
    while i < len(toks):
        tok = toks[i]
        if _is_op(tok, "(") and i in calls:
            call = calls[i]
            names = [t.string for t in toks[call.chain_start : i] if t.type == tokenize.NAME]
            first = toks[call.chain_start]
            yield Candidate(
                kind="chain",
                line=first.start[0],
                column=first.start[1] + 1,
                call_line=first.start[0],
                call_column=first.start[1] + 1,
                key_name=chain_key_name(names, options),
                text="".join(t.string for t in toks[call.chain_start : i]),
                window=_window(
                    toks,
                    depths,
                    call.chain_start,
                    i,
                    "chain",
                    first_positional=False,
                    in_kwargs=_in_kwargs(toks, call.chain_start),
                ),
                kwargs=call.kwargs,
                kwargs_unknown=call.kwargs_unknown,
                first_positional_is_string=call.first_positional is not None,
            )
            i += 1
            continue
        if (
            tok.type == tokenize.STRING
            and i > 0
            and (_is_op(toks[i - 1], "(") or _is_op(toks[i - 1], ","))
        ):
            end = _string_group(toks, i)
            value, is_bytes = _group_value(toks, i, end)
            if not is_bytes:
                opener = _enclosing_open(toks, i)
                call = calls.get(opener) if opener is not None else None
                is_first = call is not None and call.first_positional == (i, end)
                chain_first = toks[call.chain_start] if call is not None else None
                yield Candidate(
                    kind="string",
                    line=tok.start[0],
                    column=tok.start[1] + 1,
                    call_line=chain_first.start[0] if chain_first is not None else None,
                    call_column=chain_first.start[1] + 1 if chain_first is not None else None,
                    key_name=value,
                    text=" ".join(t.string for t in toks[i:end]),
                    window=_window(
                        toks,
                        depths,
                        i,
                        end,
                        "string",
                        first_positional=is_first,
                        in_kwargs=_in_kwargs(toks, i),
                    ),
                    kwargs=call.kwargs if is_first and call is not None else (),
                    kwargs_unknown=call.kwargs_unknown if is_first and call is not None else False,
                    first_positional_is_string=False,
                )
            i = end
            continue
        i += 1


def candidates_of_file(path: str, options: ExtractOptions) -> list[Candidate]:
    """Read ``path`` as UTF-8 and return its candidates (helper for scripts and tests)."""
    with open(path, encoding="utf-8") as fh:
        return list(iter_candidates(fh.read(), options))


def describe(candidate: Candidate) -> str:
    """One human-readable line per candidate (docs dumps)."""
    kwargs = ", ".join(
        f"{k.name}={k.path_value!r}" if k.path_value is not None else k.name
        for k in candidate.kwargs
    )
    call = (
        f" call@{candidate.call_line}:{candidate.call_column}"
        if candidate.call_position is not None and candidate.kind == "string"
        else ""
    )
    return (
        f"{candidate.line}:{candidate.column} {candidate.kind:<6} key={candidate.key_name!r}{call}"
        f" depth={candidate.window.depth} first_pos={int(candidate.window.first_positional)}"
        f" in_kwargs={int(candidate.window.in_kwargs)} kwargs=[{kwargs}]"
        f"{' **' if candidate.kwargs_unknown else ''}"
    )
