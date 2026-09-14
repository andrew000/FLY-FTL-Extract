"""The tokenizer offers every key the reference finds (recall 1.0), without ``ast``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fly_ftl_extract.ftl.model import ExtractOptions, kwargs_from_key
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.tokenizer.candidates import (
    WINDOW_AFTER,
    WINDOW_BEFORE,
    Candidate,
    CandidateError,
    chain_key_name,
    iter_candidates,
    string_literal_value,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _fixtures import fixture_files, fixture_names  # noqa: E402

OPTS = ExtractOptions(code_path="app", locales_path="locales")
PREFIX_OPTS = ExtractOptions(
    code_path="app", locales_path="locales", i18n_keys_prefix=frozenset({"self", "cls"})
)


def cands(source: str, options: ExtractOptions = OPTS) -> list[Candidate]:
    return list(iter_candidates(source, options))


# --------------------------------------------------------------------- recall on fixtures


@pytest.mark.parametrize("fixture", fixture_names())
def test_every_reference_key_is_a_candidate(fixture: str) -> None:
    """Each positive of ``reference/`` is offered at the same call position with the same key."""
    checked = 0
    for f in fixture_files(fixture):
        if f.source is None:
            continue
        try:
            positives = key_occurrences(f.path, f.source, f.options)
        except SyntaxError:
            continue  # deliberately broken fixture files
        offered: dict[tuple[int, int], set[str | None]] = {}
        for c in iter_candidates(f.source, f.options):
            if c.call_position is not None:
                offered.setdefault(c.call_position, set()).add(c.key_name)
        for key in positives:
            assert key.source_location is not None
            pos = (key.source_location.line, key.source_location.column)
            assert pos in offered, f"{f.fixture}/{f.path}:{pos} {key.key!r} not a candidate"
            assert key.key in offered[pos], f"{f.fixture}/{f.path}:{pos} offers {offered[pos]}"
            checked += 1
    if fixture in ("basic", "prefix", "keys_lazy", "ignore_attrs"):
        assert checked > 0


def test_positive_kwargs_match_reference() -> None:
    """For every positive, the candidate's kwargs/_path/** agree with the reference."""
    for f in fixture_files("basic"):
        assert f.source is not None
        positives = key_occurrences(f.path, f.source, f.options)
        by_pos = {
            (c.call_position, c.key_name): c
            for c in iter_candidates(f.source, f.options)
            if c.call_position is not None
        }
        for key in positives:
            loc = key.source_location
            assert loc is not None
            c = by_pos[((loc.line, loc.column), key.key)]
            names = [k.name for k in c.kwargs if k.name != "_path"]
            assert sorted(names) == sorted(kwargs_from_key(key))
            assert c.kwargs_unknown == (key.kwargs_unknown is not None)
            path = next((k.path_value for k in c.kwargs if k.name == "_path"), None)
            if path:
                assert key.path.startswith(path.split("/")[0])


# --------------------------------------------------------------------- lexical behaviour


def test_fstrings_are_not_candidates() -> None:
    source = 'i18n.get(f"dynamic-{x}")\ni18n.get(t"template-{x}")\n'
    got = cands(source)
    assert [c.kind for c in got] == ["chain", "chain"]
    assert key_occurrences("x.py", source, OPTS) == []


def test_implicit_concatenation_is_one_candidate() -> None:
    source = 'i18n.get("a-" "b")\n'
    strings = [c for c in cands(source) if c.kind == "string"]
    assert len(strings) == 1
    assert strings[0].key_name == "a-b"
    assert strings[0].text == '"a-" "b"'
    assert [k.key for k in key_occurrences("x.py", source, OPTS)] == ["a-b"]


def test_string_positions_that_are_not_positional_arguments() -> None:
    source = 'i18n.get(key="kw")\nx["idx"]\n{"k": "v"}\ni18n.get(b"bytes")\n'
    assert [c.kind for c in cands(source)] == ["chain", "chain"]


def test_call_kwargs_and_path() -> None:
    source = 'i18n.get("k", a=1, _path="w/x.ftl", when=lambda: 0, **extra)\n'
    chain, string = cands(source)
    assert chain.kind == "chain"
    assert chain.key_name is None  # i18n.get -> the key is the string
    assert chain.first_positional_is_string
    assert string.kind == "string"
    assert string.key_name == "k"
    assert string.call_position == chain.position == (1, 1)
    assert string.window.first_positional
    assert [k.name for k in string.kwargs] == ["a", "_path", "when"]
    assert string.kwargs[1].path_value == "w/x.ftl"
    assert string.kwargs_unknown
    assert chain.kwargs == string.kwargs
    for k in string.kwargs:
        assert k.window.focus_kind == "kwarg"
        assert k.window.in_kwargs
        assert k.window.focus[0].text == k.name


def test_chain_key_names() -> None:
    assert chain_key_name(["i18n", "some", "key_1"], OPTS) == "some-key_1"
    assert chain_key_name(["i18n", "get"], OPTS) is None
    assert chain_key_name(["i18n"], OPTS) is None
    assert chain_key_name(["i18n", "only_attr"], OPTS) == "only_attr"
    assert chain_key_name(["self", "i18n", "a", "b"], PREFIX_OPTS) == "a-b"
    assert chain_key_name(["self", "i18n", "get"], PREFIX_OPTS) is None
    assert chain_key_name(["obj", "self", "i18n", "x"], PREFIX_OPTS) == "self-i18n-x"
    assert chain_key_name(["i18n", "core", "get"], OPTS) == "core-get"


def test_chain_hanging_off_an_expression_is_still_a_candidate() -> None:
    source = 'i18n.get("k").upper()\n'
    got = cands(source)
    assert [(c.kind, c.text) for c in got] == [
        ("chain", "i18n.get"),
        ("string", '"k"'),
        ("chain", "upper"),
    ]


def test_nested_calls_and_depth() -> None:
    source = 'f(i18n.get("outer", inner=i18n.get("inner", n=1)))\n'
    got = {(c.kind, c.text): c for c in cands(source)}
    outer = got[("string", '"outer"')]
    inner = got[("string", '"inner"')]
    assert outer.window.depth == 2
    assert inner.window.depth == 3
    assert not outer.window.in_kwargs
    assert inner.window.in_kwargs
    assert inner.call_position == got[("chain", "i18n.get")].position or inner.call_position == (
        1,
        28,
    )
    assert [k.name for k in outer.kwargs] == ["inner"]
    assert [k.name for k in inner.kwargs] == ["n"]


def test_window_sizes_and_positions() -> None:
    lines = ["x = 1"] * 20
    lines.append('await message.answer(i18n.get("menu-main-title", name=user.name))')
    source = "\n".join(lines) + "\n"
    string = next(c for c in cands(source) if c.kind == "string")
    assert string.position == (21, 31)
    assert len(string.window.before) == WINDOW_BEFORE
    assert len(string.window.after) == WINDOW_AFTER
    assert string.window.focus[0].type == "STRING"
    assert string.window.before[-1].text == "("
    assert [t.text for t in string.window.after[:2]] == [",", "name"]


def test_keywords_are_not_chains() -> None:
    source = "if (a):\n    return (b)\nawait (c)\nprint(d)\n"
    assert [c.text for c in cands(source)] == ["print"]


def test_unterminated_source_raises() -> None:
    with pytest.raises(CandidateError):
        cands('i18n.get("unterminated\n')
    with pytest.raises(CandidateError):
        cands("i18n.get(\n")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('"plain"', "plain"),
        ("'single'", "single"),
        ('"""triple"""', "triple"),
        (r'"esc\n\t\\"', "esc\n\t\\"),
        (r'r"raw\n"', "raw\\n"),
        (r'"\x41B\U00000043"', "ABC"),
        (r'"\N{LATIN SMALL LETTER A}"', "a"),
        ('"unicode-ключ"', "unicode-ключ"),
        (r'"\101"', "A"),
        (r'"\q"', "\\q"),
    ],
)
def test_string_literal_value(text: str, value: str) -> None:
    assert string_literal_value(text) == (value, False)


def test_bytes_literal_value() -> None:
    assert string_literal_value('b"raw"') == ("raw", True)
