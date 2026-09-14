"""Generate the fly's training corpus: random Python modules labelled by the ast teacher.

Every snippet is generated under a random option configuration (``--i18n-keys``, ``-p``,
``--ignore-attributes``, ``--ignore-kwargs``); the encoder normalises with those options and
``reference/`` labels with the same options, so the same text can be a key in one snippet
and not in another.  Two tables come out:

* ``keys``   — one row per candidate, label ``is_key``;
* ``kwargs`` — one row per keyword argument of a *positive* call, label ``is_placeable``
  (``_path`` and ``--ignore-kwargs`` names are the negatives).

Odours are stored already encoded (``*.npz``) next to the metadata (``*.jsonl``), so that
``scripts/train.py`` never needs the tokenizer.  Class balance ≈ 1:3 (the majority class is
subsampled), split 80/10/10 by *snippet*, fixed seed.  Fixtures from ``tests/`` are never
part of the corpus (``tests/test_dataset.py`` checks it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fly_ftl_extract.dopamine.seed import kwarg_trial_index, trial_seed
from fly_ftl_extract.ftl.model import (
    DEFAULT_I18N_KEYS,
    DEFAULT_IGNORE_ATTRIBUTES,
    ExtractOptions,
)
from fly_ftl_extract.odor.encoder import ENCODER_VERSION, encode_many, normalize_window
from fly_ftl_extract.reference.extractor import key_occurrences
from fly_ftl_extract.reference.labels import LabelError, label_candidates, label_kwargs
from fly_ftl_extract.tokenizer.candidates import Candidate, CandidateError, iter_candidates

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "fly_ftl_extract" / "data" / "dataset"
GENERATOR_VERSION = "grammar-3"
DEFAULT_SNIPPETS = 20_000
DEFAULT_SEED = 20240914
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
NEGATIVE_RATIO = 3  # majority : minority
SPLIT_NAMES = ("train", "val", "test")
PREFIXED_BARE_CALL = 0.6  # share of bare calls written as self.L(...) / cls.LazyProxy(...)
PREFIX_GET_CALL = 0.08

# ------------------------------------------------------------------- vocabulary

WORDS = (  # noqa: SIM905
    "menu main title hello user balance info wallet start help error name count amount "
    "currency welcome back settings lang profile order item total price date time status ok "
    "cancel confirm delete edit save load send message button label text value result page "
    "next prev list detail summary note warning success failed admin chat group invite "
    "reply photo file card pay refund bonus level rank score task done retry limit"
).split()
I18N_POOL = (
    "i18n",
    "L",
    "LazyProxy",
    "LazyFilter",
    "LF",
    "I18nFormat",
    "tr",
    "t",
    "gettext",
    "translator",
)
PREFIX_POOL = ("self", "cls")
CUSTOM_IGNORE_ATTRS = ("core", "reload", "setup", "bind", "context", "plural")
IGNORE_KWARG_POOL = ("when", "ctx", "default", "locale", "fmt")
LOOKALIKE_ROOTS = ("i18n_utils", "my_i18n", "i18n2", "l10n", "i18n_ctx", "translations")
PLAIN_CALLEES = ("print", "logger.info", "bot.send", "message.answer", "render", "Const", "make")
KWARG_NAMES = (
    "name",
    "user",
    "amount",
    "count",
    "x",
    "y",
    "n",
    "who",
    "value",
    "first",
    "second",
    "id",
)


@dataclass(frozen=True)
class OptionConfig:
    """The option subset a snippet was generated (and labelled, and encoded) under."""

    i18n_keys: tuple[str, ...]
    i18n_keys_prefix: tuple[str, ...]
    ignore_attributes: tuple[str, ...]
    ignore_kwargs: tuple[str, ...]

    def options(self) -> ExtractOptions:
        return ExtractOptions(
            code_path="app",
            locales_path="locales",
            i18n_keys=frozenset(self.i18n_keys),
            i18n_keys_prefix=frozenset(self.i18n_keys_prefix),
            ignore_attributes=frozenset(self.ignore_attributes),
            ignore_kwargs=frozenset(self.ignore_kwargs),
        )


@dataclass(frozen=True)
class Snippet:
    """One generated module."""

    id: int
    source: str
    config: OptionConfig


def random_config(rng: random.Random) -> OptionConfig:
    r = rng.random()
    if r < 0.55:  # noqa: PLR2004
        keys: tuple[str, ...] = tuple(sorted(DEFAULT_I18N_KEYS))
    else:
        keys = tuple(sorted(rng.sample(I18N_POOL, rng.randint(1, 4))))
        if rng.random() < 0.5 and "i18n" not in keys:  # noqa: PLR2004
            keys = (*keys, "i18n")
    prefixes: tuple[str, ...] = ()
    if rng.random() < 0.5:  # noqa: PLR2004
        prefixes = tuple(sorted(rng.sample(PREFIX_POOL, rng.randint(1, 2))))
    r = rng.random()
    if r < 0.6:  # noqa: PLR2004
        ignore: tuple[str, ...] = tuple(sorted(DEFAULT_IGNORE_ATTRIBUTES))
    elif r < 0.85:  # noqa: PLR2004
        ignore = tuple(
            sorted(
                DEFAULT_IGNORE_ATTRIBUTES | set(rng.sample(CUSTOM_IGNORE_ATTRS, rng.randint(1, 2)))
            )
        )
    else:
        ignore = tuple(sorted(rng.sample(CUSTOM_IGNORE_ATTRS, rng.randint(1, 2))))
    ignore_kwargs: tuple[str, ...] = ()
    if rng.random() < 0.35:  # noqa: PLR2004
        ignore_kwargs = tuple(sorted(rng.sample(IGNORE_KWARG_POOL, rng.randint(1, 2))))
    return OptionConfig(keys, prefixes, ignore, ignore_kwargs)


# ------------------------------------------------------------------- grammar


class Grammar:
    """Random Python modules with the constructs listed in CLAUDE.md / the Phase 5 brief."""

    def __init__(self, rng: random.Random, config: OptionConfig) -> None:
        self.rng = rng
        self.config = config
        self.in_async = False
        self.in_function = False
        self.in_method: str | None = None  # "self" / "cls" inside a class body

    # -- atoms -------------------------------------------------------------------------
    def word(self) -> str:
        return self.rng.choice(WORDS)

    def key(self) -> str:
        parts = [self.word() for _ in range(self.rng.randint(1, 3))]
        sep = self.rng.choice(["-", "-", "-", "_", "."])
        text = sep.join(parts)
        if self.rng.random() < 0.15:  # noqa: PLR2004
            text += f"_{self.rng.randint(1, 9)}"
        if self.rng.random() < 0.05:  # noqa: PLR2004
            text = "ключ-" + text
        return text

    def ident(self) -> str:
        return "_".join(self.word() for _ in range(self.rng.randint(1, 2)))

    def string(self) -> str:
        quote = self.rng.choice(['"', '"', "'"])
        return f"{quote}{self.key()}{quote}"

    def root(self) -> str:
        """A callee root: a configured i18n name, possibly behind a prefix."""
        name = self.rng.choice(self.config.i18n_keys)
        r = self.rng.random()
        if self.in_method and r < 0.6:  # noqa: PLR2004
            return f"{self.in_method}.{name}"
        if r < 0.7:  # noqa: PLR2004
            return name
        return f"{self.rng.choice(['self', 'cls', 'obj', 'ctx', 'app'])}.{name}"

    def negative_root(self) -> str:
        r = self.rng.random()
        if r < 0.4:  # noqa: PLR2004
            return self.rng.choice(LOOKALIKE_ROOTS)
        if r < 0.7:  # noqa: PLR2004
            unused = [n for n in I18N_POOL if n not in self.config.i18n_keys]
            return self.rng.choice(unused) if unused else "translate"
        if r < 0.85:  # noqa: PLR2004
            return f"{self.rng.choice(['obj', 'other', 'request', 'self'])}.{self.rng.choice(self.config.i18n_keys)}"
        # a prefix name behind another object is not a root (obj.self.i18n.get is no key)
        return f"{self.rng.choice(['obj', 'request', 'ctx'])}.{self.rng.choice(PREFIX_POOL)}.{self.rng.choice(self.config.i18n_keys)}"

    def kwarg_name(self) -> str:
        r = self.rng.random()
        if r < 0.12:  # noqa: PLR2004
            return "_path"
        if r < 0.3 and self.config.ignore_kwargs:  # noqa: PLR2004
            return self.rng.choice(self.config.ignore_kwargs)
        if r < 0.36:  # noqa: PLR2004
            return self.rng.choice(IGNORE_KWARG_POOL)
        return self.rng.choice(KWARG_NAMES)

    def value(self, depth: int = 0) -> str:
        r = self.rng.random()
        if r < 0.25:  # noqa: PLR2004
            return str(self.rng.randint(0, 99))
        if r < 0.45:  # noqa: PLR2004
            return self.string()
        if r < 0.6:  # noqa: PLR2004
            return self.ident()
        if r < 0.72:  # noqa: PLR2004
            return f"{self.ident()}.{self.ident()}"
        if r < 0.8 and depth < 2:  # noqa: PLR2004
            return self.get_call(depth + 1)
        if r < 0.88:  # noqa: PLR2004
            return f"{self.ident()}({self.value(depth + 1)})"
        if r < 0.94:  # noqa: PLR2004
            return f"lambda: {self.ident()}"
        return f'f"{self.word()}-{{{self.ident()}}}"'

    def kwargs(self, depth: int = 0, *, allow_stars: bool = True) -> list[str]:
        items = []
        used: set[str] = set()
        for _ in range(self.rng.randint(0, 4)):
            name = self.kwarg_name()
            if name in used:
                continue
            used.add(name)
            if name == "_path":
                value = self.rng.choice(
                    ['"wallet/balance.ftl"', '"wallet"', '""', '"a/b/c.ftl"', self.ident()]
                )
            else:
                value = self.value(depth)
            items.append(f"{name}={value}")
        if allow_stars and self.rng.random() < 0.08:  # noqa: PLR2004
            items.append(f"**{self.ident()}")
        return items

    # -- calls -------------------------------------------------------------------------
    def call_args(self, first: str | None, depth: int) -> str:
        parts = [] if first is None else [first]
        if first is not None and self.rng.random() < 0.06:  # noqa: PLR2004
            parts.append(f"*{self.ident()}")
        parts.extend(self.kwargs(depth))
        return ", ".join(parts)

    def get_call(self, depth: int = 0, root: str | None = None) -> str:
        root = root or self.root()
        r = self.rng.random()
        if r < 0.6:  # noqa: PLR2004
            first: str | None = self.string()
        elif r < 0.68:  # noqa: PLR2004
            first = self.ident()
        elif r < 0.74:  # noqa: PLR2004
            first = f'f"{self.word()}-{{{self.ident()}}}"'
        elif r < 0.79:  # noqa: PLR2004
            first = f'"{self.word()}-" "{self.word()}"'
        elif r < 0.84:  # noqa: PLR2004
            first = f'"{self.word()}-" + {self.ident()}'
        elif r < 0.9:  # noqa: PLR2004
            first = f"{self.ident()}({self.string()})"
        elif r < 0.94:  # noqa: PLR2004
            first = f"*{self.ident()}"
        elif r < 0.97:  # noqa: PLR2004
            first = None
        else:
            return f"{root}.get(key={self.string()})"
        return f"{root}.get({self.call_args(first, depth)})"

    def attr_call(self, root: str | None = None) -> str:
        root = root or self.root()
        n = self.rng.randint(1, 4)
        attrs = [self.ident() for _ in range(n)]
        r = self.rng.random()
        if r < 0.15:  # noqa: PLR2004
            attrs[0] = self.rng.choice(self.config.ignore_attributes)
        elif r < 0.25:  # noqa: PLR2004
            attrs[0] = self.rng.choice(("set_locale", "use_locale", *CUSTOM_IGNORE_ATTRS))
        if self.rng.random() < 0.2 and n > 1:  # noqa: PLR2004
            attrs[-1] = "get"
        first = self.string() if attrs[-1] == "get" or self.rng.random() < 0.1 else None  # noqa: PLR2004
        return f"{root}.{'.'.join(attrs)}({self.call_args(first, 0)})"

    def bare_call(self, root: str | None = None) -> str:
        # self.L("k") / cls.LazyProxy("k") are keys with -p, plain LF("k") always
        prefixed = self.rng.random() < PREFIXED_BARE_CALL
        root = root or (self.root() if prefixed else self.rng.choice(self.config.i18n_keys))
        first = self.string() if self.rng.random() < 0.85 else self.ident()  # noqa: PLR2004
        return f"{root}({self.call_args(first, 0)})"

    def ignore_call(self) -> str:
        attr = self.rng.choice(
            (
                "set_locale",
                "use_locale",
                "use_context",
                "set_context",
                *self.config.ignore_attributes,
            )
        )
        arg = self.rng.choice(['"uk"', '"en"', "", f"{self.ident()}=1"])
        if self.rng.random() < 0.3:  # noqa: PLR2004
            return f"{self.root()}.{attr}.{self.ident()}({arg})"
        return f"{self.root()}.{attr}({arg})"

    def lookalike_call(self) -> str:
        root = self.negative_root()
        r = self.rng.random()
        if r < 0.12:  # noqa: PLR2004
            # a prefix name is not a key by itself: self.get("x"), cls("x")
            prefix = self.rng.choice(PREFIX_POOL)
            return (
                f"{prefix}.get({self.call_args(self.string(), 0)})"
                if r < PREFIX_GET_CALL
                else f"{prefix}({self.call_args(self.string(), 0)})"
            )
        if r < 0.5:  # noqa: PLR2004
            return self.get_call(root=root)
        if r < 0.8:  # noqa: PLR2004
            return self.attr_call(root=root)
        return f"{root}({self.call_args(self.string(), 0)})"

    def plain_call(self) -> str:
        callee = self.rng.choice(PLAIN_CALLEES)
        parts = [self.string() if self.rng.random() < 0.7 else self.ident()]  # noqa: PLR2004
        if self.rng.random() < 0.5:  # noqa: PLR2004
            parts.append(self.value())
        parts.extend(self.kwargs(1, allow_stars=False))
        return f"{callee}({', '.join(parts)})"

    def i18n_expr(self) -> str:
        r = self.rng.random()
        if r < 0.45:  # noqa: PLR2004
            return self.get_call()
        if r < 0.7:  # noqa: PLR2004
            return self.attr_call()
        if r < 0.82:  # noqa: PLR2004
            return self.bare_call()
        if r < 0.9:  # noqa: PLR2004
            return self.ignore_call()
        return self.lookalike_call()

    # -- statements --------------------------------------------------------------------
    def wrap(self, expr: str) -> list[str]:
        r = self.rng.random()
        if self.in_async and r < 0.3:  # noqa: PLR2004
            return [f"await {self.rng.choice(['bot.send', 'message.answer', 'reply'])}({expr})"]
        if r < 0.45:  # noqa: PLR2004
            return [f"{self.ident()} = {expr}"]
        if r < 0.6 and self.in_function:  # noqa: PLR2004
            return [f"return {expr}"]
        if r < 0.7:  # noqa: PLR2004
            return [f"{self.rng.choice(PLAIN_CALLEES)}({self.ident()}, {expr})"]
        if r < 0.78:  # noqa: PLR2004
            return [f"{self.ident()} = (lambda: {expr})()"]
        if r < 0.84:  # noqa: PLR2004
            return [f"{self.ident()} = lambda: {expr}"]
        if r < 0.9:  # noqa: PLR2004
            return [f"{expr}.{self.ident()}()"]
        return [expr]

    def multiline_call(self) -> list[str]:
        root = self.root()
        kwargs = self.kwargs(1)
        if not kwargs:
            kwargs = [f"{self.rng.choice(KWARG_NAMES)}={self.value(1)}"]
        first = self.string() if self.rng.random() < 0.85 else self.attr_call(root="").split("(")[0]  # noqa: PLR2004
        head = (
            f"{self.ident()} = {root}.get("
            if first.startswith(('"', "'"))
            else f"{self.ident()} = {root}.{self.ident()}("
        )
        lines = [head]
        if first.startswith(('"', "'")):
            lines.append(f"    {first},")
        lines.extend(f"    {k}," for k in kwargs)
        lines.append(")")
        return lines

    def other_string_statement(self) -> list[str]:
        r = self.rng.random()
        if r < 0.15:  # noqa: PLR2004
            return [f"{self.ident()} = {self.string()}"]
        if r < 0.3:  # noqa: PLR2004
            return [f"{self.ident()}[{self.string()}] = {self.value()}"]
        if r < 0.45:  # noqa: PLR2004
            return [self.plain_call()]
        if r < 0.55:  # noqa: PLR2004
            return [f"{self.ident()} = [{self.string()}, {self.string()}, {self.ident()}]"]
        if r < 0.65:  # noqa: PLR2004
            return [
                f"{self.ident()} = {{{self.string()}: {self.string()}, {self.string()}: {self.value()}}}"
            ]
        if r < 0.72:  # noqa: PLR2004
            return [f"assert {self.ident()}, {self.string()}"]
        if r < 0.8:  # noqa: PLR2004
            return [f"raise ValueError({self.string()})"]
        if r < 0.88:  # noqa: PLR2004
            return [f"{self.ident()}: {self.string()} = {self.value()}"]
        if r < 0.94:  # noqa: PLR2004
            return [f"# {self.get_call()}"]
        return [f'{self.ident()} = f"{self.word()} {{{self.ident()}}} {self.word()}"']

    def statement(self) -> list[str]:
        r = self.rng.random()
        if r < 0.5:  # noqa: PLR2004
            return self.wrap(self.i18n_expr())
        if r < 0.58:  # noqa: PLR2004
            return self.multiline_call()
        if r < 0.9:  # noqa: PLR2004
            return self.other_string_statement()
        return [f"{self.ident()} = {self.value()}"]

    def body(self, n: int) -> list[str]:
        lines: list[str] = []
        for _ in range(n):
            lines.extend(self.statement())
        if all(ln.lstrip().startswith("#") for ln in lines):
            lines.append("pass")
        return lines

    # -- blocks ------------------------------------------------------------------------
    def decorator(self) -> list[str]:
        r = self.rng.random()
        if r < 0.5:  # noqa: PLR2004
            return []
        if r < 0.75:  # noqa: PLR2004
            return [
                f"@{self.ident()}.{self.rng.choice(['message', 'route', 'callback'])}({self.string()})"
            ]
        return [f"@{self.ident()}"]

    def function(self) -> list[str]:
        self.in_async = self.rng.random() < 0.4  # noqa: PLR2004
        self.in_function = True
        prev = self.in_method
        self.in_method = None
        params = ", ".join(
            dict.fromkeys(
                [
                    *(self.ident() for _ in range(self.rng.randint(0, 3))),
                    *self.rng.sample(["i18n", "message", "user"], self.rng.randint(0, 2)),
                ]
            )
        )
        lines = [
            *self.decorator(),
            f"{'async ' if self.in_async else ''}def {self.ident()}({params}):",
        ]
        if self.rng.random() < 0.3:  # noqa: PLR2004
            lines.append(f'    """{self.word().capitalize()} {self.word()}: {self.get_call()}."""')
        lines.extend(f"    {ln}" for ln in self.body(self.rng.randint(1, 6)))
        self.in_async = False
        self.in_function = False
        self.in_method = prev
        return lines

    def klass(self) -> list[str]:
        lines = [f"class {self.ident().title().replace('_', '')}:"]
        if self.rng.random() < 0.5:  # noqa: PLR2004
            lines.append("    def __init__(self, i18n):\n        self.i18n = i18n")
        for _ in range(self.rng.randint(1, 3)):
            kind = self.rng.choice(["self", "self", "cls"])
            self.in_method = kind
            self.in_function = True
            self.in_async = kind == "self" and self.rng.random() < 0.4  # noqa: PLR2004
            head = "    @classmethod\n" if kind == "cls" else ""
            params = ", ".join(
                dict.fromkeys([kind, *(self.ident() for _ in range(self.rng.randint(0, 2)))])
            )
            lines.append(
                f"{head}    {'async ' if self.in_async else ''}def {self.ident()}({params}):"
            )
            lines.extend(f"        {ln}" for ln in self.body(self.rng.randint(1, 5)))
            self.in_method = None
            self.in_function = False
            self.in_async = False
        return lines

    def module(self) -> str:
        r = self.rng.random()
        # 8 % of modules are 1-4 lines and start with a statement: keys at file start
        # (no context before the candidate) must be smelled too
        target = self.rng.randint(1, 4) if r < 0.08 else self.rng.randint(5, 40)  # noqa: PLR2004
        lines: list[str] = []
        if r >= 0.08 and self.rng.random() < 0.55:  # noqa: PLR2004
            names = ", ".join(
                sorted(
                    set(
                        self.rng.sample(
                            [*self.config.i18n_keys, "I18nContext", "Router"],
                            min(3, len(self.config.i18n_keys) + 2),
                        )
                    )
                )
            )
            lines.append(
                f"from {self.rng.choice(['aiogram_i18n', 'app.i18n', 'utils'])} import {names}"
            )
        while len(lines) < target:
            r = self.rng.random()
            if r < 0.35:  # noqa: PLR2004
                lines.extend(self.function())
            elif r < 0.55:  # noqa: PLR2004
                lines.extend(self.klass())
            else:
                lines.extend(self.body(self.rng.randint(1, 3)))
            lines.append("")
        return "\n".join(lines).rstrip("\n") + "\n"


# ------------------------------------------------------------------- generation


def generate_snippets(n: int, seed: int) -> tuple[list[Snippet], int]:
    """``n`` compilable snippets with at least one candidate; also how many were rejected."""
    rng = random.Random(seed)  # noqa: S311 — a corpus, not a secret
    out: list[Snippet] = []
    rejected = 0
    while len(out) < n:
        config = random_config(rng)
        source = Grammar(rng, config).module()
        try:
            compile(source, "<snippet>", "exec", dont_inherit=True)
        except SyntaxError:
            rejected += 1
            continue
        out.append(Snippet(len(out), source, config))
    return out, rejected


@dataclass
class Row:
    """One dataset row before balancing."""

    snippet: int
    index: int
    label: bool
    seed: int
    meta: dict[str, object]
    odor: np.ndarray


def rows_of_snippet(snippet: Snippet) -> tuple[list[Row], list[Row]] | None:
    options = snippet.config.options()
    try:
        candidates = list(iter_candidates(snippet.source, options))
        occurrences = key_occurrences("snippet.py", snippet.source, options)
        labels, positive_index = label_candidates(candidates, occurrences)
    except (CandidateError, SyntaxError, LabelError) as err:
        print(f"snippet {snippet.id}: {type(err).__name__}: {err}", file=sys.stderr)
        return None
    if not candidates:
        return None
    content = snippet.source.encode("utf-8")
    odors = encode_many([c.window for c in candidates], options)
    keys = [
        Row(
            snippet.id,
            i,
            labels[i],
            trial_seed(content, i, ENCODER_VERSION),
            {
                "line": c.line,
                "column": c.column,
                "kind": c.kind,
                "key_name": c.key_name,
                "text": c.text,
                "window": " ".join(t for t, _ in normalize_window(c.window, options)),
            },
            odors[i],
        )
        for i, c in enumerate(candidates)
    ]
    kwargs: list[Row] = []
    occurrence_of = dict(zip(positive_index, occurrences, strict=True))
    for i in positive_index:
        c: Candidate = candidates[i]
        if not c.kwargs:
            continue
        placeable = label_kwargs(c, occurrence_of[i])
        kw_odors = encode_many([k.window for k in c.kwargs], options)
        kwargs.extend(
            Row(
                snippet.id,
                kwarg_trial_index(i, k),
                placeable[k],
                trial_seed(content, kwarg_trial_index(i, k), ENCODER_VERSION),
                {
                    "candidate": i,
                    "kwarg": k,
                    "name": kw.name,
                    "key_name": c.key_name,
                    "window": " ".join(t for t, _ in normalize_window(kw.window, options)),
                },
                kw_odors[k],
            )
            for k, kw in enumerate(c.kwargs)
        )
    return keys, kwargs


def balance(rows: list[Row], rng: np.random.Generator) -> list[Row]:
    """Subsample the majority class down to ``NEGATIVE_RATIO`` × the minority class."""
    pos = [r for r in rows if r.label]
    neg = [r for r in rows if not r.label]
    minority, majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)
    cap = NEGATIVE_RATIO * len(minority)
    if len(majority) > cap:
        keep = rng.choice(len(majority), cap, replace=False)
        majority = [majority[i] for i in sorted(keep)]
    return sorted(minority + majority, key=lambda r: (r.snippet, r.index))


def write_table(
    name: str, rows: list[Row], split_of: dict[int, int], out_dir: Path
) -> dict[str, int]:
    np.savez_compressed(
        out_dir / f"{name}.npz",
        odors=np.stack([r.odor for r in rows]).astype(np.float32),
        label=np.array([r.label for r in rows], dtype=bool),
        split=np.array([split_of[r.snippet] for r in rows], dtype=np.int8),
        snippet=np.array([r.snippet for r in rows], dtype=np.int32),
        index=np.array([r.index for r in rows], dtype=np.int64),
        seed=np.array([r.seed for r in rows], dtype=np.uint64),
    )
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {"snippet": r.snippet, "index": r.index, "label": r.label, **r.meta},
                    ensure_ascii=False,
                )
                + "\n"
            )
    counts = {"rows": len(rows), "positive": int(sum(r.label for r in rows))}
    for s, split_name in enumerate(SPLIT_NAMES):
        counts[split_name] = int(sum(split_of[r.snippet] == s for r in rows))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snippets", type=int, default=DEFAULT_SNIPPETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    snippets, rejected = generate_snippets(args.snippets, args.seed)
    t_generate = time.perf_counter() - t0
    print(
        f"generated {len(snippets)} snippets ({rejected} rejected by compile) in {t_generate:.1f} s",
        flush=True,
    )

    t1 = time.perf_counter()
    key_rows: list[Row] = []
    kwarg_rows: list[Row] = []
    skipped = 0
    for snippet in snippets:
        result = rows_of_snippet(snippet)
        if result is None:
            skipped += 1
            continue
        keys, kwargs = result
        key_rows.extend(keys)
        kwarg_rows.extend(kwargs)
        if snippet.id % 2000 == 0:
            print(
                f"  snippet {snippet.id}: {len(key_rows)} key rows, {len(kwarg_rows)} kwarg rows",
                flush=True,
            )
    t_label = time.perf_counter() - t1
    raw_counts = {
        "keys": {"rows": len(key_rows), "positive": int(sum(r.label for r in key_rows))},
        "kwargs": {"rows": len(kwarg_rows), "positive": int(sum(r.label for r in kwarg_rows))},
    }
    print(f"labelled + encoded in {t_label:.1f} s: {raw_counts} (skipped {skipped})", flush=True)

    rng = np.random.default_rng(args.seed)
    key_rows = balance(key_rows, rng)
    kwarg_rows = balance(kwarg_rows, rng)
    order = rng.permutation(len(snippets))
    n_train = int(SPLIT_FRACTIONS[0] * len(snippets))
    n_val = int(SPLIT_FRACTIONS[1] * len(snippets))
    split_of = {int(sid): 0 for sid in order[:n_train]}
    split_of.update({int(sid): 1 for sid in order[n_train : n_train + n_val]})
    split_of.update({int(sid): 2 for sid in order[n_train + n_val :]})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "snippets.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for s in snippets:
            fh.write(
                json.dumps(
                    {
                        "id": s.id,
                        "split": SPLIT_NAMES[split_of[s.id]],
                        "source": s.source,
                        "config": asdict(s.config),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    counts = {
        "keys": write_table("keys", key_rows, split_of, args.out),
        "kwargs": write_table("kwargs", kwarg_rows, split_of, args.out),
    }
    meta = {
        "generator_version": GENERATOR_VERSION,
        "encoder_version": ENCODER_VERSION,
        "seed": args.seed,
        "snippets": len(snippets),
        "rejected_by_compile": rejected,
        "skipped": skipped,
        "raw_counts": raw_counts,
        "balanced_counts": counts,
        "split_fractions": SPLIT_FRACTIONS,
        "negative_ratio": NEGATIVE_RATIO,
        "timing_s": {
            "generate": t_generate,
            "label_and_encode": t_label,
            "total": time.perf_counter() - t0,
        },
        "sha256_snippets": hashlib.sha256((args.out / "snippets.jsonl").read_bytes()).hexdigest(),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: meta[k]
                for k in ("snippets", "rejected_by_compile", "balanced_counts", "timing_s")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
