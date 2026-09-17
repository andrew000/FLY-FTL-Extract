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
from collections.abc import Callable
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
from fly_ftl_extract.tokenizer.candidates import (
    Candidate,
    CandidateError,
    Window,
    iter_candidates,
)

REPO = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO / "fly_ftl_extract" / "data" / "dataset"
GENERATOR_VERSION = "grammar-5"
DEFAULT_SNIPPETS = 20_000
DEFAULT_SEED = 20240914
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
NEGATIVE_RATIO = 3  # majority : minority
SPLIT_NAMES = ("train", "val", "test")
PREFIXED_BARE_CALL = 0.6  # share of bare calls written as self.L(...) / cls.LazyProxy(...)
PREFIX_GET_CALL = 0.08
MUTATION_SHARE = 0.45
"""Share of statements drawn from the mutation families (grammar-4 and grammar-5) below.

grammar-4 (auditor's decision after attempt 9): one token class of a positive production
changed, the label always from ``reference/``.  The two fixture files the attempt-9 fly got
wrong (``i18n.nested.set_locale()`` is a key, ``self.other.get("x")`` is not) were
constructs the grammar-3 corpus almost never produced.

grammar-5 (auditor's decision after Phase 7 on a real aiogram bot, ``docs/REAL_PROJECT.md``):
the i18n call in *positions* the grammar never produced — a dict value, a keyword argument
of another constructor, a decorator argument, a sequence element, a return/yield value — and
their look-alike negatives.  0.30 → 0.45 so that the grammar-4 families keep a useful
absolute count next to the ten new ones."""
GRAMMAR5_SHARE = 0.6
"""Share of mutation draws that are grammar-5 families: 10 families × ≈ 6 % of all family
occurrences each (auditor: ~5–7 % each), the 14 grammar-4 families ≈ 2.9 % each."""
FILE_START_MUTATION = 0.1
"""Share of modules whose very first line is a mutation statement (no import header)."""
MUTATION_FAMILIES = (
    "ignore-L1",  # <I18N>.<IGNORE>()            — ignored (first attribute)
    "ignore-L2-last",  # <I18N>.<NAME>.<IGNORE>() — a key: only the first attribute counts
    "ignore-L2-first",  # <I18N>.<IGNORE>.<NAME>() — ignored
    "ignore-L3",  # <I18N>.<NAME>.<NAME>.<IGNORE>() — a key
    "prefix-i18n-get",  # <PREFIX>.<I18N>.get("x")        — key with -p, not without
    "prefix-name-get",  # <PREFIX>.<NAME>.get("x")        — never a key
    "name-prefix-i18n-get",  # <NAME>.<PREFIX>.<I18N>.get("x") — never a key
    "prefix-prefix-i18n-get",  # <PREFIX>.<PREFIX>.<I18N>.get("x") — never a key
    "prefix-i18n-attr",  # <PREFIX>.<I18N>.<NAME>()        — key with -p
    "prefix-name-attr",  # <PREFIX>.<NAME>.<NAME>()        — never a key
    "ignore-kw-first",  # <I18N>.get("x", <IGNORE_KW>=…, a=…)
    "ignore-kw-middle",  # <I18N>.get("x", a=…, <IGNORE_KW>=…, b=…)
    "ignore-kw-last",  # <I18N>.get("x", a=…, <IGNORE_KW>=…)
    "ignore-kw-only",  # <I18N>.get("x", <IGNORE_KW>=…)
)
MUTATION_CONTEXTS = (
    "ctx-plain",  # the usual statement wrappers
    "ctx-return-list",  # return [expr, "s"]
    "ctx-list",  # x = ["s", expr]
    "ctx-dict",  # x = {"k": expr, "s": v}
    "ctx-arg",  # other(ident, expr) / other(expr, "s")
    "ctx-file-start",  # the module's first line
)
GRAMMAR5_FAMILIES = (
    "g5-dict-value-str",  # {"k": <I18N>("x"), …}                — key (dict value)
    "g5-dict-value-enum",  # {Kind.X: <I18N>("x", _path=…), …}   — key (dict value)
    "g5-kwarg-value",  # Item(name=<I18N>("a"), description=<I18N>("b", …)) — keys
    "g5-decorator-arg",  # @router.message(<I18N>("k", …)) def … — key
    "g5-seq-element",  # [<I18N>("a"), …] / (…, <I18N>("b")) / {…} — keys
    "g5-return-yield",  # return <I18N>("x") / yield <I18N>("x")  — key
    "g5-neg-seq-str",  # (("s", data.get("s")), …) — strings, no i18n call
    "g5-neg-dict-key",  # {"k": Kind.X, "s": 1}   — strings as dict keys
    "g5-neg-decorator-plain",  # @router.message(Command("s")) — no i18n name
    "g5-neg-tuple-in-arg",  # other(("s", 1)) — a tuple of strings as an argument
)
"""grammar-5 families; each statement carries its own context and one of
:data:`GRAMMAR5_LAYOUTS` (one line / broken over lines).  Labels — from the teacher, as
always: the grammar only puts the constructs in front of it."""
GRAMMAR5_LAYOUTS = ("g5-one-line", "g5-multi-line")
ENUM_POOL = ("Kind", "Gender", "State", "Mode", "Category", "Slot")
CTOR_POOL = ("Item", "Resource", "Button", "Entry", "Option", "Field")
I18N_KWARG_NAMES = ("name", "description", "title", "label", "hint", "text")
ROUTER_POOL = ("router", "dp", "app", "handlers")
DECORATOR_METHODS = ("message", "callback_query", "inline_query")
PLAIN_FILTERS = (
    'Command("{s}")',
    'Command("{s}", "{t}")',
    'F.text == "{s}"',
    'F.data.startswith("{s}")',
    'StateFilter("{s}")',
    'Text("{s}")',
)
DATA_POOL = ("data", "state", "payload", "store", "workflow_data", "settings")

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
    families: tuple[str, ...] = ()
    """grammar-4 mutation families and contexts the module contains (one tag per use)."""


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
        self.families: list[str] = []

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

    # -- grammar-4 mutation families ---------------------------------------------------
    def ignore_attr(self) -> str:
        """A configured ignore attribute — or, 1 in 4, an ignore-looking name that this
        snippet's options do *not* ignore (``-i core`` makes ``set_locale`` a key)."""
        pool = self.config.ignore_attributes
        if self.rng.random() < 0.25 or not pool:  # noqa: PLR2004
            others = [
                a for a in (*DEFAULT_IGNORE_ATTRIBUTES, *CUSTOM_IGNORE_ATTRS) if a not in pool
            ]
            return self.rng.choice(others) if others else self.ident()
        return self.rng.choice(pool)

    def ignore_kw(self) -> str:
        """A configured ignore kwarg — or, without any, a name from the pool (then it is an
        ordinary, placeable kwarg: the label comes from the teacher either way)."""
        if self.config.ignore_kwargs and self.rng.random() < 0.8:  # noqa: PLR2004
            return self.rng.choice(self.config.ignore_kwargs)
        return self.rng.choice(IGNORE_KWARG_POOL)

    def plain_kwargs(self, n: int) -> list[str]:
        names = self.rng.sample(KWARG_NAMES, n)
        return [f"{name}={self.value(1)}" for name in names]

    def mutation(self) -> str:
        """One expression from :data:`MUTATION_FAMILIES` (the family tag is recorded)."""
        family = self.rng.choice(MUTATION_FAMILIES)
        self.families.append(family)
        i18n = self.rng.choice(self.config.i18n_keys)
        name, name2 = self.ident(), self.ident()
        prefix = self.rng.choice(PREFIX_POOL)
        prefix2 = "cls" if prefix == "self" else "self"
        args = ", ".join(self.kwargs(1, allow_stars=False)) if self.rng.random() < 0.5 else ""  # noqa: PLR2004
        # the i18n root itself may sit behind a prefix (self.i18n) like anywhere else
        root = f"{self.in_method}.{i18n}" if self.in_method and self.rng.random() < 0.4 else i18n  # noqa: PLR2004
        ign = self.ignore_attr()
        s = self.string()
        if family == "ignore-L1":
            return f"{root}.{ign}({args})"
        if family == "ignore-L2-last":
            return f"{root}.{name}.{ign}({args})"
        if family == "ignore-L2-first":
            return f"{root}.{ign}.{name}({args})"
        if family == "ignore-L3":
            return f"{root}.{name}.{name2}.{ign}({args})"
        if family == "prefix-i18n-get":
            return f"{prefix}.{i18n}.get({self.call_args(s, 1)})"
        if family == "prefix-name-get":
            return f"{prefix}.{name}.get({self.call_args(s, 1)})"
        if family == "name-prefix-i18n-get":
            return f"{name}.{prefix}.{i18n}.get({self.call_args(s, 1)})"
        if family == "prefix-prefix-i18n-get":
            return f"{prefix}.{prefix2}.{i18n}.get({self.call_args(s, 1)})"
        if family == "prefix-i18n-attr":
            return f"{prefix}.{i18n}.{name}({args})"
        if family == "prefix-name-attr":
            return f"{prefix}.{name}.{name2}({args})"
        kw = f"{self.ignore_kw()}={self.value(1)}"
        if family == "ignore-kw-first":
            items = [kw, *self.plain_kwargs(self.rng.randint(1, 2))]
        elif family == "ignore-kw-middle":
            a, b = self.plain_kwargs(2)
            items = [a, kw, b]
        elif family == "ignore-kw-last":
            items = [*self.plain_kwargs(self.rng.randint(1, 2)), kw]
        else:  # ignore-kw-only
            items = [kw]
        return f"{root}.get({', '.join([s, *items])})"

    # -- grammar-5 families: the i18n call in positions the earlier grammars never used --
    def i18n_call(self) -> str:
        """A call the teacher usually calls a key: ``L("x")``, ``L("x", _path=…)``,
        ``i18n.get("x", a=1)``, ``i18n.a.b()`` — with the snippet's own ``-k``/``-K`` names
        (``root()`` may put it behind a prefix inside a method)."""
        r = self.rng.random()
        if r < 0.55:  # noqa: PLR2004
            root = self.rng.choice(self.config.i18n_keys)
            if self.in_method and self.rng.random() < 0.3:  # noqa: PLR2004
                root = f"{self.in_method}.{root}"
            return f"{root}({self.call_args(self.string(), 1)})"
        if r < 0.8:  # noqa: PLR2004
            return self.get_call(1)
        return self.attr_call()

    def enum_key(self) -> str:
        return f"{self.rng.choice(ENUM_POOL)}.{self.word().upper()}"

    def _lay_out(self, head: str, items: list[str], tail: str, *, multi: bool) -> list[str]:
        """``head item, item tail`` on one line or one item per line."""
        self.families.append(GRAMMAR5_LAYOUTS[1] if multi else GRAMMAR5_LAYOUTS[0])
        if multi:
            return [head, *(f"    {item}," for item in items), tail]
        return [f"{head}{', '.join(items)}{tail}"]

    def _annotated_target(self, key_type: str) -> str:
        r = self.rng.random()
        if r < 0.25:  # noqa: PLR2004
            return f"{self.ident().upper()}: Final[dict[{key_type}, Any]]"
        if r < 0.4:  # noqa: PLR2004
            return f"{self.ident()}: dict[{key_type}, str]"
        return self.ident().upper() if r < 0.7 else self.ident()  # noqa: PLR2004

    def _ctor_call(self, *, multi: bool, indent: str = "") -> list[str]:
        """``Item(name=<I18N>("a"), description=<I18N>("b"), price=3)`` — two or three
        i18n keyword values, the second/third after a line break in the multi-line form
        (the shape the Phase-7 fly missed: ``docs/REAL_PROJECT.md`` № 8–10)."""
        names = self.rng.sample(I18N_KWARG_NAMES, self.rng.randint(2, 3))
        items = [f"{n}={self.i18n_call()}" for n in names]
        extras = [
            f"{self.rng.choice(['price', 'weight', 'order', 'limit'])}={self.rng.randint(1, 99)}",
            f"kind={self.enum_key()}",
        ]
        for extra in extras[: self.rng.randint(0, 2)]:
            items.insert(self.rng.randint(0, len(items)), extra)
        if self.rng.random() < 0.3:  # noqa: PLR2004
            items.insert(0, self.ident())
        ctor = self.rng.choice(CTOR_POOL)
        if multi:
            return [f"{ctor}(", *(f"{indent}    {item}," for item in items), f"{indent})"]
        return [f"{ctor}({', '.join(items)})"]

    def g5_dict_value(self, *, enum: bool) -> list[str]:
        n = self.rng.randint(2, 5)
        items = []
        for _ in range(n):
            key = self.enum_key() if enum else self.string()
            value = self.i18n_call() if self.rng.random() < 0.8 else self.value(1)  # noqa: PLR2004
            items.append(f"{key}: {value}")
        target = self._annotated_target(self.rng.choice(ENUM_POOL) if enum else "str")
        return self._lay_out(f"{target} = {{", items, "}", multi=self.rng.random() < 0.6)  # noqa: PLR2004

    def g5_kwarg_value(self) -> list[str]:
        multi = self.rng.random() < 0.6  # noqa: PLR2004
        r = self.rng.random()
        if r < 0.4:  # noqa: PLR2004
            # nested as a dict value (a private aiogram bot: Kind.X: Resource(name=L(…), description=L(…)))
            lines = [f"{self._annotated_target(self.rng.choice(ENUM_POOL))} = {{"]
            for _ in range(self.rng.randint(1, 3)):
                ctor = self._ctor_call(multi=multi, indent="    ")
                ctor[0] = f"    {self.enum_key()}: {ctor[0]}"
                ctor[-1] += ","
                lines.extend(ctor)
            lines.append("}")
            self.families.append(GRAMMAR5_LAYOUTS[1] if multi else GRAMMAR5_LAYOUTS[0])
            return lines
        ctor = self._ctor_call(multi=multi)
        self.families.append(GRAMMAR5_LAYOUTS[1] if multi else GRAMMAR5_LAYOUTS[0])
        if r < 0.7 or not self.in_function:  # noqa: PLR2004
            ctor[0] = f"{self.ident()} = {ctor[0]}"
        elif r < 0.85:  # noqa: PLR2004
            ctor[0] = f"return {ctor[0]}"
        else:
            ctor[0] = f"{self.rng.choice(PLAIN_CALLEES)}({ctor[0]}"
            ctor[-1] += ")"
        return ctor

    def _decorated_def(self, filters: list[str], *, multi: bool) -> list[str]:
        router = self.rng.choice(ROUTER_POOL)
        method = self.rng.choice(DECORATOR_METHODS)
        lines = self._lay_out(f"@{router}.{method}(", filters, ")", multi=multi)
        is_async = self.rng.random() < 0.7  # noqa: PLR2004
        params = ", ".join(
            dict.fromkeys(
                [
                    "message" if method == "message" else "query",
                    *self.rng.sample(["i18n", "state", "user"], self.rng.randint(0, 2)),
                ]
            )
        )
        lines.append(f"{'async ' if is_async else ''}def {self.ident()}({params}):")
        target = "message" if method == "message" else "query.message"
        r = self.rng.random()
        if r < 0.3:  # noqa: PLR2004
            body = "pass"
        elif r < 0.6 and is_async:  # noqa: PLR2004
            arg = self.i18n_call() if self.rng.random() < 0.6 else self.string()  # noqa: PLR2004
            body = f"await {target}.answer({arg})"
        elif r < 0.8:  # noqa: PLR2004
            body = f"return {self.value(1)}"
        else:
            body = f"{self.ident()} = {self.i18n_call()}"
        lines.append(f"    {body}")
        return lines

    def g5_decorator_arg(self) -> list[str]:
        filters = [self.i18n_call()]
        if self.rng.random() < 0.4:  # noqa: PLR2004
            s, t = self.key(), self.key()
            filters.append(self.rng.choice(PLAIN_FILTERS).format(s=s, t=t))
        if self.rng.random() < 0.25:  # noqa: PLR2004
            filters.insert(0, f'F.chat.type == "{self.rng.choice(["private", "group"])}"')
        return self._decorated_def(filters, multi=self.rng.random() < 0.6)  # noqa: PLR2004

    def g5_neg_decorator_plain(self) -> list[str]:
        s, t = self.key(), self.key()
        filters = [self.rng.choice(PLAIN_FILTERS).format(s=s, t=t)]
        if self.rng.random() < 0.3:  # noqa: PLR2004
            filters.append(f"{self.rng.choice(['flags', 'magic'])}={self.value(1)}")
        return self._decorated_def(filters, multi=self.rng.random() < 0.5)  # noqa: PLR2004

    def g5_seq_element(self) -> list[str]:
        n = self.rng.randint(2, 4)
        items = [self.i18n_call()]
        items.extend(
            self.i18n_call() if self.rng.random() < 0.6 else self.value(1)  # noqa: PLR2004
            for _ in range(n - 1)
        )
        self.rng.shuffle(items)
        multi = self.rng.random() < 0.6  # noqa: PLR2004
        open_, close = self.rng.choice([("[", "]"), ("(", ")"), ("{", "}"), ("[", "]")])
        target = self.ident() if self.rng.random() < 0.8 else self.ident().upper()  # noqa: PLR2004
        return self._lay_out(f"{target} = {open_}", items, close, multi=multi)

    def g5_return_yield(self) -> list[str]:
        # an async def with both `return value` and `yield` does not compile
        keyword = "return" if self.in_async else self.rng.choice(["return", "return", "yield"])
        r = self.rng.random()
        if r < 0.5:  # noqa: PLR2004
            body = self._lay_out(f"{keyword} ", [self.i18n_call()], "", multi=False)
        elif r < 0.75:  # noqa: PLR2004
            body = self._lay_out(
                f"{keyword} (", [self.i18n_call(), self.string()], ")", multi=False
            )
        else:
            body = self._lay_out(
                f"{keyword} [",
                [self.i18n_call() for _ in range(self.rng.randint(1, 3))],
                "]",
                multi=True,
            )
        if self.in_function:
            return body
        return [f"def {self.ident()}():", *(f"    {ln}" for ln in body)]

    def g5_neg_seq_str(self) -> list[str]:
        data = self.rng.choice(DATA_POOL)
        r = self.rng.random()
        if r < 0.5:  # noqa: PLR2004
            # (("s", data.get("s")), ("t", data.get("t"))) — a private aiogram bot № 1
            items = []
            for _ in range(self.rng.randint(1, 3)):
                s = self.string()
                items.append(f"({s}, {data}.get({s}))")
        elif r < 0.75:  # noqa: PLR2004
            items = [self.string() for _ in range(self.rng.randint(2, 4))]
            if self.rng.random() < 0.5:  # noqa: PLR2004
                items.append(self.ident())
        else:
            items = [f"({self.string()}, {self.rng.randint(0, 9)})" for _ in range(2)]
        open_, close = self.rng.choice([("(", ")"), ("[", "]")])
        return self._lay_out(
            f"{self.ident()} = {open_}",
            items,
            close,
            multi=self.rng.random() < 0.5,  # noqa: PLR2004
        )

    def g5_neg_dict_key(self) -> list[str]:
        items = []
        for _ in range(self.rng.randint(2, 4)):
            r = self.rng.random()
            if r < 0.4:  # noqa: PLR2004
                value = self.enum_key()
            elif r < 0.7:  # noqa: PLR2004
                value = str(self.rng.randint(0, 99))
            elif r < 0.85:  # noqa: PLR2004
                value = self.ident()
            else:
                value = f"{self.rng.choice(CTOR_POOL)}({self.ident()}={self.rng.randint(1, 9)})"
            items.append(f"{self.string()}: {value}")
        target = self._annotated_target("str")
        return self._lay_out(f"{target} = {{", items, "}", multi=self.rng.random() < 0.6)  # noqa: PLR2004

    def g5_neg_tuple_in_arg(self) -> list[str]:
        callee = self.rng.choice((*PLAIN_CALLEES, "add", "register", "dict", "sorted"))
        r = self.rng.random()
        if r < 0.4:  # noqa: PLR2004
            items = [f"({self.string()}, {self.rng.randint(0, 9)})"]
        elif r < 0.7:  # noqa: PLR2004
            items = [f"({self.string()}, {self.ident()})", f"key={self.string()}"]
        else:
            items = [f"({self.string()}, {self.rng.randint(0, 9)})" for _ in range(2)]
        return self._lay_out(f"{callee}(", items, ")", multi=self.rng.random() < 0.5)  # noqa: PLR2004

    def grammar5_statement(self) -> list[str]:
        family = self.rng.choice(GRAMMAR5_FAMILIES)
        self.families.append(family)
        if family == "g5-dict-value-str":
            return self.g5_dict_value(enum=False)
        if family == "g5-dict-value-enum":
            return self.g5_dict_value(enum=True)
        if family == "g5-kwarg-value":
            return self.g5_kwarg_value()
        if family == "g5-decorator-arg":
            return self.g5_decorator_arg()
        if family == "g5-seq-element":
            return self.g5_seq_element()
        if family == "g5-return-yield":
            return self.g5_return_yield()
        if family == "g5-neg-seq-str":
            return self.g5_neg_seq_str()
        if family == "g5-neg-dict-key":
            return self.g5_neg_dict_key()
        if family == "g5-neg-decorator-plain":
            return self.g5_neg_decorator_plain()
        return self.g5_neg_tuple_in_arg()

    def mutation_statement(self, context: str | None = None) -> list[str]:
        """A mutation expression placed into one of :data:`MUTATION_CONTEXTS`, or (with
        :data:`GRAMMAR5_SHARE`) a grammar-5 statement that carries its own context."""
        if self.rng.random() < GRAMMAR5_SHARE:
            return self.grammar5_statement()
        expr = self.mutation()
        ctx = context or self.rng.choice(MUTATION_CONTEXTS[:-1])
        if ctx == "ctx-return-list" and not self.in_function:
            ctx = "ctx-list"
        self.families.append(ctx)
        if ctx == "ctx-plain":
            return self.wrap(expr)
        if ctx == "ctx-return-list":
            return [f"return [{expr}, {self.string()}]"]
        if ctx == "ctx-list":
            return [f"{self.ident()} = [{self.string()}, {expr}]"]
        if ctx == "ctx-dict":
            return [
                f"{self.ident()} = {{{self.string()}: {expr}, {self.string()}: {self.value(1)}}}"
            ]
        if ctx == "ctx-arg":
            callee = self.rng.choice(PLAIN_CALLEES)
            if self.rng.random() < 0.5:  # noqa: PLR2004
                return [f"{callee}({self.ident()}, {expr})"]
            return [f"{callee}({expr}, {self.string()})"]
        return [expr]  # ctx-file-start: the bare expression on the first line

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
        if self.rng.random() < MUTATION_SHARE:
            return self.mutation_statement()
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
        if self.rng.random() < FILE_START_MUTATION:
            # grammar-4: the construct on the very first line of the file
            lines.extend(self.mutation_statement("ctx-file-start"))
        elif r >= 0.08 and self.rng.random() < 0.55:  # noqa: PLR2004
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
        grammar = Grammar(rng, config)
        source = grammar.module()
        try:
            compile(source, "<snippet>", "exec", dont_inherit=True)
        except SyntaxError:
            rejected += 1
            continue
        out.append(Snippet(len(out), source, config, tuple(grammar.families)))
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


Encoder = Callable[[list[Window], ExtractOptions], np.ndarray]


def rows_of_snippet(
    snippet: Snippet, encode: Encoder = encode_many
) -> tuple[list[Row], list[Row]] | None:
    """Labelled rows of one snippet; ``encode`` is injectable so that the encoder proxy
    can put other encoders through the very same labelling path."""
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
    odors = encode([c.window for c in candidates], options)
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
        kw_odors = encode([k.window for k in c.kwargs], options)
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


def family_shares(snippets: list[Snippet]) -> dict[str, dict[str, float | int]]:
    """Per mutation family / context / layout: occurrences, ``share`` within its own group
    (grammar-4 families, contexts, grammar-5 families, layouts), ``share_all`` among *all*
    family occurrences (grammar-4 + grammar-5 — the auditor's «~5–7 % each»), and the
    share of snippets that contain it (docs/METRICS.md §6b/§6c)."""
    groups = (MUTATION_FAMILIES, MUTATION_CONTEXTS, GRAMMAR5_FAMILIES, GRAMMAR5_LAYOUTS)
    counts: dict[str, int] = dict.fromkeys((tag for group in groups for tag in group), 0)
    in_snippets: dict[str, int] = dict.fromkeys(counts, 0)
    for s in snippets:
        for tag in s.families:
            counts[tag] = counts.get(tag, 0) + 1
        for tag in set(s.families):
            in_snippets[tag] = in_snippets.get(tag, 0) + 1
    totals = {tag: sum(counts[t] for t in group) or 1 for group in groups for tag in group}
    total_all = sum(counts[f] for f in (*MUTATION_FAMILIES, *GRAMMAR5_FAMILIES)) or 1
    out: dict[str, dict[str, float | int]] = {}
    for tag, n in counts.items():
        out[tag] = {
            "occurrences": n,
            "share": n / totals.get(tag, total_all),
            "share_all": n / total_all,
            "snippets": in_snippets[tag],
            "snippet_share": in_snippets[tag] / max(len(snippets), 1),
        }
    return out


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


def balance_and_split(
    key_rows: list[Row], kwarg_rows: list[Row], n_snippets: int, seed: int
) -> tuple[list[Row], list[Row], dict[int, int]]:
    """Balanced tables and the snippet → split (0 train, 1 val, 2 test) map.

    One generator seeded with ``seed`` does the subsampling and then the permutation, so
    the split depends only on the corpus (candidates and labels), never on the encoder:
    the encoder proxy gets the very same rows and split as the dataset.
    """
    rng = np.random.default_rng(seed)
    key_rows = balance(key_rows, rng)
    kwarg_rows = balance(kwarg_rows, rng)
    order = rng.permutation(n_snippets)
    n_train = int(SPLIT_FRACTIONS[0] * n_snippets)
    n_val = int(SPLIT_FRACTIONS[1] * n_snippets)
    split_of = {int(sid): 0 for sid in order[:n_train]}
    split_of.update({int(sid): 1 for sid in order[n_train : n_train + n_val]})
    split_of.update({int(sid): 2 for sid in order[n_train + n_val :]})
    return key_rows, kwarg_rows, split_of


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

    key_rows, kwarg_rows, split_of = balance_and_split(
        key_rows, kwarg_rows, len(snippets), args.seed
    )

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
                        "families": list(s.families),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    counts = {
        "keys": write_table("keys", key_rows, split_of, args.out),
        "kwargs": write_table("kwargs", kwarg_rows, split_of, args.out),
    }
    shares = family_shares(snippets)
    meta = {
        "generator_version": GENERATOR_VERSION,
        "mutation_share": MUTATION_SHARE,
        "grammar5_share": GRAMMAR5_SHARE,
        "grammar4_families": {tag: shares[tag] for tag in (*MUTATION_FAMILIES, *MUTATION_CONTEXTS)},
        "grammar5_families": {tag: shares[tag] for tag in (*GRAMMAR5_FAMILIES, *GRAMMAR5_LAYOUTS)},
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
