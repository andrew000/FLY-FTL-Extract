"""``pyproject.toml`` (``[tool.ftl-extract.extract]``) loading and option resolution.

Priority is CLI > pyproject > built-in defaults, field by field, exactly like the
original's ``main.rs``: a list given on the CLI replaces the config list (so ``-K`` on
the CLI hides ``i18n-keys-append`` from the config), booleans are OR-ed, relative paths
from the config are joined to the directory of the ``pyproject.toml``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from typing import Any, cast

from fly_ftl_extract.ftl.model import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_FTL_FILENAME,
    DEFAULT_I18N_KEYS,
    DEFAULT_IGNORE_ATTRIBUTES,
    DEFAULT_IGNORE_KWARGS,
    DEFAULT_LANGUAGE,
    CommentKeysMode,
    ExtractOptions,
    LineEndings,
)

CONFIG_SECTION = ("tool", "ftl-extract", "extract")


class ConfigError(Exception):
    """Printed as ``[ERROR cli] Configuration error: …`` with exit code 2."""


@dataclass
class ExtractOverrides:
    """What the command line supplied; ``None`` / empty means "not given"."""

    code_path: str | None = None
    locales_path: str | None = None
    language: list[str] | None = None
    i18n_keys: list[str] | None = None
    i18n_keys_append: list[str] | None = None
    i18n_keys_prefix: list[str] | None = None
    exclude_dirs: list[str] | None = None
    exclude_dirs_append: list[str] | None = None
    ignore_attributes: list[str] | None = None
    append_ignore_attributes: list[str] | None = None
    ignore_kwargs: list[str] | None = None
    default_ftl_file: str | None = None
    comment_keys_mode: str | None = None
    line_endings: str | None = None
    dry_run: bool = False
    cache: bool = False
    cache_path: str | None = None
    clear_cache: bool = False
    allow_parse_errors: bool = False


@dataclass
class LoadedConfig:
    """The ``extract`` section of a ``pyproject.toml`` plus the directory it lives in."""

    section: dict[str, Any]
    base_dir: str


def find_pyproject(start: str) -> str | None:
    """Search ``pyproject.toml`` from ``start`` upward, like the original."""
    current = os.path.abspath(start)
    while True:
        candidate = os.path.join(current, "pyproject.toml")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def load_pyproject(explicit: str | None, cwd: str) -> LoadedConfig | None:
    """Load the config named by ``--config`` or discovered from ``cwd``; ``None`` if none."""
    path = explicit if explicit is not None else find_pyproject(cwd)
    if path is None:
        return None
    if not os.path.exists(path):
        raise ConfigError(f"Config file `{path}` does not exist")
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except OSError as err:
        raise ConfigError(f"Failed to read config file `{path}`") from err
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"Failed to parse config file `{path}`") from err
    section: Any = data
    for part in CONFIG_SECTION:
        section = section.get(part) if isinstance(section, dict) else None
        if section is None:
            break
    base_dir = os.path.dirname(path) or "."
    return LoadedConfig(section if isinstance(section, dict) else {}, base_dir)


def _cfg_list(section: dict[str, Any], name: str) -> list[str] | None:
    value = section.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"Invalid `{name}` value in pyproject.toml: expected a list of strings")
    return cast("list[str]", value)


def _cfg_str(section: dict[str, Any], name: str) -> str | None:
    value = section.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Invalid `{name}` value in pyproject.toml: expected a string")
    return value


def _cfg_bool(section: dict[str, Any], name: str) -> bool:
    value = section.get(name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ConfigError(f"Invalid `{name}` value in pyproject.toml: expected a boolean")
    return value


def _cli_or_config_list(
    cli: list[str] | None, config: list[str] | None, default: list[str]
) -> list[str]:
    if cli:
        return cli
    if config is not None:
        return config
    return default


def _resolve_config_path(value: str | None, base_dir: str) -> str | None:
    if value is None:
        return None
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def _enum(
    cli: str | None, config: str | None, field_name: str, allowed: tuple[str, ...]
) -> str | None:
    if cli is not None:
        return cli
    if config is None:
        return None
    if config.lower() not in allowed:
        raise ConfigError(
            f"Invalid `{field_name}` value `{config}`. Expected one of: {', '.join(allowed)}"
        )
    return config.lower()


def resolve_options(overrides: ExtractOverrides, loaded: LoadedConfig | None) -> ExtractOptions:
    """Merge CLI overrides, the pyproject section and defaults into :class:`ExtractOptions`."""
    section = loaded.section if loaded is not None else {}
    base_dir = loaded.base_dir if loaded is not None else "."

    code_path = overrides.code_path or _resolve_config_path(
        _cfg_str(section, "code-path"), base_dir
    )
    if code_path is None:
        raise ConfigError(
            "Missing code path. Pass it as an argument or set tool.ftl-extract.extract.code-path"
        )
    locales_path = overrides.locales_path or _resolve_config_path(
        _cfg_str(section, "locales-path"), base_dir
    )
    if locales_path is None:
        raise ConfigError(
            "Missing locales path. Pass locales path as an argument or set "
            "tool.ftl-extract.extract.locales-path"
        )
    default_ftl_file = (
        overrides.default_ftl_file or _cfg_str(section, "default-ftl-file") or DEFAULT_FTL_FILENAME
    )
    comment_mode = (
        _enum(
            overrides.comment_keys_mode,
            _cfg_str(section, "comment-keys-mode"),
            "comment-keys-mode",
            ("comment", "warn"),
        )
        or "comment"
    )
    line_endings = (
        _enum(
            overrides.line_endings,
            _cfg_str(section, "line-endings"),
            "line-endings",
            ("default", "lf", "cr", "crlf"),
        )
        or "default"
    )

    i18n_keys = set(
        _cli_or_config_list(
            overrides.i18n_keys, _cfg_list(section, "i18n-keys"), sorted(DEFAULT_I18N_KEYS)
        )
    )
    i18n_keys.update(
        _cli_or_config_list(overrides.i18n_keys_append, _cfg_list(section, "i18n-keys-append"), [])
    )
    exclude = set(
        _cli_or_config_list(
            overrides.exclude_dirs, _cfg_list(section, "exclude-dirs"), sorted(DEFAULT_EXCLUDE_DIRS)
        )
    )
    exclude.update(
        _cli_or_config_list(
            overrides.exclude_dirs_append, _cfg_list(section, "exclude-dirs-append"), []
        )
    )
    ignore_attrs = set(
        _cli_or_config_list(
            overrides.ignore_attributes,
            _cfg_list(section, "ignore-attributes"),
            sorted(DEFAULT_IGNORE_ATTRIBUTES),
        )
    )
    ignore_attrs.update(
        _cli_or_config_list(
            overrides.append_ignore_attributes, _cfg_list(section, "ignore-attributes-append"), []
        )
    )
    cache_path = overrides.cache_path or _resolve_config_path(
        _cfg_str(section, "cache-path"), base_dir
    )
    clear_cache = overrides.clear_cache or _cfg_bool(section, "clear-cache")

    return ExtractOptions(
        code_path=code_path,
        locales_path=locales_path,
        languages=tuple(
            _cli_or_config_list(
                overrides.language, _cfg_list(section, "languages"), [DEFAULT_LANGUAGE]
            )
        ),
        i18n_keys=frozenset(i18n_keys),
        i18n_keys_prefix=frozenset(
            _cli_or_config_list(
                overrides.i18n_keys_prefix, _cfg_list(section, "i18n-keys-prefix"), []
            )
        ),
        exclude_dirs=frozenset(exclude),
        ignore_attributes=frozenset(ignore_attrs),
        ignore_kwargs=frozenset(
            _cli_or_config_list(
                overrides.ignore_kwargs,
                _cfg_list(section, "ignore-kwargs"),
                sorted(DEFAULT_IGNORE_KWARGS),
            )
        ),
        default_ftl_file=default_ftl_file,
        comment_keys_mode=cast("CommentKeysMode", comment_mode),
        line_endings=cast("LineEndings", line_endings),
        dry_run=overrides.dry_run or _cfg_bool(section, "dry-run"),
        allow_parse_errors=overrides.allow_parse_errors or _cfg_bool(section, "allow-parse-errors"),
        cache=overrides.cache
        or _cfg_bool(section, "cache")
        or cache_path is not None
        or clear_cache,
        cache_path=cache_path,
        clear_cache=clear_cache,
    )


OVERRIDE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ExtractOverrides))
