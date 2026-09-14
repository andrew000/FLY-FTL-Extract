"""Parse ``ftl extract`` argv (as stored in ``args.json``) into :class:`ExtractOverrides`.

Test/script helper: the real CLI (Phase 6) is click-based; here a minimal argparse mirror
of the original's options is enough to replay the golden runs.
"""

from __future__ import annotations

import argparse

from fly_ftl_extract.cli.config import ExtractOverrides


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ftl extract", add_help=False)
    p.add_argument("--config", dest="config", default=None)
    p.add_argument("code_path", nargs="?", default=None)
    p.add_argument("locales_path", nargs="?", default=None)
    p.add_argument("-l", "--language", action="append", dest="language")
    p.add_argument("-k", "--i18n-keys", action="append", dest="i18n_keys")
    p.add_argument("-K", "--i18n-keys-append", action="append", dest="i18n_keys_append")
    p.add_argument("-p", "--i18n-keys-prefix", action="append", dest="i18n_keys_prefix")
    p.add_argument("-e", "--exclude-dirs", action="append", dest="exclude_dirs")
    p.add_argument("-E", "--exclude-dirs-append", action="append", dest="exclude_dirs_append")
    p.add_argument("-i", "--ignore-attributes", action="append", dest="ignore_attributes")
    p.add_argument(
        "-I", "--append-ignore-attributes", action="append", dest="append_ignore_attributes"
    )
    p.add_argument("--ignore-kwargs", action="append", dest="ignore_kwargs")
    p.add_argument("--default-ftl-file", dest="default_ftl_file", default=None)
    p.add_argument("--comment-keys-mode", dest="comment_keys_mode", choices=["comment", "warn"])
    p.add_argument("--line-endings", dest="line_endings", choices=["default", "lf", "cr", "crlf"])
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--cache", dest="cache", action="store_true")
    p.add_argument("--cache-path", dest="cache_path", default=None)
    p.add_argument("--clear-cache", dest="clear_cache", action="store_true")
    p.add_argument("--allow-parse-errors", dest="allow_parse_errors", action="store_true")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    return p


def parse_extract_argv(argv: list[str]) -> tuple[ExtractOverrides, str | None]:
    """Return ``(overrides, --config path)`` for the given ``ftl extract`` arguments."""
    ns = _parser().parse_args(argv)
    overrides = ExtractOverrides(
        code_path=ns.code_path,
        locales_path=ns.locales_path,
        language=ns.language,
        i18n_keys=ns.i18n_keys,
        i18n_keys_append=ns.i18n_keys_append,
        i18n_keys_prefix=ns.i18n_keys_prefix,
        exclude_dirs=ns.exclude_dirs,
        exclude_dirs_append=ns.exclude_dirs_append,
        ignore_attributes=ns.ignore_attributes,
        append_ignore_attributes=ns.append_ignore_attributes,
        ignore_kwargs=ns.ignore_kwargs,
        default_ftl_file=ns.default_ftl_file,
        comment_keys_mode=ns.comment_keys_mode,
        line_endings=ns.line_endings,
        dry_run=ns.dry_run,
        cache=ns.cache,
        cache_path=ns.cache_path,
        clear_cache=ns.clear_cache,
        allow_parse_errors=ns.allow_parse_errors,
    )
    return overrides, ns.config
