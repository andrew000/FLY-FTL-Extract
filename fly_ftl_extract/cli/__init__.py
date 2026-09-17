"""Command-line entry points (``ftl`` and ``fly-ftl``): a drop-in for ``ftl-extract`` 0.12.1.

Same commands, same options, same output (``docs/FORMAT.md``); the classifier behind
``extract`` is the fly (``cli/extract.py``).  Extra options all start with ``--fly-``.
"""

from __future__ import annotations

import sys

import click

from fly_ftl_extract import __version__
from fly_ftl_extract.cli.config import ExtractOverrides
from fly_ftl_extract.cli.extract import DEFAULT_WORKERS, FlyOptions, run_command
from fly_ftl_extract.cli.sample import sample_text

NOT_TRAINED_MSG = "the fly has not been trained for this yet — use the original ftl-extract"
NOT_TRAINED_EXIT_CODE = 2


def _utf8_streams() -> None:
    """Write UTF-8 like the Rust original does, even into a cp1251 pipe on Windows.

    The statistics end with ``✅ Done``; without this a redirected stderr on Windows would
    raise ``UnicodeEncodeError`` instead of printing it.  Line ends stay LF (no CRLF
    translation): the original's ``config sample`` and logs are LF-only bytes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace", newline="\n")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to pyproject.toml with [tool.ftl-extract.<command>] config",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(
    __version__, "-V", "--version", prog_name="ftl", message="%(prog)s %(version)s"
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Extract Fluent keys from Python code using a simulated fly brain."""
    _utf8_streams()
    ctx.obj = {"config": config_path, "verbose": verbose}


@main.command()
@click.argument("code_path", required=False, default=None)
@click.argument("locales_path", required=False, default=None)
@click.option("-l", "--language", "language", help="Language codes to extract", multiple=True)
@click.option(
    "-k",
    "--i18n-keys",
    "i18n_keys",
    help="Names of function that is used to get translation",
    multiple=True,
)
@click.option(
    "-K",
    "--i18n-keys-append",
    "i18n_keys_append",
    help="Append names of function that is used to get translation",
    multiple=True,
)
@click.option(
    "-p",
    "--i18n-keys-prefix",
    "i18n_keys_prefix",
    help="Prefix names of function that is used to get translation. `self.i18n.*()`",
    multiple=True,
)
@click.option("-e", "--exclude-dirs", "exclude_dirs", help="Exclude directories", multiple=True)
@click.option(
    "-E",
    "--exclude-dirs-append",
    "exclude_dirs_append",
    help="Append directories to exclude",
    multiple=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "-i",
    "--ignore-attributes",
    "ignore_attributes",
    help="Ignore attributes, e.g. `i18n.set_locale()`",
    multiple=True,
)
@click.option(
    "-I",
    "--append-ignore-attributes",
    "append_ignore_attributes",
    help="Append attributes to ignore",
    multiple=True,
)
@click.option(
    "--ignore-kwargs",
    "ignore_kwargs",
    help="Ignore kwargs, like `when` from `aiogram_dialog.I18nFormat(..., when=...)`",
    multiple=True,
)
@click.option("--default-ftl-file", default=None, help="Default FTL filename")
@click.option(
    "--comment-keys-mode",
    type=click.Choice(["comment", "warn"]),
    default=None,
    help="Comment keys mode",
)
@click.option(
    "--line-endings",
    type=click.Choice(["default", "lf", "cr", "crlf"]),
    default=None,
    help="Line endings in output FTL files",
)
@click.option("--dry-run", is_flag=True, help="Dry run, do not write to files")
@click.option("--cache", is_flag=True, help="Cache Python extraction results between runs")
@click.option("--cache-path", default=None, help="Directory or file path for the extraction cache")
@click.option("--clear-cache", is_flag=True, help="Clear the extraction cache before running")
@click.option(
    "--allow-parse-errors",
    is_flag=True,
    help="Skip Python files that cannot be read or parsed instead of aborting",
)
@click.option(
    "--fly-audit",
    is_flag=True,
    help="Run the ast reference next to the fly; print differences to stderr, exit 1 if any",
)
@click.option(
    "--fly-seed",
    type=int,
    default=0,
    show_default=True,
    help="Salt for every trial seed (another nose; 0 = production seeds)",
)
@click.option("--fly-no-tui", is_flag=True, help="Plain log output even on a terminal")
@click.option(
    "--fly-trials",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Sniffs per candidate before the resniff rule (margins are summed)",
)
@click.option(
    "--fly-batch",
    type=click.IntRange(min=1),
    default=None,
    help="Trials per brain call (default 256 in-process, 64 per worker)",
)
@click.option(
    "--fly-workers",
    type=click.IntRange(min=1),
    default=DEFAULT_WORKERS,
    show_default=True,
    help="Processes for the brain (1 = in-process; small runs stay in-process anyway)",
)
@click.pass_context
def extract(  # noqa: PLR0917
    ctx: click.Context,
    code_path: str | None,
    locales_path: str | None,
    language: tuple[str, ...],
    i18n_keys: tuple[str, ...],
    i18n_keys_append: tuple[str, ...],
    i18n_keys_prefix: tuple[str, ...],
    exclude_dirs: tuple[str, ...],
    exclude_dirs_append: tuple[str, ...],
    verbose: bool,
    ignore_attributes: tuple[str, ...],
    append_ignore_attributes: tuple[str, ...],
    ignore_kwargs: tuple[str, ...],
    default_ftl_file: str | None,
    comment_keys_mode: str | None,
    line_endings: str | None,
    dry_run: bool,
    cache: bool,
    cache_path: str | None,
    clear_cache: bool,
    allow_parse_errors: bool,
    fly_audit: bool,
    fly_seed: int,
    fly_no_tui: bool,
    fly_trials: int,
    fly_batch: int | None,
    fly_workers: int,
) -> None:
    """Extract Fluent keys from Python code into locale files."""
    overrides = ExtractOverrides(
        code_path=code_path,
        locales_path=locales_path,
        language=list(language) or None,
        i18n_keys=list(i18n_keys) or None,
        i18n_keys_append=list(i18n_keys_append) or None,
        i18n_keys_prefix=list(i18n_keys_prefix) or None,
        exclude_dirs=list(exclude_dirs) or None,
        exclude_dirs_append=list(exclude_dirs_append) or None,
        ignore_attributes=list(ignore_attributes) or None,
        append_ignore_attributes=list(append_ignore_attributes) or None,
        ignore_kwargs=list(ignore_kwargs) or None,
        default_ftl_file=default_ftl_file,
        comment_keys_mode=comment_keys_mode,
        line_endings=line_endings,
        dry_run=dry_run,
        cache=cache,
        cache_path=cache_path,
        clear_cache=clear_cache,
        allow_parse_errors=allow_parse_errors,
    )
    fly = FlyOptions(
        workers=fly_workers,
        batch=fly_batch,
        trials=fly_trials,
        seed=fly_seed,
        tui=not fly_no_tui,
        audit=fly_audit,
        verbose=verbose or bool(ctx.obj and ctx.obj.get("verbose")),
    )
    config_path = ctx.obj.get("config") if ctx.obj else None
    ctx.exit(run_command(overrides, config_path, fly))


@main.group()
def config() -> None:
    """Configuration helpers."""


@config.command()
@click.option(
    "--command",
    type=click.Choice(["extract", "stub", "check"]),
    default=None,
    help="Print only one command-specific pyproject.toml section",
)
def sample(command: str | None) -> None:
    """Print a sample pyproject.toml configuration."""
    click.echo(sample_text(command), nl=False)


def run() -> None:
    """Console-script entry point (``ftl``, ``fly-ftl``, ``python -m fly_ftl_extract``).

    click expands ``*``/``?`` arguments itself on Windows (``windows_expand_args``); the
    original passes ``-E '**/tests/**'`` through untouched, so that is switched off.
    """
    main(prog_name="ftl", windows_expand_args=False)


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def stub(args: tuple[str, ...]) -> None:
    """Not implemented in v1 (see CLAUDE.md)."""
    del args
    click.echo(NOT_TRAINED_MSG, err=True)
    raise SystemExit(NOT_TRAINED_EXIT_CODE)


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def check(args: tuple[str, ...]) -> None:
    """Not implemented in v1 (see CLAUDE.md)."""
    del args
    click.echo(NOT_TRAINED_MSG, err=True)
    raise SystemExit(NOT_TRAINED_EXIT_CODE)


__all__ = ["main", "run"]
