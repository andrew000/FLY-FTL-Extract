"""Command-line entry points (``ftl`` and ``fly-ftl``).

Only the skeleton exists in Phase 0; the real ``extract`` command arrives in Phase 6.
"""

import click

from fly_ftl_extract import __version__

NOT_TRAINED_MSG = "the fly has not been trained for this yet — use the original ftl-extract"
NOT_TRAINED_EXIT_CODE = 2


@click.group()
@click.version_option(
    __version__, "-V", "--version", prog_name="ftl", message="%(prog)s %(version)s"
)
def main() -> None:
    """Extract Fluent keys from Python code using a simulated fly brain."""


@main.command()
def stub() -> None:
    """Not implemented in v1 (see CLAUDE.md)."""
    click.echo(NOT_TRAINED_MSG, err=True)
    raise SystemExit(NOT_TRAINED_EXIT_CODE)


@main.command()
def check() -> None:
    """Not implemented in v1 (see CLAUDE.md)."""
    click.echo(NOT_TRAINED_MSG, err=True)
    raise SystemExit(NOT_TRAINED_EXIT_CODE)


__all__ = ["main"]
