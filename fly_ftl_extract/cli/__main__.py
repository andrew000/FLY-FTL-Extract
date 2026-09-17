"""``python -m fly_ftl_extract.cli`` — the ``ftl`` command."""

from fly_ftl_extract.cli import run

if __name__ == "__main__":  # spawn workers re-import __main__; the guard keeps them quiet
    run()
