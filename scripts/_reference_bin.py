"""Locate the *reference* ``ftl`` binary (original ftl-extract 0.12.1).

The reference lives in a ``uv tool`` environment, never in the project venv, so the
two ``ftl`` executables cannot be confused.  Override with ``FTL_REFERENCE_BIN``.
"""

import os

REFERENCE_VERSION = "0.12.1"


def reference_ftl_command() -> list[str]:
    """Return the argv prefix that runs the original ``ftl`` binary."""
    override = os.environ.get("FTL_REFERENCE_BIN")
    if override:
        return [override]
    return ["uv", "tool", "run", "--from", f"ftl-extract=={REFERENCE_VERSION}", "ftl"]
