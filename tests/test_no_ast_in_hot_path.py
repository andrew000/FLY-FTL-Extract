"""Guard rail: the extraction hot path must never touch ``ast`` or ``reference/``.

CLAUDE.md rule 2. The fly is the only classifier; ``fly_ftl_extract.reference`` (the ast
teacher) may be imported only from ``reference/`` itself, ``scripts/`` and ``tests/``.
"""

import re
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "fly_ftl_extract"

FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*import\s+.*\bast\b"),
    re.compile(r"^\s*from\s+ast\b"),
    re.compile(r"fly_ftl_extract\.reference\b"),
    re.compile(r"^\s*from\s+\.+\s*reference\b"),
    re.compile(r"^\s*from\s+\.+\s+import\s+.*\breference\b"),
)


def _hot_path_files() -> list[Path]:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    return [f for f in files if "reference" not in f.relative_to(PACKAGE_DIR).parts]


def test_hot_path_files_exist() -> None:
    assert _hot_path_files(), "package skeleton is missing"


def test_no_forbidden_imports_in_hot_path_sources() -> None:
    offenders: list[str] = []
    for path in _hot_path_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(line) for p in FORBIDDEN_PATTERNS):
                offenders.append(f"{path.relative_to(PACKAGE_DIR.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "ast / reference leaked into the hot path:\n" + "\n".join(offenders)


def test_importing_cli_does_not_load_reference_or_ast() -> None:
    # Run in a fresh interpreter: other tests legitimately import ``reference`` in-process,
    # so an in-process ``sys.modules`` check would be order-dependent.
    code = (
        "import sys\n"
        "import fly_ftl_extract.cli\n"
        "leaked = sorted(m for m in sys.modules "
        "if m == 'fly_ftl_extract.reference' or m.startswith('fly_ftl_extract.reference.'))\n"
        "print('REFERENCE_LOADED' if leaked else 'CLEAN')\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.splitlines()[0] == "CLEAN", result.stdout
