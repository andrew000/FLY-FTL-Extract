"""Guard rail: the extraction hot path must never touch ``ast`` or ``reference/``.

CLAUDE.md rules 2 and 2b. The fly is the only classifier; ``fly_ftl_extract.reference`` (the
ast teacher) may be imported only from ``reference/`` itself, ``audit/`` (the ``--fly-audit``
runner), ``scripts/`` and ``tests/``.  A plain ``ftl extract`` must load neither ``reference``
nor ``audit``.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "fly_ftl_extract"
FIXTURE_APP = PACKAGE_DIR.parent / "tests" / "fixtures" / "projects" / "basic" / "app"
GREP_EXEMPT_PACKAGES = frozenset({"reference", "audit"})

FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*import\s+.*\bast\b"),
    re.compile(r"^\s*from\s+ast\b"),
    re.compile(r"fly_ftl_extract\.reference\b"),
    re.compile(r"^\s*from\s+\.+\s*reference\b"),
    re.compile(r"^\s*from\s+\.+\s+import\s+.*\breference\b"),
)


COMPILE_CALL = re.compile(r"(?<![\w.])compile\(")
COMPILE_ALLOWED = Path("ftl") / "pyerrors.py"


def _hot_path_files() -> list[Path]:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    return [f for f in files if not GREP_EXEMPT_PACKAGES & set(f.relative_to(PACKAGE_DIR).parts)]


def test_hot_path_files_exist() -> None:
    assert _hot_path_files(), "package skeleton is missing"


def test_no_forbidden_imports_in_hot_path_sources() -> None:
    offenders: list[str] = []
    for path in _hot_path_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(line) for p in FORBIDDEN_PATTERNS):
                offenders.append(f"{path.relative_to(PACKAGE_DIR.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "ast / reference leaked into the hot path:\n" + "\n".join(offenders)


def test_bare_compile_only_in_pyerrors() -> None:
    """CLAUDE.md rule 2c: ``compile()`` may detect parse errors in one place only."""
    offenders: list[str] = []
    for path in _hot_path_files():
        if path.relative_to(PACKAGE_DIR) == COMPILE_ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if COMPILE_CALL.search(line):
                offenders.append(f"{path.relative_to(PACKAGE_DIR.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "compile() outside ftl/pyerrors.py:\n" + "\n".join(offenders)
    assert any(
        COMPILE_CALL.search(line)
        for line in (PACKAGE_DIR / COMPILE_ALLOWED).read_text(encoding="utf-8").splitlines()
    )


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


_LEAK_PROBE = """
import runpy, sys
sys.argv = ["ftl", *sys.argv[1:]]
try:
    runpy.run_module("fly_ftl_extract.cli", run_name="__main__", alter_sys=True)
except SystemExit:
    pass
leaked = sorted(
    m for m in sys.modules
    if m.split(".")[:2] in (["fly_ftl_extract", "reference"], ["fly_ftl_extract", "audit"])
)
print("LEAKED " + ",".join(leaked) if leaked else "CLEAN")
"""


def test_plain_extract_does_not_load_reference_or_audit(tmp_path: Path) -> None:
    """A real ``ftl extract`` in a fresh interpreter: the fly decides, the teacher stays out."""
    assert (PACKAGE_DIR / "cli" / "extract.py").exists()
    assert (PACKAGE_DIR / "__main__.py").exists()
    app = tmp_path / "app"
    shutil.copytree(FIXTURE_APP, app)
    locales = tmp_path / "locales"
    locales.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", _LEAK_PROBE, "extract", str(app), str(locales), "--fly-no-tui"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.stdout.strip().splitlines()[-1] == "CLEAN", result.stdout + result.stderr
