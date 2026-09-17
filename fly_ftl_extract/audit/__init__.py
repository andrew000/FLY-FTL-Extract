"""``--fly-audit``: the ast teacher (``reference/``) run next to the fly, differences listed.

CLAUDE.md rule 2b: this is the only package that imports both the reference and the
extraction pipeline.  ``cli/`` imports it lazily, only inside the ``--fly-audit`` branch;
a plain ``ftl extract`` never loads it (``tests/test_no_ast_in_hot_path.py``).

The comparison is on the merged result the writer would receive: every key's ``.ftl``
path, placeable variables, ``**kwargs`` flag and call site, plus the diagnostics and the
count of files with keys.  Differences become ``[WARN  fly::audit]`` lines, the summary an
``INFO`` (none) or ``ERROR`` (some) line; the CLI exits 1 when there are any.
"""

from __future__ import annotations

from fly_ftl_extract.ftl.merge import CodeExtraction
from fly_ftl_extract.ftl.model import ExtractOptions, FluentKey, kwargs_from_key
from fly_ftl_extract.ftl.pipeline import LogLine
from fly_ftl_extract.reference.extractor import extract_code

AUDIT_TARGET = "fly::audit"


def _describe(key: FluentKey) -> str:
    kwargs = ", ".join(kwargs_from_key(key)) or "no kwargs"
    star = ", **kwargs" if key.kwargs_unknown is not None else ""
    return f"{key.path} [{kwargs}{star}] at {key.source_location}"


def compare_extractions(fly: CodeExtraction, teacher: CodeExtraction) -> list[str]:
    """Human-readable differences between the fly's and the teacher's merged extraction."""
    differences: list[str] = []
    fly_keys = dict(fly.keys.items())
    teacher_keys = dict(teacher.keys.items())
    for name in sorted(set(fly_keys) | set(teacher_keys)):
        a = fly_keys.get(name)
        b = teacher_keys.get(name)
        if a is None and b is not None:
            differences.append(f"key `{name}` missed by the fly: teacher has {_describe(b)}")
        elif b is None and a is not None:
            differences.append(f"key `{name}` invented by the fly: {_describe(a)}")
        elif a is not None and b is not None:
            if a.path_key != b.path_key:
                differences.append(f"key `{name}` path: fly {a.path}, teacher {b.path}")
            if sorted(kwargs_from_key(a)) != sorted(kwargs_from_key(b)):
                differences.append(
                    f"key `{name}` kwargs: fly [{', '.join(kwargs_from_key(a))}], "
                    f"teacher [{', '.join(kwargs_from_key(b))}]"
                )
            if (a.kwargs_unknown is None) != (b.kwargs_unknown is None):
                differences.append(
                    f"key `{name}` **kwargs flag: fly {a.kwargs_unknown}, "
                    f"teacher {b.kwargs_unknown}"
                )
            if a.source_location != b.source_location:
                differences.append(
                    f"key `{name}` call site: fly {a.source_location}, teacher {b.source_location}"
                )
    fly_diag = sorted(str(d) for d in fly.diagnostics)
    teacher_diag = sorted(str(d) for d in teacher.diagnostics)
    differences.extend(
        f"diagnostic only from the fly: {d}" for d in fly_diag if d not in teacher_diag
    )
    differences.extend(
        f"diagnostic only from the teacher: {d}" for d in teacher_diag if d not in fly_diag
    )
    if fly.py_files_with_keys != teacher.py_files_with_keys:
        differences.append(
            f"files with keys: fly {fly.py_files_with_keys}, teacher {teacher.py_files_with_keys}"
        )
    return differences


def compare(options: ExtractOptions, fly: CodeExtraction) -> list[str]:
    """Run the teacher with the same options and compare it with the fly's extraction."""
    return compare_extractions(fly, extract_code(options))


def report_lines(differences: list[str]) -> list[LogLine]:
    """Log lines for the audit result."""
    lines = [LogLine("WARN", AUDIT_TARGET, d) for d in differences]
    if differences:
        n = len(differences)
        lines.append(
            LogLine(
                "ERROR",
                AUDIT_TARGET,
                f"Audit: {n} {'difference' if n == 1 else 'differences'} from the ast reference.",
            )
        )
    else:
        lines.append(LogLine("INFO", AUDIT_TARGET, "Audit: 0 differences from the ast reference."))
    return lines


__all__ = ["AUDIT_TARGET", "compare", "compare_extractions", "report_lines"]
