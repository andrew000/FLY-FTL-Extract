"""``ftl config sample``: the *sample* ``pyproject.toml`` sections of the original 0.12.1.

Captured from the reference binary (``ftl config sample``); it is a sample, not the
defaults.  Every section ends with a blank line, exactly like the original prints it.
"""

from __future__ import annotations

SAMPLE_EXTRACT = """\
[tool.ftl-extract.extract]
code-path = "app/bot"
locales-path = "app/bot/locales"
languages = ["en", "uk"]
i18n-keys-append = ["LF", "LazyProxy"]
ignore-attributes-append = ["core"]
exclude-dirs-append = ["./tests/*"]
ignore-kwargs = ["when"]
comment-keys-mode = "comment"
line-endings = "lf"
cache = true

"""

SAMPLE_STUB = """\
[tool.ftl-extract.stub]
locales-path = "app/bot/locales/en"
stub-path = "app/bot/stub.pyi"
export-tree = false

"""

SAMPLE_CHECK = """\
[tool.ftl-extract.check]
locales-path = "app/bot/locales"
code-path = "app/bot"
languages = ["uk", "pl"]
checks = ["all"]
suggest-from = ["en"]
fail-on = ["error"]
report-path = "reports/ftl-check"
report-format = "json"

# Per-check severity overrides. Defaults: stale and untranslated are warnings,
# everything else is an error.
# severity = { stale = "error", untranslated = "warn" }

# Check presets:
# checks = ["all"]
# checks = ["untranslated"] # Does not require code-path.
# checks = ["syntax"]       # Does not require code-path.
# checks = ["references"]   # Does not require code-path.
# checks = ["missing"]      # Requires code-path.
# checks = ["stale"]        # Requires code-path.
# checks = ["kwargs"]       # Requires code-path.
# checks = ["syntax", "references", "missing", "kwargs"]

"""

SAMPLES: dict[str, str] = {
    "extract": SAMPLE_EXTRACT,
    "stub": SAMPLE_STUB,
    "check": SAMPLE_CHECK,
}


def sample_text(command: str | None) -> str:
    """The sample for one command, or all three in the original's order."""
    if command is not None:
        return SAMPLES[command]
    return "".join(SAMPLES.values())
