# ftl-extract: ignore stale
dynamic-key = Built from an f-string in Python
# ftl-extract: ignore all
brand = brand
# ftl-extract: ignore kwargs
flex-kwargs = Has { $x } but code passes y
# ftl-extract: ignore untranslated
plain = plain
# Some note
# ftl-extract: ignore stale
kept-with-note = Kept
# ftl-extract: ignore stale
stale-parent = references { stale-child }
stale-child = child kept through ignore-stale parent
regular-stale = will be commented
# ftl-extract: ignored
not-a-marker = will be commented too
# FTL-Extract: Ignore Stale, Kwargs
mixed-case = { $q }
# ignore
legacy-ignore = legacy
