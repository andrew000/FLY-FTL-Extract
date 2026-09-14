"""fly-ftl-extract: ftl-extract, but the classifier is a simulated Drosophila mushroom body."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fly-ftl-extract")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = "0+unknown"
