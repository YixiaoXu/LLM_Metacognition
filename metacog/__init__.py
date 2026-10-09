"""Tools for studying continuous metacognitive signals in language models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-metacog")
except PackageNotFoundError:  # Source checkout without an editable install.
    __version__ = "0.1.0"

__all__ = ["__version__"]
