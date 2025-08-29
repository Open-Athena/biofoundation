"""An ecosystem for biological foundation models.

This package provides tools for working with biological foundation models,
including data transformation, model wrappers, and inference utilities.
"""

try:
    from importlib.metadata import version

    __version__ = version("biofoundation")
except ImportError:
    __version__ = "unknown"
