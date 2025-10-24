"""Provides simply tensors for testing tensor manipulation algorithms."""

from importlib.metadata import PackageNotFoundError, version

from .generate_3d import generate_cross_3d

# Optional visualization import with graceful fallback
try:
    from .visualize import visualize_3d_tensor, create_demo_notebook
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

    def visualize_3d_tensor(*args, **kwargs):
        """Visualization function placeholder."""
        raise ImportError(
            "Visualization requires optional dependencies. "
            "Install with: pip install test-tensors[viz]"
        )

    def create_demo_notebook(*args, **kwargs):
        """Demo notebook function placeholder."""
        raise ImportError(
            "Demo notebook requires optional dependencies. "
            "Install with: pip install test-tensors[viz]"
        )

try:
    __version__ = version("test-tensors")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

__all__ = ["generate_cross_3d", "visualize_3d_tensor", "create_demo_notebook"]
