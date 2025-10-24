"""Provides simply tensors for testing tensor manipulation algorithms"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("test-tensors")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"
