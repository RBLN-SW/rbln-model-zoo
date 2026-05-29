"""RBLN Model Zoo — CLI and model registry."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rbln_zoo")
except PackageNotFoundError:
    __version__ = "0.0.0"
