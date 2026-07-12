"""Baseline model adapters for CellDiffA."""

from .base_adapter import BaseAdapter
from .adapter_gears import GEARSAdapter
from .adapter_cpa import CPAAdapter
from .adapter_perturbdiff import PerturbDiffAdapter
from .adapter_scdfm import ScDFMAdapter

__all__ = [
    "BaseAdapter",
    "GEARSAdapter",
    "CPAAdapter",
    "PerturbDiffAdapter",
    "ScDFMAdapter",
]
