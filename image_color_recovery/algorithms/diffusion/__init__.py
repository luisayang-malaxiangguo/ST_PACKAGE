"""
Diffusion-based PDE colorization for image color recovery.
"""

from .builder import build_neighbor_indices_and_weights
from .solver import pde_colorize_3ch_with_error
from .main import run_pde_demo

__all__ = [
    "build_neighbor_indices_and_weights",
    "pde_colorize_3ch_with_error",
    "run_pde_demo"
]
