"""
MRF-based colorization subpackage.

Provides functionality to colorize images by building and solving an MRF system.
"""

from .builder import build_mrf_system
from .solver import solve_channel_direct, solve_channel_iterative
from .main import run_mrf_demo  # Exposing a function from main.py
from .utils import (
    compute_mrf_normalized_error,
    compute_mrf_normalized_error_channel
)
