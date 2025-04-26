"""
CSRBF interpolation subpackage.

Provides functions for generating the kernel matrix, solving for coefficients,
and reconstructing the image using CSRBF.
"""

from .kernel import generate_kernel_matrix_gpu, csrbf_phi
from .solver import solve_coefficients_gpu_direct
from .recovery import recover_color_image_gpu
from .config import DEFAULT_PARAMS

# Utility functions from utils.py
from .utils import (
    compute_distances_gpu_chunked,
    load_image,
    generate_random_indices
)
