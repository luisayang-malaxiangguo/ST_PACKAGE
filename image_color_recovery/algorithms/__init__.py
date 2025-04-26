"""
Algorithms subpackage.

Currently available:
- CSRBF interpolation
- Gaussian Diffusion-Propagation colorization
- MRF-based colorization
"""

# CSRBF algorithm
from .csrbf import (
    kernel as csrbf_kernel,
    solver as csrbf_solver,
    recovery as csrbf_recovery,
    utils as csrbf_utils,
    config as csrbf_config
)

# Gaussian Diffusion-Propagation algorithm
from .diffusion import (
    builder as diffusion_builder,
    solver as diffusion_solver,
    main as diffusion_main
)

# MRF algorithm
from .mrf import (
    builder as mrf_builder,
    solver as mrf_solver,
    utils as mrf_utils,
    main as mrf_main
)
