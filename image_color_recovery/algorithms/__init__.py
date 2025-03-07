"""
Algorithms subpackage.

Currently available:
- CSRBF interpolation
- Diffusion-based PDE colorization
- MRF-based colorization
"""

# CSRBF algorithm
from .csrbf import (
    kernel as csrbf_kernel,
    solver as csrbf_solver,
    recovery as csrbf_recovery,
    utils as csrbf_utils,
    csrbf_visualization,
    config as csrbf_config
)

# Diffusion-based PDE algorithm
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
