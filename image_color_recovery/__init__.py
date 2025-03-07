"""
Image Color Recovery package.

Provides algorithms and utilities to recover color in images using:
- CSRBF interpolation 
- Diffusion-based PDE colorization
- MRF-based colorization
"""

from .algorithms import csrbf, diffusion, mrf