"""
Default configuration parameters for image color recovery.
"""

DEFAULT_PARAMS = {
    "percentage": 1,       # % of pixels selected (random)
    "sigma1": 100,          # Spatial scaling factor
    "sigma2": 100,          # Grayscale scaling factor
    "p": 0.5,               # Exponent for grayscale distance
    "delta": 2e-4,          # Regularization parameter
    "batch_size": 10000
}
