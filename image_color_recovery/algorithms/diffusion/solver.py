"""
Solver for diffusion-based PDE colorization with error tracking.
"""

import cupy as cp
from tqdm import trange
from .builder import build_neighbor_indices_and_weights


def pde_colorize_3ch_with_error(
    gray_gpu,
    known_mask_gpu,
    known_values_3ch_gpu,
    original_3ch_gpu,
    sigma=0.1,
    max_iters=35
):
    """
    Perform PDE colorization (in [0,1] scale) with error tracking.

    Parameters
    ----------
    gray_gpu : cp.ndarray of shape (H, W), float32 in [0,1]
        Grayscale intensities on the GPU.
    known_mask_gpu : cp.ndarray of shape (H, W), bool
        Boolean mask of known pixels (True where color is known).
    known_values_3ch_gpu : cp.ndarray of shape (H, W, 3), float32 in [0,1]
        Known RGB color values at the known pixels.
    original_3ch_gpu : cp.ndarray of shape (H, W, 3), float32 in [0,1]
        Ground truth color image (for error measurement).
    sigma : float, optional
        Gaussian standard deviation for neighbor weights.
    max_iters : int, optional
        Number of diffusion iterations.

    Returns
    -------
    recovered_gpu : cp.ndarray of shape (H, W, 3)
        Recovered color image on GPU in [0,1].
    error_values : list of float
        Mean-squared error (MSE) at each iteration.
    """
    neighbors, weights, height, width = build_neighbor_indices_and_weights(
        gray_gpu, sigma=sigma
    )
    num_pixels = height * width

    known_mask_flat = known_mask_gpu.ravel()
    known_vals_flat = known_values_3ch_gpu.reshape(num_pixels, 3)
    original_flat = original_3ch_gpu.reshape(num_pixels, 3)

    # Initialize color buffer with known values
    color_buffer = cp.zeros((num_pixels, 3), dtype=cp.float32)
    color_buffer[known_mask_flat] = known_vals_flat[known_mask_flat]

    unknown_mask_flat = ~known_mask_flat
    error_values = []

    for iteration in trange(max_iters, desc="Diffusion Iterations"):
        color_old = color_buffer.copy()
        accum = cp.zeros_like(color_old)
        wsum = cp.zeros(num_pixels, dtype=cp.float32)

        for k in range(4):
            nbr_indices = neighbors[:, k]
            w_k = weights[:, k]
            valid_mask = nbr_indices >= 0
            idx_valid = cp.where(valid_mask)[0]
            nbr_idx_valid = nbr_indices[idx_valid]
            w_expand = w_k[idx_valid, None]
            accum[idx_valid] += w_expand * color_old[nbr_idx_valid]
            wsum[idx_valid] += w_k[idx_valid]

        color_new = color_old.copy()
        idx_unknown = cp.where(unknown_mask_flat)[0]
        denom = wsum[idx_unknown]
        safe_mask = denom > 1e-12
        idx_safe = idx_unknown[safe_mask]
        color_new[idx_safe] = accum[idx_safe] / denom[safe_mask, None]

        color_buffer = color_new

        # Compute MSE in [0,1] scale
        diff = color_buffer - original_flat
        mse = cp.mean(diff * diff)
        error_values.append(float(mse.get()))

    recovered_gpu = color_buffer.reshape(height, width, 3)
    return recovered_gpu, error_values
