"""
Neighbor and weight builder for Gaussian Diffusion-Propagation colorization.
"""

import cupy as cp


def build_neighbor_indices_and_weights(gray_gpu, sigma=0.1):
    """
    Build neighbor indices and weights for Gaussian Diffusion-Propagation 
    colorization using a 4-neighbor system. The weights are Gaussian-based
    on intensity differences.

    Parameters
    ----------
    gray_gpu : cp.ndarray of shape (H, W)
        Grayscale image on GPU in float32 [0,1].
    sigma : float, optional
        Gaussian standard deviation for weighting neighbors.

    Returns
    -------
    neighbors : cp.ndarray of shape (H*W, 4), int32
        Indices of the 4 neighbors for each pixel (-1 if out of bounds).
    weights : cp.ndarray of shape (H*W, 4), float32
        Gaussian weights for each neighbor.
    height : int
        Height of the image.
    width : int
        Width of the image.
    """
    height, width = gray_gpu.shape
    num_pixels = height * width

    neighbors = cp.full((num_pixels, 4), -1, dtype=cp.int32)
    weights = cp.zeros((num_pixels, 4), dtype=cp.float32)

    gray_flat = gray_gpu.ravel()
    offset_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            center_val = gray_flat[idx]
            for k, (dy, dx_) in enumerate(offset_list):
                ny = y + dy
                nx = x + dx_
                if 0 <= ny < height and 0 <= nx < width:
                    nbr_idx = ny * width + nx
                    diff = center_val - gray_flat[nbr_idx]
                    w = cp.exp(-(diff * diff) / (2 * sigma * sigma))
                    # Convert to float in CPU to avoid unnecessary array allocation
                    w = float(w)
                    neighbors[idx, k] = nbr_idx
                    weights[idx, k] = w
                else:
                    neighbors[idx, k] = -1
                    weights[idx, k] = 0.0

    return neighbors, weights, height, width
