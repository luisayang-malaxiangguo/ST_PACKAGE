"""
Solver for CSRBF interpolation.

Provides the function to solve for interpolation coefficients on the GPU.
"""

import cupy as cp


def solve_coefficients_gpu_direct(kernel_matrix, sampled_colors, regularization):
    """
    Solve for interpolation coefficients on the GPU for each color channel.

    Parameters
    ----------
    kernel_matrix : cp.ndarray
        Kernel matrix of shape (N, N).
    sampled_colors : cp.ndarray
        Known color values at the sampled points (N x 3).
    regularization : float
        Regularization parameter (delta).

    Returns
    -------
    cp.ndarray
        Interpolation coefficients (N x 3).
    """
    num_points = kernel_matrix.shape[0]
    # (delta * num_points) * Identity ensures stability
    system_matrix = kernel_matrix + (regularization * num_points) * cp.eye(num_points, dtype=cp.float32)

    coefficients = cp.zeros((num_points, 3), dtype=cp.float32)
    for channel_idx in range(3):
        channel_values = sampled_colors[:, channel_idx]
        solution_channel = cp.linalg.solve(system_matrix, channel_values)
        coefficients[:, channel_idx] = solution_channel
    return coefficients
