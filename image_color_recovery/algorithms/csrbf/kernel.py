"""
Kernel functions for CSRBF interpolation.

Contains the CSRBF function and the routine for generating the kernel matrix on the GPU.
"""

import cupy as cp
from image_color_recovery.algorithms.csrbf.utils import compute_distances_gpu_chunked


def csrbf_phi(distance):
    """
    Compute the Compactly Supported Radial Basis Function (CSRBF).

    Parameters
    ----------
    distance : cp.ndarray
        Distance values.

    Returns
    -------
    cp.ndarray
        CSRBF evaluated at the given distances.
    """
    return cp.power(cp.maximum(1 - distance, 0), 4) * (4 * distance + 1)


def generate_kernel_matrix_gpu(
    sample_indices, gray_image, sigma_spatial, sigma_gray,
    gray_exponent, chunk_size=4000
):
    """
    Generate the kernel matrix on the GPU using CSRBF with spatial and greyscale info.

    Parameters
    ----------
    sample_indices : tuple of np.ndarray
        Pixel indices (rows, columns) for sampled points.
    gray_image : np.ndarray
        Grayscale image.
    sigma_spatial : float
        Spatial scaling factor.
    sigma_gray : float
        Grayscale scaling factor.
    gray_exponent : float
        Exponent for grayscale distance.
    chunk_size : int, optional
        Chunk size for GPU computation.

    Returns
    -------
    cp.ndarray
        The kernel matrix (float32).
    """
    sampled_rows = cp.asarray(sample_indices[0], dtype=cp.int32)
    sampled_cols = cp.asarray(sample_indices[1], dtype=cp.int32)
    gray_cp = cp.asarray(gray_image, dtype=cp.float32)

    coords = cp.stack([sampled_rows, sampled_cols], axis=1)
    gray_values = gray_cp[sampled_rows, sampled_cols]

    spatial_distances = compute_distances_gpu_chunked(coords, coords, chunk_size=chunk_size)
    gray_values_2d = gray_values[:, None]
    gray_distances = compute_distances_gpu_chunked(
        gray_values_2d, gray_values_2d, chunk_size=chunk_size
    )
    gray_distances = cp.abs(gray_distances) ** gray_exponent

    r_spatial = spatial_distances / sigma_spatial
    r_gray = gray_distances / sigma_gray

    kernel_matrix = csrbf_phi(r_spatial) * csrbf_phi(r_gray)
    return kernel_matrix.astype(cp.float32)
