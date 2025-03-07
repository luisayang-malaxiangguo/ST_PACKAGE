"""
Image recovery for CSRBF interpolation.

Contains the function to reconstruct the color image from CSRBF coefficients on the GPU.
"""

import cupy as cp
from tqdm import tqdm
from image_color_recovery.algorithms.csrbf.kernel import csrbf_phi
from image_color_recovery.algorithms.csrbf.utils import compute_distances_gpu_chunked


def recover_color_image_gpu(
    sample_indices, gray_image, coefficients,
    sigma_spatial, sigma_gray, gray_exponent,
    batch_size=10000
):
    """
    Reconstruct the color image from the CSRBF coefficients on the GPU.

    Parameters
    ----------
    sample_indices : tuple of np.ndarray
        Pixel indices (rows, columns) for sampled points.
    gray_image : np.ndarray
        Grayscale image.
    coefficients : cp.ndarray
        Interpolation coefficients (N x 3).
    sigma_spatial : float
        Spatial scaling factor.
    sigma_gray : float
        Grayscale scaling factor.
    gray_exponent : float
        Exponent for grayscale distance.
    batch_size : int, optional
        Batch size for image reconstruction.

    Returns
    -------
    np.ndarray
        Recovered image (uint8) with clamped pixel values.
    """
    sampled_rows = cp.asarray(sample_indices[0], dtype=cp.int32)
    sampled_cols = cp.asarray(sample_indices[1], dtype=cp.int32)
    gray_cp = cp.asarray(gray_image, dtype=cp.float32)

    coords_sampled = cp.stack([sampled_rows, sampled_cols], axis=1)
    rows, cols = gray_image.shape

    grid_y, grid_x = cp.meshgrid(
        cp.arange(rows, dtype=cp.float32),
        cp.arange(cols, dtype=cp.float32),
        indexing="ij"
    )
    all_pixels = cp.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
    all_gray_values = gray_cp.ravel()

    recovered_image = cp.zeros((rows, cols, 3), dtype=cp.float32)
    total_pixels = all_pixels.shape[0]

    print("Reconstructing image in batches...")
    for start in tqdm(range(0, total_pixels, batch_size), desc="Reconstructing Image"):
        end = min(start + batch_size, total_pixels)
        pixel_chunk = all_pixels[start:end]
        gray_chunk = all_gray_values[start:end, None]

        spatial_distances = compute_distances_gpu_chunked(pixel_chunk, coords_sampled)
        gray_sample_values = gray_cp[sampled_rows, sampled_cols][:, None]
        gray_distances = compute_distances_gpu_chunked(gray_chunk, gray_sample_values)
        gray_distances = cp.abs(gray_distances) ** gray_exponent

        r_spatial = spatial_distances / sigma_spatial
        r_gray = gray_distances / sigma_gray

        phi_spatial = csrbf_phi(r_spatial)
        phi_gray = csrbf_phi(r_gray)
        kernel_batch = phi_spatial * phi_gray

        recovered_chunk = cp.dot(kernel_batch, coefficients)
        pixel_chunk_int = pixel_chunk.astype(cp.int32)
        recovered_image[
            pixel_chunk_int[:, 0],
            pixel_chunk_int[:, 1],
            :
        ] = cp.clip(recovered_chunk, 0, 255)

    recovered_image = cp.clip(recovered_image, 0, 255)
    return cp.asnumpy(recovered_image.astype(cp.uint8))
