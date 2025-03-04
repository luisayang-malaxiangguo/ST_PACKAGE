import cupy as cp
import numpy as np
from .kernel import compute_kernel_batch


def recover_image_gpu(selected_points, image_gray, coeffs, sigma_1,
                      rescale=True, batch_size=10000):
    """
    Reconstruct a color image from coefficients, using spatial kernel info.

    Parameters
    ----------
    selected_points : tuple of np.ndarray
        Two arrays (rows, cols) indicating the selected points.
    image_gray : np.ndarray
        Grayscale image (H, W).
    coeffs : cp.ndarray (N, 3)
        Coefficients computed by the solver.
    sigma_1 : float
        Spatial scaling parameter for the kernel.
    rescale : bool, optional
        If True, rescale the final image to 0-255.
    batch_size : int, optional
        Number of pixels processed in each batch.

    Returns
    -------
    np.ndarray
        Reconstructed image of shape (H, W, 3) in uint8 format.
    """
    rows_gpu = cp.asarray(selected_points[0], dtype=cp.int32)
    cols_gpu = cp.asarray(selected_points[1], dtype=cp.int32)

    coords_selected = cp.stack([rows_gpu, cols_gpu], axis=1)

    rows, cols = image_gray.shape
    image_gray_cp = cp.asarray(image_gray, dtype=cp.float32)

    grid_y, grid_x = cp.meshgrid(cp.arange(rows, dtype=cp.float32),
                                 cp.arange(cols, dtype=cp.float32),
                                 indexing='ij')
    all_pixels = cp.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
    recovered = cp.zeros((rows, cols, 3), dtype=cp.float32)

    total_pixels = all_pixels.shape[0]

    for start in range(0, total_pixels, batch_size):
        end = min(start + batch_size, total_pixels)
        batch_pixels = all_pixels[start:end]
        kernel_batch = compute_kernel_batch(batch_pixels, coords_selected,
                                            sigma_1)
        recovered_batch = kernel_batch @ coeffs
        batch_pixels_int = batch_pixels.astype(cp.int32)
        recovered[batch_pixels_int[:, 0], batch_pixels_int[:, 1], :] = cp.clip(
            recovered_batch, 0, 255
        )

    if rescale:
        min_val = cp.min(recovered)
        max_val = cp.max(recovered)
        recovered = 255 * (recovered - min_val) / (max_val - min_val)
    else:
        recovered = cp.clip(recovered, 0, 255)

    return cp.asnumpy(recovered.astype(np.uint8))
