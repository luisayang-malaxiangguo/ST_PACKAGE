import cupy as cp
from .distance import cdist_gpu_chunked


def generate_kernel_gpu(selected_points, image_gray, sigma_1):
    """
    Build a Gaussian kernel matrix (spatial only) on the GPU.

    Parameters
    ----------
    selected_points : tuple of np.ndarray
        Two arrays (rows, cols) for the selected points.
    image_gray : np.ndarray
        Grayscale image (unused here, but included for consistency).
    sigma_1 : float
        Scaling parameter for spatial distances.

    Returns
    -------
    cp.ndarray
        Kernel matrix of shape (N, N) in float32.
    """
    rows_gpu = cp.asarray(selected_points[0], dtype=cp.int32)
    cols_gpu = cp.asarray(selected_points[1], dtype=cp.int32)

    coords = cp.stack([rows_gpu, cols_gpu], axis=1)
    spatial_dists = cdist_gpu_chunked(coords, coords)
    r_spatial = spatial_dists / sigma_1

    kernel_matrix = cp.exp(-(r_spatial**2))
    return kernel_matrix.astype(cp.float32)


def compute_kernel_batch(batch_pixels, coords_selected, sigma_1):
    """
    Compute kernel values for a batch of pixels, spatial only.

    Parameters
    ----------
    batch_pixels : cp.ndarray (batch_size, 2)
        Coordinates of the batch of pixels.
    coords_selected : cp.ndarray (N, 2)
        Coordinates of selected points.
    sigma_1 : float
        Scaling parameter for spatial distances.

    Returns
    -------
    cp.ndarray
        Kernel batch of shape (batch_size, N).
    """
    spatial_dists = cdist_gpu_chunked(batch_pixels, coords_selected)
    r_spatial = spatial_dists / sigma_1
    kernel_batch = cp.exp(-(r_spatial**2))
    return kernel_batch
