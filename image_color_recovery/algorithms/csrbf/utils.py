"""
Utility functions for image color recovery.
"""

import cv2
import numpy as np
from skimage import color
import cupy as cp


def compute_distances_gpu_chunked(a, b, chunk_size=4000):
    """
    Compute pairwise Euclidean distances on the GPU in chunks.

    Parameters
    ----------
    a : cp.ndarray of shape (m, d)
        First set of points.
    b : cp.ndarray of shape (n, d)
        Second set of points.
    chunk_size : int, optional
        Number of rows in 'a' to process at once.

    Returns
    -------
    cp.ndarray
        Euclidean distances.
    """
    m = a.shape[0]
    n = b.shape[0]
    distances = cp.zeros((m, n), dtype=cp.float32)
    b_squared = cp.sum(b ** 2, axis=1).reshape(1, -1)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        a_chunk = a[start:end]
        a_squared = cp.sum(a_chunk ** 2, axis=1).reshape(-1, 1)
        ab_dot = a_chunk @ b.T
        dist_chunk = cp.sqrt(a_squared - 2 * ab_dot + b_squared)
        distances[start:end, :] = dist_chunk

    return distances


def load_image(image_path):
    """
    Load an image and convert it to RGB and grayscale.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple of np.ndarray
        Original RGB image and grayscale image (uint8).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = color.rgb2gray(image) * 255
    return image, gray_image.astype(np.uint8)


def generate_random_indices(image_shape, sample_percentage):
    """
    Generate random pixel indices for sampling.

    Parameters
    ----------
    image_shape : tuple
        Shape of the grayscale image (rows, cols).
    sample_percentage : float
        Percentage of pixels to sample.

    Returns
    -------
    tuple of np.ndarray
        Row and column indices of selected pixels.
    """
    total_pixels = image_shape[0] * image_shape[1]
    num_points = int((sample_percentage / 100) * total_pixels)
    indices = np.random.choice(total_pixels, num_points, replace=False)
    return np.unravel_index(indices, image_shape)
