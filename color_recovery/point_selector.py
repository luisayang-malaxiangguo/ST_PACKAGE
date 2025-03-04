import numpy as np


def generate_random_points(image_shape, percentage):
    """
    Generate random points in an image according to a given percentage.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the grayscale image (height, width).
    percentage : float
        Percentage of total pixels to select.

    Returns
    -------
    tuple of np.ndarray
        Two arrays (rows, cols) indicating the selected pixel coordinates.
    """
    total_pixels = image_shape[0] * image_shape[1]
    num_points = int((percentage / 100) * total_pixels)
    indices = np.random.choice(total_pixels, num_points, replace=False)
    return np.unravel_index(indices, image_shape)
