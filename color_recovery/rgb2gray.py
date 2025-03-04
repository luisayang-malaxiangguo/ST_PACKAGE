import numpy as np


def rgb2gray(image_rgb):
    """
    Convert an RGB image to grayscale using a standard weighting.

    Parameters
    ----------
    image_rgb : np.ndarray (H, W, 3)
        RGB image in uint8 format.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) in uint8 format.
    """
    red = image_rgb[:, :, 0].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    blue = image_rgb[:, :, 2].astype(np.float32)

    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    gray = np.clip(gray, 0, 255)

    return gray.astype(np.uint8)
