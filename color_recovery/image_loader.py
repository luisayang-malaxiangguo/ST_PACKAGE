import cv2
from .rgb2gray import rgb2gray


def load_image(image_path):
    """
    Load an image from disk, convert from BGR to RGB, then to grayscale.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple
        A tuple (image_rgb, image_gray) where image_rgb is the original
        RGB image in uint8 format, and image_gray is the grayscale
        version in uint8 format.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_rgb = image_bgr[..., ::-1]
    image_gray = rgb2gray(image_rgb)

    return image_rgb, image_gray
