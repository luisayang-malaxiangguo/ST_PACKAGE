import numpy as np
from color_recovery.rgb2gray import rgb2gray


def test_rgb2gray_basic():
    """
    Test rgb2gray with a small 2x2 RGB image.
    """
    rgb_image = np.array([
        [[255,   0,   0], [0, 255,   0]],
        [[  0,   0, 255], [255, 255, 255]]
    ], dtype=np.uint8)

    gray_image = rgb2gray(rgb_image)

    assert gray_image.shape == (2, 2)
    assert gray_image.dtype == np.uint8
    # Optionally, you can compare specific pixel values if needed.
