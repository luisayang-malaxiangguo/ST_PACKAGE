import os
import cv2
import numpy as np
import pytest
from color_recovery.image_loader import load_image


def test_load_image(tmp_path):
    """
    Test load_image by creating a small 2x2 BGR image and loading it.
    """
    test_path = os.path.join(tmp_path, "test_img.png")

    # 2x2 BGR image
    bgr_image = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr_image[0, 0] = [255, 0, 0]     # Blue
    bgr_image[0, 1] = [0, 255, 0]     # Green
    bgr_image[1, 0] = [0, 0, 255]     # Red
    bgr_image[1, 1] = [255, 255, 255] # White

    cv2.imwrite(test_path, bgr_image)

    rgb_loaded, gray_loaded = load_image(test_path)

    assert rgb_loaded.shape == (2, 2, 3)
    assert gray_loaded.shape == (2, 2)
    assert rgb_loaded.dtype == np.uint8
    assert gray_loaded.dtype == np.uint8
