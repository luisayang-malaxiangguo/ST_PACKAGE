import cupy as cp
import numpy as np
from color_recovery.reconstructor import recover_image_gpu


def test_recover_image_gpu_small():
    """
    Test recover_image_gpu with a small 4x4 grayscale image
    and 2 selected points.
    """
    image_gray = np.zeros((4, 4), dtype=np.uint8)
    selected_points = (np.array([0, 3]), np.array([0, 3]))

    # Suppose solver gave us 2x3 coefficients
    coeffs_cpu = np.array([[10, 20, 30],
                           [40, 50, 60]], dtype=np.float32)
    coeffs_gpu = cp.asarray(coeffs_cpu)

    sigma_1 = 1.0
    reconstructed = recover_image_gpu(selected_points, image_gray,
                                      coeffs_gpu, sigma_1,
                                      rescale=True, batch_size=2)

    assert reconstructed.shape == (4, 4, 3)
    assert reconstructed.dtype == np.uint8
