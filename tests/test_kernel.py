import cupy as cp
import numpy as np
from color_recovery.kernel import generate_kernel_gpu, compute_kernel_batch


def test_generate_kernel_gpu_small():
    """
    Test generate_kernel_gpu with two selected points in a 4x4 image.
    """
    selected_rows = np.array([0, 3])
    selected_cols = np.array([0, 3])
    selected_points = (selected_rows, selected_cols)
    image_gray = np.zeros((4, 4), dtype=np.uint8)
    sigma_1 = 1.0

    kernel_matrix = generate_kernel_gpu(selected_points, image_gray, sigma_1)
    kernel_cpu = kernel_matrix.get()

    # Kernel should be 2x2
    assert kernel_cpu.shape == (2, 2)
    # Diagonal entries should be ~1 (distance=0 => exp(-0)=1)
    assert np.isclose(kernel_cpu[0, 0], 1.0, atol=1e-5)
    assert np.isclose(kernel_cpu[1, 1], 1.0, atol=1e-5)


def test_compute_kernel_batch():
    """
    Test compute_kernel_batch with a small batch of 2 pixels.
    """
    coords_selected = cp.array([[0, 0],
                                [2, 2]], dtype=cp.float32)
    batch_pixels = cp.array([[0, 2],
                             [2, 0]], dtype=cp.float32)
    sigma_1 = 1.0

    kernel_batch = compute_kernel_batch(batch_pixels, coords_selected, sigma_1)
    kernel_cpu = kernel_batch.get()

    assert kernel_cpu.shape == (2, 2)
