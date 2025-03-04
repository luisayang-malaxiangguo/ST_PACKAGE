import cupy as cp
import numpy as np
from color_recovery.solver import solve_coeffs_gpu


def test_solve_coeffs_gpu_small():
    """
    Test solve_coeffs_gpu with a small 2x2 kernel matrix.
    """
    kd_cpu = np.array([[2.0, 1.0],
                       [1.0, 2.0]], dtype=np.float32)
    kd_gpu = cp.asarray(kd_cpu)

    color_vals_cpu = np.array([[10, 20, 30],
                               [40, 50, 60]], dtype=np.float32)
    color_vals_gpu = cp.asarray(color_vals_cpu)

    delta = 0.01
    coeffs = solve_coeffs_gpu(kd_gpu, color_vals_gpu, delta)

    assert coeffs.shape == (2, 3)

    # Check that (KD + delta*N*I)*coeffs ~ color_vals within tolerance
    n_points = kd_gpu.shape[0]
    matrix_a = kd_gpu + (delta * n_points) * cp.eye(n_points,
                                                    dtype=cp.float32)
    color_vals_est = matrix_a @ coeffs
    assert cp.allclose(color_vals_est, color_vals_gpu, atol=1e-3)
