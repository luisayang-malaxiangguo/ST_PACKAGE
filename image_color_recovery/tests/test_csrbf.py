"""
Unit tests for the CSRBF interpolation algorithm.
"""

import unittest
import numpy as np
import cupy as cp
from image_color_recovery.algorithms.csrbf import kernel, solver, recovery
from image_color_recovery.algorithms.csrbf.config import DEFAULT_PARAMS


class TestCSRBF(unittest.TestCase):

    def test_generate_kernel_matrix_gpu_shape(self):
        # Create a dummy grayscale image and a few random points.
        gray_image = np.full((100, 100), 128, dtype=np.uint8)
        sample_indices = (np.array([10, 20, 30]), np.array([10, 20, 30]))

        kernel_matrix = kernel.generate_kernel_matrix_gpu(
            sample_indices, gray_image,
            DEFAULT_PARAMS["sigma1"],
            DEFAULT_PARAMS["sigma2"],
            DEFAULT_PARAMS["p"]
        )
        # Expect a 3x3 kernel matrix
        self.assertEqual(kernel_matrix.shape, (3, 3))

    def test_solver_output_shape(self):
        # Create a simple kernel matrix and test solver output shape.
        dummy_kernel = cp.eye(3, dtype=cp.float32)
        # Create dummy color values at the sample points
        sampled_colors = cp.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ], dtype=cp.float32)

        coefficients = solver.solve_coefficients_gpu_direct(
            dummy_kernel, sampled_colors, DEFAULT_PARAMS["delta"]
        )
        self.assertEqual(coefficients.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
