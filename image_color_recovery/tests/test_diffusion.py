import unittest
import numpy as np
import cupy as cp
from image_color_recovery.algorithms.diffusion import builder, solver


class TestDiffusion(unittest.TestCase):
    def test_builder(self):
        gray_cpu = np.array([[0.0, 0.5], [0.2, 0.7]], dtype=np.float32)
        gray_gpu = cp.asarray(gray_cpu)
        neighbors, weights, h, w = builder.build_neighbor_indices_and_weights(
            gray_gpu, sigma=0.1
        )
        self.assertEqual(h, 2)
        self.assertEqual(w, 2)

    def test_solver(self):
        gray_cpu = np.zeros((2, 2), dtype=np.float32)
        gray_gpu = cp.asarray(gray_cpu)
        known_mask_cpu = np.array([[True, False], [False, False]])
        known_mask_gpu = cp.asarray(known_mask_cpu)
        known_vals_cpu = np.zeros((2, 2, 3), dtype=np.float32)
        known_vals_cpu[0, 0] = [1.0, 0.0, 0.0]
        known_vals_gpu = cp.asarray(known_vals_cpu)
        original_3ch_cpu = np.zeros((2, 2, 3), dtype=np.float32)
        original_3ch_cpu[0, 0] = [1.0, 0.0, 0.0]
        original_3ch_gpu = cp.asarray(original_3ch_cpu)

        recovered_gpu, errors = solver.pde_colorize_3ch_with_error(
            gray_gpu,
            known_mask_gpu,
            known_vals_gpu,
            original_3ch_gpu,
            sigma=0.1,
            max_iters=5
        )
        self.assertEqual(recovered_gpu.shape, (2, 2, 3))
        self.assertEqual(len(errors), 5)


if __name__ == "__main__":
    unittest.main()
