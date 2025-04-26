"""
Unit tests for the MRF-based colorization subpackage.
"""

import unittest
import numpy as np
import scipy.sparse as sp

from image_color_recovery.algorithms.mrf import builder, solver, utils


class TestMRFBuilder(unittest.TestCase):
    def test_build_mrf_system_known_pixels(self):
        """
        Check known pixels produce diagonal = 1 and correct b_vector.
        """
        height, width = 2, 2
        lightness_channel = np.full((height, width), 50.0, dtype=np.float32)
        channel_known = np.array([[0.0, np.nan],
                                  [np.nan, 1.0]], dtype=np.float32)

        matrix_csr, b_vec = builder.build_mrf_system(
            lightness_channel, channel_known,
            height, width, sigma=5.0
        )

        self.assertEqual(matrix_csr.shape, (4, 4))
        self.assertEqual(b_vec.shape, (4,))

        # Known pixels => row i has diagonal=1, b_vec[i] = known_value
        # top-left (row=0,col=0) => 0.0, bottom-right => 1.0
        # Linear index: (0,0) -> 0, (1,1) -> 3
        self.assertAlmostEqual(b_vec[0], 0.0, places=6)
        self.assertAlmostEqual(b_vec[3], 1.0, places=6)

        # Check diagonal for known pixels
        # matrix_csr[0,0] => should be 1
        # matrix_csr[3,3] => should be 1
        self.assertAlmostEqual(matrix_csr[0, 0], 1.0, places=6)
        self.assertAlmostEqual(matrix_csr[3, 3], 1.0, places=6)

    def test_build_mrf_system_unknown_pixels(self):
        """
        Check unknown pixels get 1 + epsilon on diagonal.
        """
        height, width = 2, 2
        lightness_channel = np.array([[40.0, 45.0],
                                      [50.0, 55.0]], dtype=np.float32)
        channel_known = np.full((height, width), np.nan, dtype=np.float32)

        matrix_csr, b_vec = builder.build_mrf_system(
            lightness_channel, channel_known,
            height, width, sigma=5.0, epsilon=0.001
        )

        # All pixels are unknown => diagonal = 1 + epsilon
        diag_vals = matrix_csr.diagonal()
        for val in diag_vals:
            self.assertAlmostEqual(val, 1.001, places=6)

        # b_vec should be all zeros
        self.assertTrue(np.allclose(b_vec, 0.0))


class TestMRFSolver(unittest.TestCase):
    def test_solve_channel_direct(self):
        """
        Check direct solver returns an array of the correct shape.
        """
        height, width = 3, 3
        lightness_channel = np.full((height, width), 50.0, dtype=np.float32)
        channel_known = np.zeros((height, width), dtype=np.float32)
        # Mark the center pixel as known=10.0
        channel_known[1, 1] = 10.0
        # Others are unknown => set to NaN
        mask = np.ones_like(channel_known, dtype=bool)
        mask[1, 1] = False
        channel_known[mask] = np.nan

        recovered = solver.solve_channel_direct(lightness_channel, channel_known, sigma=5.0)
        self.assertEqual(recovered.shape, (3, 3))

        # The known pixel should remain ~10
        self.assertAlmostEqual(recovered[1, 1], 10.0, places=5)

    def test_solve_channel_iterative(self):
        """
        Check iterative solver (CG) runs and returns correct shape.
        """
        height, width = 2, 2
        lightness_channel = np.zeros((height, width), dtype=np.float32)
        channel_known = np.array([[5.0, np.nan],
                                  [np.nan, 0.0]], dtype=np.float32)

        recovered, info = solver.solve_channel_iterative(
            lightness_channel, channel_known, sigma=2.0,
            tol=1e-5, maxiter=10
        )
        self.assertEqual(recovered.shape, (2, 2))
        # CG info=0 => successful convergence
        self.assertEqual(info, 0)


class TestMRFUtils(unittest.TestCase):
    def test_compute_mrf_normalized_error(self):
        """
        Check the normalized error function for a trivial difference.
        """
        a_rec = np.zeros((2, 2))
        b_rec = np.zeros((2, 2))
        a_gt = np.ones((2, 2))
        b_gt = np.ones((2, 2))

        # Each channel difference is 1 => squared difference=1
        # For 2 channels => sum=2
        # MSE => mean(2)=2
        # Normalized by 32768 => 2/32768
        expected = 2.0 / 32768.0
        actual = utils.compute_mrf_normalized_error(a_rec, b_rec, a_gt, b_gt)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_compute_mrf_normalized_error_channel(self):
        """
        Check the single-channel error function for a simple difference.
        """
        rec = np.zeros((2, 2))
        gt = np.ones((2, 2))
        # difference => 1 => squared=1 => MSE=1 => normalized by 16384 => 1/16384
        expected = 1.0 / 16384.0
        actual = utils.compute_mrf_normalized_error_channel(rec, gt)
        self.assertAlmostEqual(actual, expected, places=7)


if __name__ == "__main__":
    unittest.main()
