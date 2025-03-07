"""
Utility functions for MRF colorization, including error metrics.
"""

import numpy as np


def compute_mrf_normalized_error(a_rec, b_rec, a_gt, b_gt):
    """
    Compute a normalized error comparing (a_rec, b_rec) to (a_gt, b_gt).

    Each channel (a, b) is in [-128, 128], so max difference = 256 for one channel,
    meaning the max squared difference for 2 channels is 2*(128^2) = 32768.

    Parameters
    ----------
    a_rec : np.ndarray, shape (height, width)
        Recovered a channel.
    b_rec : np.ndarray, shape (height, width)
        Recovered b channel.
    a_gt : np.ndarray, shape (height, width)
        Ground truth a channel.
    b_gt : np.ndarray, shape (height, width)
        Ground truth b channel.

    Returns
    -------
    normalized_error : float
        MSE / 32768.
    """
    diff_a = a_rec - a_gt
    diff_b = b_rec - b_gt
    raw_mse = np.mean(diff_a**2 + diff_b**2)
    max_sq_diff_2_channels = 2 * (128**2)  # 32768
    return raw_mse / max_sq_diff_2_channels


def compute_mrf_normalized_error_channel(rec_channel, gt_channel):
    """
    Compute normalized error for a single channel in [-128, 128].
    Max squared difference for one channel = 128^2 = 16384.

    Parameters
    ----------
    rec_channel : np.ndarray
        Recovered channel.
    gt_channel : np.ndarray
        Ground truth channel.

    Returns
    -------
    float
        Normalized error in [0, 1].
    """
    raw_mse = np.mean((rec_channel - gt_channel)**2)
    max_sq_diff = 128**2  # 16384
    return raw_mse / max_sq_diff
