"""
Build the sparse linear system for MRF colorization.
"""

import numpy as np
import scipy.sparse as sp


def build_mrf_system(
    lightness_channel, channel_known, height, width,
    sigma, epsilon=0.001
):
    """
    Build the sparse system A*x = b for one color channel in MRF colorization.

    Known pixel i => row i: x_i = known_value
    Unknown pixel i => (1+epsilon)*x_i - sum_j(weights)*x_j = 0

    Parameters
    ----------
    lightness_channel : np.ndarray, shape (height, width)
        The L channel in Lab space (used to compute neighbor weights).
    channel_known : np.ndarray, shape (height, width)
        Contains known color values for this channel (a or b), or NaN for unknown.
    height : int
        Image height.
    width : int
        Image width.
    sigma : float
        Gaussian parameter for weighting neighbors.
    epsilon : float, optional
        Small constant added to the diagonal to stabilize the system.

    Returns
    -------
    matrix_csr : scipy.sparse.csr_matrix of shape (height*width, height*width)
        Sparse matrix representing MRF constraints.
    b_vector : np.ndarray of shape (height*width,)
        Right-hand side vector.
    """
    num_pixels = height * width
    matrix_lil = sp.lil_matrix((num_pixels, num_pixels), dtype=np.float64)
    b_vector = np.zeros(num_pixels, dtype=np.float64)

    def linear_index(row, col):
        return row * width + col

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(height):
        for col in range(width):
            row_id = linear_index(row, col)
            if not np.isnan(channel_known[row, col]):
                # Known pixel => x_i = known_value
                matrix_lil[row_id, row_id] = 1.0
                b_vector[row_id] = channel_known[row, col]
            else:
                # Unknown pixel => Weighted average with neighbors
                matrix_lil[row_id, row_id] = 1.0 + epsilon
                weights = []
                neighbors = []
                center_lightness = lightness_channel[row, col]
                for d_row, d_col in directions:
                    nbr_row = row + d_row
                    nbr_col = col + d_col
                    if 0 <= nbr_row < height and 0 <= nbr_col < width:
                        diff = center_lightness - lightness_channel[nbr_row, nbr_col]
                        weight_val = np.exp(- (diff**2) / (2 * sigma**2))
                        weights.append(weight_val)
                        neighbors.append(linear_index(nbr_row, nbr_col))

                total_weight = np.sum(weights)
                if total_weight > 1e-12:
                    weights = np.array(weights, dtype=np.float64) / total_weight
                for w_val, nbr_idx in zip(weights, neighbors):
                    matrix_lil[row_id, nbr_idx] -= w_val

                b_vector[row_id] = 0.0

    matrix_csr = matrix_lil.tocsr()
    return matrix_csr, b_vector
