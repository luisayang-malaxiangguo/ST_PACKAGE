"""
Solvers for MRF colorization, including direct and iterative methods.
"""

import numpy as np
import scipy.sparse.linalg as spla
from .builder import build_mrf_system


def solve_channel_direct(lightness_channel, channel_known, sigma):
    """
    Solve for a single channel (a or b) using a direct sparse solver.

    Parameters
    ----------
    lightness_channel : np.ndarray, shape (height, width)
        L channel in Lab space.
    channel_known : np.ndarray, shape (height, width)
        Known values for the channel; NaN where unknown.
    sigma : float
        Gaussian parameter for weighting neighbors.

    Returns
    -------
    channel_recovered : np.ndarray, shape (height, width)
        Recovered channel.
    """
    height, width = lightness_channel.shape
    matrix_csr, b_vector = build_mrf_system(
        lightness_channel, channel_known, height, width, sigma
    )
    solution = spla.spsolve(matrix_csr, b_vector)
    channel_recovered = solution.reshape(height, width)
    return channel_recovered


def solve_channel_iterative(
    lightness_channel, channel_known, sigma,
    tol=1e-6, maxiter=200, callback_list=None
):
    """
    Solve for a single channel using Conjugate Gradient (CG), tracking partial solutions.

    Parameters
    ----------
    lightness_channel : np.ndarray, shape (height, width)
        L channel in Lab space.
    channel_known : np.ndarray, shape (height, width)
        Known values for the channel; NaN where unknown.
    sigma : float
        Gaussian parameter for weighting neighbors.
    tol : float, optional
        Tolerance for CG convergence.
    maxiter : int, optional
        Maximum number of CG iterations.
    callback_list : list, optional
        If provided, each iteration's solution vector will be appended here.

    Returns
    -------
    channel_recovered : np.ndarray, shape (height, width)
        Final recovered channel.
    info : int
        CG solver info (0 means successful convergence).
    """
    height, width = lightness_channel.shape
    matrix_csr, b_vector = build_mrf_system(
        lightness_channel, channel_known, height, width, sigma
    )

    def cg_callback(xk):
        if callback_list is not None:
            callback_list.append(xk.copy())

    x0 = np.zeros_like(b_vector)
    x_final, info = spla.cg(
        matrix_csr, b_vector, x0, tol,
        maxiter, callback=cg_callback
    )
    channel_recovered = x_final.reshape(height, width)
    return channel_recovered, info
