import cupy as cp


def conjugate_gradient(matrix_a, vector_b, tol=1e-6, max_iter=1000):
    """
    Solve matrix_a * x = vector_b using the Conjugate Gradient method.

    Parameters
    ----------
    matrix_a : cp.ndarray (N, N)
        System matrix (assumed symmetric positive-definite).
    vector_b : cp.ndarray (N,)
        Right-hand side vector.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    cp.ndarray
        Approximate solution x of shape (N,).
    """
    x = cp.zeros_like(vector_b)
    r = vector_b - matrix_a @ x
    p = r.copy()
    rs_old = cp.dot(r, r)

    for _ in range(max_iter):
        ap = matrix_a @ p
        alpha = rs_old / cp.dot(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = cp.dot(r, r)
        if cp.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def solve_coeffs_gpu(kernel_matrix, color_values, delta):
    """
    Solve (kernel_matrix + delta*N*I) * coeffs = color_values for each channel.

    Parameters
    ----------
    kernel_matrix : cp.ndarray (N, N)
        Spatial kernel matrix.
    color_values : cp.ndarray (N, 3)
        Color values at the selected points (float32).
    delta : float
        Regularization parameter.

    Returns
    -------
    cp.ndarray
        Coefficients of shape (N, 3).
    """
    n_points = kernel_matrix.shape[0]
    matrix_a = kernel_matrix + (delta * n_points) * cp.eye(n_points,
                                                           dtype=cp.float32)
    coeffs = cp.zeros((n_points, 3), dtype=cp.float32)

    for channel_idx in range(3):
        b_channel = color_values[:, channel_idx]
        x_channel = conjugate_gradient(matrix_a, b_channel)
        coeffs[:, channel_idx] = x_channel

    return coeffs
