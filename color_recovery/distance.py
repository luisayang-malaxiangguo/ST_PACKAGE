import cupy as cp


def cdist_gpu_chunked(points_a, points_b, chunk_size=4000):
    """
    Compute pairwise Euclidean distances on the GPU in chunks.

    Parameters
    ----------
    points_a : cp.ndarray (m, d)
        First set of points (float32).
    points_b : cp.ndarray (n, d)
        Second set of points (float32).
    chunk_size : int, optional
        Number of rows of points_a to process at once.

    Returns
    -------
    cp.ndarray
        Array of shape (m, n) containing pairwise distances.
    """
    m = points_a.shape[0]
    n = points_b.shape[0]
    dists = cp.zeros((m, n), dtype=cp.float32)
    b_sq = cp.sum(points_b**2, axis=1).reshape(1, -1)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        a_chunk = points_a[start:end]
        a_sq = cp.sum(a_chunk**2, axis=1).reshape(-1, 1)
        ab = a_chunk @ points_b.T
        dist_chunk = cp.sqrt(a_sq - 2 * ab + b_sq)
        dists[start:end, :] = dist_chunk

    return dists
