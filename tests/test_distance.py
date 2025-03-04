import pytest
import cupy as cp
import numpy as np
from color_recovery.distance import cdist_gpu_chunked


def test_cdist_gpu_chunked_small():
    """
    Test cdist_gpu_chunked with small input arrays.
    """
    array_a_cpu = np.array([[0, 0], [3, 4]], dtype=np.float32)
    array_b_cpu = np.array([[0, 0], [6, 8]], dtype=np.float32)

    array_a_gpu = cp.asarray(array_a_cpu)
    array_b_gpu = cp.asarray(array_b_cpu)

    # chunk_size=1 to force chunked processing
    dists_gpu = cdist_gpu_chunked(array_a_gpu, array_b_gpu, chunk_size=1)
    dists_cpu = dists_gpu.get()

    # Expected distances:
    # (0,0)->(0,0) = 0
    # (0,0)->(6,8) = sqrt(36+64)=10
    # (3,4)->(0,0) = 5
    # (3,4)->(6,8) = sqrt(3^2+4^2)=5
    expected = np.array([[0, 10],
                         [5,  5]], dtype=np.float32)

    assert dists_cpu.shape == (2, 2)
    assert np.allclose(dists_cpu, expected, atol=1e-5)


def test_cdist_gpu_chunked_empty():
    """
    Test cdist_gpu_chunked with empty arrays.
    """
    array_a_gpu = cp.zeros((0, 2), dtype=cp.float32)
    array_b_gpu = cp.zeros((0, 2), dtype=cp.float32)

    dists_gpu = cdist_gpu_chunked(array_a_gpu, array_b_gpu)
    assert dists_gpu.shape == (0, 0)
