import numpy as np
from color_recovery.point_selector import generate_random_points


def test_generate_random_points():
    """
    Test generate_random_points for correct shape and quantity of points.
    """
    shape = (10, 10)
    percentage = 10.0
    rows, cols = generate_random_points(shape, percentage)

    total_pixels = shape[0] * shape[1]
    expected_points = int((percentage / 100) * total_pixels)

    assert len(rows) == expected_points
    assert len(cols) == expected_points

    assert (rows >= 0).all() and (rows < shape[0]).all()
    assert (cols >= 0).all() and (cols < shape[1]).all()
