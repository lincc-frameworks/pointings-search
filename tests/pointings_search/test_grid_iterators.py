import numpy as np

from pointings_search.grid_iterators import DistanceGridIterator, GridIterator


def test_grid_iterator():
    itr = GridIterator([1.0, 2.0, -1.0], [11.0, 7.0, 10.0], [5, 5, 5])

    # Manually check the first two points.
    first_point = next(itr)
    assert first_point[0] == 1.0
    assert first_point[1] == 2.0
    assert first_point[2] == -1.0

    next_point = next(itr)
    assert next_point[0] == 1.0
    assert next_point[1] == 2.0
    assert next_point[2] == 1.75

    # Check the bounds of the rest and that we get the full number of points.
    count = 2
    for point in itr:
        count += 1
        assert point[0] >= 1.0
        assert point[1] >= 2.0
        assert point[2] >= -1.0

        assert point[0] <= 11.0
        assert point[1] <= 7.0
        assert point[2] <= 10.0
    assert count == 125


def test_distance_iterator():
    itr = DistanceGridIterator(5.0, 20.0, 10)

    # Check the distances of the sample points.
    for point in itr:
        dist = np.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
        assert dist <= 20.0
        assert dist >= 5.0
