import numpy as np

from pointings_search.pointing_table import PointingTable
from pointings_search.search_algorithms import (
    DistanceGridIterator,
    GridIterator,
    distance_grid_search,
    exhaustive_search,
    helio_project_grid_search,
    helio_project_search,
)


def _make_fake_pointings(num_steps=25):
    ras = []
    decs = []
    obstimes = []
    obstime = 60253.1
    for r in np.linspace(15.0, 275.0, num_steps):
        for d in np.linspace(-85.0, 0.0, num_steps):
            ras.append(r)
            decs.append(d)
            obstimes.append(obstime)
            obstime += 0.01

    data_dict = {
        "ra": ras,
        "dec": decs,
        "obstime": obstimes,
        "fov": [3.0] * len(ras),
        "id": [i for i in range(len(ras))],
    }
    data = PointingTable.from_dict(data_dict)
    return data


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


def test_grid_search_given():
    data = _make_fake_pointings()

    # Search 1/8th of the sky. There should be no matches in this region.
    itr = GridIterator([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10, 10, 10])
    results = exhaustive_search(data, itr)
    assert len(results) == 0

    # Search another 1/8th of the sky. There should be matches in this region.
    itr = GridIterator([0.0, 0.0, 0.0], [10.0, 10.0, -10.0], [10, 10, 10])
    results = exhaustive_search(data, itr)
    assert len(results) > 0


def test_distance_grid_search():
    data = _make_fake_pointings()
    results = distance_grid_search(data, num_steps=25)
    assert len(results) > 0
    assert results[0][0] >= results[-1][0]


def test_helio_project_search():
    """Search for projected points at 40 AU"""
    data = _make_fake_pointings()
    results = helio_project_search(data, est_distance=40.0)
    assert len(results) > 0
    assert results[0][0] >= results[-1][0]


def test_helio_project_grid_search():
    """Search for projected points defined by a bounding box at 5 AU."""
    data = _make_fake_pointings()
    results = helio_project_grid_search(data, est_distance=5.0)
    assert len(results) > 0
    assert results[0][0] >= results[-1][0]
