import numpy as np

from pointings_search.grid_iterators import GridIterator
from pointings_search.pointing_table import PointingTable
from pointings_search.search_algorithms import (
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
