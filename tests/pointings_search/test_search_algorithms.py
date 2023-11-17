import numpy as np

from pointings_search.pointing_table import PointingTable
from pointings_search.search_algorithms import grid_search


def test_grid_search():
    # Generate fake pointings as a sweep a patch of the southern
    ras = []
    decs = []
    obstimes = []
    obstime = 60253.1
    for r in np.linspace(15.0, 275.0, 25):
        for d in np.linspace(-85.0, 0.0, 25):
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

    # Search for results on a smallish grid.
    results = grid_search(data, num_steps=25)
    assert len(results) > 0
    assert results[0][0] >= results[-1][0]
