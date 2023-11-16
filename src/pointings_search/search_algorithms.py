"""Functions for performing searches over a PointingTable for good candidates."""

import heapq

import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord, get_body_barycentric

from pointings_search.pointing_table import PointingTable


def grid_search(
    data,
    min_dist=5.0,
    max_dist=50.0,
    num_steps=10,
    min_images=5,
    max_results=100,
):
    """Perform a brute force grid search over barycentric space. Takes in a minimum
    and maximum distance to prune points outside sphere defined by maximum distance
    and inside the sphere defined by minimum distance.

    Current approach is turly brute force and makes no attempt at early pruning.

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    min_dist : `float`
        The minimum distance in AU from the barycenter.
        (default = 5.0)
    max_dist : `float`
        The maximum distance in AU from the barycenter.
        (default = 50.0)
    num_steps : `int`
        The number of steps between min_offset and max_offset. (default=10)
    min_images : `int`
        The minimum number of images to consider this a hit. (default=5)
    max_results : `int`
        The maximum number of results to return. (Default = 100)

    Returns
    -------
    results : `list`
        A sorted list that stores a tuple (num_images, (x, y, z))
        for each match.
    """
    results = []

    for x in np.linspace(-max_dist, max_dist, num_steps):
        for y in np.linspace(-max_dist, max_dist, num_steps):
            for z in np.linspace(-max_dist, max_dist, num_steps):
                # Prune based on distance.
                dist = np.sqrt(x * x + y * y + z * z)
                if dist < min_dist or dist > max_dist:
                    continue

                xyz_res = data.search_heliocentric_xyz([x, y, z])
                if len(xyz_res) >= min_images:
                    heapq.heappush(results, (len(xyz_res), (x, y, z)))
                    if len(results) > max_results:
                        heapq.heappop(results)

    results.sort(reverse=True)
    return results
