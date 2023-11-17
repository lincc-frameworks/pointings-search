"""Functions for performing searches over a PointingTable for good candidates."""

import heapq
import itertools

import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord, get_body_barycentric

from pointings_search.pointing_table import PointingTable


class GridIterator:
    """Iterate over a uniform grid of points in D-dimensional space.

    Attributes
    ----------
    coord_start : `list`, `tuple`, or `array`
        A length D data structure holding the starting coordinates for
        x, y, and z respectively.
    coord_end : `list`, `tuple`, or `array`
        A length D data structure holding the ending coordinates for
        x, y, and z respectively.
    num_steps : `list`, `tuple`, or `array`
        A length D data structure holding the number of steps between
        coord_start and coord_end.
    """

    def __init__(self, coord_start, coord_end, num_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_dim = len(coord_start)
        if self.num_dim != len(coord_end) or self.num_dim != len(num_steps):
            raise ValueError("Incompatible dimensions.")

        # Create a single product iterator over all the dimensions.
        self.iter = itertools.product(
            *[np.linspace(coord_start[i], coord_end[i], num_steps[i]) for i in range(self.num_dim)]
        )

    def __iter__(self):
        return self

    def __next__(self):
        """Get next value from the iterator.

        Returns
        -------
        `list`
            Returns a D-dimensional list representing a sample point.
        """
        return next(self.iter)


class DistanceGridIterator(GridIterator):
    """Iterate over a full grid of points with distance limits. Performs a brute force grid
    search over 3-d cartesian barycentric space. Takes in a minimum and maximum distance
    to prune points outside sphere defined by maximum distance and inside the sphere defined
    by minimum distance.

    Attributes
    ----------
    min_dist : `float`
        The minimum distance in AU from the barycenter.
        (default = 5.0)
    max_dist : `float`
        The maximum distance in AU from the barycenter.
        (default = 50.0)
    num_steps : `int`
        The number of steps between min_offset and max_offset. (default=10)
    """

    def __init__(self, min_dist=5.0, max_dist=50.0, num_steps=10, *args, **kwargs):
        super().__init__(
            [-max_dist] * 3,  # coord_start
            [max_dist] * 3,  # coord_end
            [num_steps] * 3,  # num_steps
            *args,
            **kwargs,
        )
        self.min_dist = min_dist
        self.max_dist = max_dist

    def __next__(self):
        """Get next value from the iterator.

        Returns
        -------
        `list`
            Returns a 3-dimensional list [x, y, z] representing a sample point.
        """
        # Relies on the GridIterator's __next__() function to compute the
        # point, increment the counters, and call StopIteration.
        current_step = super().__next__()
        dist = np.sqrt(
            current_step[0] * current_step[0]
            + current_step[1] * current_step[1]
            + current_step[2] * current_step[2]
        )

        # Keep skipping points outside the distance range.
        while dist < self.min_dist or dist > self.max_dist:
            current_step = super().__next__()
            dist = np.sqrt(
                current_step[0] * current_step[0]
                + current_step[1] * current_step[1]
                + current_step[2] * current_step[2]
            )
        return current_step


def exhaustive_search(data, options, min_images=5, max_results=100):
    """Perform a linear search over the options defined by a BaseSearchIterator.

    Current approach is turly brute force and makes no attempt at early pruning.

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    options : iterable
        The iterator over search options to test.
    min_images : `int`
        The minimum number of images to consider this a hit. (default=5)
    max_results : `int`
        The maximum number of results to return. (Default = 100)

    Returns
    -------
    results : `list`
        A sorted list that stores a tuple (num_images, [x, y, z])
        for each match.
    """
    results = []
    for point in options:
        xyz_res = data.search_heliocentric_xyz(point)
        if len(xyz_res) >= min_images:
            heapq.heappush(results, (len(xyz_res), point))
            if len(results) > max_results:
                heapq.heappop(results)

    results.sort(reverse=True)
    return results


def distance_grid_search(
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

    Current approach is truly brute force and makes no attempt at early pruning.

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
        A sorted list that stores a tuple (num_images, [x, y, z])
        for each match.
    """
    options = DistanceGridIterator(min_dist, max_dist, num_steps)
    return exhaustive_search(data, options, min_images, max_results)


def helio_project_search(
    data,
    est_distance=10.0,
    num_steps=10,
    min_images=5,
    max_results=100,
):
    """Project the pointings into 3-d points in barycentric space (given an estimated
    distance) and use those as search points.

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    est_distance : `float`
        The estimated geocentric distance of a potential object in AU.
        (default = 10.0)
    num_steps : `int`
        The number of steps between min_offset and max_offset. (default=10)
    min_images : `int`
        The minimum number of images to consider this a hit. (default=5)
    max_results : `int`
        The maximum number of results to return. (Default = 100)

    Returns
    -------
    results : `list`
        A sorted list that stores a tuple (num_images, [x, y, z])
        for each match.
    """
    data.preprocess_pointing_info()
    data.append_earth_pos()

    # Compute the barycentric cartesian points of of the estimated object positions.
    bary_pts = np.array(
        [
            est_distance * data.pointings["unit_vec_x"] + data.pointings["earth_vec_x"],
            est_distance * data.pointings["unit_vec_y"] + data.pointings["earth_vec_y"],
            est_distance * data.pointings["unit_vec_z"] + data.pointings["earth_vec_z"],
        ]
    ).T
    return exhaustive_search(data, bary_pts.tolist(), min_images, max_results)


def helio_project_grid_search(
    data,
    est_distance=10.0,
    num_steps=10,
    min_images=5,
    max_results=100,
):
    """Project the pointings into 3-d points in barycentric space (given an estimated
    distance) and use those to define bounds for a grid search.

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    est_distance : `float`
        The estimated geocentric distance of a potential object in AU.
        (default = 10.0)
    num_steps : `int`
        The number of steps between min_offset and max_offset. (default=10)
    min_images : `int`
        The minimum number of images to consider this a hit. (default=5)
    max_results : `int`
        The maximum number of results to return. (Default = 100)

    Returns
    -------
    results : `list`
        A sorted list that stores a tuple (num_images, [x, y, z])
        for each match.
    """
    data.preprocess_pointing_info()
    data.append_earth_pos()

    # Compute the barycentric cartesian points of of the estimated object positions.
    x_pts = est_distance * data.pointings["unit_vec_x"] + data.pointings["earth_vec_x"]
    y_pts = est_distance * data.pointings["unit_vec_y"] + data.pointings["earth_vec_y"]
    z_pts = est_distance * data.pointings["unit_vec_z"] + data.pointings["earth_vec_z"]
    itr = GridIterator(
        [np.min(x_pts), np.min(y_pts), np.min(z_pts)],
        [np.max(x_pts), np.max(y_pts), np.max(z_pts)],
        [num_steps] * 3,
    )

    return exhaustive_search(data, itr, min_images, max_results)
