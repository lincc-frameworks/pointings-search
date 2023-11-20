"""Classes for iterating over grids of coordinates Uses for the various grid searches."""

import itertools

import numpy as np


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
