"""A spatial data structure for storing and querying vectors representing
pointing data."""

import numpy as np

from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy import units as u

from pointings_search.pointing_table import PointingTable


class SearchStats:
    """A data class to store the search statistics

    Attributes
    ----------
    nodes_checked : `int`
        The number of nodes tested.
    points_checked : `int`
        The number of points tested.
    """

    def __init__(self):
        self.nodes_checked = 0
        self.points_checked = 0


class PointingTree:
    """A PointingTree is a kd-tree over data with data with the following
    columns:
        0: index in original table
        1: earth_vec_x
        2: earth_vec_y
        3: earth_vec_z
        4: unit_vec_x
        5: unit_vec_y
        6: unit_vec_z
        7: field of view [optional]

    Attributes
    ----------
    pointing : `numpy.ndarray` or None
        The array of minimal pointing information. Set to None for internal nodes
        and non-None for leaf nodes.
    num_points : `int`
        The number of points in this tree.
    pos_center : `numpy.ndarray`
        A 3d array with the center of the position (earth vectors).
    pos_radius : `float`
        The radius of the node's spatial footprint: the maximum distance
        from pos_center to any earth_vec in the node.
    low_bnd : `numpy.ndarray`
        The lowest value in each column of the points in this tree.
    high_bnd : `numpy.ndarray`
        The highest value in each column of the points in this tree.
    view_center : `numpy.ndarray`
        A 3d array representing the "central pointing" of the node (a unit vector).
    view_radius : `numpy.ndarray`
        The radius of the node's viewing cone: the maximum angular distance
        from view_center to any unit_vec in the node.
    left_child : `PointingTree`
        A reference to the left subtree (all values where
        pointing[split_col] < split_value). None for leaf nodes.
    right_child : `PointingTree`
        A reference to the right subtree (all values where
        pointing[split_col] >= split_value). None for leaf nodes.
    split_col : `int`
        For debugging: The column along which this node is split. -1 for leaf nodes.
    split_value : `float`
        for debugging: The value along which the node is split.
    """

    def __init__(self, pointings):
        if pointings.shape[1] < 7:
            raise ValueError("Incorrect shape for pointing tree data: {pointings.shape}")

        # Set the node up as a leaf with no children and all the pointings.
        self.pointings = pointings
        self.num_points = len(pointings)
        self.left_child = None
        self.right_child = None
        self.split_col = -1
        self.split_value = 0.0

        # Compute the center of the positions and the viewing cone. Make the center of
        # the viewing cone a unit vector handling the edge case where the average is exactly zero.
        self.low_bnd = np.min(pointings, axis=0)
        self.high_bnd = np.max(pointings, axis=0)
        self.pos_center = 0.5 * (self.high_bnd[1:4] + self.low_bnd[1:4])
        self.view_center = 0.5 * (self.high_bnd[4:7] + self.low_bnd[4:7])
        if self.view_center[0] == 0.0 and self.view_center[1] == 0.0 and self.view_center[2] == 0.0:
            self.view_center[0] == 1.0
        else:
            normalizer = np.sqrt(np.sum(np.square(self.view_center)))
            self.view_center = self.view_center / normalizer

        # Compute the radius of the node's positional footprint.
        distances = np.sqrt(np.sum(np.square(self.pointings[:, 1:4] - self.pos_center), axis=1))
        self.pos_radius = np.max(distances)

        # Compute the angular radius of the node's viewing cone (in degrees).
        # Since we are only dealing with unit vectors the magnitudes are all 1
        # Clip to 1.0 to avoid problems with numerical precision.
        dots = np.dot(self.pointings[:, 4:7], self.view_center)
        dots[dots > 1.0] = 1.0
        distances = np.arccos(dots) * (180.0 / np.pi)

        if pointings.shape[1] == 8:
            # Add the field of views to the distances before computing the max.
            distances += pointings[:, 7]

        self.view_radius = np.max(distances)

    def recursive_split_dist(self, effective_dist=1.0, max_points=10, min_width=1e-6):
        """Split the node to break up the amount of space covered. Project the points
        out to an effective_distance and use that for partitioning.

        Parameters
        ----------
        effective_dist : `float`
            The effective distance of the search.
        max_points : `int`
            The maximum number of points to include in a leaf node.
        min_width : `float`
            The minimal normalized width in each dimension.

        Returns
        -------
        A Boolean indicating whether the node was split.
        """
        # Do not split an internal node.
        if self.pointings is None:
            return False

        # Check the stopping criteria based on number of points.
        if self.num_points <= max_points:
            return False

        # Project the points out to their effective distances and compute the resulting bounding box.
        projected = self.pointings[:, 1:4] + effective_dist * self.pointings[:, 4:7]
        prj_low = np.min(projected, axis=0)
        prj_high = np.max(projected, axis=0)

        # Compute the widest dimension and check that stopping criteria.
        widths = prj_high - prj_low
        if np.max(widths) < min_width:
            return False

        # Compute the split.
        self.split_col = np.argmax(widths)
        self.split_value = 0.5 * (prj_high[self.split_col] + prj_low[self.split_col])
        right_pointings = self.pointings[projected[:, self.split_col] >= self.split_value]
        left_pointings = self.pointings[projected[:, self.split_col] < self.split_value]

        # Remove the local copy of the pointings. Need to do this before creating the children
        # to avoid exploding memory.
        self.pointings = None

        # Create a right child.
        self.right_child = PointingTree(right_pointings)
        self.right_child.recursive_split_dist(effective_dist, max_points, min_width)

        # Create a left child.
        self.left_child = PointingTree(left_pointings)
        self.left_child.recursive_split_dist(effective_dist, max_points, min_width)

        return True

    def recursive_split_kd(self, max_widths, max_points=10, min_width=1e-6):
        """Split the node along the 'widest' spatial column (ignoring the index column)
        where each dimension is normalized by max_widths. We recursively
        split the node until either it has fewer than max_points or is smaller than
        min_width in each dimension.

        Parameters
        ----------
        max_widths : `numpy.ndarray`
            A length 6 - numpy array with the normalization factors for each column,
            including the index column.
        max_points : `int`
            The maximum number of points to include in a leaf node.
        min_width : `float`
            The minimal normalized width in each dimension.

        Returns
        -------
        A Boolean indicating whether the node was split.
        """
        # Do not split an internal node.
        if self.pointings is None:
            return False

        # Check the stopping criteria based on number of points.
        if self.num_points <= max_points:
            return False

        # Compute the widest dimension and check that stopping criteria.
        widths = np.divide(self.high_bnd[1:7] - self.low_bnd[1:7], max_widths)
        if np.max(widths) < min_width:
            return False

        # Compute the split.
        self.split_col = np.argmax(widths) + 1
        self.split_value = 0.5 * (self.high_bnd[self.split_col] + self.low_bnd[self.split_col])
        right_pointings = self.pointings[self.pointings[:, self.split_col] >= self.split_value]
        left_pointings = self.pointings[self.pointings[:, self.split_col] < self.split_value]

        # Remove the local copy of the pointings. Need to do this before creating the children
        # to avoid exploding memory.
        self.pointings = None

        # Create a right child.
        self.right_child = PointingTree(right_pointings)
        self.right_child.recursive_split_kd(max_widths, max_points, min_width)

        # Create a left child.
        self.left_child = PointingTree(left_pointings)
        self.left_child.recursive_split_kd(max_widths, max_points, min_width)

        return True

    def prune(self, target, fov=0.0):
        """Determines whether or not to prune the node based on whether
        any pointing in this subtree could see the target point.

        Parameters
        ----------
        target : `numpy.ndarray`
            A length 3 numpy array indicating a (x, y, z) point.
        fov : `float`
            The maximum field of view for the pointings in degrees.
            Should be zero if per-pointing FOV was included in the input data.

        Returns
        -------
        prune : `bool`
            A Boolean indicating whether to prune the node.
        """
        # Skip the pruning test if the angular radius is too large. Avoids problems where the
        # distance along the ray can be negative because the cone is over 180 degrees wide.
        if self.view_radius >= 90.0:
            return False

        # If the target point is in the positional sphere, do not prune.
        dist_vect = self.pos_center - target
        dist_to_center = np.sqrt(np.dot(dist_vect, dist_vect))
        if dist_to_center <= self.pos_radius:
            return False

        # Using a cone coming from the target point and centered on the *inverse* of
        # the node's central ray: X = P - alpha * view_center
        # compute how far along the ray is the closest point Q to the center.
        dist_along_ray = np.dot(dist_vect, -self.view_center)
        if dist_along_ray < 0:
            return True
        pt_Q = target - dist_along_ray * self.view_center

        # If point Q falls within the node's positional sphere, don't prune.
        dist_vect2 = pt_Q - self.pos_center
        dist_to_center2 = np.sqrt(np.dot(dist_vect2, dist_vect2))
        if dist_to_center2 <= self.pos_radius:
            return False

        # Compute the closest point C on the node's positional sphere to point Q.
        unit_vect = dist_vect2 / np.sqrt(np.dot(dist_vect2, dist_vect2))
        pt_C = self.pos_center + self.pos_radius * unit_vect

        # Check whether the angle from the target point to point C is within the cone.
        TC_vect = pt_C - target
        unit_TC = TC_vect / np.sqrt(np.dot(TC_vect, TC_vect))
        ang_dist = np.arccos(np.dot(unit_TC, -self.view_center)) * 180.0 / np.pi
        return ang_dist > self.view_radius + fov

    def search_heliocentric_xyz(self, target, fov=0.0, stats=None):
        """Search for pointings that would overlap a given heliocentric
        point (x, y, z). Allows a single field of view or per pointing
        field of views.

        Parameters
        ----------
        point : tuple, list, or array
            The point represented as (x, y, z) on which to compute the distances.
        fov : `float` (optional)
            The field of view of the individual pointings. If None
            tries to retrieve from table.
        stats : `SearchStats` (optional)
            A data structure for performance statistics on the search.

        Returns
        -------
        An astropy table with information for the matching pointings.

        Raises
        ------
        ValueError if no field of view is provided.
        """
        if stats is not None:
            stats.nodes_checked += 1

        # Test if we can prune the node.
        if self.prune(target, fov):
            return []

        # Recursively explore children
        if self.left_child is not None:
            r_res = self.right_child.search_heliocentric_xyz(target, fov, stats)
            l_res = self.left_child.search_heliocentric_xyz(target, fov, stats)
            if len(r_res) == 0:
                return l_res
            if len(l_res) == 0:
                return r_res
            return np.append(r_res, l_res, axis=0)

        # Compute the geocentric cartesian positions of the point.
        geo_pts = target - self.pointings[:, 1:4]

        magnitudes = np.sqrt(np.sum(np.square(geo_pts), axis=1))

        dot_products = np.sum(np.multiply(geo_pts, self.pointings[:, 4:7]), axis=1)

        dist = np.arccos(np.divide(dot_products, magnitudes)) * (180.0 / np.pi)

        # Compute the angular distance from the dot product of the vectors. This can be slightly
        # inaccurate very close to 0, but is sufficient for pruning. Since we are using the vectors,
        # (instead of RA, dec) we do not need to worry about the poles.
        # norm = geo_pts.norm()
        # dot = geo_pts.dot(pointing_pts)
        # dist = np.arccos(dot / norm).to(u.deg)

        if fov > 0.0:
            res = self.pointings[dist <= fov, :]
        elif self.pointings.shape[1] == 8:
            res = self.pointings[dist <= self.pointings[:, 7], :]
        else:
            raise ValueError("No field of view provided.")

        if stats is not None:
            stats.points_checked += 1

        return res

    def search_heliocentric_pointing(self, point, fov=0.0, stats=None):
        """Search for pointings that would overlap a given heliocentric
        pointing and estimated distance. Allows a single field of view
        or per pointing field of views.

        Parameters
        ----------
        point : `astropy.coordinates.SkyCoord`
            a barycentric pointing with at least RA, dec, and distance.
        fov : `float` (optional)
            The field of view of the individual pointings. If None
            tries to retrieve from table.
        stats : `SearchStats` (optional)
            A data structure for performance statistics on the search.

        Returns
        -------
        An astropy table with information for the matching pointings.

        Raises
        ------
        ValueError if no field of view is provided.
        """
        cart_pt = point.cartesian
        helio_pt = [cart_pt.x.value, cart_pt.y.value, cart_pt.z.value]
        return self.search_heliocentric_xyz(helio_pt, fov, stats)


def build_pointing_tree(data, effective_dist=-1.0, max_points=10, min_width=1e-6):
    """Create a PointingTree from a PointingTable

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    effective_dist : `float`
        The effective distance to use. For any value < 0.0 use the weighted dimension splitting.
        For 0.0 use the unweighted dimensional splitting.
    max_points : `int`
        The maximum number of points to include in a leaf node.
    min_width : `float`
        The minimal normalized width in each dimension.

    Returns
    -------
    root : `PointingTree`
        The root node of the resulting pointing tree.

    Raises
    ------
    KeyError if either the Earth's location data or the pointing unit vector data
    is missing from the PointingTable.
    """
    if "earth_vec_x" not in data.pointings.columns:
        raise KeyError("PointingTable missing Earth data. Call append_earth_pos()")
    if "unit_vec_x" not in data.pointings.columns:
        raise KeyError("PointingTable missing pointing data. Call preprocess_pointing_info()")

    # Create the numpy array of the data for the tree.
    if "fov" in data.pointings.columns:
        arr_data = np.array(
            [
                np.arange(0, len(data)),
                data.pointings["earth_vec_x"].value,
                data.pointings["earth_vec_y"].value,
                data.pointings["earth_vec_z"].value,
                data.pointings["unit_vec_x"].value,
                data.pointings["unit_vec_y"].value,
                data.pointings["unit_vec_z"].value,
                data.pointings["fov"].value,
            ]
        )
    else:
        arr_data = np.array(
            [
                np.arange(0, len(data)),
                data.pointings["earth_vec_x"].value,
                data.pointings["earth_vec_y"].value,
                data.pointings["earth_vec_z"].value,
                data.pointings["unit_vec_x"].value,
                data.pointings["unit_vec_y"].value,
                data.pointings["unit_vec_z"].value,
            ]
        )

    # Allocate the tree root.
    root = PointingTree(arr_data.T)

    # Compute the normalization factors and split the tree.
    if effective_dist <= 0.0:
        if effective_dist == 0.0:
            max_widths = np.array([1.0] * 6)
        else:
            max_widths = root.high_bnd[1:7] - root.low_bnd[1:7]
        root.recursive_split_kd(max_widths, max_points, min_width)
    else:
        root.recursive_split_dist(effective_dist, max_points, min_width)

    return root
