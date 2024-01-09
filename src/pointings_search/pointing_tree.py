"""A spatial data structure for storing and querying vectors representing
pointing data. Uses a hybrid approach between a kd-tree (for splitting)
and a ball tree (for bounds and pruning)."""

import math
import numpy as np

from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy import units as u

from pointings_search.pointing_table import PointingTable


class PointingTree:
    """A PointingTree is a kd-tree over 3-d pointing vectors defined by
        - A 3-dimensional vector of vertex position (the position of the observatory).
        - A 3-dimensional unit vector indicating the pointing direction
        - A pointing field of view (in degrees).

    Attributes
    ----------
    root : `PointingTreeNode`
        The root of the tree.
    search_nodes_checked : `int`
        A stats counter used for performance monitoring
    search_points_checked : `int`
        A stats counter used for performance monitoring
    """

    def __init__(self, root):
        self.root = root

        # Stats information only.
        self.search_nodes_checked = 0
        self.search_points_checked = 0

    def reset_stats(self):
        """Reset the stats counters."""
        self.search_nodes_checked = 0
        self.search_points_checked = 0

    def count_nodes(self):
        """Count the number of nodes in the tree"""
        return self.root.subtree_count()

    @classmethod
    def from_numpy_array(cls, arr_data, max_points=10, min_width=1e-6, angle_weight=1.0):
        """Create a PointingTree from a numpy array with the following columns:
            0: index in original table
            1: earth_vec_x (in AU)
            2: earth_vec_y (in AU)
            3: earth_vec_z (in AU)
            4: unit_vec_x
            5: unit_vec_y
            6: unit_vec_z
            7: field of view (in degrees)

        Parameters
        ----------
        data : `numpy.ndarray`
            The matrix of pointings over which to search.
        max_points : `int`
            The maximum number of points to include in a leaf node.
        min_width : `float`
            The minimal width in each dimension needed to continue splitting the nodes.
        angle_weight : `float`
            The relative weight of the angle (unit vector) dimensions
            compared to the spatial dimensions. Used for tuning the splits.

        Returns
        -------
        root : `PointingTree`
            The root node of the resulting pointing tree.

        Raises
        ------
        KeyError if either the Earth's location data or the pointing unit vector data
        is missing from the PointingTable.
        """
        # Allocate and then split the tree root.
        root = PointingTreeNode(arr_data)
        root.recursive_split_kd(max_points, min_width, angle_weight)
        return PointingTree(root)

    @classmethod
    def from_pointing_table(cls, data, max_points=10, min_width=1e-6, angle_weight=1.0, global_fov=0.0):
        """Create a PointingTree from a PointingTable

        Parameters
        ----------
        data : `PointingTable`
            The table of pointings over which to search.
        max_points : `int`
            The maximum number of points to include in a leaf node.
        min_width : `float`
            The minimal width in each dimension needed to continue splitting the nodes.
        angle_weight : `float`
            The relative weight of the angle (unit vector) dimensions
            compared to the spatial dimensions. Used for tuning the splits.
        global_fov : `float`
            A global field of view (in degrees) to use if no FOV column is provided.

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

        # If no field of view is provided, add a column of zeros.
        if "fov" in data.pointings.columns:
            fov_col = data.pointings["fov"].value
        else:
            fov_col = np.repeat(global_fov, len(data))

        # Create the numpy array of the data for the tree.
        arr_data = np.array(
            [
                np.arange(0, len(data)),
                data.pointings["earth_vec_x"].value,
                data.pointings["earth_vec_y"].value,
                data.pointings["earth_vec_z"].value,
                data.pointings["unit_vec_x"].value,
                data.pointings["unit_vec_y"].value,
                data.pointings["unit_vec_z"].value,
                fov_col,
            ]
        )

        # Create the tree from the numpy array.
        return cls.from_numpy_array(arr_data.T, max_points, min_width, angle_weight)

    def search_heliocentric_xyz(self, target, extra_fov=0.0):
        """Search for pointings that would overlap a given heliocentric
        point (x, y, z). Allows a single field of view or per pointing
        field of views.

        Parameters
        ----------
        point : tuple, list, or array
            The point represented as (x, y, z) on which to compute the distances.
        extra_fov : `float`
            Additional FOV to use for the search. If set to 0.0 uses only the FOV
            from the input array/table.

        Returns
        -------
        A list of indices (in the original table) matching the search.
        """
        self.reset_stats()
        results_indices = []

        # Use a stack-based DFS to avoid recursive calls.
        node_stack = [self.root]
        while len(node_stack) > 0:
            current = node_stack.pop()

            if current.left_child is not None:  # Internal node.
                if not current.left_child.prune(target, extra_fov):
                    node_stack.append(current.left_child)
                if not current.right_child.prune(target, extra_fov):
                    node_stack.append(current.right_child)
                self.search_nodes_checked += 2
            else:  # Leaf node.
                leaf_results = current.find_leaf_matches(target, extra_fov)
                if len(leaf_results) > 0:
                    results_indices.extend(leaf_results.tolist())
                self.search_points_checked += current.num_points

        return results_indices

    def search_heliocentric_pointing(self, point, extra_fov=0.0):
        """Search for pointings that would overlap a given heliocentric
        pointing and estimated distance. Allows a single field of view
        or per pointing field of views.

        Parameters
        ----------
        point : `astropy.coordinates.SkyCoord`
            a barycentric pointing with at least RA, dec, and distance.
        extra_fov : `float`
            Additional FOV to use for the search. If set to 0.0 uses just the per-pointing FOV
            from the input array/table.

        Returns
        -------
        A list of indices (in the original table) matching the search.
        """
        cart_pt = point.cartesian
        helio_pt = [cart_pt.x.value, cart_pt.y.value, cart_pt.z.value]
        return self.search_heliocentric_xyz(helio_pt, extra_fov)


class PointingTreeNode:
    """A PointingTree is a kd-tree over data with data with the following
    columns:
        0: index in original table
        1: earth_vec_x (in AU)
        2: earth_vec_y (in AU)
        3: earth_vec_z (in AU)
        4: unit_vec_x
        5: unit_vec_y
        6: unit_vec_z
        7: field of view (in degrees)

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
        This is the central ray of the viewing cone.
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
        if pointings.shape[1] != 8:
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
        # Clip to +/- 1.0 to avoid problems with numerical precision.
        dots = np.dot(self.pointings[:, 4:7], self.view_center)
        dots[dots > 1.0] = 1.0
        dots[dots < -1.0] = -1.0
        distances = np.arccos(dots) * (180.0 / np.pi)

        # Add the field of views to the distances before computing the max.
        distances += pointings[:, 7]
        self.view_radius = np.max(distances)

    def subtree_count(self):
        """Count the number of nodes (including this one) in the current subtree."""
        if self.pointings is not None:
            return 1
        return self.left_child.subtree_count() + self.right_child.subtree_count() + 1

    def recursive_split_kd(self, max_points=10, min_width=1e-6, angle_weight=1.0):
        """Split the node along the 'widest' spatial column (ignoring the index column).
        We recursively split the node until either it has fewer than max_points or
        is smaller than min_width in each dimension.

        Parameters
        ----------
        max_points : `int`
            The maximum number of points to include in a leaf node.
        min_width : `float`
            The minimal normalized width in each dimension.
        angle_weight : `float`
            The relative weight of the angle (unit vector) dimensions
            compared to the spatial dimensions. Used for tuning the splits.

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
        widths = self.high_bnd[1:7] - self.low_bnd[1:7]
        if np.max(widths) < min_width:
            return False

        # Adjust the weight of the unit vector directions for splitting.
        widths[3:6] *= angle_weight

        # Compute the split.
        self.split_col = np.argmax(widths) + 1
        self.split_value = 0.5 * (self.high_bnd[self.split_col] + self.low_bnd[self.split_col])
        right_pointings = self.pointings[self.pointings[:, self.split_col] >= self.split_value]
        left_pointings = self.pointings[self.pointings[:, self.split_col] < self.split_value]

        # Remove the local copy of the pointings. Need to do this before creating the children
        # to avoid exploding memory as we recursively split.
        self.pointings = None

        # Create a right child.
        self.right_child = PointingTreeNode(right_pointings)
        self.right_child.recursive_split_kd(max_points, min_width, angle_weight)

        # Create a left child.
        self.left_child = PointingTreeNode(left_pointings)
        self.left_child.recursive_split_kd(max_points, min_width, angle_weight)

        return True

    def prune(self, target, extra_fov=0.0):
        """Determines whether or not to prune the node based on whether
        any pointing in this subtree could see the target point.

        Code adapted from Geometric Tools "Intersection of a Sphere and a Cone"
        by David Eberly
        https://www.geometrictools.com/Documentation/IntersectionSphereCone.pdf
        which is covered under the Creative Commons Attribution 4.0 International License

        Parameters
        ----------
        target : `numpy.ndarray`
            A length 3 numpy array indicating a (x, y, z) point.
        extra_fov : `float`
            Additional FOV to use for the search in degrees. Should be set to 0.0
            to use the per-pointing FOV directly. Generally used to provide a global
            FOV when no per-pointing FOV is given. But can also be used to provide
            a pruning buffer.

        Returns
        -------
        prune : `bool`
            A Boolean indicating whether to prune the node.
        """
        # Skip the pruning test if the angular radius is too large. Avoids problems where the
        # distance along the ray can be negative because the cone is over 180 degrees wide.
        if self.view_radius + extra_fov >= 90.0:
            return False

        # The pruning uses the fact that if there exists a cone with the total angle theta
        # and origin in the node's positional sphere, then a cone in the opposite direction
        # (-self.view_center) from the target point must intersect the sphere.
        theta = (self.view_radius + extra_fov) * (np.pi / 180.0)

        # Pruning code for checking whether a cone intersects a sphere adapted from
        # David Eberly's "Intersection of a Sphere and a Cone". The code uses only the
        # first two checks because: those prunes the majority of the cases and we are okay
        # with a few false negatives (unpruned nodes) to keep the average cost low.
        U_vect = target - (self.pos_radius / np.sin(theta)) * (-self.view_center)
        UtoC = self.pos_center - U_vect
        dist_along_ray = np.dot(-self.view_center, UtoC)
        if dist_along_ray < 0:
            return True

        if dist_along_ray * dist_along_ray < np.dot(UtoC, UtoC) * (np.cos(theta) ** 2):
            return True

        return False

    def find_leaf_matches(self, target, extra_fov=0.0):
        """Return a list of points in a leaf node that match the search query.

        Parameters
        ----------
        target : `numpy.ndarray`
            A length 3 numpy array indicating a (x, y, z) point.
        extra_fov : `float`
            Additional FOV to use for the search in degrees. If set to 0.0 uses
            only the per-pointing FOV from the input array/table.

        Returns
        -------
        results : `numpy.ndarray`
            A numpy array of the indices of the matching pointings.

        Raises
        ------
        ValueError if the node is not a leaf.
        """
        if self.pointings is None:
            raise ValueError("Matching called at a non-leaf node.")

        # Compute the geocentric cartesian positions of the point and their (scaled) dot products
        # with each of the viewing vectors in the leaf node.
        geo_pts = target - self.pointings[:, 1:4]
        dot_products = np.sum(np.multiply(geo_pts, self.pointings[:, 4:7]), axis=1)
        magnitudes = np.sqrt(np.sum(np.square(geo_pts), axis=1))
        scaled = np.divide(dot_products, magnitudes)

        # Clip the scaled dot_products to +/- 1 to avoid slight errors that can arise due
        # to numerical precision.
        scaled[scaled > 1.0] = 1.0
        scaled[scaled < -1.0] = -1.0
        dist = np.arccos(scaled) * (180.0 / np.pi)

        res = self.pointings[dist <= self.pointings[:, 7] + extra_fov, 0]

        return res
