"""A spatial data structure for storing and querying 6-d vectors representing
pointing data."""

import numpy as np

from pointings_search.pointing_table import PointingTable


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

    Attributes
    ----------
    pointing : `numpy.ndarray` or None
        The array of minimal pointing information. Set to None for internal nodes
        and non-None for leaf nodes.
    num_points : `int`
        The number of points in this tree.
    low_bnd : `numpy.ndarray`
        The lowest value in each column of the points in this tree.
    high_bnd : `numpy.ndarray`
        The highest value in each column of the points in this tree.
    split_col : `int`
        The column along which this node is split. -1 for leaf nodes.
    split_value : `float`
        The value along which the node is split.
    left_child : `PointingTree`
        A reference to the left subtree (all values where
        pointing[split_col] < split_value). None for leaf nodes.
    right_child : `PointingTree`
        A reference to the right subtree (all values where
        pointing[split_col] >= split_value). None for leaf nodes.
    """

    def __init__(self, pointings):
        if pointings.shape[1] != 7:
            raise ValueError("Incorrect shape for pointing tree data: {pointings.shape}")

        self.pointings = pointings
        self.num_points = len(pointings)
        self.low_bnd = np.min(pointings, axis=0)
        self.high_bnd = np.max(pointings, axis=0)

        self.split_col = -1
        self.split_value = 0.0
        self.left_child = None
        self.right_child = None

    def recursive_split(self, max_widths, max_points=10, min_width=1e-6):
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
        self.right_child.recursive_split(max_widths, max_points, min_width)

        # Create a left child.
        self.left_child = PointingTree(left_pointings)
        self.left_child.recursive_split(max_widths, max_points, min_width)

        return True


def build_pointing_tree(data, max_points=10, min_width=1e-6):
    """Create a PointingTree from a PointingTable

    Parameters
    ----------
    data : `PointingTable`
        The table of pointings over which to search.
    max_points : `int`
        The maximum number of points to include in a leaf node.
    min_width : `float`
        The minimal normalized width in each dimension.
    """
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
        ]
    ).T

    # allocate the tree root.
    root = PointingTree(arr_data)

    # Compute the normalization factors and split the tree.
    max_widths = root.high_bnd[1:7] - root.low_bnd[1:7]
    root.recursive_split(max_widths, max_points, min_width)

    return root
