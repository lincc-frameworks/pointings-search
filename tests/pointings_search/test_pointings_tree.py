import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from pointings_search.pointing_table import PointingTable
from pointings_search.pointing_tree import PointingTree, PointingTreeNode


def test_build_pointing_tree_node():
    data = np.array(
        [
            [0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1, 0.99, 0.01, -0.01, 1.0, 0.0, 0.0, 0.0],
            [2, 0.98, 0.04, 0.02, 0.0, 1.0, 0.0, 0.0],
            [3, 1.0, 0.04, -0.01, 0.5, 0.8, 0.0, 0.0],
        ]
    )
    tree_node = PointingTreeNode(data)

    assert tree_node.pointings is not None
    assert tree_node.num_points == 4
    assert tree_node.left_child is None
    assert tree_node.right_child is None

    assert np.allclose(tree_node.low_bnd, [0, 0.98, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(tree_node.high_bnd, [3, 1.0, 0.04, 0.02, 1.0, 1.0, 0.0, 0.0])

    assert np.allclose(tree_node.pos_center, [0.99, 0.02, 0.005])
    assert np.allclose(tree_node.view_center, [0.70710678, 0.70710678, 0.0])
    assert np.isclose(tree_node.pos_radius, 0.026925824035672525)
    assert np.isclose(tree_node.view_radius, 45.0)

    # Cannot create a node without the correct number of columns.
    bad_data = np.array(
        [
            [0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1, 0.99, 0.01, -0.01, 1.0, 0.0, 0.0],
            [2, 0.98, 0.04, 0.02, 0.0, 1.0, 0.0],
            [3, 1.0, 0.04, -0.01, 0.5, 0.8, 0.0],
        ]
    )
    with pytest.raises(ValueError) as exception_info:
        tree_node = PointingTreeNode(bad_data)
    assert "Incorrect shape" in str(exception_info.value)


def test_build_pointing_tree_numpy():
    data = np.array(
        [
            [0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1, 0.99, 0.01, -0.01, 1.0, 0.0, 0.0, 0.0],
            [2, 0.98, 0.04, 0.02, 0.0, 1.0, 0.0, 0.0],
            [3, 1.0, 0.04, -0.01, 0.5, 0.8, 0.0, 0.0],
        ]
    )
    # Global FOV defaults to 0.0 if not provided.
    tree = PointingTree.from_numpy_array(data)

    assert tree.root.pointings is not None
    assert tree.root.num_points == 4
    assert tree.root.left_child is None
    assert tree.root.right_child is None

    assert np.allclose(tree.root.low_bnd, [0, 0.98, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(tree.root.high_bnd, [3, 1.0, 0.04, 0.02, 1.0, 1.0, 0.0, 0.0])

    assert np.allclose(tree.root.pos_center, [0.99, 0.02, 0.005])
    assert np.allclose(tree.root.view_center, [0.70710678, 0.70710678, 0.0])
    assert np.isclose(tree.root.pos_radius, 0.026925824035672525)
    assert np.isclose(tree.root.view_radius, 45.0)


def test_build_pointing_tree_pointing_table():
    data_dict = {
        "obsid": [1, 2, 3, 4, 5, 6],
        "ra": [219.63063, 219.63063, 219.63063, 219.63063, 25.51, 356.24],
        "dec": [-15.45532, -16.45532, -15.7, -15.45532, 15.45532, -1.6305],
        "obstime": [60253.1, 60253.1, 60253.1, 60353.5, 60253.1, 60253.1],
    }
    data = PointingTable.from_dict(data_dict)

    with pytest.raises(KeyError) as exception_info:
        _ = PointingTree.from_pointing_table(data, max_points=2)
    assert "append_earth_pos" in str(exception_info.value)
    data.append_earth_pos()

    with pytest.raises(KeyError) as exception_info:
        _ = PointingTree.from_pointing_table(data, max_points=2)
    assert "preprocess_pointing_info" in str(exception_info.value)
    data.preprocess_pointing_info()

    tree = PointingTree.from_pointing_table(data, max_points=10)
    assert tree is not None
    assert tree.root is not None
    assert tree.root.pointings is not None
    assert tree.root.pointings.shape[0] == 6
    assert tree.root.pointings.shape[1] == 8

    # FOVs should be zero.
    assert tree.root.low_bnd[7] == 0.0
    assert tree.root.high_bnd[7] == 0.0

    # Build a tree with a global FOV
    tree = PointingTree.from_pointing_table(data, max_points=10, global_fov=0.5)
    assert tree is not None
    assert tree.root is not None
    assert tree.root.pointings is not None
    assert tree.root.pointings.shape[0] == 6
    assert tree.root.pointings.shape[1] == 8
    assert tree.root.low_bnd[7] == 0.5
    assert tree.root.high_bnd[7] == 0.5


def test_pointing_tree_node_recursive_split_kd():
    data = np.array(
        [
            [0, 0.5, 0.5, 0.2, 0.68041382, 0.68041382, 0.28216553, 0.5],
            [1, 0.5, 0.6, 0.2, 0.68041382, 0.68041382, 0.28216553, 0.5],
            [2, 0.5, 0.7, 0.2, 0.68041382, 0.68041382, 0.28216553, 0.5],
            [3, 0.5, 0.8, 0.2, 0.68041382, 0.68041382, 0.27216553, 0.5],
            [4, 0.5, 0.4, 0.2, 0.68041382, 0.68041382, 0.27216553, 0.5],
            [5, 0.5, 0.3, 0.2, 0.68041382, 0.68041382, 0.27216553, 0.5],
        ]
    )
    tree1 = PointingTreeNode(data)
    assert tree1.num_points == 6

    # Check the stop pruning conditions
    assert not tree1.recursive_split_kd(max_points=10)
    assert not tree1.recursive_split_kd(max_points=2, min_width=1.0)

    # Check a true prune
    assert tree1.recursive_split_kd(max_points=4)
    assert tree1.pointings is None
    assert tree1.num_points == 6
    assert tree1.split_col == 2
    assert np.isclose(tree1.split_value, 0.55)

    # Check the children
    assert tree1.left_child is not None
    assert tree1.left_child.num_points == 3
    for i in range(3):
        assert tree1.left_child.pointings[i, 2] < 0.55

    assert tree1.right_child is not None
    assert tree1.right_child.num_points == 3
    for i in range(3):
        assert tree1.right_child.pointings[i, 2] > 0.55

    # We cannot split a middle node again
    assert not tree1.recursive_split_kd(max_points=4)

    # Split the node on pointing this time.
    tree2 = PointingTreeNode(data)
    assert tree2.num_points == 6
    assert tree2.recursive_split_kd(max_points=4, angle_weight=100.0)
    assert tree2.pointings is None
    assert tree2.num_points == 6
    assert tree2.split_col == 6
    assert np.isclose(tree2.split_value, 0.27716553)

    # Check the children
    assert tree2.left_child is not None
    assert tree2.left_child.num_points == 3
    for i in range(3):
        assert tree2.left_child.pointings[i, 6] < 0.27716553

    assert tree2.right_child is not None
    assert tree2.right_child.num_points == 3
    for i in range(3):
        assert tree2.right_child.pointings[i, 6] > 0.27716553


def test_pointing_tree_node_prune():
    # Two pointings close in position with pointings that differ by ~0.57 degrees.
    data = np.array(
        [
            [0, 0.98, 0.11, 0.05, 0.0, 1.0, 0.0, 1.0],
            [1, 0.96, 0.11, 0.05, 0.0099995, 0.99995, 0.0, 1.0],
        ]
    )
    tree = PointingTreeNode(data)
    center = tree.pos_center

    # Don't prune points in the node positional sphere.
    assert not tree.prune(np.array([0.97, 0.11, 0.05]))
    assert not tree.prune(np.array([0.975, 0.11, 0.05]))
    assert not tree.prune(np.array([0.97, 0.105, 0.05]))
    assert not tree.prune(np.array([0.97, 0.11, 0.055]))
    assert not tree.prune(np.array([0.98, 0.11, 0.05]))

    # Don't prune points directly in front of the center (y direction)
    assert not tree.prune(np.array([0.97, 1.0, 0.05]))
    assert not tree.prune(np.array([0.97, 10.0, 0.05]))
    assert not tree.prune(np.array([0.97, 100.0, 0.05]))

    # Prune points behind the center (y direction)
    assert tree.prune(np.array([0.97, -1.0, 0.05]))
    assert tree.prune(np.array([0.97, -10.0, 0.05]))

    # Prune points offset along x or z axis
    assert tree.prune(np.array([1.97, 0.11, 0.05]))
    assert tree.prune(np.array([0.97, 0.11, 1.05]))
    assert tree.prune(np.array([-0.97, 0.11, 0.05]))
    assert tree.prune(np.array([0.97, 0.11, -1.05]))
    assert tree.prune(np.array([1.97, 0.11, 1.05]))

    # Check an angle that is inside the cone.
    assert not tree.prune(tree.pos_center + 1.0 * np.array([0.019996, 0.99980006, 0.0]))

    # Check an angle just outside the cone (1.4 degrees from the center)
    assert tree.prune(tree.pos_center + 1.0 * np.array([0.04993762, 0.99875234, 0.0]))

    # The angle is outside the cone from the center, but we can compensate with a
    # different positional point in the node. Note the radius of the node is 0.01 so
    # projection of 0.02 puts the target point outside the node itself.
    assert not tree.prune(tree.pos_center + 0.02 * np.array([0.04993762, 0.99875234, 0.0]))

    # Try a few manually computed test cases
    assert not tree.prune(np.array([0.97999963, 2.109975, 0.09]))  # within cone
    assert not tree.prune(np.array([0.97999963, 2.109975, 0.10]))  # within cone + pos_radius
    assert tree.prune(np.array([0.97999963, 2.109975, 0.11]))
    assert tree.prune(np.array([0.97999963, 2.109975, 0.12]))


def test_pointing_tree_node_search_leaf():
    # Data at two different positions with 6 pointings each.
    # One pointing has a wide FOV.
    data = np.array(
        [
            [0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [2, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [3, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0],
            [4, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0],
            [5, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 5.0],
            [6, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 15.0],
        ]
    )
    tree = PointingTreeNode(data)

    # Test right ahead.
    matches = tree.find_leaf_matches(np.array([-1.0, 1.0, 0.0]))
    assert len(matches) == 3
    assert matches[0] == 0
    assert matches[1] == 3
    assert matches[2] == 4

    # Test off to the side a little
    matches = tree.find_leaf_matches(np.array([-1.0, 1.0, 0.02]))
    assert len(matches) == 2
    assert matches[0] == 3
    assert matches[1] == 4

    # Test that we account for the offset with extra FOV buffer.
    matches = tree.find_leaf_matches(np.array([-1.0, 1.0, 0.02]), extra_fov=10.0)
    assert len(matches) == 3
    assert matches[0] == 0
    assert matches[1] == 3
    assert matches[2] == 4

    # Test off to the side a little more.
    matches = tree.find_leaf_matches(np.array([-1.0, 1.0, 0.1]))
    assert len(matches) == 1
    assert matches[0] == 4

    # Test that we account for the offset with extra FOV buffer.
    matches = tree.find_leaf_matches(np.array([-1.0, 1.0, 0.1]), extra_fov=15.0)
    assert len(matches) == 3
    assert matches[0] == 0
    assert matches[1] == 3
    assert matches[2] == 4


def test_pointing_tree_search_heliocentric_xyz():
    # Data at two different positions with 6 pointings each.
    # One pointing has a wide FOV.
    data = np.array(
        [
            [0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            [1, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0],
            [2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
            [3, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 2.0],
            [4, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [5, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0],
            [6, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            [7, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 30.0],
            [8, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0],
            [9, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 2.0],
            [10, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [11, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 2.0],
        ]
    )
    tree = PointingTree.from_numpy_array(data, max_points=3)

    # Basic tests
    for i in range(12):
        target = data[i, 1:4] + 5.0 * data[i, 4:7]

        matches = tree.search_heliocentric_xyz(target)
        if i == 1:
            assert len(matches) == 2
            assert matches[0] == 1
            assert matches[1] == 7
        else:
            assert len(matches) == 1
            assert matches[0] == i


def test_pointing_tree_search_heliocentric_pointing():
    # The first observation is effectively looking at the sun.
    data_dict = {
        "obsid": [1, 2, 3, 4, 5, 6],
        "ra": [219.63063, 219.63063, 219.63063, 219.63063, 25.51, 356.24],
        "dec": [-15.45532, -16.45532, -15.7, -15.45532, 15.45532, -1.6305],
        "obstime": [60253.1, 60253.1, 60253.1, 60353.5, 60253.1, 60253.1],
    }
    data = PointingTable.from_dict(data_dict)
    data.append_earth_pos()
    data.preprocess_pointing_info()

    # Create a tree with a few branches and no global FOV. We will insert different ones during
    # the search itself.
    tree = PointingTree.from_pointing_table(data, max_points=2, min_width=1e-6, global_fov=0.0)

    # Check the pointings compared to the position of the sun.
    sun_pos = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=0.0 * u.au)
    match_list = tree.search_heliocentric_pointing(sun_pos, extra_fov=0.9)
    assert len(match_list) == 2
    assert match_list[0] == 0
    assert match_list[1] == 2

    # Check the pointings 10 AU out from the sun.
    other_pos = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=10.0 * u.au)
    match_list = tree.search_heliocentric_pointing(other_pos, extra_fov=0.9)
    assert len(match_list) == 1
    assert match_list[0] == 5

    # At 10,000 AU from the sun, the heliocentric point should approximately match the geocentric one
    other_pos = SkyCoord(ra=219.63063 * u.deg, dec=-15.7 * u.deg, distance=10000.0 * u.au)
    match_list = tree.search_heliocentric_pointing(other_pos, extra_fov=0.01)
    assert len(match_list) == 1
    assert match_list[0] == 2

    other_pos = SkyCoord(ra=25.51 * u.deg, dec=15.45532 * u.deg, distance=10000.0 * u.au)
    match_list = tree.search_heliocentric_pointing(other_pos, extra_fov=0.01)
    assert len(match_list) == 1
    assert match_list[0] == 4
