import numpy as np

from pointings_search.pointing_table import PointingTable
from pointings_search.pointing_tree import build_pointing_tree, PointingTree


def test_build_node():
    data = np.array(
        [
            [0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1, 0.99, 0.01, -0.01, 1.0, 0.0, 0.0],
            [2, 0.98, 0.04, 0.02, 0.0, 1.0, 0.0],
            [3, 1.0, 0.04, -0.01, 0.5, 0.8, 0.0],
        ]
    )
    tree = PointingTree(data)

    assert tree.pointings is not None
    assert tree.num_points == 4
    assert tree.left_child is None
    assert tree.right_child is None

    assert np.allclose(tree.low_bnd, [0, 0.98, 0.0, -0.01, 0.0, 0.0, 0.0])
    assert np.allclose(tree.high_bnd, [3, 1.0, 0.04, 0.02, 1.0, 1.0, 0.0])

    assert np.allclose(tree.pos_center, [0.99, 0.02, 0.005])
    assert np.allclose(tree.view_center, [0.70710678, 0.70710678, 0.0])
    assert np.isclose(tree.pos_radius, 0.026925824035672525)
    assert np.isclose(tree.view_radius, 45.0)


def test_recursive_split_kd():
    equal_widths = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ang_widths = np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])

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
    tree1 = PointingTree(data)
    assert tree1.num_points == 6

    # Check the stop pruning conditions
    assert not tree1.recursive_split_kd(equal_widths, max_points=10)
    assert not tree1.recursive_split_kd(equal_widths, max_points=2, min_width=1.0)

    # Check a true prune
    assert tree1.recursive_split_kd(equal_widths, max_points=4)
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
    assert not tree1.recursive_split_kd(equal_widths, max_points=4)

    # Split the node on pointing this time.
    tree2 = PointingTree(data)
    assert tree2.num_points == 6
    assert tree2.recursive_split_kd(ang_widths, max_points=4)
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


def test_recursive_split_dist():
    # Data with a bunch of increasing y-values and interleaved pointings
    data = np.array(
        [
            [0, 0.5, 0.51, 0.01, 0.68041382, 0.68041382, 0.27216553, 0.5],
            [1, 0.5, 0.52, 0.01, 0.62017367, 0.74420841, 0.24806947, 0.5],
            [2, 0.5, 0.53, 0.04, 0.68041382, 0.68041382, 0.27216553, 0.5],
            [3, 0.5, 0.54, 0.03, 0.62017367, 0.74420841, 0.24806947, 0.5],
            [4, 0.5, 0.55, 0.02, 0.68041382, 0.68041382, 0.27216553, 0.5],
            [5, 0.5, 0.56, -0.01, 0.62017367, 0.74420841, 0.24806947, 0.5],
        ]
    )

    # Check the stop pruning conditions
    tree1 = PointingTree(data)
    assert tree1.num_points == 6
    assert not tree1.recursive_split_dist(effective_dist=1.0, max_points=10)
    assert not tree1.recursive_split_dist(effective_dist=1.0, max_points=2, min_width=10.0)

    # Check a true prune with a small effective distance (prioritizes the
    # Earth's position when splitting).
    assert tree1.recursive_split_dist(effective_dist=0.001, max_points=4)
    assert tree1.pointings is None
    assert tree1.num_points == 6

    # Check the children
    assert tree1.left_child is not None
    assert tree1.left_child.num_points == 3
    for i in range(3):
        assert tree1.left_child.pointings[i, 2] < 0.535

    assert tree1.right_child is not None
    assert tree1.right_child.num_points == 3
    for i in range(3):
        assert tree1.right_child.pointings[i, 2] > 0.535

    # We cannot split a middle node again
    assert not tree1.recursive_split_dist(effective_dist=0.001, max_points=4)

    # Create a new node and use a large maximum distance, which will prioritize
    # the viewing angle.
    tree2 = PointingTree(data)
    assert tree2.recursive_split_dist(effective_dist=100.0, max_points=4)
    assert tree2.pointings is None
    assert tree2.num_points == 6

    # Check the children
    assert tree2.left_child is not None
    assert tree2.left_child.num_points == 3
    for i in range(3):
        assert tree2.left_child.pointings[i, 6] > 0.25

    assert tree2.right_child is not None
    assert tree2.right_child.num_points == 3
    for i in range(3):
        assert tree2.right_child.pointings[i, 6] < 0.25
