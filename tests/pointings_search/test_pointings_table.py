from astropy.time import Time
import astropy.units as u
from os import path
import numpy as np
import pytest
import tempfile

from pointings_search.pointing_table import PointingTable


def test_check_and_rename_column():
    d = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
    }
    data = PointingTable.from_dict(d)

    # The column is there.
    assert data._check_and_rename_column("ra", [], True)
    assert data._check_and_rename_column("ra", [], False)

    # The column is not there, but can be (and is) renamed.
    assert "dec" not in data.pointings.columns
    assert data._check_and_rename_column("dec", ["Dec", "DEC", "declin"], True)
    assert "dec" in data.pointings.columns

    # A column is missing without a valid replacement.
    assert not data._check_and_rename_column("time", ["mjd", "MJD", "obstime"], False)
    with pytest.raises(Exception):
        data._check_and_rename_column("time", ["mjd", "MJD", "obstime"], True)


def test_validate_and_standardize():
    d = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
        "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
        "brightness": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(d)

    assert len(data.pointings.columns) == 4
    assert "ra" in data.pointings.columns
    assert "dec" not in data.pointings.columns
    assert "obstime" not in data.pointings.columns
    assert "flux" not in data.pointings.columns

    data.validate_and_standardize()
    assert len(data.pointings.columns) == 4
    assert "ra" in data.pointings.columns
    assert "dec" in data.pointings.columns
    assert "obstime" in data.pointings.columns
    assert "flux" not in data.pointings.columns

    data.validate_and_standardize({"flux": ["brightness"]})
    assert len(data.pointings.columns) == 4
    assert "ra" in data.pointings.columns
    assert "dec" in data.pointings.columns
    assert "obstime" in data.pointings.columns
    assert "flux" in data.pointings.columns


def test_from_csv(test_data_dir):
    filename = path.join(test_data_dir, "test_pointings.csv")
    data = PointingTable.from_csv(filename)
    assert len(data.pointings) == 5
    assert len(data.pointings.columns) == 5


def test_to_csv():
    d = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
        "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
        "flux": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(d)

    with tempfile.TemporaryDirectory() as dir_name:
        filename = path.join(dir_name, "test.csv")
        data.to_csv(filename)

        # Check that we can reload it.
        data2 = PointingTable.from_csv(filename)
        assert len(data.pointings) == 5
        assert len(data.pointings.columns) == 4


def test_append_sun_pos():
    d = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [60253.0 + i / 24.0 for i in range(5)],
    }
    data = PointingTable.from_dict(d)
    assert "sun_pos" not in data.pointings.columns
    assert "sun_vec" not in data.pointings.columns

    # Check that the data is corrected.
    data.append_sun_pos()
    assert "sun_pos" in data.pointings.columns
    assert "sun_vec" in data.pointings.columns
    assert len(data.pointings["sun_pos"]) == 5
    assert len(data.pointings["sun_vec"]) == 5

    # Check that the sun's distance is reasonable and consistent between the
    # angular and cartesian representations.
    for i in range(5):
        assert data.pointings["sun_pos"][i].distance < 1.1 * u.AU
        assert data.pointings["sun_pos"][i].distance > 0.9 * u.AU
        vec_dist = np.linalg.norm(data.pointings["sun_vec"][i])
        assert np.isclose(data.pointings["sun_pos"][i].distance.value, vec_dist)


def test_append_unit_vector():
    d = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
    }
    data = PointingTable.from_dict(d)
    assert "unit_vec" not in data.pointings.columns

    data.append_unit_vector()
    assert "unit_vec" in data.pointings.columns
    assert np.allclose(data.pointings["unit_vec"][:, 0], [1.0, 0.0, 0.707106781, 0.0, 0.0])
    assert np.allclose(data.pointings["unit_vec"][:, 1], [0.0, 0.0, 0.707106781, 0.707106781, -1.0])
    assert np.allclose(data.pointings["unit_vec"][:, 2], [0.0, 1.0, 0.0, 0.707106781, 0.0])


def test_angular_dist_3d_heliocentric():
    # The first observation is effectively looking at the sun and the second is
    # looking 1 degree away.
    d = {
        "ra": [219.63062629578198, 219.63062629578198],
        "dec": [-15.455316915908792, -16.455316915908792],
        "obstime": [60253.1, 60253.1],
    }
    data = PointingTable.from_dict(d)

    # Check the pointings compared to the position of the sun.
    ang_dist = data.angular_dist_3d_heliocentric([0.0, 0.0, 0.0])
    assert np.allclose(ang_dist, [0.0, 1.0], atol=1e-5)

    # Check an object that is 1 AU from the sun along the x-axis
    ang_dist = data.angular_dist_3d_heliocentric([1.0, 0.0, 0.0])
    assert np.allclose(ang_dist, [69.587114, 69.283768], atol=1e-5)

    # Check an object in a known location in geocentric space [0.5, 0.5, 0.0] when looking
    # out at RA=0.0 and dec=0.0
    d2 = {"ra": [0.0], "dec": [0.0], "obstime": [60253.1]}
    data2 = PointingTable.from_dict(d2)
    ang_dist = data2.angular_dist_3d_heliocentric(
        [1.2361460125166166, 1.1096560270277159, 0.2642697128572278]
    )
    assert np.allclose(ang_dist, 45.0, atol=1e-5)


def test_angular_dist_3d_heliocentric():
    # The first observation is effectively looking at the sun.
    d = {
        "obsid": [1, 2, 3, 4, 5, 6],
        "ra": [219.63063, 219.63063, 219.63063, 219.63063, 25.51, 356.24],
        "dec": [-15.45532, -16.45532, -15.7, -15.45532, 15.45532, -1.6305],
        "obstime": [60253.1, 60253.1, 60253.1, 60353.5, 60253.1, 60253.1],
    }
    data = PointingTable.from_dict(d)

    # Check the pointings compared to the position of the sun.
    match_table = data.search_heliocentric_pointing(0.0, 0.0, 0.0, 0.9)
    assert len(match_table) == 2
    assert np.allclose(match_table["obsid"], [1, 3])

    # Check the pointings 10 AU out from the sun.
    match_table = data.search_heliocentric_pointing(0.0, 0.0, 10.0, 0.9)
    assert len(match_table) == 1
    assert np.allclose(match_table["obsid"], [6])
