import tempfile
from os import path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, get_body, get_sun
from astropy.time import Time

from pointings_search.pointing_table import PointingTable


def test_check_and_rename_column():
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
    }
    data = PointingTable.from_dict(data_dict)

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
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
        "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
        "brightness": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(data_dict)

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
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "DEC": [0.0, 90.0, 0.0, 45.0, 0.0],
        "MJD": [0.0, 1.0, 2.0, 3.0, 4.0],
        "flux": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(data_dict)

    with tempfile.TemporaryDirectory() as dir_name:
        filename = path.join(dir_name, "test.csv")
        data.to_csv(filename)

        # Check that we can reload it.
        data2 = PointingTable.from_csv(filename)
        assert len(data.pointings) == 5
        assert len(data.pointings.columns) == 4


def test_append_earth_pos():
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [60253.0 + i / 24.0 for i in range(5)],
    }
    data = PointingTable.from_dict(data_dict)
    assert "earth_pos" not in data.pointings.columns
    assert "earth_vec" not in data.pointings.columns

    # Check that the data is corrected.
    data.append_earth_pos()
    assert "earth_pos" in data.pointings.columns
    assert "earth_vec" in data.pointings.columns
    assert len(data.pointings["earth_pos"]) == 5
    assert len(data.pointings["earth_vec"]) == 5

    # Check that the sun's distance is reasonable and consistent between the
    # angular and cartesian representations.
    for i in range(5):
        assert data.pointings["earth_pos"][i].distance < 1.1 * u.AU
        assert data.pointings["earth_pos"][i].distance > 0.9 * u.AU
        vec_dist = np.linalg.norm(data.pointings["earth_vec"][i])
        assert np.isclose(data.pointings["earth_pos"][i].distance.value, vec_dist)


def test_preprocess_pointing_info():
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [60261.0, 60261.1, 60261.2, 60261.3, 60261.4],
    }
    data = PointingTable.from_dict(data_dict)
    assert "pointing" not in data.pointings.columns

    data.preprocess_pointing_info()
    assert "pointing" in data.pointings.columns

    # Check that everything was correctlu copied to the SkyCoord.
    assert np.allclose(data.pointings["pointing"].ra.deg, data_dict["ra"])
    assert np.allclose(data.pointings["pointing"].dec.deg, data_dict["dec"])
    assert np.allclose(data.pointings["pointing"].obstime.mjd, data_dict["obstime"])


def test_angular_dist_3d_heliocentric():
    # The first observation is effectively looking at the sun, the second is
    # looking 1 degree away, and the third is spot on at a later time.
    obstimes = [60253.1, 60253.1, 60263.7]
    sun_pos = get_sun(Time(obstimes, format="mjd"))
    data_dict = {
        "ra": sun_pos.ra.deg,
        "dec": sun_pos.dec.deg,
        "obstime": obstimes,
    }
    data_dict["dec"][1] += 1.0
    data = PointingTable.from_dict(data_dict)

    # Check the pointings compared to the position of the sun. Allow 0.5 degrees
    # error because we are comparing the sun's position and the barycenter's position.
    ang_dist = data.angular_dist_3d_heliocentric([0.0, 0.0, 0.0])
    assert np.allclose(ang_dist.value, [0.0, 1.0, 0.0], atol=0.5)

    # Check an object that is 1 AU from the sun along the x-axis
    # Answer computed manually from sunpos
    ang_dist = data.angular_dist_3d_heliocentric([1.0, 0.0, 0.0])
    assert np.allclose(ang_dist.value, [70.27581206, 70.5852368, 64.81695263], atol=0.5)

    # Check data for Mars
    mars_pos_ang = get_body("mars", Time(60253.1, format="mjd")).transform_to("icrs")
    data_dict2 = {
        "obsid": [1],
        "ra": [mars_pos_ang.ra.deg],
        "dec": [mars_pos_ang.dec.deg],
        "obstime": [60253.1],
    }
    data2 = PointingTable.from_dict(data_dict2)

    # Use the true barycentric position as queried by JPL's Horizons as the offset.
    mars_helio = [-0.945570054569104, -1.127108800070284, -0.4913967118719473]
    ang_dist = data2.angular_dist_3d_heliocentric(mars_helio)
    assert np.allclose(ang_dist.value, [0.0], atol=0.2)


def test_search_heliocentric_pointing():
    # The first observation is effectively looking at the sun.
    data_dict = {
        "obsid": [1, 2, 3, 4, 5, 6],
        "ra": [219.63063, 219.63063, 219.63063, 219.63063, 25.51, 356.24],
        "dec": [-15.45532, -16.45532, -15.7, -15.45532, 15.45532, -1.6305],
        "obstime": [60253.1, 60253.1, 60253.1, 60353.5, 60253.1, 60253.1],
    }
    data = PointingTable.from_dict(data_dict)

    # Check the pointings compared to the position of the sun.
    sun_pos = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=0.0 * u.au)
    match_table = data.search_heliocentric_pointing(sun_pos, 0.9)
    assert len(match_table) == 2
    assert np.allclose(match_table["obsid"], [1, 3])

    # Check the pointings 10 AU out from the sun.
    other_pos = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=10.0 * u.au)
    match_table = data.search_heliocentric_pointing(other_pos, 0.9)
    assert len(match_table) == 1
    assert np.allclose(match_table["obsid"], [6])
