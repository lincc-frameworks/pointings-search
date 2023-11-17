import tempfile
from os import path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import (
    CartesianRepresentation,
    SkyCoord,
    SphericalRepresentation,
    get_body,
    get_body_barycentric,
    get_sun,
)
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
    """Confirm that we can save and reload the basic data."""
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [0.0, 1.0, 2.0, 3.0, 4.0],
        "flux": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(data_dict)

    with tempfile.TemporaryDirectory() as dir_name:
        filename = path.join(dir_name, "test.csv")
        data.to_csv(filename)

        # Check that we can reload it.
        data2 = PointingTable.from_csv(filename)
        assert len(data2.pointings) == 5
        assert len(data2.pointings.columns) == 4
        assert np.allclose(data2.pointings["ra"], data_dict["ra"])
        assert np.allclose(data2.pointings["dec"], data_dict["dec"])
        assert np.allclose(data2.pointings["obstime"], data_dict["obstime"])
        assert np.allclose(data2.pointings["flux"], data_dict["flux"])


def test_to_csv_cached():
    """Confirm that we can save and reload the cached data."""
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [0.0, 1.0, 2.0, 3.0, 4.0],
        "flux": [10.0, 10.0, 10.0, 10.0, 10.0],
    }
    data = PointingTable.from_dict(data_dict)
    data.preprocess_pointing_info()
    data.append_earth_pos()

    with tempfile.TemporaryDirectory() as dir_name:
        filename = path.join(dir_name, "test2.csv")
        data.to_csv(filename)

        # Check that we can reload it.
        data2 = PointingTable.from_csv(filename)
        assert len(data2.pointings) == 5
        assert len(data2.pointings.columns) == 10
        assert np.allclose(data2.pointings["ra"], data_dict["ra"])
        assert np.allclose(data2.pointings["dec"], data_dict["dec"])
        assert np.allclose(data2.pointings["obstime"], data_dict["obstime"])
        assert np.allclose(data2.pointings["flux"], data_dict["flux"])

        assert "unit_vec_x" in data2.pointings.columns
        assert "unit_vec_y" in data2.pointings.columns
        assert "unit_vec_z" in data2.pointings.columns
        assert "earth_vec_x" in data2.pointings.columns
        assert "earth_vec_y" in data2.pointings.columns
        assert "earth_vec_z" in data2.pointings.columns


def test_filter_time():
    data_dict = {
        "ra": [0.0] * 10,
        "dec": [0.0] * 10,
        "obstime": [60253.0 + i for i in range(10)],
    }
    data = PointingTable.from_dict(data_dict)
    assert len(data) == 10

    data.filter_on_time(min_obstime=60255.0)
    assert len(data) == 8
    for i in range(len(data)):
        assert data.pointings["obstime"][i] >= 60255.0

    data.filter_on_time(max_obstime=60257.5)
    assert len(data) == 3
    for i in range(len(data)):
        assert data.pointings["obstime"][i] <= 60257.5


def test_append_earth_pos():
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [60253.0 + i / 24.0 for i in range(5)],
    }
    data = PointingTable.from_dict(data_dict)
    assert "earth_vec_x" not in data.pointings.columns
    assert "earth_vec_y" not in data.pointings.columns
    assert "earth_vec_z" not in data.pointings.columns

    # Check that the data is correct
    data.append_earth_pos()
    assert "earth_vec_x" in data.pointings.columns
    assert "earth_vec_y" in data.pointings.columns
    assert "earth_vec_z" in data.pointings.columns
    assert len(data.pointings["earth_vec_x"]) == 5
    assert len(data.pointings["earth_vec_y"]) == 5
    assert len(data.pointings["earth_vec_z"]) == 5

    # Check that the sun's distance is reasonable and consistent between the
    # angular and cartesian representations.
    for i in range(5):
        x = data.pointings["earth_vec_x"][i]
        y = data.pointings["earth_vec_y"][i]
        z = data.pointings["earth_vec_z"][i]
        dist = np.sqrt(x * x + y * y + z * z)

        assert dist < 1.1
        assert dist > 0.9


def test_preprocess_pointing_info():
    data_dict = {
        "ra": [0.0, 90.0, 45.0, 90.0, 270.0],
        "dec": [0.0, 90.0, 0.0, 45.0, 0.0],
        "obstime": [60261.0, 60261.1, 60261.2, 60261.3, 60261.4],
    }
    data = PointingTable.from_dict(data_dict)
    assert "unit_vec_x" not in data.pointings.columns
    assert "unit_vec_y" not in data.pointings.columns
    assert "unit_vec_z" not in data.pointings.columns

    data.preprocess_pointing_info()
    assert "unit_vec_x" in data.pointings.columns
    assert "unit_vec_y" in data.pointings.columns
    assert "unit_vec_z" in data.pointings.columns
    assert np.allclose(data.pointings["unit_vec_x"], [1.0, 0.0, 0.707106781, 0.0, 0.0])
    assert np.allclose(data.pointings["unit_vec_y"], [0.0, 0.0, 0.707106781, 0.707106781, -1.0])
    assert np.allclose(data.pointings["unit_vec_z"], [0.0, 1.0, 0.0, 0.707106781, 0.0])


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

    # Position the fake object relative (+1.0, -0.2, +0.5) relative to the Earth.
    earth_pos = get_body_barycentric("earth", Time(60153.2, format="mjd"))
    other_pos = [
        earth_pos.x.value + 1.0,
        earth_pos.y.value - 0.2,
        earth_pos.z.value + 0.5,
    ]
    pointing_dir = CartesianRepresentation(x=1.0, y=-0.2, z=0.5)
    pointing_ang = SkyCoord(SphericalRepresentation.from_cartesian(pointing_dir))
    data_dict3 = {
        "obsid": [1],
        "ra": [pointing_ang.ra.deg],
        "dec": [pointing_ang.dec.deg],
        "obstime": [60153.2],
    }
    data3 = PointingTable.from_dict(data_dict3)
    ang_dist = data3.angular_dist_3d_heliocentric(other_pos)
    assert np.allclose(ang_dist.value, [0.0], atol=0.01)


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

    # At 10,000 AU from the sun, the heliocentric point should approximately match the geocentric one
    other_pos = SkyCoord(ra=219.63063 * u.deg, dec=-15.7 * u.deg, distance=10000.0 * u.au)
    match_table = data.search_heliocentric_pointing(other_pos, 0.01)
    assert len(match_table) == 1
    assert np.allclose(match_table["obsid"], [3])

    other_pos = SkyCoord(ra=25.51 * u.deg, dec=15.45532 * u.deg, distance=10000.0 * u.au)
    match_table = data.search_heliocentric_pointing(other_pos, 0.01)
    assert len(match_table) == 1
    assert np.allclose(match_table["obsid"], [5])
