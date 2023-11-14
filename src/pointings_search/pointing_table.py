"""A class for loading, storing and querying the pointings data."""

import sqlite3

import numpy as np
import pandas as pd
from astropy import io
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    get_body_barycentric,
    SkyCoord,
    SphericalRepresentation,
)
from astropy.table import Table
from astropy.time import Time

from pointings_search.geometry import ang2unitvec, angular_distance


class PointingTable:
    """PointingTable is a wrapper around astropy's Table that enforces
    required columns, name remapping, and a variety of specific query functions.

    Note: The current code builds PointingTable from a CSV, but we can extend this
    to read from a sqllite database, Butler, parquet files, etc.

    Attributes
    ----------
    pointings: `astropy.table.Table`
        The table of all the pointings.
    """

    def __init__(self, pointings):
        self.pointings = pointings

    @classmethod
    def from_dict(cls, data):
        """Create the pointings data from a dictionary.

        Parameters
        ----------
        data : `dict`
            The dictionary to use.
        """
        pointing_table = Table(data)
        return PointingTable(pointing_table)

    @classmethod
    def from_csv(cls, filename, alt_names=None):
        """Read the pointings data from a CSV file.

        Parameters
        ----------
        filename : `str`
            The name of the CSV file.
        alt_names : dict, optional
            A dictionary mapping final column names to known alternatives.
            Example: If we know the `ra` column could be called
            `RightAscension (deg)` or `RA (deg)` we would pass
            alt_names={"ra": ["RightAscension (deg)", "RA (deg)"]}
        """
        csv_table = io.ascii.read(filename, format="csv")
        data = PointingTable(csv_table)
        data.validate_and_standardize(alt_names)
        return data

    def from_sqllite(self, db_name, table_name, columns_map):
        """Create the PointingTable from a sqllite database file.

        The columns map must have the following

        Parameters
        ----------
        db_name : `str`
            The file location of pointing database.
        table_name : `str`
            The table to query from the database.
        columns_map : `dict`
            A dictionary indcating which columns to load and mapping the
            column name in the PointingTable to the column name in the SQL database.
            Must contain all required columns.
        """
        db_query = "SELECT"
        for key in columns_map:
            if len(db_query) <= 8:
                db_query += f" {columns_map[key]} as {key}"
            else:
                db_query += f", {columns_map[key]} as {key}"
        db_query += f" FROM {table_name}"

        # Read the data from the database.
        connection = sqlite3.connect(db_name)
        pandas_df = pd.read_sql_query(db_query, connection)

        # Convert the data into an astropy Table (from the Panda's table).
        ap_table = Table.from_pandas(pandas_df)
        data = PointingTable(ap_table)
        data.validate_and_standardize()
        return data

    def _check_and_rename_column(self, col_name, alt_names, required=True):
        """Check if the column is included using multiple possible names
        and renaming to a single canonical name.

        Parameters
        ----------
        col_name: str
            The canonical name we will use to represent the column.
        alt_names: list of str
            Others things a column could be called. The column will be
            renamed to `col_name` in place.
        required: bool
            Is the column required.

        Returns
        -------
        bool
            Whether the column is in the dataframe.

        Raises
        ------
        KeyError is the column is required and not present.
        """
        if col_name in self.pointings.columns:
            return True

        # Check if the column is using an alternate name and, if so, rename.
        for alt in alt_names:
            if alt in self.pointings.columns:
                self.pointings.rename_column(alt, col_name)
                return True

        if required:
            raise KeyError(f"Required column `{col_name}` missing.")
        return False

    def validate_and_standardize(self, alt_names=None):
        """Validate that the data has the required columns expected operations.

        Parameters
        ----------
        alt_names : dict, optional
            A dictionary mapping final column names to known alternatives.
            Example: If we know the `ra` column could be called
            `RightAscension (deg)` or `RA (deg)` we would pass
            alt_names={"ra": ["RightAscension (deg)", "RA (deg)"]}

        Raises
        ------
        KeyError is the column is required and not present.
        """
        if alt_names is None:
            alt_names = {
                "ra": ["RA"],
                "dec": ["DEC", "Dec"],
                "obstime": ["time", "mjd", "MJD"],
            }

        self._check_and_rename_column("ra", alt_names.get("ra", ["RA"]), required=True)
        self._check_and_rename_column("dec", alt_names.get("dec", ["DEC", "Dec"]), required=True)
        self._check_and_rename_column(
            "obstime", alt_names.get("obstime", ["time", "mjd", "MJD"]), required=True
        )
        for key in alt_names.keys():
            self._check_and_rename_column(key, alt_names[key], required=False)

    def append_earth_pos(self, recompute=False):
        """Compute an approximate position of the Earth (relative to solar systems's
        barycenter and append this the table. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if "earth_pos" in self.pointings.columns and not recompute:
            return

        # Compute and save the (RA, dec, dist) coordinates.
        # Astropy computes the pointing in the ICRS frame.
        times = Time(self.pointings["obstime"], format="mjd")
        earth_pos_cart = get_body_barycentric("earth", times)
        earth_pos_spherical = SphericalRepresentation.from_cartesian(earth_pos_cart)
        self.pointings["earth_pos"] = SkyCoord(earth_pos_cart, frame="icrs", obstime=times)

        # Compute and save the geocentric cartesian coordinates in AU as a float
        # no astropy units.
        self.pointings["earth_vec"] = np.array(
            [
                earth_pos_cart.x.value,
                earth_pos_cart.y.value,
                earth_pos_cart.z.value,
            ]
        ).T

    def preprocess_pointing_info(self, recompute=False):
        """Convert the raw RA, dec, and time columns into a astropy
        SkyCoord. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if "pointing" in self.pointings.columns and not recompute:
            return

        times = Time(self.pointings["obstime"], format="mjd")
        self.pointings["pointing"] = SkyCoord(
            ra=self.pointings["ra"].data * u.deg,
            dec=self.pointings["dec"].data * u.deg,
            obstime=times,
            frame="icrs",
        )

    def angular_dist_3d_heliocentric(self, cart_pt):
        """Compute the angular offset (in degrees) between the pointing and
        the 3-d location (specified as heliocentric coordinates and AU) of a point.

        Parameters
        ----------
        cart_pt : tuple, list, or array
            The point represented as (x, y, z) on which to compute the distances.

        Returns
        -------
        ang_dist : numpy array
            A length K numpy array with the angular distances in degrees.

        Raises
        ------
        ValueError if the list of points is not width=3.
        """
        if len(cart_pt) != 3:
            raise ValueError(f"Expected 3 dimensions, found {len(cart_pt)}")

        if "earth_pos" not in self.pointings.columns:
            self.append_earth_pos()
        if "pointing" not in self.pointings.columns:
            self.preprocess_pointing_info()

        # Compute the geocentric cartesian position of the point in the ICRS frame.
        geo_pts = CartesianRepresentation(
            (cart_pt[0] - self.pointings["earth_vec"][:, 0]) * u.au,
            (cart_pt[1] - self.pointings["earth_vec"][:, 1]) * u.au,
            (cart_pt[2] - self.pointings["earth_vec"][:, 2]) * u.au,
        )

        # Convert to geocentric spherical coordinates in the ICRS frame.
        geo_pos = SkyCoord(SphericalRepresentation.from_cartesian(geo_pts), frame="icrs")

        # Compute the angular distance of the point with each pointing.
        ang_dist = geo_pos.separation(self.pointings["pointing"])
        return ang_dist

    def search_heliocentric_pointing(self, point, fov=None):
        """Search for pointings that would overlap a given heliocentric
        pointing and estimated distance. Allows a single field of view
        or per pointing field of views.

        Note
        ----
        Currently uses a linear algorithm that computes all distances. It
        is likely we can accelerate this with better indexing.

        Parameters
        ----------
        point: a barycentric pointing with at least RA, dec, and distance.
        fov : `float` (optional)
            The field of view of the individual pointings. If None
            tries to retrieve from table.

        Returns
        -------
        An astropy table with information for the matching pointings.

        Raises
        ------
        ValueError if no field of view is provided.
        """
        if fov is None and "fov" not in self.pointings.columns:
            raise ValueError("No field of view provided.")

        # Create the query point in 3-d heliocentric cartesian space.
        cart_pt = point.cartesian
        helio_pt = [cart_pt.x.value, cart_pt.y.value, cart_pt.z.value]

        # Compare the angular distance of the query point to each pointing.
        ang_dist = self.angular_dist_3d_heliocentric(helio_pt)
        if fov is None:
            inds = ang_dist.value < self.pointings["fov"]
        else:
            inds = ang_dist.value < fov
        return self.pointings[inds]

    def to_csv(self, filename, overwrite=False):
        """Write the pointings data to a CSV file.

        Parameters
        ----------
        filename : `str`
            The name of the CSV file.
        overwrite: `bool`
            Whether to overwrite an existing file.
        """
        self.pointings.write(filename, overwrite=overwrite)
