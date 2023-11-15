"""A class for loading, storing and querying the pointings data."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import io
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord, get_body_barycentric
from astropy.table import Table
from astropy.time import Time

from .fits_utils import pointing_from_hdu


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
        result = PointingTable(csv_table)
        result.validate_and_standardize(alt_names)
        return result

    @classmethod
    def from_sqllite(cls, db_name, table_name, columns_map):
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
        result = PointingTable(ap_table)
        result.validate_and_standardize()
        return result

    @classmethod
    def from_fits(self, base_dir, file_pattern, layer=-1, verbose=False):
        """Create a PointingTable from multiple of FITS files.
        Requires each FITS file to have a valid timestamp(s) and a valid WCS
        for each layer indicating an observation.

        Parameters
        ----------
        base_dir : `str`
            The base directory in which to search.
        pattern : `str`
            The pattern of the filenames to read. Can be a single filename.
        layer : `int`
            The layer in which to read the WCS and obstime. If no layer is given
            it will try to read a WCS from all layers.
        verbose : `bool`
            Log verbose information for debugging or status.

        Returns
        -------
        `bool`
            A Boolean indicating whether the file was successfully read and added.
        """
        ras = []
        decs = []
        obstimes = []
        filenames = []
        layers = []

        all_files = Path(base_dir).glob(file_pattern)
        for fpath in all_files:
            # Skip directories or other non-files
            if not fpath.is_file():
                if verbose:
                    print(f"Skipping non-file: {str(fpath)}")
                continue

            with io.fits.open(fpath) as hdu_list:
                if verbose:
                    print(f"Reading {str(fpath)}")

                # Check for a common (layer 0) obstime for all images.
                base_mjd = -1
                if "MJD" in hdu_list[0].header:
                    base_mjd = hdu_list[0].header["MJD"]

                # If we only want a single layer otherwise iterate through
                # all of the layers
                if layer != -1:
                    layers_to_check = [layer]
                else:
                    layers_to_check = [i for i in range(len(hdu_list))]

                for i in layers_to_check:
                    # Read the WCS and skip this layer if there is an error.
                    coords = pointing_from_hdu(hdu_list[i])
                    if coords is None:
                        continue

                    # Read the observation time.
                    if "MJD" in hdu_list[i].header:
                        mjd = hdu_list[i].header["MJD"]
                    elif base_mjd > 0.0:
                        mjd = base_mjd
                    else:
                        continue

                    # Append the data.
                    ras.append(coords[0])
                    decs.append(coords[1])
                    obstimes.append(mjd)
                    filenames.append(str(fpath))
                    layers.append(i)

        # Use the raw data to create the pointing table.
        data_dict = {
            "id": [i for i in range(len(ras))],
            "ra": ras,
            "dec": decs,
            "obstime": obstimes,
            "filename": filenames,
            "layer": layers,
        }
        return PointingTable.from_dict(data_dict)

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
        barycenter) and append this the table. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if "earth_vec_x" in self.pointings.columns and not recompute:
            return

        # Compute and save the (RA, dec, dist) coordinates.
        # Astropy computes the pointing in the ICRS frame.
        times = Time(self.pointings["obstime"], format="mjd")
        earth_pos_cart = get_body_barycentric("earth", times)

        # Compute and save the geocentric cartesian coordinates in AU as a float
        # no astropy units.
        self.pointings["earth_vec_x"] = earth_pos_cart.x.value
        self.pointings["earth_vec_y"] = earth_pos_cart.y.value
        self.pointings["earth_vec_z"] = earth_pos_cart.z.value

    def preprocess_pointing_info(self, recompute=False):
        """Convert the raw RA, dec, and time columns into a astropy
        SkyCoord. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if "unit_vec_x" in self.pointings.columns and not recompute:
            return

        # Compute the pointing vector.
        times = Time(self.pointings["obstime"], format="mjd")
        pointings_ang = SkyCoord(
            ra=self.pointings["ra"].data * u.deg,
            dec=self.pointings["dec"].data * u.deg,
            obstime=times,
            frame="icrs",
        )

        # Save the unit vector of the pointings.
        pointings_cart = pointings_ang.cartesian
        self.pointings["unit_vec_x"] = pointings_cart.x.value
        self.pointings["unit_vec_y"] = pointings_cart.y.value
        self.pointings["unit_vec_z"] = pointings_cart.z.value

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

        if "earth_vec_x" not in self.pointings.columns:
            self.append_earth_pos()
        if "unit_vec_x" not in self.pointings.columns:
            self.preprocess_pointing_info()

        # Compute the geocentric cartesian position of the point in the ICRS frame.
        geo_pts = CartesianRepresentation(
            (cart_pt[0] - self.pointings["earth_vec_x"]),
            (cart_pt[1] - self.pointings["earth_vec_y"]),
            (cart_pt[2] - self.pointings["earth_vec_z"]),
        )

        # Put the pointing data into a CartesianRepresentation so we can use astropy's functions.
        pointing_pts = CartesianRepresentation(
            self.pointings["unit_vec_x"],
            self.pointings["unit_vec_y"],
            self.pointings["unit_vec_z"],
        )

        # Compute the angular distance from the dot product of the vectors. This can be slightly
        # inaccurate very close to 0, but is sufficient for pruning. Since we are using the vectors,
        # (instead of RA, dec) we do not need to worry about the poles.
        norm = geo_pts.norm()
        dot = geo_pts.dot(pointing_pts)
        dist = np.arccos(dot / norm).to(u.deg)

        return dist

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
