"""Functions for extracting pointing information from a FITs file."""

from pathlib import Path

import numpy as np
from astropy import io
from astropy.wcs import WCS


def pointing_from_hdu(hdu):
    """Extract a pointing information from the HDU. Returns the
    RA and dec for the pointing or None if HDU is not valid
    (no WCS, no data, etc.)

    Parameters
    ----------
    hdu :
        The HDU to read.

    Returns
    -------
    tuple or None
        Returns a tuple of (RA, dec, fov) in degrees or None if the
        HDU does not represent a valid pointing.
    """
    # Skip cases where there is no image data at all.
    if hdu.data is None or hdu.header is None:
        return None

    # Try parsing the WCS from the header, catching any errors.
    try:
        wcs = WCS(hdu.header)
    except Exception:
        return None
    if wcs is None:
        return None

    # Confirm we have at least 2 pixel dimensions.
    if wcs.pixel_shape is None or len(wcs.pixel_shape) < 2:
        return None

    num_r = wcs.pixel_shape[0]
    num_c = wcs.pixel_shape[1]
    if num_r < 1 or num_c < 1:
        return None

    # Try to get the RA, dec of the center of the image.
    sky_pos = wcs.pixel_to_world(int(num_r / 2), int(num_c / 2))

    # Check that we produced a valid SkyCoord result.
    if type(sky_pos) is list:
        return None

    # Compute an estimated FOV from the corners.
    max_dist = np.max(
        [
            sky_pos.separation(wcs.pixel_to_world(0, 0)).degree,
            sky_pos.separation(wcs.pixel_to_world(num_r, 0)).degree,
            sky_pos.separation(wcs.pixel_to_world(0, num_c)).degree,
            sky_pos.separation(wcs.pixel_to_world(num_r, num_c)).degree,
        ]
    )
    return (sky_pos.ra.deg, sky_pos.dec.deg, max_dist)


def obstime_from_hdu(hdu, default=-1.0):
    """Extract an observation time from the FITS file using common
    header keys: MJD, DATE-AVE.

    Parameters
    ----------
    hdu :
        The HDU to read.

    Returns
    -------
    obstime : `float`
        The obstime. This is set to default if no matching header
        is found.
    """
    obstime = default
    if "MJD" in hdu.header:
        obstime = hdu.header["MJD"]
    elif "DATE-AVE" in hdu.header:
        obstime = hdu.header["DATE-AVE"]
    return obstime


def extend_pointing_dict_from_fits_file(filename, data_dict, extension=-1):
    """Append to a pointings dictionary from a single FITS file. Checks either
    a specific extension or all extensions.

    Parameters
    ----------
    filename : `str`
        The location of the file.
    data_dict : `dict`
        The pointing dictionary to extend. Contains ra, dec, obstime, filename,
        layer, and fov. Modified in place.
    extension : `int`
        The extension in which to read the WCS and obstime. If no extension is
        given it will try to read a WCS from all layers.

    Returns
    -------
    num_added : `int`
        The number of entries added.
    """
    num_added = 0
    with io.fits.open(filename) as hdu_list:
        # Check for a common (layer 0) obstime for all images.
        base_mjd = obstime_from_hdu(hdu_list[0], -1.0)

        # If we only want a single layer otherwise iterate through all layers.
        if extension != -1:
            if extension >= len(hdu_list):
                return
            layers_to_check = [extension]
        else:
            layers_to_check = [i for i in range(len(hdu_list))]

        for i in layers_to_check:
            # Read the WCS and skip this layer if there is an error.
            coords = pointing_from_hdu(hdu_list[i])
            if coords is None:
                continue

            # Check for an extension specific timestamp.
            obstime = obstime_from_hdu(hdu_list[i], base_mjd)
            if obstime < 0.0:
                continue

            # Append the data.
            data_dict["ra"].append(coords[0])
            data_dict["dec"].append(coords[1])
            data_dict["obstime"].append(obstime)
            data_dict["filename"].append(filename)
            data_dict["layer"].append(i)
            num_added += 1
    return num_added


def pointing_dict_from_fits_files(base_dir, file_pattern, extension=-1):
    """Create a pointings dictionary from all the FITS files matching a pattern.
    Checks either a specific extension per file or all extensions.

    Parameters
    ----------
    base_dir : `str`
        The base directory in which to search.
    pattern : `str`
        The pattern of the filenames to read. Can be a single filename.
    extension : `int`
        The layer in which to read the WCS and obstime. If no layer is given
        it will try to read a WCS from all layers.

    Returns
    -------
    data_dict : `dict`
        The pointing dictionary containing ra, dec, obstime, filename,
        layer, and fov.
    """
    data_dict = {
        "ra": [],
        "dec": [],
        "obstime": [],
        "filename": [],
        "layer": [],
        # "fov": [],
    }

    all_files = Path(base_dir).glob(file_pattern)
    for fpath in all_files:
        # Skip directories or other non-files
        if not fpath.is_file():
            continue

        extend_pointing_dict_from_fits_file(str(fpath), data_dict, extension)
    return data_dict
