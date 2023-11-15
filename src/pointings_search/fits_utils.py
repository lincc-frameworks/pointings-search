"""Functions for extracting pointing information from a FITs file."""

from pathlib import Path

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
        Returns a tuple of (RA, dec) in degrees or None if the
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
    if wcs.pixel_shape[0] < 1 or wcs.pixel_shape[1] < 1:
        return None

    # Try to get the RA, dec of the center of the image.
    mid_r = int(wcs.pixel_shape[0] / 2)
    mid_c = int(wcs.pixel_shape[1] / 2)
    sky_pos = wcs.pixel_to_world(mid_r, mid_c)

    # Check that we produced a valid SkyCoord result.
    if type(sky_pos) is list:
        return None

    return (sky_pos.ra.deg, sky_pos.dec.deg)
