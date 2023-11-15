import numpy as np
import pytest
from astropy.io import fits

from pointings_search.fits_utils import pointing_from_hdu


def make_test_hdu(num_row=10, num_col=15):
    # Create fake image data.
    hdu1 = fits.ImageHDU()
    hdu1.data = np.zeros((num_row, num_col))
    hdu1.header["NAXIS"] = 2
    hdu1.header["NAXIS1"] = num_row
    hdu1.header["NAXIS2"] = num_col

    # Create fake WCS data.
    hdu1.header["WCSAXES"] = 2
    hdu1.header["CTYPE1"] = "RA---TAN-SIP"
    hdu1.header["CTYPE2"] = "DEC--TAN-SIP"
    hdu1.header["CRVAL1"] = 200.614997245422
    hdu1.header["CRVAL2"] = -7.78878863332778
    hdu1.header["CRPIX1"] = 1033.934327
    hdu1.header["CRPIX2"] = 2043.548284
    hdu1.header["CD1_1"] = -1.13926485986789e-07
    hdu1.header["CD1_2"] = 7.31839748843125e-05
    hdu1.header["CD2_1"] = -7.30064978350695e-05
    hdu1.header["CD2_2"] = -1.27520156332774e-07
    hdu1.header["CTYPE1A"] = "LINEAR  "
    hdu1.header["CTYPE2A"] = "LINEAR  "
    hdu1.header["CUNIT1A"] = "PIXEL   "
    hdu1.header["CUNIT2A"] = "PIXEL   "

    return hdu1


def test_pointing_from_hdu():
    test_hdu = make_test_hdu()
    result = pointing_from_hdu(test_hdu)
    assert result is not None
    assert np.isclose(result[0], 200.46478614563296)
    assert np.isclose(result[1], -7.713457029036169)


def test_pointing_from_hdu_no_data():
    test_hdu = make_test_hdu()
    test_hdu.data = None
    result = pointing_from_hdu(test_hdu)
    assert result is None


def test_pointing_from_hdu_empty_data():
    test_hdu = make_test_hdu()
    test_hdu.data = np.array([[]])
    test_hdu.header["NAXIS1"] = 0
    test_hdu.header["NAXIS2"] = 0
    result = pointing_from_hdu(test_hdu)
    assert result is None


def test_pointing_bad_wcs():
    test_hdu = make_test_hdu()
    del test_hdu.header["CTYPE1"]
    del test_hdu.header["CTYPE2"]
    result = pointing_from_hdu(test_hdu)
    assert result is None


def test_pointing_long_image():
    """Test a case where the center of the height is outside the width"""
    test_hdu = make_test_hdu(200, 20)
    result = pointing_from_hdu(test_hdu)
    assert result is not None


def test_pointing_wide_image():
    """Test a case where the center of the width is outside the height"""
    test_hdu = make_test_hdu(20, 200)
    result = pointing_from_hdu(test_hdu)
    assert result is not None
