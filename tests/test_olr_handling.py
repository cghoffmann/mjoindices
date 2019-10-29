# -*- coding: utf-8 -*-

""" """

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindices

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Contact: christoph.hoffmann@uni-greifswald.de

import math
import os.path
from pathlib import Path

import numpy as np
import pytest

import mjoindices.olr_handling as olr

olr_data_filename = Path(__file__).resolve().parent / "testdata" / "olr.day.mean.nc"


def test_OLRData_basic_properties():

    time = np.arange("2018-01-01", "2018-01-10", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(9, 2, 4)
    target = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    if not np.all(target.time == time):
        errors.append("Time property is incorrect.")
    if not np.all(target.lat == lat):
        errors.append("Lat property is incorrect.")
    if not np.all(target.long == long):
        errors.append("Long property is incorrect.")
    if not np.all(target.olr == olrmatrix):
        errors.append("OLR property is incorrect.")

    # Check wrong lengths of grids
    olrmatrix = np.random.rand(12, 2, 4)
    with pytest.raises(ValueError) as e:
        target = olr.OLRData(olrmatrix, time, lat, long)
    if "first" not in str(e.value):
        errors.append("Time grid test failed")

    olrmatrix = np.random.rand(9, 3, 4)
    with pytest.raises(ValueError) as e:
        target = olr.OLRData(olrmatrix, time, lat, long)
    if "second" not in str(e.value):
        errors.append("Lat grid test failed")

    olrmatrix = np.random.rand(9, 2, 3)
    with pytest.raises(ValueError) as e:
        target = olr.OLRData(olrmatrix, time, lat, long)
    if "third" not in str(e.value):
        errors.append("Long grid test failed")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))



@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available")
def test_loadNOAAInterpolatedOLR():
    errors = []
    target = olr.load_noaa_interpolated_olr(olr_data_filename)

    # Check time grid
    # Period always starts on 1974/06/01, whereas the ending date
    # changes when file is updated
    if not target.time[0] == np.datetime64("1974-06-01"):
        errors.append("First date does not match")
    if not ((target.time[1] - target.time[0]).astype('timedelta64[D]')/np.timedelta64(1,"D")) == 1:
        errors.append("Temporal spacing does not match 1 day")

    # Check latitude grid
    if not target.lat[0] == 90:
        errors.append("First latitude entry does not match.")
    if not target.lat[3] == 82.5:
        errors.append("Forth latitude entry does not match.")
    if not target.lat[-1] == -90:
        errors.append("Last latitude entry does not match.")
    if not (target.lat[0] - target.lat[1]) == 2.5:
        errors.append("Latitudinal spacing does not meet the expectation")

    # Check longitude grid
    if not target.long[0] == 0:
        errors.append("First longitude entry does not match.")
    if not target.long[-1] == 357.5:
        errors.append("Last longitude entry does not match.")
    if not target.long[1] - target.long[0] == 2.5:
        errors.append("Longitudinal spacing does not meet the expectation")

    # Check OLR Data
    # OLR samples extracted from file using Panoply viewer, which directly
    # applies scaling and offset values
    if not math.isclose(target.olr[0,0,0], 205.450):
        errors.append("First OLR sample value does not match")
    if not math.isclose(target.olr[0, 3, 0], 207.860):
        errors.append("Second OLR sample value does not match")
    if not math.isclose(target.olr[4, 3, 15], 216.700):
        errors.append("Third OLR sample value does not match")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_resampleOLRToOriginalSpatialGrid():
    # Basic assumption of the test is, that both grids are identical and only the direction of latitude is reversed.
    # Furthermore, the original grid only covers the tropics between -20 and 20 deg lat.
    origOLR = olr.load_noaa_interpolated_olr(olr_data_filename)
    target = olr.interpolate_spatial_grid_to_original(origOLR)
    errors = []
    print(origOLR.lat[44:27:-1])
    if not np.all(target.lat == origOLR.lat[44:27:-1]):
        errors.append("Latitude grid does not match middle of original one")
    if not np.all(target.long == origOLR.long):
        errors.append("Logitude grid does not match original one")
    if not np.all(target.time == origOLR.time):
        errors.append("Time grid does not match original one")
    if not np.all(target.olr == origOLR.olr[:, 44:27:-1, :]):
        errors.append("OLR data does not match original one in the tropics")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_resample_spatial_grid():
    time = np.arange("2018-01-01", "2018-01-03", dtype='datetime64[D]')
    lat = np.array([-2.5, 0., 2.5])
    long = np.array([10., 20., 30., 40.])
    olrmatrix = np.array([((1., 2., 3., 4.),
                           (5., 6., 7., 8.),
                           (9., 10., 11., 12.)),
                          ((10., 20., 30., 40.),
                           (50., 60., 70., 80.),
                           (90., 100., 110., 120.))])
    testdata = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    target_lat = np.array([-1.25, 1.25])
    target_long = np.array([10., 20., 30., 40.])
    target = olr.interpolate_spatial_grid(testdata, target_lat, target_long)
    if not target.olr[0, 0, 0] == (1.+5.)/2.:
        errors.append("Latitude interpolation with longitude constant incorrect (setup 1).")
    if not target.olr[0, 1, 0] == (5.+9.)/2.:
        errors.append("Latitude interpolation with longitude constant incorrect (setup 2).")
    if not target.olr[0, 1, 2] == (7.+11.)/2.:
        errors.append("Latitude interpolation with longitude constant incorrect (setup 3).")
    if not target.olr[1, 1, 0] == (50.+90.)/2.:
        errors.append("Latitude interpolation with longitude constant incorrect (setup 4).")


    target_lat =np.array([-2.5, 0., 2.5])
    target_long = np.array([15., 25., 35.])
    target = olr.interpolate_spatial_grid(testdata, target_lat, target_long)
    if not target.olr[0, 0, 0] == (1. + 2.) / 2.:
        errors.append("Longitude interpolation with latitude constant incorrect (setup 1).")
    if not target.olr[0, 0, 1] == (2. + 3.) / 2.:
        errors.append("Longitude interpolation with latitude constant incorrect (setup 2).")
    if not target.olr[0, 1, 1] == (6. + 7.) / 2.:
        errors.append("Longitude interpolation with latitude constant incorrect (setup 3).")
    if not target.olr[1, 0, 1] == (20. + 30.) / 2.:
        errors.append("Longitude interpolation with latitude constant incorrect (setup 3).")

    target_lat = np.array([-1.25, 1.25])
    target_long = np.array([15., 25., 35.])
    target = olr.interpolate_spatial_grid(testdata, target_lat, target_long)
    if not target.olr[0, 0, 0] == 3.5:
        errors.append("Simultaneous lat/long interpolation incorrect (setup 1).")
    if not target.olr[0, 1, 0] == 7.5:
        errors.append("Simultaneous lat/long interpolation incorrect (setup 2).")
    if not target.olr[0, 1, 1] == 8.5:
        errors.append("Simultaneous lat/long interpolation incorrect (setup 2).")
    if not target.olr[1, 1, 1] == 85:
        errors.append("Simultaneous lat/long interpolation incorrect (setup 2).")

    target_lat = np.array([-1.25, 1.25])
    target_long = np.array([15., 25., 35., 45])
    with pytest.raises(ValueError) as e:
        target = olr.interpolate_spatial_grid(testdata, target_lat, target_long)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_restrict_time_coverage():
    origOLR = olr.load_noaa_interpolated_olr(olr_data_filename)

    errors = []

    target = olr.restrict_time_coverage(origOLR, np.datetime64("1974-06-01"), np.datetime64("1974-06-03"))
    if not np.all(target.lat == origOLR.lat):
        errors.append("Latitude grid does not match original one")
    if not np.all(target.long == origOLR.long):
        errors.append("Logitude grid does not match original one")
    if not np.all(target.time == origOLR.time[:3]):
        errors.append("Time grid does not match the beginning of the original one")
    if not np.all(target.olr == origOLR.olr[:3, :, :]):
        errors.append("OLR data does not match the beginning of the original one")

    target = olr.restrict_time_coverage(origOLR, np.datetime64("1974-06-03"), np.datetime64("1974-06-05"))
    if not np.all(target.lat == origOLR.lat):
        errors.append("Latitude grid does not match original one")
    if not np.all(target.long == origOLR.long):
        errors.append("Logitude grid does not match original one")
    if not np.all(target.time == origOLR.time[2:5]):
        errors.append("Time grid does not match the beginning of the original one")
    if not np.all(target.olr == origOLR.olr[2:5, :, :]):
        errors.append("OLR data does not match the beginning of the original one")

    # Test period that is not covered by data set.
    with pytest.raises(ValueError) as e:
        target = olr.restrict_time_coverage(origOLR, np.datetime64("1973-06-01"), np.datetime64("1973-06-03"))

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_save_to_npzfile_restore_from_npzfile(tmp_path):
    filename = tmp_path / "OLRSaveTest.npz"
    time = np.arange("2018-01-01", "2018-01-04", dtype='datetime64[D]')
    lat= np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix=np.random.rand(3, 2, 4)
    testdata = olr.OLRData(olrmatrix, time, lat, long)
    testdata.save_to_npzfile(filename)

    target = olr.restore_from_npzfile(filename)

    errors = []
    if not np.all(target.olr == olrmatrix):
        errors.append("OLR data incorrect")
    if not np.all(target.time == time):
        errors.append("Time incorrect")
    if not np.all(target.lat == lat):
        errors.append("Latiturde incorrect")
    if not np.all(target.long == long):
        errors.append("Longitude incorrect")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_get_olr_for_date():
    time = np.arange("2018-01-01", "2018-01-04", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(3, 2, 4)
    testdata = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    target = testdata.get_olr_for_date(np.datetime64("2018-01-01"))
    if not np.all(target == np.squeeze(olrmatrix[0, :, :])):
        errors.append("Returned wrong OLR data for index 0.")

    target = testdata.get_olr_for_date(np.datetime64("2018-01-02"))
    if not np.all(target == np.squeeze(olrmatrix[1, :, :])):
        errors.append("Returned wrong OLR data for index 1.")

    target = testdata.get_olr_for_date(np.datetime64("2017-01-01"))
    print(target)
    if target is not None:
        errors.append("Returned wrong OLR data for unknown date.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_extract_olr_matrix_for_doy_range():

    time = np.arange("2018-01-01", "2018-01-10", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(9, 2, 4)
    testdata = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    target = testdata.extract_olr_matrix_for_doy_range(4, 2, strict_leap_year_treatment=True)
    if not np.all(target == np.squeeze(olrmatrix[1:6, :, :])):
        errors.append("Returned wrong OLR data for DOY 4, length 2.")

    target = testdata.extract_olr_matrix_for_doy_range(4, 3, strict_leap_year_treatment=True)
    if not np.all(target == np.squeeze(olrmatrix[0:7, :, :])):
        errors.append("Returned wrong OLR data for DOY 4, length 3.")

    # test period of two years (DOY is found twice)
    time = np.arange("2018-01-01", "2019-01-10", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(9+365, 2, 4)
    testdata = olr.OLRData(olrmatrix, time, lat, long)
    target = testdata.extract_olr_matrix_for_doy_range(4, 2, strict_leap_year_treatment=True)
    inds = np.concatenate((np.arange(1, 6, 1), np.arange(1, 6, 1) + 365))
    if not np.all(target == np.squeeze(olrmatrix[inds, :, :])):
        errors.append("Returned wrong OLR data for DOY 4, length 2 oder 2 years")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_equality_operator():
    time = np.arange("2018-01-01", "2018-01-04", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(3, 2, 4)
    control = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    target = olr.OLRData(olrmatrix, time, lat, long)
    if not target == control:
        errors.append("equality not detected")

    target = olr.OLRData(olrmatrix + 5., time, lat, long)
    if target == control:
        errors.append("inequality of olr not detected")

    target = olr.OLRData(olrmatrix, np.arange("2017-01-01", "2017-01-04", dtype='datetime64[D]'), lat, long)
    if target == control:
        errors.append("inequality of time not detected")

    target = olr.OLRData(olrmatrix, time, np.array([-2.7, 2.5]), long)
    if target == control:
        errors.append("inequality of lat not detected")

    target = olr.OLRData(olrmatrix, time, lat, np.array([11, 20, 30, 40]))
    if target == control:
        errors.append("inequality of long not detected")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_close():
    time = np.arange("2018-01-01", "2018-01-04", dtype='datetime64[D]')
    lat = np.array([-2.5, 2.5])
    long = np.array([10, 20, 30, 40])
    olrmatrix = np.random.rand(3, 2, 4)
    control = olr.OLRData(olrmatrix, time, lat, long)

    errors = []

    target = olr.OLRData(olrmatrix, time, lat, long)
    if not target.close(control):
        errors.append("equality not detected")

    target = olr.OLRData(olrmatrix + 5., time, lat, long)
    if target.close(control):
        errors.append("inequality of olr not detected")

    target = olr.OLRData(olrmatrix, np.arange("2017-01-01", "2017-01-04", dtype='datetime64[D]'), lat, long)
    if target.close(control):
        errors.append("inequality of time not detected")

    target = olr.OLRData(olrmatrix, time, np.array([-2.7, 2.5]), long)
    if target.close(control):
        errors.append("inequality of lat not detected")

    target = olr.OLRData(olrmatrix, time, lat, np.array([11, 20, 30, 40]))
    if target.close(control):
        errors.append("inequality of long not detected")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))