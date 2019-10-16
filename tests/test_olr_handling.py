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


#FIXME: Test basic properties of OLRData

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
    target = olr.resample_spatial_grid_to_original(origOLR)
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

#FIXME: Add spatial resampling test, with a really different grid
#FIXME: also test resample_spatial_grid
@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_restrict_time_coverage():
    origOLR = olr.load_noaa_interpolated_olr(olr_data_filename)
    target = olr.restrict_time_coverage(origOLR, np.datetime64("1974-06-01"), np.datetime64("1974-06-03"))
    errors = []
    print(target.time)
    print(origOLR.time[:3])
    if not np.all(target.lat == origOLR.lat):
        errors.append("Latitude grid does not match original one")
    if not np.all(target.long == origOLR.long):
        errors.append("Logitude grid does not match original one")
    if not np.all(target.time == origOLR.time[:3]):
        errors.append("Time grid does not match the beginning of the original one")
    if not np.all(target.olr == origOLR.olr[:3,:,:]):
        errors.append("OLR data does not match the beginning of the original one")
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

