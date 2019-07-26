# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:17 2019

@author: ch
"""
import os
import os.path
import pytest
import numpy as np
import math

import mjoindex_omi.olr_handling as olr

olr_data_filename = os.path.dirname(__file__) + os.path.sep + "testdata" + os.path.sep + "olr.day.mean.nc"

@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_loadNOAAInterpolatedOLR():
    errors = []
    target = olr.loadNOAAInterpolatedOLR(olr_data_filename)

    # Check time grid
    # Period always starts on 1974/06/01, whereas the ending date
    # changes when file is updated
    if not target.time[0] == np.datetime64("1974-06-01"):
        errors.append("First date does not match")
    if not ((target.time[1] - target.time[0]).astype('timedelta64[D]')/np.timedelta64(1,"D")) == 1:
        errors.append("Temporal spacing does not match 1 day")

    # Check latitude grid
    # First latitude in file is 90deg. but order is reversed to be
    # consistent with original implemenation by G. Kiladis
    if not target.lat[0] == -90:
        errors.append("First latitude entry does not matched reversed order")
    if not target.lat[3] == -82.5:
        errors.append("Forth latitude entry does not matched reversed order")
    if not target.lat[-1] == 90:
        errors.append("Last latitude entry does not matched reversed order")
    if not (target.lat[0] - target.lat[1]) == -2.5:
        errors.append("Latitudnal spacing does not meet the expectation")

    # Check longitude grid
    if not target.long[0] == 0:
        errors.append("First latitude entry does not matched reversed order")
    if not target.long[-1] == 357.5:
        errors.append("Last latitude entry does not matched reversed order")
    if not target.long[1] - target.long[0] == 2.5:
        errors.append("Longitudinal spacing does not meet the expectation")

    # Check OLR Data
    # OLR samples extracted from file using Panoply viewer, which directly
    # applies scaling and offset values
    # Reversed order of latitude grid has been considered manually
    if not math.isclose(target.olr[0,-1,0],205.450):
         errors.append("First OLR sample value does not match")
    if not math.isclose(target.olr[0, 3, 0],117.600):
         errors.append("Second OLR sample value does not match")
    if not math.isclose(target.olr[4, 3, 15],122.700):
         errors.append("Third OLR sample value does not match")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_resampleOLRToOriginalSpatialGrid():
    origOLR = olr.loadNOAAInterpolatedOLR(olr_data_filename)
    target = olr.resampleOLRToOriginalSpatialGrid(origOLR)
    errors = []
    if not np.all(target.lat == origOLR.lat[28:45]):
        errors.append("Latitude grid does not match middle of original one")
    if not np.all(target.long == origOLR.long):
        errors.append("Logitude grid does not match original one")
    if not np.all(target.time == origOLR.time):
        errors.append("Time grid does not match original one")
    if not np.all(target.olr == origOLR.olr[:,28:45,:]):
        errors.append("OLR data does not match original one in the tropics")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

#FIXME: Add spatial resampling test, with a really different grid

@pytest.mark.skipif(not os.path.isfile(olr_data_filename),
                    reason="OLR data file not available")
def test_restrictOLRDataToTimeRange():
    origOLR = olr.loadNOAAInterpolatedOLR(olr_data_filename)
    target = olr.restrictOLRDataToTimeRange(origOLR, np.datetime64("1974-06-01"), np.datetime64("1974-06-03"))
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
    assert not errors, "errors occured:\n{}".format("\n".join(errors))