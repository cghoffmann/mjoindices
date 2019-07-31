# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:17 2019

@author: ch
"""
import math
import os

import pytest

import mjoindex_omi.io as omiio

originalOMIDataDirname = (os.path.dirname(__file__)
                          + os.path.sep
                          + "testdata"
                          + os.path.sep
                          + "OriginalOMI")

eof1Dirname = (originalOMIDataDirname
               + os.path.sep
               + "eof1")
eof2Dirname = (originalOMIDataDirname
               + os.path.sep
               + "eof2")

origOMIPCsFilename = (originalOMIDataDirname
                      + os.path.sep
                      + "omi.1x.txt")

@pytest.mark.skipif(not os.path.isdir(eof1Dirname),
                    reason="EOF1 data not available")
@pytest.mark.skipif(not os.path.isdir(eof2Dirname),
                    reason="EOF2 data not available")
def test_load_eofs_for_doy():
    target1_eof1, target1_eof2 = omiio.load_eofs_for_doy(originalOMIDataDirname,1)
    target10_eof1, target10_eof2 = omiio.load_eofs_for_doy(originalOMIDataDirname, 10)
    target366_eof1, target366_eof2 = omiio.load_eofs_for_doy(originalOMIDataDirname, 366)

    errors = []
    if not math.isclose(target1_eof1[0], 0.00022178496):
        errors.append("EOF1 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target1_eof1[10], -0.0023467445):
        errors.append("EOF1 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target1_eof1[-1], 0.013897266):
        errors.append("EOF1 of DOY 1 is incorrect (Last position)")
    if not math.isclose(target1_eof2[0], 0.0042107304):
        errors.append("EOF2 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target1_eof2[10], 0.015404793):
        errors.append("EOF2 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target1_eof2[-1], 0.012487547):
        errors.append("EOF2 of DOY 1 is incorrect (Last position)")

    if not math.isclose(target10_eof1[0], 0.00016476621):
        errors.append("EOF1 of DOY 10 is incorrect (Position 0)")
    if not math.isclose(target10_eof2[0], 0.0044616843):
        errors.append("EOF2 of DOY 10 is incorrect (Position 0)")

    if not math.isclose(target366_eof1[-1], 0.013874311):
        errors.append("EOF1 of DOY 366 is incorrect (Last position)")
    if not math.isclose(target366_eof2[-1], 0.012473147):
        errors.append("EOF2 of DOY 366 is incorrect (Last position)")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))