# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:17 2019

@author: ch
"""
import os
import math
import pytest
import numpy as np
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




def test_save_pcs_to_txt_file_and_load_pcs_from_txt_file(tmp_path):
    filename = tmp_path / "test_save_pcs_to_txt_file.txt"
    test_pc1 = np.array([0.12345678, 0.33333333, 0.555555555])
    test_pc2 = np.array([0.38462392, 0.44444444, 0.666666666])
    test_dates= np.array([np.datetime64("2019-06-10"),np.datetime64("2019-06-11"),np.datetime64("2019-06-12"),])
    omiio.save_pcs_to_txt_file(test_dates, test_pc1, test_pc2, filename)

    target_dates, target_pc1, target_pc2 = omiio.load_pcs_from_txt_file(filename)

    errors = []
    if not np.all(test_dates == target_dates):
        errors.append("Dates do not match.")
    if not np.allclose(np.array([0.12346, 0.33333, 0.55556]), target_pc1):
        errors.append("PC1 values do not match.")
    if not np.allclose(np.array([0.38462, 0.44444, 0.66667]), target_pc2):
        errors.append("PC2 values do not match.")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not os.path.isfile(origOMIPCsFilename),
                    reason="Original OMI PCs not available for comparison")
def test_load_original_pcs_from_txt_file():
    #works with original data file that ends on August 28, 2018.
    target_dates, target_pc1, target_pc2 = omiio.load_original_pcs_from_txt_file(origOMIPCsFilename)

    errors = []
    if not target_pc1[0] == 0.16630:
        errors.append("First Entry of PC1 wrong!")
    if not target_pc2[0] == 0.76455:
        errors.append("First Entry of PC2 wrong!")
    if not target_dates[0] == np.datetime64("1979-01-01"):
        errors.append("First Entry of Dates wrong!")

    index_somewhere = 10
    if not target_pc1[index_somewhere] == -1.49757:
        errors.append("Some entry of PC1 wrong!")
    if not target_pc2[index_somewhere] == 0.30697:
        errors.append("Some entry of PC2 wrong!")
    if not target_dates[index_somewhere] == np.datetime64("1979-01-11"):
        errors.append("Some entry of Dates wrong!")

    if not target_pc1[-1] == 0.23704:
        errors.append("Last Entry of PC1 wrong!")
    if not target_pc2[-1] == 0.17256:
        errors.append("Last Entry of PC2 wrong!")
    if not target_dates[-1] == np.datetime64("2018-08-28"):
        errors.append("Last Entry of Dates wrong!")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))