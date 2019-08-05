# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:42:52 2019

@author: ch
"""

from pathlib import Path

import numpy as np
import pytest

import mjoindex_omi.olr_handling as olr
import mjoindex_omi.omi_calculator as omi
import mjoindex_omi.principal_components as pc

olr_data_filename = Path(__file__).parent / "testdata" /"olr.day.mean.nc"
originalOMIDataDirname = Path(__file__).parent / "testdata" / "OriginalOMI"
eof1Dirname = originalOMIDataDirname / "eof1"
eof2Dirname = originalOMIDataDirname / "eof2"
origOMIPCsFilename = originalOMIDataDirname / "omi.1x.txt"


setups = [(True, 0.99, 0.99), (False, 0.999, 0.999)]
@pytest.mark.slow
@pytest.mark.parametrize("useQuickTemporalFilter, expectedCorr1, expectedCorr2", setups)
@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available")
@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available")
@pytest.mark.skipif(not origOMIPCsFilename.is_file(), reason="Original OMI PCs not available for comparison")
def test_calculatePCsFromOLRWithOriginalConditions_Quickfilter(useQuickTemporalFilter, expectedCorr1, expectedCorr2):

    orig_omi = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)
    olrData = olr.load_noaa_interpolated_olr(olr_data_filename)

    target = omi.calculatePCsFromOLRWithOriginalConditions(olrData,
                                                           originalOMIDataDirname,
                                                           useQuickTemporalFilter=useQuickTemporalFilter)
    errors = []
    if not np.all(target.time == orig_omi.time):
        errors.append("Test is not reasonable, because temporal coverages of original OMI and recalculation do not "
                      "fit. Maybe wrong original file downloaded? Supported is the one with coverage until August 28, "
                      "2018.")

    corr1 = (np.corrcoef(orig_omi.pc1, target.pc1))[0, 1]
    if not corr1 > expectedCorr1:
        errors.append("Correlation of PC1 too low!")

    corr2 = (np.corrcoef(orig_omi.pc2, target.pc2))[0, 1]
    if not corr2 > expectedCorr2:
        errors.append("Correlation of PC2 too low!")

    # FIXME: Define more test criteria: e.g., max deviation etc.

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))