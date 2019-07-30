# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:01:37 2019

@author: ch
"""

import os
import warnings
import numpy as np
import mjoindex_omi.omi_calculator as omi
import mjoindex_omi.olr_handling as olr
import mjoindex_omi.plotting as plotting


def compare_Recalc_OMI_PCs_OriginalOLROriginalEOFs():
    """ Calulates and plots OMI PCs, which are compareable to the original PCs.

    The calculations is based based on the original EOFs and the original OLR
    dataset. Both have to be downloaded and stored locally before the example
    is executeable

    Furthermore, the original OMI PC file is needed to be able to procude the
    comparison plot.

    See tests/testdata/README for download links and local storage directories.

    """

    olrDataFilename = (os.path.dirname(__file__)
                       + os.path.sep
                       + "tests"
                       + os.path.sep
                       + "testdata"
                       + os.path.sep
                       + "olr.day.mean.nc")
    if not os.path.isfile(olrDataFilename):
        raise Exception("OLR data file not available. Expected file: %s" % olrDataFilename)

    originalOMIDataDirname = (os.path.dirname(__file__)
                              + os.path.sep
                              + "tests"
                              + os.path.sep
                              + "testdata"
                              + os.path.sep
                              + "OriginalOMI")
    if not os.path.isdir(originalOMIDataDirname):
        raise Exception("Path to original OMI EOFs is missing. Expected path: %s" % originalOMIDataDirname)

    origOMIPCsFilename = (originalOMIDataDirname
                          + os.path.sep
                          + "omi.1x.txt")
    if not os.path.isfile(origOMIPCsFilename):
        warnings.warn(
            "File with the original OMI PCs are missing. Generation of the comparison plot will fail. Expected file: %s" % origOMIPCsFilename)

    resultfile = (os.path.dirname(__file__)
                  + os.path.sep
                  + "example_data"
                  + os.path.sep
                  + "RecalcPCsOrigOLROrigEOF.txt")
    resultfigfile = (os.path.dirname(__file__)
                     + os.path.sep
                     + "example_data"
                     + os.path.sep
                     + "RecalcPCsOrigOLROrigEOF")

    olrData = olr.loadNOAAInterpolatedOLR(olrDataFilename)
    (target_pc1, target_pc2) = omi.calculatePCsFromOLRWithOriginalConditions(
        olrData,
        originalOMIDataDirname,
        np.datetime64("1979-01-01"),
        np.datetime64("2018-08-28"),
        resultfile,
        useQuickTemporalFilter=True)

    fig = plotting.plotComparisonOrigRecalcPCs(resultfile, origOMIPCsFilename, np.datetime64("2011-06-01"),
                                               np.datetime64("2011-12-31"))
    fig.show()
    fig.savefig(resultfigfile + ".png", bbox_inches='tight')
    fig.savefig(resultfigfile + ".pdf", bbox_inches='tight')


if __name__ == '__main__':
    compare_Recalc_OMI_PCs_OriginalOLROriginalEOFs()
