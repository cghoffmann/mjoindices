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

import warnings
from pathlib import Path

import numpy as np

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.principal_components as pc


#FIXME: Place contents of this file into examples directory

def compare_Recalc_OMI_PCs_OriginalOLROriginalEOFs():
    """ Calulates and plots OMI PCs, which are compareable to the original PCs.

    The calculations is based based on the original EOFs and the original OLR
    dataset. Both have to be downloaded and stored locally before the example
    is executeable

    Furthermore, the original OMI PC file is needed to be able to procude the
    comparison plot.

    See tests/testdata/README for download links and local storage directories.

    """

    olr_data_filename = Path(__file__).resolve().parent / "tests" / "testdata" / "olr.day.mean.nc"
    originalOMIDataDirname = Path(__file__).resolve().parent / "tests" / "testdata" / "OriginalOMI"
    origOMIPCsFilename = originalOMIDataDirname / "omi.1x.txt"

    if not olr_data_filename.is_file():
        raise Exception("OLR data file not available. Expected file: %s" % olr_data_filename)

    if not originalOMIDataDirname.is_dir():
        raise Exception("Path to original OMI EOFs is missing. Expected path: %s" % originalOMIDataDirname)

    if not origOMIPCsFilename.is_file():
        warnings.warn(
            "File with the original OMI PCs are missing. Generation of the comparison plot will fail. Expected file: %s" % origOMIPCsFilename)

    resultfile = Path(__file__).resolve().parent / "example_data" / "RecalcPCsOrigOLROrigEOF.txt"
    
    resultfigfile = Path(__file__).resolve().parent / "example_data" / "RecalcPCsOrigOLROrigEOF"

    olrData = olr.load_noaa_interpolated_olr(olr_data_filename)
    target = omi.calculatePCsFromOLRWithOriginalConditions(
        olrData,
        originalOMIDataDirname,
        useQuickTemporalFilter=True)
    target.save_pcs_to_txt_file(resultfile)

    orig_pcs = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)
    fig = pc.plot_comparison_orig_calc_pcs(target, orig_pcs, np.datetime64("2011-06-01"), np.datetime64("2011-12-31"))
    fig.show()
    fig.savefig(resultfigfile.with_suffix(".png"), bbox_inches='tight')
    fig.savefig(resultfigfile.with_suffix(".pdf"), bbox_inches='tight')


if __name__ == '__main__':
    compare_Recalc_OMI_PCs_OriginalOLROriginalEOFs()
