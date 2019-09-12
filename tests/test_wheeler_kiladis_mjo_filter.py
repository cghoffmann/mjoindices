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

import subprocess
from pathlib import Path

import numpy as np

from src import mjoindices as wkfilter

testdata_dir = Path(__file__).parent / "testdata"

def test_filter_MJOCondition_lat0deg():
    reference_dir = testdata_dir / "WKFilterReference" / "lat0degPyIdx36"
    # FIXME Try to use common OLR data
    # olr_dir = Path("/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
    # raw_olr = loadKiladisBinaryOLRDataTwicePerDay(olr_dir / "olr.2x.7918.b")
    # test_olr = np.squeeze(raw_olr.olr[:, 36, :])
    test_olr = wkfilter.loadKiladisOriginalOLR(reference_dir / "OLROriginal.b")

    validator = wkfilter.WKFilterValidator(test_olr, reference_dir, do_plot=0, atol=1e-8, rtol=100.)
    errors = validator.validate_WKFilter_perform2dimSpectralSmoothing_MJOConditions()

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def configure_and_run_fortran_code(lat_index_fortran: int):
    fortranfile = "/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/stfilt_CHDebugOutput_MJOConditions_Automatic.f"
    scriptfile = "/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/compileAndStartFilter_CHDebugOutputMJOCond_Automatic.sh"
    with open(fortranfile, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[112] = "      parameter (soutcalc=%i,noutcalc=%i)  ! Region of output 90ns AUTOMATIC CHANGE!\n" %(lat_index_fortran,lat_index_fortran)

    # and write everything back
    with open(fortranfile, 'w') as file:
        file.writelines(data)
    out = subprocess.call([scriptfile],cwd="/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
    print(out)


def check_test_input_OLRData():
    data_exchange_dir = Path("/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
    kiladis_olr = wkfilter.loadKiladisBinaryOLRDataTwicePerDay(data_exchange_dir / "olr.2x.7918.b")
    k_inputOLR = wkfilter.loadKiladisOriginalOLR(data_exchange_dir / "OLROriginal.b")

    found = None
    for i in range(0, kiladis_olr.olr.shape[1]):
        if np.all(np.isclose(np.squeeze(kiladis_olr.olr[:,i,:]), k_inputOLR)):
            found = i
    print(found)
    testdata = np.squeeze(kiladis_olr.olr[:, found, :])  # select one latitude
    print(np.mean(testdata - k_inputOLR))


#configure_and_run_fortran_code(29)
#check_test_input_OLRData()