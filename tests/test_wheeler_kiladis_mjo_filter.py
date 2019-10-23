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
import pytest
import numpy as np

import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindices.olr_handling as olr

testdata_dir = Path(__file__).resolve().parent / "testdata"

olr_data_filename = Path(__file__).resolve().parent / "testdata" / "olr.day.mean.nc"

reference_file_filterOLRForMJO_EOF_Calculation_lat0 = Path(__file__).resolve().parent / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat0.npz"
reference_file_filterOLRForMJO_EOF_Calculation_lat5 = Path(__file__).resolve().parent / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat5.npz"
reference_file_filterOLRForMJO_EOF_Calculation_latmin10 = Path(__file__).resolve().parent / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat-10.npz"



@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.exists(), reason="OLR data file not available")
def test_mjoindices_reference_validation_filterOLRForMJO_EOF_Calculation():

    errors = []

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.resample_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_lat0)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 0 not identical")

    lat = np.array([5])
    test_olr_part = olr.resample_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_lat5)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 5 not identical")

    lat = np.array([-10.])
    test_olr_part = olr.resample_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_latmin10)
    if not target.close(control):
        errors.append("Filtered OLR for latitude -10 not identical")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skip
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

def generateReferenceData():

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.resample_spatial_grid(test_olr,lat, orig_long)
    olrdata_filtered = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_lat0) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([5])
    test_olr_part = olr.resample_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_lat5) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([-10])
    test_olr_part = olr.resample_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filterOLRForMJO_EOF_Calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_latmin10) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)




#configure_and_run_fortran_code(29)
#check_test_input_OLRData()
#generateReferenceData()