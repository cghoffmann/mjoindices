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
import os.path

import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindices.olr_handling as olr

testdata_dir = Path(os.path.abspath('')) / "testdata"

olr_data_filename = Path(os.path.abspath('')) / "testdata" / "olr.day.mean.nc"

reference_file_filterOLRForMJO_EOF_Calculation_lat0 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat0.npz"
reference_file_filterOLRForMJO_EOF_Calculation_lat5 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat5.npz"
reference_file_filterOLRForMJO_EOF_Calculation_latmin10 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOEOFCond_lat-10.npz"

reference_file_filterOLRForMJO_PC_Calculation_lat0 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOPCCond_lat0.npz"
reference_file_filterOLRForMJO_PC_Calculation_lat5 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOPCCond_lat5.npz"
reference_file_filterOLRForMJO_PC_Calculation_latmin10 = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "olr_ref_filteredForMJOPCCond_lat-10.npz"


@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.exists(), reason="OLR data file not available")
def test_mjoindices_reference_validation_filterOLRForMJO_EOF_Calculation():

    errors = []

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_lat0)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 0 not identical")

    lat = np.array([5])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_lat5)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 5 not identical")

    lat = np.array([-10.])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_EOF_Calculation_latmin10)
    if not target.close(control):
        errors.append("Filtered OLR for latitude -10 not identical")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.exists(), reason="OLR data file not available")
def test_mjoindices_reference_validation_filterOLRForMJO_PC_Calculation():

    errors = []

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_PC_Calculation_lat0)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 0 not identical")

    lat = np.array([5])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_PC_Calculation_lat5)
    if not target.close(control):
        errors.append("Filtered OLR for latitude 5 not identical")

    lat = np.array([-10])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    target = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    control = olr.restore_from_npzfile(reference_file_filterOLRForMJO_PC_Calculation_latmin10)
    if not target.close(control):
        errors.append("Filtered OLR for latitude -10 not identical")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def generate_reference_data_for_eof_filter_tests():

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_lat0) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([5])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_lat5) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([-10])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_eof_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_EOF_Calculation_latmin10) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)


def generate_reference_data_for_pc_filter_tests():

    orig_long = np.arange(0., 359.9, 2.5)

    test_olr = olr.load_noaa_interpolated_olr(olr_data_filename)

    lat = np.array([0])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_PC_Calculation_lat0) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([5])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_PC_Calculation_lat5) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)

    lat = np.array([-10])
    test_olr_part = olr.interpolate_spatial_grid(test_olr, lat, orig_long)
    olrdata_filtered = wkfilter.filter_olr_for_mjo_pc_calculation(test_olr_part)
    filename = Path(str(reference_file_filterOLRForMJO_PC_Calculation_latmin10) + ".newcalc")
    olrdata_filtered.save_to_npzfile(filename)


if __name__ == '__main__':
    generate_reference_data_for_eof_filter_tests()
    generate_reference_data_for_pc_filter_tests()