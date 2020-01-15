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

from pathlib import Path
import os.path

import numpy as np
import pytest
import importlib


import mjoindices.omi.omi_calculator as omi
import mjoindices.principal_components as pc
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.evaluation_tools
import mjoindices.olr_handling as olr

olr_data_filename = Path(os.path.abspath('')) / "testdata" / "olr.day.mean.nc"
originalOMIDataDirname = Path(os.path.abspath('')) / "testdata" / "OriginalOMI"
eof1Dirname = originalOMIDataDirname / "eof1"
eof2Dirname = originalOMIDataDirname / "eof2"
origOMIPCsFilename = originalOMIDataDirname / "omi.1x.txt"
mjoindices_reference_eofs_filename_strict = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_strict.npz"
mjoindices_reference_pcs_filename_strict = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "PCs_strict.txt"
mjoindices_reference_eofs_filename_coarsegrid = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_coarsegrid.npz"
mjoindices_reference_pcs_filename_coarsegrid = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "PCs_coarsegrid.txt"
mjoindices_reference_eofs_filename_strict_eofs = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_eofs_package.npz"
mjoindices_reference_pcs_filename_strict_eofs = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "PCs_eofs_package.txt"
mjoindices_reference_eofs_filename = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs.npz"
mjoindices_reference_pcs_filename = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "PCs.txt"
original_omi_explained_variance_file = Path(os.path.abspath('')) / "testdata" / "OriginalOMI" / "omi_var.txt"



setups = [(True, 0.99, 0.99), (False, 0.999, 0.999)]
@pytest.mark.slow
@pytest.mark.parametrize("use_quick_temporal_filter, expectedCorr1, expectedCorr2", setups)
@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available")
@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available")
@pytest.mark.skipif(not origOMIPCsFilename.is_file(), reason="Original OMI PCs not available for comparison")
def test_calculatePCsFromOLRWithOriginalEOFs(use_quick_temporal_filter, expectedCorr1, expectedCorr2):
# This test is quicker than the complete integration tests below. The quality check is very simple and just checks
# the correlations of the PC time series. Hence, this test in thought the get a first idea and more intensive testing
# should be done using the tests below.
    orig_omi = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)
    olrData = olr.load_noaa_interpolated_olr(olr_data_filename)

    target = omi.calculate_pcs_from_olr_original_conditions(olrData,
                                                            originalOMIDataDirname,
                                                            use_quick_temporal_filter=use_quick_temporal_filter)
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

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available.")
@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available not available for comparison.")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available not available for comparison.")
@pytest.mark.skipif(not origOMIPCsFilename.is_file(), reason="Original OMI PCs not available for comparison.")
def test_completeOMIReproduction_strict_leap_year_treatment(tmp_path):

    errors = []

    # Calculate EOFs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
    interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)
    eofs = omi.calc_eofs_from_olr(interpolated_olr,
                                  sign_doy1reference=True,
                                  interpolate_eofs=True,
                                  strict_leap_year_treatment=True)
    eofs.save_all_eofs_to_npzfile(tmp_path / "test_completeOMIReproduction_strict_leap_year_treatment_EOFs.npz")

    # Validate EOFs against original (results are inexact but close)
    orig_eofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)

    corr_1, diff_mean_1, diff_std_1, diff_abs_percent68_1, diff_abs_percent95_1, diff_abs_percent99_1 = mjoindices.evaluation_tools.calc_comparison_stats_for_eofs_all_doys(orig_eofs, eofs, 1,  exclude_doy366=True, percentage=False, do_print=False)
    if not np.all(corr_1 > 0.994):
        errors.append("original-validation: Correlation for EOF1 at least for one DOY too low!")
    if not np.all(diff_abs_percent99_1 < 0.0084):
        errors.append("original-validation: 99% percentile for EOF1 at least for one DOY too high!")
    if not np.all(diff_abs_percent68_1 < 0.0018):
        errors.append("original-validation: 68% percentile for EOF1 at least for one DOY too high!")

    corr_2, diff_mean_2, diff_std_2, diff_abs_percent68_2, diff_abs_percent95_2, diff_abs_percent99_2 = mjoindices.evaluation_tools.calc_comparison_stats_for_eofs_all_doys(orig_eofs, eofs, 2,  exclude_doy366=True, percentage=False, do_print=False)
    if not np.all(corr_2 > 0.993):
        errors.append("original-validation: Correlation for EOF2 at least for one DOY too low!")
    if not np.all(diff_abs_percent99_2 < 0.0065):
        errors.append("original-validation: 99% percentile for EOF2 at least for one DOY too high!")
    if not np.all(diff_abs_percent68_2 < 0.0018):
        errors.append("original-validation: 68% percentile for EOF2 at least for one DOY too high!")

    # Validate explained variance against original (results are inexact but close)
    orig_explained_variance_1, orig_explained_variance_2 = mjoindices.evaluation_tools.load_omi_explained_variance(original_omi_explained_variance_file)

    corr_var1, diff_mean_var1, diff_std_var1, diff_vec_var1, diff_abs_percent68_var1, diff_abs_percent95_var1, diff_abs_percent99_var1 = mjoindices.evaluation_tools.calc_comparison_stats_for_explained_variance(orig_explained_variance_1, eofs.explained_variance1_for_all_doys(), do_print=False, exclude_doy366=True)
    if not diff_std_var1 < 0.0007:
        errors.append("original-validation: Std.Dev. of the difference of both explained variances for EOF1 is to too high!")
    if not diff_abs_percent99_var1 < 0.0013:
        errors.append("original-validation: 99% percentile of the difference of both explained variances for EOF1 is to too high!")
    if not diff_abs_percent68_var1 < 0.0007:
        errors.append("original-validation: 68% percentile of the difference of both explained variances for EOF1 is to too high!")

    corr_var2, diff_mean_var2, diff_std_var2, diff_vec_var2, diff_abs_percent68_var2, diff_abs_percent95_var2, diff_abs_percent99_var2 = mjoindices.evaluation_tools.calc_comparison_stats_for_explained_variance(
        orig_explained_variance_2, eofs.explained_variance2_for_all_doys(), do_print=False, exclude_doy366=True)
    if not diff_std_var2 < 0.0006:
        errors.append(
            "original-validation: Std.Dev. of the difference of both explained variances for EOF2 is to too high!")
    if not diff_abs_percent99_var2 < 0.0014:
        errors.append(
            "original-validation: 99% percentile of the difference of both explained variances for EOF2 is to too high!")
    if not diff_abs_percent68_var2 < 0.0007:
        errors.append(
            "original-validation: 68% percentile of the difference of both explained variances for EOF2 is to too high!")

    # Validate EOFs against mjoindices own reference (results should be equal)
    mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_strict)
    for idx, target_eof in enumerate(eofs.eof_list):
        if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
            errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

    # Calculate PCs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    pcs = omi.calculate_pcs_from_olr(raw_olr,
                                     eofs,
                                     np.datetime64("1979-01-01"),
                                     np.datetime64("2018-08-28"),
                                     use_quick_temporal_filter=False)
    pc_temp_file = tmp_path / "test_completeOMIReproduction_strict_leap_year_treatment_PCs.txt"
    pcs.save_pcs_to_txt_file(pc_temp_file)

    # Validate PCs  against original (results are inexact but close)
    orig_pcs = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)
    # Reload pcs instead of using the calculated ones, because the saving routine has truncated some decimals of the
    # reference values. So do the same with the testing target pcs.
    pcs = pc.load_pcs_from_txt_file(pc_temp_file)

    tempa, tempb, corr_pc1, diff_mean_pc1, diff_std_pc1, diff_ts_abs_pc1, diff_abs_percent68_pc1, diff_abs_percent95_pc1, diff_abs_percent99_pc1 = mjoindices.evaluation_tools.calc_timeseries_agreement(orig_pcs.pc1, orig_pcs.time, pcs.pc1, pcs.time, exclude_doy366=True, do_print=False)
    if not corr_pc1 > 0.998:
        errors.append("original-validation: Correlation for PC1 timeseries is to too low!")
    if not diff_std_pc1 < 0.0449:
        errors.append("original-validation: Std.Dev. of the difference of both PC1 timeseries is to too high!")
    if not diff_abs_percent99_pc1 < 0.1523:
        errors.append("original-validation: 99% percentile of the difference of both PC1 timeseries is to too high!")
    if not diff_abs_percent68_pc1 < 0.0319:
        errors.append("original-validation: 68% percentile of the difference of both PC1 timeseries is to too high!")

    tempa, tempb, corr_pc2, diff_mean_pc2, diff_std_pc2, diff_ts_abs_pc2, diff_abs_percent68_pc2, diff_abs_percent95_pc2, diff_abs_percent99_pc2 = mjoindices.evaluation_tools.calc_timeseries_agreement(
        orig_pcs.pc2, orig_pcs.time, pcs.pc2, pcs.time, exclude_doy366=True, do_print=False)
    if not corr_pc2 > 0.998:
        errors.append("original-validation: Correlation for PC2 timeseries is to too low!")
    if not diff_std_pc2 < 0.0484:
        errors.append("original-validation: Std.Dev. of the difference of both PC2 timeseries is to too high!")
    if not diff_abs_percent99_pc2 < 0.1671:
        errors.append("original-validation: 99% percentile of the difference of both PC2 timeseries is to too high!")
    if not diff_abs_percent68_pc2 < 0.0350:
        errors.append("original-validation: 68% percentile of the difference of both PC2 timeseries is to too high!")


    strength = np.sqrt(np.square(pcs.pc1) + np.square(pcs.pc2))
    orig_strength = np.sqrt(np.square(orig_pcs.pc1) + np.square(orig_pcs.pc2))

    tempa, tempb, corr_strength, diff_mean_strength, diff_std_strength, diff_ts_abs_strength, diff_abs_percent68_strength, diff_abs_percent95_strength, diff_abs_percent99_strength = mjoindices.evaluation_tools.calc_timeseries_agreement(
        orig_strength, orig_pcs.time, strength, pcs.time, exclude_doy366=True, do_print=False)
    if not corr_strength > 0.9998:
        errors.append("original-validation: Correlation for strength timeseries is to too low!")
    if not diff_std_strength < 0.0105:
        errors.append("original-validation: Std.Dev. of the difference of both strength timeseries is to too high!")
    if not diff_abs_percent99_strength < 0.0350:
        errors.append("original-validation: 99% percentile of the difference of both strength timeseries is to too high!")
    if not diff_abs_percent68_strength < 0.0081:
        errors.append("original-validation: 68% percentile of the difference of both strength timeseries is to too high!")

    # Validate PCs against mjoindices own reference (results should be equal)
    mjoindices_reference_pcs = pc.load_pcs_from_txt_file(mjoindices_reference_pcs_filename_strict)
    if not np.all(mjoindices_reference_pcs.time == pcs.time):
        errors.append("mjoindices-reference-validation: Dates of PCs do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc1, pcs.pc1):
        errors.append("mjoindices-reference-validation: PC1 values do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc2, pcs.pc2):
        errors.append("mjoindices-reference-validation: PC2 values do not match.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available.")
@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available not available for comparison.")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available not available for comparison.")
@pytest.mark.skipif(not origOMIPCsFilename.is_file(), reason="Original OMI PCs not available for comparison.")
def test_completeOMIReproduction(tmp_path):

        errors = []

        # Calculate EOFs
        raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
        shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
        interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)
        eofs = omi.calc_eofs_from_olr(interpolated_olr,
                                      sign_doy1reference=True,
                                      interpolate_eofs=True,
                                      strict_leap_year_treatment=False)
        eofs.save_all_eofs_to_npzfile(tmp_path / "test_completeOMIReproduction_EOFs.npz")

        # Validate EOFs against original (results are inexact but close)
        orig_eofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)

        corr_1, diff_mean_1, diff_std_1, diff_abs_percent68_1, diff_abs_percent95_1, diff_abs_percent99_1 = mjoindices.evaluation_tools.calc_comparison_stats_for_eofs_all_doys(
            orig_eofs, eofs, 1, exclude_doy366=False, percentage=False, do_print=False)
        if not np.all(corr_1 > 0.994):
            errors.append("original-validation: Correlation for EOF1 at least for one DOY too low!")
        if not np.all(diff_abs_percent99_1 < 0.0084):
            errors.append("original-validation: 99% percentile for EOF1 at least for one DOY too high!")
        if not np.all(diff_abs_percent68_1 < 0.0018):
            errors.append("original-validation: 68% percentile for EOF1 at least for one DOY too high!")

        corr_2, diff_mean_2, diff_std_2, diff_abs_percent68_2, diff_abs_percent95_2, diff_abs_percent99_2 = mjoindices.evaluation_tools.calc_comparison_stats_for_eofs_all_doys(
            orig_eofs, eofs, 2, exclude_doy366=False, percentage=False, do_print=False)
        if not np.all(corr_2 > 0.993):
            errors.append("original-validation: Correlation for EOF2 at least for one DOY too low!")
        if not np.all(diff_abs_percent99_2 < 0.0065):
            errors.append("original-validation: 99% percentile for EOF2 at least for one DOY too high!")
        if not np.all(diff_abs_percent68_2 < 0.0018):
            errors.append("original-validation: 68% percentile for EOF2 at least for one DOY too high!")

        # Validate explained variance against original (results are inexact but close)
        orig_explained_variance_1, orig_explained_variance_2 = mjoindices.evaluation_tools.load_omi_explained_variance(
            original_omi_explained_variance_file)

        corr_var1, diff_mean_var1, diff_std_var1, diff_vec_var1, diff_abs_percent68_var1, diff_abs_percent95_var1, diff_abs_percent99_var1 = mjoindices.evaluation_tools.calc_comparison_stats_for_explained_variance(
            orig_explained_variance_1, eofs.explained_variance1_for_all_doys(), do_print=False, exclude_doy366=False)
        if not diff_std_var1 < 0.0008:
            errors.append(
                "original-validation: Std.Dev. of the difference of both explained variances for EOF1 is to too high!")
        if not diff_abs_percent99_var1 < 0.0017:
            errors.append(
                "original-validation: 99% percentile of the difference of both explained variances for EOF1 is to too high!")
        if not diff_abs_percent68_var1 < 0.0009:
            errors.append(
                "original-validation: 68% percentile of the difference of both explained variances for EOF1 is to too high!")

        corr_var2, diff_mean_var2, diff_std_var2, diff_vec_var2, diff_abs_percent68_var2, diff_abs_percent95_var2, diff_abs_percent99_var2 = mjoindices.evaluation_tools.calc_comparison_stats_for_explained_variance(
            orig_explained_variance_2, eofs.explained_variance2_for_all_doys(), do_print=False, exclude_doy366=False)
        if not diff_std_var2 < 0.0008:
            errors.append(
                "original-validation: Std.Dev. of the difference of both explained variances for EOF2 is to too high!")
        if not diff_abs_percent99_var2 < 0.0018:
            errors.append(
                "original-validation: 99% percentile of the difference of both explained variances for EOF2 is to too high!")
        if not diff_abs_percent68_var2 < 0.001:
            errors.append(
                "original-validation: 68% percentile of the difference of both explained variances for EOF2 is to too high!")

        # Validate EOFs against mjoindices own reference (results should be equal)
        mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename)
        for idx, target_eof in enumerate(eofs.eof_list):
            if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
                errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

        # Calculate PCs
        raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
        pcs = omi.calculate_pcs_from_olr(raw_olr,
                                         eofs,
                                         np.datetime64("1979-01-01"),
                                         np.datetime64("2018-08-28"),
                                         use_quick_temporal_filter=False)
        pc_temp_file = tmp_path / "test_completeOMIReproduction_PCs.txt"
        pcs.save_pcs_to_txt_file(pc_temp_file)

        # Validate PCs  against original (results are inexact but close)
        orig_pcs = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)
        # Reload pcs instead of using the calculated ones, because the saving routine has truncated some decimals of the
        # reference values. So do the same with the testing target pcs.
        pcs = pc.load_pcs_from_txt_file(pc_temp_file)

        tempa, tempb, corr_pc1, diff_mean_pc1, diff_std_pc1, diff_ts_abs_pc1, diff_abs_percent68_pc1, diff_abs_percent95_pc1, diff_abs_percent99_pc1 = mjoindices.evaluation_tools.calc_timeseries_agreement(
            orig_pcs.pc1, orig_pcs.time, pcs.pc1, pcs.time, exclude_doy366=False, do_print=False)
        if not corr_pc1 > 0.998:
            errors.append("original-validation: Correlation for PC1 timeseries is to too low!")
        if not diff_std_pc1 < 0.0458:
            errors.append("original-validation: Std.Dev. of the difference of both PC1 timeseries is to too high!")
        if not diff_abs_percent99_pc1 < 0.157:
            errors.append(
                "original-validation: 99% percentile of the difference of both PC1 timeseries is to too high!")
        if not diff_abs_percent68_pc1 < 0.0327:
            errors.append(
                "original-validation: 68% percentile of the difference of both PC1 timeseries is to too high!")

        tempa, tempb, corr_pc2, diff_mean_pc2, diff_std_pc2, diff_ts_abs_pc2, diff_abs_percent68_pc2, diff_abs_percent95_pc2, diff_abs_percent99_pc2 = mjoindices.evaluation_tools.calc_timeseries_agreement(
            orig_pcs.pc2, orig_pcs.time, pcs.pc2, pcs.time, exclude_doy366=False, do_print=False)
        if not corr_pc2 > 0.998:
            errors.append("original-validation: Correlation for PC2 timeseries is to too low!")
        if not diff_std_pc2 < 0.0488:
            errors.append("original-validation: Std.Dev. of the difference of both PC2 timeseries is to too high!")
        if not diff_abs_percent99_pc2 < 0.1704:
            errors.append(
                "original-validation: 99% percentile of the difference of both PC2 timeseries is to too high!")
        if not diff_abs_percent68_pc2 < 0.0353:
            errors.append(
                "original-validation: 68% percentile of the difference of both PC2 timeseries is to too high!")

        strength = np.sqrt(np.square(pcs.pc1) + np.square(pcs.pc2))
        orig_strength = np.sqrt(np.square(orig_pcs.pc1) + np.square(orig_pcs.pc2))

        tempa, tempb, corr_strength, diff_mean_strength, diff_std_strength, diff_ts_abs_strength, diff_abs_percent68_strength, diff_abs_percent95_strength, diff_abs_percent99_strength = mjoindices.evaluation_tools.calc_timeseries_agreement(
            orig_strength, orig_pcs.time, strength, pcs.time, exclude_doy366=False, do_print=False)
        if not corr_strength > 0.9998:
            errors.append("original-validation: Correlation for strength timeseries is to too low!")
        if not diff_std_strength < 0.0103:
            errors.append("original-validation: Std.Dev. of the difference of both strength timeseries is to too high!")
        if not diff_abs_percent99_strength < 0.0341:
            errors.append(
                "original-validation: 99% percentile of the difference of both strength timeseries is to too high!")
        if not diff_abs_percent68_strength < 0.0079:
            errors.append(
                "original-validation: 68% percentile of the difference of both strength timeseries is to too high!")

        # Validate PCs against mjoindices own reference (results should be equal)
        mjoindices_reference_pcs = pc.load_pcs_from_txt_file(mjoindices_reference_pcs_filename)
        if not np.all(mjoindices_reference_pcs.time == pcs.time):
            errors.append("mjoindices-reference-validation: Dates of PCs do not match.")
        if not np.allclose(mjoindices_reference_pcs.pc1, pcs.pc1):
            errors.append("mjoindices-reference-validation: PC1 values do not match.")
        if not np.allclose(mjoindices_reference_pcs.pc2, pcs.pc2):
            errors.append("mjoindices-reference-validation: PC2 values do not match.")

        assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_completeOMIReproduction_coarsegrid(tmp_path):

    errors = []

    # Calculate EOFs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
    coarse_lat = np.arange(-20., 20.1, 8.0)
    coarse_long = np.arange(0., 359.9, 20.0)
    interpolated_olr = olr.interpolate_spatial_grid(shorter_olr, coarse_lat, coarse_long)
    eofs = omi.calc_eofs_from_olr(interpolated_olr,
                                  sign_doy1reference=True,
                                  interpolate_eofs=True,
                                  strict_leap_year_treatment=True)
    eofs.save_all_eofs_to_npzfile(tmp_path / "test_completeOMIReproduction_coarsegrid_EOFs.npz")

    # Validate EOFs against mjoindices own reference (results should be equal)
    mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_coarsegrid)
    for idx, target_eof in enumerate(eofs.eof_list):
        if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
            errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

    # Calculate PCs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    pcs = omi.calculate_pcs_from_olr(raw_olr,
                                     eofs,
                                     np.datetime64("1979-01-01"),
                                     np.datetime64("2018-08-28"),
                                     use_quick_temporal_filter=False)
    pc_temp_file = tmp_path / "test_completeOMIReproduction_coarsegrid_PCs.txt"
    pcs.save_pcs_to_txt_file(pc_temp_file)

    # Validate PCs against mjoindices own reference (results should be equal)
    # Reload pcs instead of using the calculated ones, because the saving routine has truncated some decimals of the
    # reference values. So do the same with the testing target pcs.
    pcs = pc.load_pcs_from_txt_file(pc_temp_file)
    mjoindices_reference_pcs = pc.load_pcs_from_txt_file(mjoindices_reference_pcs_filename_coarsegrid)
    if not np.all(mjoindices_reference_pcs.time == pcs.time):
        errors.append("mjoindices-reference-validation: Dates of PCs do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc1, pcs.pc1):
        errors.append("mjoindices-reference-validation: PC1 values do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc2, pcs.pc2):
        errors.append("mjoindices-reference-validation: PC2 values do not match.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

eofs_spec = importlib.util.find_spec("eofs")
@pytest.mark.slow
@pytest.mark.skipif(not olr_data_filename.is_file(), reason="OLR data file not available.")
@pytest.mark.skipif(eofs_spec is None, reason="Optional eofs package is not available.")
def test_completeOMIReproduction_eofs_package_strict_leap_year(tmp_path):

    errors = []

    # Calculate EOFs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
    interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)
    eofs = omi.calc_eofs_from_olr(interpolated_olr,
                                  sign_doy1reference=True,
                                  interpolate_eofs=True,
                                  strict_leap_year_treatment=True,
                                  implementation="eofs_package")
    eofs.save_all_eofs_to_npzfile(tmp_path / "test_completeOMIReproduction_strict_leap_year_treatment_EOFs.npz")

    # Validate EOFs against mjoindices own reference (results should be equal)
    mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_strict_eofs)
    for idx, target_eof in enumerate(eofs.eof_list):
        if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
            errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

    # Calculate PCs
    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    pcs = omi.calculate_pcs_from_olr(raw_olr,
                                     eofs,
                                     np.datetime64("1979-01-01"),
                                     np.datetime64("2018-08-28"),
                                     use_quick_temporal_filter=False)
    pc_temp_file = tmp_path / "test_completeOMIReproduction_eofs_package_strict_leap_year_PCs.txt"
    pcs.save_pcs_to_txt_file(pc_temp_file)

    # Validate PCs against mjoindices own reference (results should be equal)
    # Reload pcs instead of using the calculated ones, because the saving routine has truncated some decimals of the
    # reference values. So do the same with the testing target pcs.
    pcs = pc.load_pcs_from_txt_file(pc_temp_file)
    mjoindices_reference_pcs = pc.load_pcs_from_txt_file(mjoindices_reference_pcs_filename_strict_eofs)
    if not np.all(mjoindices_reference_pcs.time == pcs.time):
        errors.append("mjoindices-reference-validation: Dates of PCs do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc1, pcs.pc1):
        errors.append("mjoindices-reference-validation: PC1 values do not match.")
    if not np.allclose(mjoindices_reference_pcs.pc2, pcs.pc2):
        errors.append("mjoindices-reference-validation: PC2 values do not match.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_preprocess_olr_warning():
    time = np.arange("2018-01-01", "2018-01-03", dtype='datetime64[D]')
    lat = np.array([-2.5, 0., 2.5])
    long = np.array([10., 20., 30., 40.])
    olrmatrix = np.array([((1., 2., 3., 4.),
                           (5., 6., 7., 8.),
                           (9., 10., 11., 12.)),
                          ((10., 20., 30., 40.),
                           (50., 60., 70., 80.),
                           (90., 100., 110., 120.))])
    olrmatrix *= -1.
    testdata = olr.OLRData(olrmatrix, time, lat, long)
    with pytest.warns(UserWarning, match="OLR data apparently given in negative numbers. Here it is assumed that OLR is positive."):
        target = omi.preprocess_olr(testdata)
