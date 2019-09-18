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

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import numpy as np

# This example reproduces the original OMI values described in
# Kiladis, G.N., J. Dias, K.H. Straub, M.C. Wheeler, S.N. Tulich, K. Kikuchi, K.M. Weickmann, and M.J. Ventrice, 2014:
# A Comparison of OLR and Circulation-Based Indices for Tracking the MJO.
# Mon. Wea. Rev., 142, 1697–1715, https://doi.org/10.1175/MWR-D-13-00301.1

# ################ Settings. Change with respect to your system ###################

# Download the data file from ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc to your local file system and
# adjust the local path below.
olr_data_filename = Path(__file__).parents[1] / "tests" / "testdata" / "olr.day.mean.nc"

# The following directory should contain the two subdirectories "eof1" and "eof2", which should contain the files
# downloaded from ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
# ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/ respectively
originalOMIDataDirname = Path(__file__).parents[1] / "tests" / "testdata" / "OriginalOMI"

# Download the original OMI values from https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt to your local file system
# and adjust the local path below.
originalOMIPCFile = Path(__file__).parents[1] / "tests" / "testdata" / "OriginalOMI" / "omi.1x.txt"

# Files to store the results:
# The EOFs
eofnpzfile = Path(__file__).parent / "example_data" / "EOFs.npz"
# The PCs
pctxtfile = Path(__file__).parent / "example_data" / "PCs.txt"

# ############## Calculation of the EOFs ###################

# Load the OLR data
raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Restrict dataset to the original length for the EOF calculation (Kiladis, 2014)
shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
# Make sure that the spatial sampling resembles the original one (This should not be necessary here, since we use
# the original data file. Nevertheless, we want to be sure.)
interpolated_olr = olr.resample_spatial_grid_to_original(shorter_olr)

# Diagnosis plot of the loaded OLR data
olr.plot_olr_map_for_date(interpolated_olr, np.datetime64("2010-01-01"))

# In order to adjust the signs of the EOFs to fit to the original ones, we need to know the original ones
# Note that signs switch arbitrarily and are also adjusted in the original approach (Kiladis, 2014).
# This step will probably be replced in future
orig_eofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)

# Calculate the eofs. In the postprocessing, the signs of the EOFs are adjusted and the the EOF  in a period
# around DOY 300 are replaced by an interpolation see Kiladis, 2014). The periods is somewhat braoder than stated in
# Kiladis (2014) to achieve good agreement. The reason for this is still unclear.
eofs= omi.calc_eofs_from_olr(interpolated_olr,
                             sign_doy1reference = orig_eofs.eofdata_for_doy(1),
                             interpolate_eofs=True,
                             interpolation_start_doy=294,
                             interpolation_end_doy=315)
eofs.save_all_eofs_to_npzfile(eofnpzfile)

# ### Some diagnostic plots to evaluate the calculated EOFs
# Load precalculated EOFs first
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
# Check correlation with original EOFs
eof.plot_correlation_with_original_eofs(eofs, orig_eofs)
# Check the explained variance by the EOFS. Values are lower than in Kiladis, 2014, which is correct!
eof.plot_explained_variance_for_all_doys(eofs)

# Check details of the EOF pair for a particular doy in the following
doy=50
# Plot EOFs for this DOY
eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
# Plot EOF pair in comparison to the original one fpr this DOY
eof.plot_individual_eof_map_comparison(orig_eofs.eofdata_for_doy(doy), eofs.eofdata_for_doy(doy), doy)
# Plot the explained variance for the first 10 EOFs of this DOY to check to drop of explained variance after EOF2
eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)


# ############## Calculation of the PCs ##################

# Load the OLR data
olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Load EOFs
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# Calculate the PCs
# Restrict calculation to the length of the official OMI time series
pcs = omi.calculate_pcs_from_olr(olr,
                                 eofs,
                                 np.datetime64("1979-01-01"),
                                 np.datetime64("2018-08-28"),
                                 useQuickTemporalFilter=False)
# Save PCs
pcs.save_pcs_to_txt_file(pctxtfile)

# ### Diagnostic plot: Comparison to original PCs
pcs = pc.load_pcs_from_txt_file(pctxtfile)
orig_pcs = pc.load_original_pcs_from_txt_file(originalOMIPCFile)
pc.plot_comparison_orig_calc_pcs(pcs, orig_pcs)
