# -*- coding: utf-8 -*-

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

"""
This example reproduces the original OMI values described in Kiladis, G.N., J. Dias, K.H. Straub, M.C. Wheeler,
S.N. Tulich, K. Kikuchi, K.M. Weickmann, and M.J. Ventrice, 2014: A Comparison of OLR and Circulation-Based Indices
for Tracking the MJO. Mon. Wea. Rev., 142, 1697–1715, https://doi.org/10.1175/MWR-D-13-00301.1

A scientific description of the software package used by this example (mjoindices) is found in
Hoffmann, C.G., Kiladis, G.N., Gehne, M. and von Savigny, C., 2021: A Python Package to Calculate the
OLR-Based Index of the Madden-Julian-Oscillation (OMI) in Climate Science and Weather Forecasting.
Journal of Open Research Software, 9(1), p.9. DOI: https://doi.org/10.5334/jors.331

We kindly ask you to cite both papers if you use computed results in your scientific publications.

Furthermore, we have received the first adaptions of the OMI algorithm from the community. These modifications are not
covered by Kiladis (2014), but are typically described in other scientific publications before their integration into the
package (if they are not only of technical nature). While we designed the default settings such that the original OMI values
will be reproduced (which also applies to this example), you might want to check the documentation
(https://cghoffmann.github.io/mjoindices/index.html) of the major function calls below to get an overview of the available
options. Furthermore, a list of all scientific publications on which the package is based is available
(https://cghoffmann.github.io/mjoindices/references.html). Please also cite any papers on the list that describe the options
you used for the calculations. So far, possible options include an alternative post-processing approach for the EOFs,
and a possibility to work with data without leap years.

You can modify this example in order to compute OMI data from other OLR datasets (this is probably what you intend if
you use this package). For this, you only have to provide your OLR data as an mjoindices.olr_handling.OLRData object and
use this object as a replacement for the original data in two lines, which is mentioned in the comments below.

This example script may run for about 2 hours on common desktop computers.

This example also produces some diagnostic plots. More evaluation can be done afterwards with the script
evaluate_omi_reproduction.py.
"""

from pathlib import Path
import os.path

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import mjoindices.evaluation_tools
import numpy as np

# ################ Settings. Change with respect to your system ###################

# Download the data file from
# https://www.psl.noaa.gov/thredds/catalog/Datasets/interp_OLR/catalog.html?dataset=Datasets/interp_OLR/olr.day.mean.nc
# to your local file system and adjust the local path below.
# Note: If you have set up the test suite using the reference data package (https://doi.org/10.5281/zenodo.3746562) and
# if you have kept the original directory structure, the following default setting should directly work.
olr_data_filename = Path(os.path.abspath('')).parents[0] / "tests" / "testdata" / "olr.day.mean.nc"

# The following directory should contain the two subdirectories "eof1" and "eof2", which should contain the files
# downloaded from ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
# ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/ , respectively
# Note: If you have set up the test suite using the reference data package (https://doi.org/10.5281/zenodo.3746562) and
# if you have kept the original directory structure, the following default setting should directly work.
originalOMIDataDirname = Path(os.path.abspath('')).parents[0] / "tests" / "testdata" / "OriginalOMI"

# Download the original OMI values from https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt to your local file system
# and adjust the local path below.
# Note: If you have set up the test suite using the reference data package (https://doi.org/10.5281/zenodo.3746562) and
# if you have kept the original directory structure, the following default setting should directly work.
originalOMIPCFile = Path(os.path.abspath('')).parents[0] / "tests" / "testdata" / "OriginalOMI" / "omi.1x.txt"

# Files in which the results are saved. Adjust according to your system.

# The EOFs
eofnpzfile = Path(os.path.abspath('')) / "example_data" / "EOFs.npz"
# The PCs
pctxtfile = Path(os.path.abspath('')) / "example_data" / "PCs.txt"
# Directory in which the figures are saved.
fig_dir = Path(os.path.abspath('')) / "example_data" / "omi_recalc_example_plots"

# ############## There should be no need to change anything below (except if you intend to use a different OLR data as input or you are experiencing problems with the NOAA OLR file NetCDF version.)

# ############## Calculation of the EOFs ###################

if not fig_dir.exists():
    fig_dir.mkdir(parents=True, exist_ok=False)

# Load the OLR data.
# This is the first line to replace to use your own OLR data, if you want to compute OMI for a different dataset.
# ATTENTION: Note that the file format was changed by NOAA from NetCDF3 to NetCDF4 sometime
# between the years 2019 and 2021. If you are using a recent download of the data and
# experience problems, you should switch between the following two lines:
raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# raw_olr = olr.load_noaa_interpolated_olr_netcdf4(olr_data_filename)

# Restrict dataset to the original length for the EOF calculation (Kiladis, 2014)
shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
# Make sure that the spatial sampling resembles the original one (This should not be necessary here, since we use
# the original data file. Nevertheless, we want to be sure.)
interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)

# Diagnosis plot of the loaded OLR data
fig = olr.plot_olr_map_for_date(interpolated_olr, np.datetime64("2010-01-01"))
fig.show()
fig.savefig(fig_dir / "OLR_map.png")

# Calculate the EOFs.
# As a preparation, a dictionary with parameters for the EOF post-processing function is filled. Afterwards,
# the basic function omi.calc_eofs_from_olr() is called. The settings here are chosen to reproduce the original OMI values.
# See the API documentation for further options and a description of the parameters
# (start at https://cghoffmann.github.io/mjoindices/api/omi_calculator.html#mjoindices.omi.omi_calculator.calc_eofs_from_olr).

kiladis_pp_params = {"sign_doy1reference": True,
                      "interpolate_eofs": True,
                      "interpolation_start_doy": 293,
                      "interpolation_end_doy": 316}

eofs = omi.calc_eofs_from_olr(interpolated_olr,
                             leap_year_treatment="original",
                             eofs_postprocessing_type="kiladis2014",
                             eofs_postprocessing_params=kiladis_pp_params)
eofs.save_all_eofs_to_npzfile(eofnpzfile)

# ### Some diagnostic plots to evaluate the calculated EOFs.
# Load precalculated EOFs first.
orig_eofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
# Check correlation with original EOFs
fig = mjoindices.evaluation_tools.plot_comparison_stats_for_eofs_all_doys(eofs, orig_eofs, exclude_doy366=False, do_print=True)
fig.show()
fig.savefig(fig_dir / "EOFs_CorrelationWithOriginal.png")

# Check the variance explained by the EOFs. Values are by a factor of 2 lower than in Kiladis (2014), which is correct!
fig = eof.plot_explained_variance_for_all_doys(eofs)
fig.show()
fig.savefig(fig_dir / "EOFs_ExplainedVariance.png")

# Check details of the EOF pair for a particular DOY in the following.
doy = 50
# Plot EOFs for this DOY.
fig = eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
fig.show()
fig.savefig(fig_dir / "EOF_Sample.png")
# Plot EOF pair in comparison to the original one for this DOY.
fig = mjoindices.evaluation_tools.plot_individual_eof_map_comparison(orig_eofs.eofdata_for_doy(doy), eofs.eofdata_for_doy(doy), doy)
fig.show()
fig.savefig(fig_dir / "EOF_SampleComparison.png")
# Plot the explained variance for the first 10 EOFs of this DOY to check the drop of explained variance after EOF2.
fig = eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)
fig.show()
fig.savefig(fig_dir / "EOF_SampleExplainedVariance.png")


# ############## Calculation of the PCs ##################

# Load the OLR data.
# This is the second line to replace to use your own OLR data, if you want to compute OMI for a different dataset.
olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Load EOFs.
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# Calculate the PCs.
# Restrict calculation to the length of the original OMI time series (at the time when this package was initially validated
# against the original values.
pcs = omi.calculate_pcs_from_olr(olr,
                                 eofs,
                                 np.datetime64("1979-01-01"),
                                 np.datetime64("2018-08-28"),
                                 use_quick_temporal_filter=False)
# Save PCs to a file.
pcs.save_pcs_to_txt_file(pctxtfile)

# ### Diagnostic plot: Comparison to original PCs.
pcs = pc.load_pcs_from_txt_file(pctxtfile)
orig_pcs = pc.load_original_pcs_from_txt_file(originalOMIPCFile)
fig = mjoindices.evaluation_tools.plot_comparison_orig_calc_pcs(pcs, orig_pcs)
fig.savefig(fig_dir / "PCs_TimeSeries.png")
