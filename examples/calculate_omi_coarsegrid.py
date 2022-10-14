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
This example is very similar to the example recalculate_original_omi.py.
However, it computes the OMI values on a freely defined spatial grid.
While this is generally supported by the package, it is not recommended to choose a
spatial grid that differs from the original one, since it is scientifically not clear if the OMI index will have the
same characteristics on a different grid.
Nevertheless, this example can be used to quickly check the implication of a different grid.
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

# Choose a spatial grid, on which the values are computed.
coarse_lat = np.arange(-20., 20.1, 8.0)
coarse_long = np.arange(0., 359.9, 20.0)

# Download the data file from ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc to your local file system and
# adjust the local path below.
# Note: If you have set up the test suite using the reference data package (https://doi.org/10.5281/zenodo.3746562) and
# if you have kept the original directory structure, the following default setting should directly work.
olr_data_filename = Path(os.path.abspath('')).parents[0] / "tests" / "testdata" / "olr.day.mean.nc"

# Download the original OMI values from https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt to your local file system
# and adjust the local path below.
# Note: If you have set up the test suite using the reference data package (https://doi.org/10.5281/zenodo.3746562) and
# if you have kept the original directory structure, the following default setting should directly work.
originalOMIPCFile = Path(os.path.abspath('')).parents[0] / "tests" / "testdata" / "OriginalOMI" / "omi.1x.txt"

# Files to store the results:
# The EOFs
eofnpzfile = Path(os.path.abspath('')) / "example_data" / "EOFs_coarsegrid.npz"
# The PCs
pctxtfile = Path(os.path.abspath('')) / "example_data" / "PCs_coarsegrid.txt"

# Directory in which the figures are saved.
fig_dir = Path(os.path.abspath('')) / "example_data" / "omi_recalc_example_plots_coarsegrid"

# ############## Calculation of the EOFs ###################

if not fig_dir.exists():
    fig_dir.mkdir(parents=True, exist_ok=False)

# Load the OLR data.
# This is the first place to insert your own OLR data, if you want to compute OMI for a different dataset.
raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Restrict dataset to the original length for the EOF calculation (Kiladis, 2014).
shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))

# This is the line, where the spatial grid is changed.
interpolated_olr = olr.interpolate_spatial_grid(shorter_olr, coarse_lat, coarse_long)

# Diagnosis plot of the loaded OLR data.
fig = olr.plot_olr_map_for_date(interpolated_olr, np.datetime64("2010-01-01"))
fig.show()
fig.savefig(fig_dir / "OLR_map.png")

# Calculate the eofs.
kiladis_pp_params = {"sign_doy1reference": True,
                      "interpolate_eofs": True,
                      "interpolation_start_doy": 293,
                      "interpolation_end_doy": 316}

eofs = omi.calc_eofs_from_olr(interpolated_olr,
                             strict_leap_year_treatment=False,
                             eofs_postprocessing_type="kiladis2014",
                             eofs_postprocessing_params=kiladis_pp_params)
eofs.save_all_eofs_to_npzfile(eofnpzfile)

# ### Some diagnostic plots to evaluate the calculated EOFs.
# Load precalculated EOFs first.
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# Check the explained variance by the EOFs. Values are lower by about a factor of 2 than in Kiladis (2014),
# which is correct!
fig = eof.plot_explained_variance_for_all_doys(eofs)
fig.show()
fig.savefig(fig_dir / "EOFs_ExplainedVariance.png")

# Check details of the EOF pair for a particular DOY in the following.
doy = 50
# Plot EOFs for this DOY.
fig = eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
fig.show()
fig.savefig(fig_dir / "EOF_Sample.png")
# Plot the explained variance for the first 10 EOFs of this DOY to check the drop of explained variances after EOF2.
fig = eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)
fig.show()
fig.savefig(fig_dir / "EOF_SampleExplainedVariance.png")


# ############## Calculation of the PCs ##################

# Load the OLR data.
# This is the second place to insert your own OLR data, if you want to compute OMI for a different dataset.
olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Load EOFs
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# Calculate the PCs.
# Restrict calculation to the length of the official OMI time series.
pcs = omi.calculate_pcs_from_olr(olr,
                                 eofs,
                                 np.datetime64("1979-01-01"),
                                 np.datetime64("2018-08-28"),
                                 use_quick_temporal_filter=False)
# Save PCs.
pcs.save_pcs_to_txt_file(pctxtfile)

# ### Diagnostic plot: Comparison to original PCs.
pcs = pc.load_pcs_from_txt_file(pctxtfile)
orig_pcs = pc.load_original_pcs_from_txt_file(originalOMIPCFile)
fig = mjoindices.evaluation_tools.plot_comparison_orig_calc_pcs(pcs, orig_pcs)
fig.savefig(fig_dir / "PCs_TimeSeries.png")

