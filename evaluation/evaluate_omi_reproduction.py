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

# This file evaluates the quality of the OMI reproduction.
# To generate the reproduction data, execute the script examples/recalculate_original_omi.py
# Change path- and filesname below according to settings in examples/recalculate_original_omi.py

from pathlib import Path
import numpy as np

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.evaluation_tools
import mjoindices.principal_components as pc

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
eofnpzfile = Path(__file__).parents[1] / "examples" / "example_data" / "EOFs.npz"
# The PCs
pctxtfile = Path(__file__).parents[1] / "examples" / "example_data" / "PCs.txt"

### evaluate EOFs

eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
orig_eofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)

# Check correlation with original EOFs
fig = eof.plot_correlation_with_original_eofs(eofs, orig_eofs)
fig.savefig("EOFs_CorrelationWithOriginal.png")
corr1, corr2 = mjoindices.evaluation_tools.calc_correlations_of_eofs_all_doys(orig_eofs, eofs)
print(corr1.shape)
#leave out DOY 366 since this is intended to be different
corr1 = corr1[0:-1]
corr2 = corr2[0:-1]
print(corr1.shape)
print("Minimum Correlation of EOF1: %f" % np.min(corr1))
print("Minimum Correlation of EOF2: %f" % np.min(corr2))


mjoindices.evaluation_tools.plot_maxdifference_of_eofs_all_doys(orig_eofs,eofs)

#make respective plots
maxdiff_abs_1, maxdiff_abs_2, maxdiff_rel_1, maxdiff_rel_2 = mjoindices.evaluation_tools.calc_maxdifference_of_eofs_all_doys(orig_eofs, eofs)
maxdiff_abs_1 = maxdiff_abs_1[0:-1]
maxdiff_abs_2 = maxdiff_abs_2[0:-1]
maxdiff_rel_1 = maxdiff_rel_1[0:-1]
maxdiff_rel_2 = maxdiff_rel_2[0:-1]

print("Maximum rel. difference EOFs1: %f" % np.max(maxdiff_rel_1))
print("Maximum rel. difference EOFs2: %f" % np.max(maxdiff_rel_2))
print("Maximum abs. difference EOFs1: %f" % np.max(maxdiff_abs_1))
print("Maximum abs. difference EOFs2: %f" % np.max(maxdiff_abs_2))

#calc also rel. mean difference

#relative deviations of explained variance

##### PCs

#correlations
#max deviation abs and rel
#mean deviation

#



# # Check the explained variance by the EOFS. Values are lower than in Kiladis, 2014, which is correct!
# fig = eof.plot_explained_variance_for_all_doys(eofs)
# fig.savefig("EOFs_ExplainedVariance.png")
#
# # Check details of the EOF pair for a particular doy in the following
# doy=50
# # Plot EOFs for this DOY
# fig = eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
# fig.savefig("EOF_Sample.png")
# # Plot EOF pair in comparison to the original one fpr this DOY
# fig = eof.plot_individual_eof_map_comparison(orig_eofs.eofdata_for_doy(doy), eofs.eofdata_for_doy(doy), doy)
# fig.savefig("EOF_SampleComparison.png")
# # Plot the explained variance for the first 10 EOFs of this DOY to check to drop of explained variance after EOF2
# fig = eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)
# fig.savefig("EOF_SampleExplainedVariance.png")
#
