from pathlib import Path

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.evaluation_tools
import numpy as np

# Directory in which all EOFs are saved as text files
eofdir = Path(__file__).resolve().parent / "data" / "EOFs"
# Numpy file, in which EOFs are saved additionally together with statistical information.
eofnpzfile = Path(__file__).resolve().parent / "data" / "EOFs.npz"
# Text file in which the PCs are saved.
pcs_txtfile = Path(__file__).resolve().parent / "data" / "PCs.txt"

# The following directory should contain the two subdirectories "eof1" and "eof2", which should contain the files
# downloaded from ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
# ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/ respectively
original_omi_data_dirname = Path(__file__).resolve().parents[1] / "tests" / "testdata" / "OriginalOMI"

# ### EOF calculation
# choose a reasonable period to calculate the EOFs.
# The original calculation includes the years from 1979 to 2012 (Kiladis, 2014)
startDate = np.datetime64("1979-01-01")
endDate = np.datetime64("2012-12-31")

# Implement a method that loads your OLR data and returns a mjoindices.olr_handling.OLRData object
# This is the only part that you have to implement yourself
raw_olr = load_your_olr_data()
# Restrict dataset to configured period
shorter_olr = olr.restrict_time_coverage(raw_olr, startDate, endDate)
# Interpolate OLR data to original OMI calculation grid. Other grids should in principle also work.
# If you want to use other grid, just leave out this step.
# However, in this case, it is unclear to what extent the resulting index will be comparable to the original
# OMI dataset. Maybe you want to do a sensitivity study using different grids before.
interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)

# Calculate the EOFs
eofs= omi.calc_eofs_from_olr(interpolated_olr, sign_doy1reference = True, interpolate_eofs=False)

# Save the EOFs before proceeding
eofs.save_all_eofs_to_dir(eofdir)
eofs.save_all_eofs_to_npzfile(eofnpzfile)


# ### EOF Diagnosis plots
# ### Some diagnostic plots to evaluate the calculated EOFs
# Load precalculated EOFs first
orig_eofs = eof.load_all_original_eofs_from_directory(original_omi_data_dirname)
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
# Check correlation with original EOFs
mjoindices.evaluation_tools.plot_comparison_stats_for_eofs_all_doys(eofs, orig_eofs)
# Check the explained variance by the EOFS.
eof.plot_explained_variance_for_all_doys(eofs)

# Check details of the EOF pair for a particular doy in the following
doy = 50
# Plot EOFs for this DOY
eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
# Plot EOF pair in comparison to the original one fpr this DOY
mjoindices.evaluation_tools.plot_individual_eof_map_comparison(orig_eofs.eofdata_for_doy(doy), eofs.eofdata_for_doy(doy), doy)
# Plot the explained variance for the first 10 EOFs of this DOY to check to drop of explained variance after EOF2
eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)


# ### PC calculation

# Choose the period, for which the PCs should be calculated. The may differ from the EOF calculation period.
startDate = np.datetime64("1979-01-01")
endDate = np.datetime64("2018-08-28")

# Use your OLR loader also here
raw_olr = load_your_olr_data()
# Load the precalculated EOFs
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# Calculate the PCs
pcs = omi.calculatePCsFromOLR(raw_olr, eofs, startDate, endDate, useQuickTemporalFilter=True)

# Save the PCs
pcs.save_pcs_to_txt_file(pcs_txtfile)