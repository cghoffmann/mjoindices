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

#FIXME: This file is only a backup and shpuld become unneccesary

from pathlib import Path

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.evaluation_tools
import numpy as np

# This example reproduces the original OMI values described in
# Kiladis, G.N., J. Dias, K.H. Straub, M.C. Wheeler, S.N. Tulich, K. Kikuchi, K.M. Weickmann, and M.J. Ventrice, 2014:
# A Comparison of OLR and Circulation-Based Indices for Tracking the MJO.
# Mon. Wea. Rev., 142, 1697–1715, https://doi.org/10.1175/MWR-D-13-00301.1

# Download the data file from ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc to your local file system and
# adjust the local path below.
olr_data_filename = Path(__file__).resolve().parents[1] / "tests" / "testdata" / "olr.day.mean.nc"
if not olr_data_filename.is_file():
    raise Exception("OLR data file not available. Expected file: %s" % olr_data_filename)

# ############## Calculation of the EOFs ###################

originalOMIDataDirname = Path(__file__).resolve().parents[1] / "tests" / "testdata" / "OriginalOMI"
#preprocessed_olr_file = Path(__file__).resolve().parent / "example_data" / "PreprocessedOLR.npz"
#eofDir = Path(__file__).resolve().parent / "example_data" / "EOFcalc"
eofnpzfile = Path(__file__).resolve().parent / "example_data" / "EOFs.npz"
#eofnpzfile_kil = Path(__file__).resolve().parent / "example_data" / "EOFs_kil.npz"
#eofnpzfile_kil_eofs = Path(__file__).resolve().parent / "example_data" / "EOFs_kil_eofs.npz"
#eof50File = Path(__file__).resolve().parent / "example_data" / "EOF1.txt"
#eof50File_kil = Path(__file__).resolve().parent / "example_data" / "EOF1_kil.txt"

# Load the OLR data
raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
# Restrict data set to the original length for the EOF calculation (Kiladis, 2014)
shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
# Make sure that the spatial sampling resembles the original one (This should not be necessary here, since we use
# the original data file. Nevertheless, we want to be sure.)
interpolated_olr = olr.resample_spatial_grid_to_original(shorter_olr)

# Diagnosis plot of the loaded OLR data
olr.plot_olr_map_for_date(interpolated_olr, np.datetime64("2010-01-01"))

# In order to adjust the signs of the EOFs to fit to the original ones, we need to know the original ones
# Note that signs switch arbitrarily and are also adjusted in the original approach (Kiladis, 2014).
# There seems to be no way to deduce this from theory or so
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


# preprocessed_olr = omi.preprocess_olr(interpolated_olr)
# preprocessed_olr.save_to_npzfile(preprocessed_olr_file)
# preprocessed_olr = olr.restore_from_npzfile(preprocessed_olr_file)
# testeofs = omi.calc_eofs_from_preprocessed_olr(preprocessed_olr)
# testeofs.save_all_eofs_to_dir(eofDir)


# origeofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)
# eofs = omi.post_process_eofs(testeofs, sign_doy1reference = origeofs.eofdata_for_doy(1), interpolate_eofs=True, interpolation_start_doy=294, interpolation_end_doy=315)

# Some diagnostic plots
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
eof.plot_correlation_with_original_eofs(eofs, orig_eofs)
eof.plot_explained_variance_for_all_doys(eofs)
doy=300
eof.plot_individual_eof_map(eofs.eofdata_for_doy(doy), doy)
mjoindices.evaluation_tools.plot_individual_eof_map_comparison(orig_eofs.eofdata_for_doy(doy), eofs.eofdata_for_doy(doy), doy)
eof.plot_individual_explained_variance_all_eofs(eofs.eofdata_for_doy(doy), doy=doy, max_eof_number=10)



# ############# Section with OLR data, which has been filtered with G. Kiladis data

#preprocessed_kil_olr = wkfilter.loadKiladisBinaryOLRDataOncePerDay(Path("/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/olr.3096.eastw.1x.7918.b"))
#shorter_preprocessed_kil_olr = olr.restrict_time_coverage(preprocessed_kil_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
#interpolated_preprocessed_kil_olr = olr.resample_spatial_grid_to_original(shorter_preprocessed_kil_olr)

#testeofdoy50_kil = omi.calc_eofs_for_doy(interpolated_preprocessed_kil_olr, 366)
#testeofdoy50_kil_eofs = omi.calc_eofs_for_doy_using_eofs_package(interpolated_preprocessed_kil_olr, 366)
#testeofdoy50_kil.save_eofs_to_txt_file(eof50File_kil)

#eof.plot_individual_eof_map(testeofdoy50_kil, 100)
#print("GK filtering")
#print(testeofdoy50_kil.explained_variance_eof1)
#print(testeofdoy50_kil.sum_of_explained_variances)

#kileofs = omi.calc_eofs_from_preprocessed_olr(interpolated_preprocessed_kil_olr)
#kileofs.save_all_eofs_to_npzfile(eofnpzfile_kil)

#kileofs = eof.restore_all_eofs_from_npzfile(eofnpzfile_kil)
#origeofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)
#pp_eofs_kil = omi.post_process_eofs(kileofs, sign_doy1reference = origeofs.eofdata_for_doy(1), interpolate_eofs=True, interpolation_start_doy=294, interpolation_end_doy=315)

#eof.plot_correlation_with_original_eofs(pp_eofs_kil, origeofs)
#eof.plot_explained_variance_for_all_doys(pp_eofs_kil,include_total_variance=True, include_no_observations=True)
#eof.plot_explained_variance_for_all_doys(pp_eofs_kil,include_total_variance=False, include_no_observations=False)

#doy=31
#eof.plot_individual_eof_map_comparison(origeofs.eofdata_for_doy(doy), pp_eofs_kil.eofdata_for_doy(doy), doy)
#eof.plot_individual_explained_variance_all_eofs(pp_eofs_kil.eofdata_for_doy(doy), doy=doy, max_eof_number=10)
#eof.plot_individual_explained_variance_all_eofs(pp_eofs_kil.eofdata_for_doy(doy), doy=doy)


#kileofs_eofspackage = omi.calc_eofs_from_preprocessed_olr(interpolated_preprocessed_kil_olr, implementation="eofs_package")
#kileofs_eofspackage.save_all_eofs_to_npzfile(eofnpzfile_kil_eofs)

#kileofs_eofspackage = eof.restore_all_eofs_from_npzfile(eofnpzfile_kil_eofs)
#origeofs = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)
#pp_eofs_kil_eofspackage = omi.post_process_eofs(kileofs_eofspackage, sign_doy1reference = origeofs.eofdata_for_doy(1), interpolate_eofs=True, interpolation_start_doy=294, interpolation_end_doy=315)
#eof.plot_correlation_with_original_eofs(pp_eofs_kil_eofspackage, origeofs)
#eof.plot_explained_variance_for_all_doys(pp_eofs_kil_eofspackage, include_no_observations=True, include_total_variance=True)
