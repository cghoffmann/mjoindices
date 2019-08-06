# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:01:37 2019

@author: ch
"""

from pathlib import Path

import mjoindex_omi.olr_handling as olr
import mjoindex_omi.omi_calculator as omi

#FIXME: This file is totally chaotic

olr_data_filename = Path(__file__).parents[1] / "tests" / "testdata" / "olr.day.mean.nc"

if not olr_data_filename.is_file():
    raise Exception("OLR data file not available. Expected file: %s" % olr_data_filename)

originalOMIDataDirname = Path(__file__).parents[1] / "tests" / "testdata" / "OriginalOMI"
preprocessed_olr_file = Path(__file__).parent / "example_data" / "PreprocessedOLR.npz"
eofDir = Path(__file__).parent / "example_data" / "EOFcalc"
eofnpzfile = Path(__file__).parent / "example_data" / "EOFs.npz"
eof50File = Path(__file__).parent / "example_data" / "EOF1.txt"

#raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
#shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
#interpolated_olr = olr.resample_spatial_grid_to_original(shorter_olr)

#preprocessed_olr = omi.preprocess_olr(interpolated_olr)

#preprocessed_olr.save_to_npzfile(preprocessed_olr_file)

preprocessed_olr = olr.restore_from_npzfile(preprocessed_olr_file)

#testeofdoy50 = omi.calc_eofs_for_doy(preprocessed_olr, 50)
#testeofdoy50.save_eofs_to_txt_file(eof50File)

#orig_eof50 = eof.load_original_eofs_for_doy(originalOMIDataDirname,50)

#plotting.plot_eof_comparison(orig_eof50, testeofdoy50)


testeofs = omi.calc_eofs_from_preprocessed_olr(preprocessed_olr)
testeofs.save_all_eofs_to_dir(eofDir)
testeofs.save_all_eofs_to_npzfile(eofnpzfile)