# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:01:37 2019

@author: ch
"""

from pathlib import Path

import mjoindex_omi.wheeler_kiladis_mjo_filter as wkfilter

#FIXME: This file is totally chaotic





olr_data_filename = Path(__file__).parents[1] / "tests" / "testdata" / "olr.day.mean.nc"

if not olr_data_filename.is_file():
    raise Exception("OLR data file not available. Expected file: %s" % olr_data_filename)

originalOMIDataDirname = Path(__file__).parents[1] / "tests" / "testdata" / "OriginalOMI"
preprocessed_olr_file = Path(__file__).parent / "example_data" / "PreprocessedOLR.npz"

reference_dir = Path(__file__).parents[1] / "tests" / "testdata" / "WKFilterReference" / "lat0degPyIdx36"
# FIXME Try to use common OLR data
#olr_dir = Path("/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
#raw_olr = loadKiladisBinaryOLRDataTwicePerDay(olr_dir / "olr.2x.7918.b")
#test_olr = np.squeeze(raw_olr.olr[:, 36, :])
test_olr = wkfilter.loadKiladisOriginalOLR(reference_dir / "OLROriginal.b")
validator = wkfilter.WKFilterValidator(test_olr, reference_dir, do_plot=1, atol=1e-8, rtol=100.)
errors = validator.validate_WKFilter_perform2dimSpectralSmoothing_MJOConditions()


#validator = wkfilter.WKFilterValidator(data_exchange_dir="/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
#validator.validate_WKFilter_perform2dimSpectralSmoothing_MJOConditions()



#raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
#shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
#interpolated_olr = olr.resample_spatial_grid_to_original(shorter_olr)

#preprocessed_olr = omi.preprocess_olr(interpolated_olr)

#preprocessed_olr.save_to_npzfile(preprocessed_olr_file)