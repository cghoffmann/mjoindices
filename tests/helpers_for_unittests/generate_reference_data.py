# -*- coding: utf-8 -*-

# Copyright (C) 2022 Christoph G. Hoffmann. All rights reserved.

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
This file produces data, which is used as a reference later during unit test excecution.
Only developers will need this file and should only overwrite existing reference data if they really know, what they are doing

Note that each generated file, will be stored directly in the target directory for the reference data. However, the term
'.recalculated' will be added to the fill suffix. That means, that the newly generated file should be carefully checked by the developer, before
this suffix extension is removed, which will activate the file for the tests.
"""

#ToDo: this file has been introduced for a later version and does not cover a bunch of reference files that have been calculated earlier by hand
#All calculated reference data should be added in future
from pathlib import Path
import os.path

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eofs
import mjoindices.principal_components as pc
import mjoindices.evaluation_tools
import numpy as np

olr_data_filename = Path(os.path.abspath('')).parent / "testdata" / "olr.day.mean.nc"

mjoindices_reference_eofs_filename_raw = Path(os.path.abspath('')).parent / "testdata" / "mjoindices_reference" / "EOFs_raw.npz.recalculated"


if __name__ == "__main__":

    raw_olr = olr.load_noaa_interpolated_olr(olr_data_filename)
    shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))
    interpolated_olr = olr.interpolate_spatial_grid_to_original(shorter_olr)

    # Calculate the EOFs. In the postprocessing, the signs of the EOFs are adjusted and the EOFs in a period
    # around DOY 300 are replaced by an interpolation see Kiladis (2014).
    # The switch strict_leap_year_treatment has major implications only for the EOFs calculated for DOY 366 and causes only
    # minor differences for the other DOYs. While the results for setting strict_leap_year_treatment=False are closer to the
    # original values, the calculation strict_leap_year_treatment=True is somewhat more stringently implemented using
    # built-in datetime functionality.
    # See documentation of mjoindices.tools.find_doy_ranges_in_dates() for details.

    preprocessed_olr = omi.preprocess_olr(interpolated_olr)
    raw_eofs = omi.calc_eofs_from_preprocessed_olr(preprocessed_olr, implementation="internal",
                                               strict_leap_year_treatment=False)
    eofs.save_all_eofs_to_npzfile(mjoindices_reference_eofs_filename_raw)
