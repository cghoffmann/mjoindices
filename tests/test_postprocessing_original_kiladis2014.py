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

from pathlib import Path
import os.path
import pytest

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.omi.postprocessing_original_kiladis2014 as pp_kil2014
import mjoindices.principal_components as pc
import mjoindices.evaluation_tools
import numpy as np


mjoindices_reference_eofs_filename_raw = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_raw.npz"
mjoindices_reference_eofs_filename = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs.npz"


@pytest.mark.filterwarnings("ignore:References for the sign of the EOFs for DOY1 have to be interpolated")
def test_if_refdata_isfound_for_correct_spontaneous_sign_changes_in_eof_series():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    eofs = eof.EOFDataForAllDOYs(eofs, no_leap=False)

    try:
        target = pp_kil2014.correct_spontaneous_sign_changes_in_eof_series(eofs, True)
    except OSError:
        pytest.fail("Function failed with OS Error, hence the reference data has probably not been found, which points "
                    "to an installation problem of the package: ".format(OSError))


def test_post_process_eofs_original_kiladis_approach():

    errors = []

    raw_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_raw)

    eofs=pp_kil2014.post_process_eofs_original_kiladis_approach(raw_eofs, interpolate_eofs=True)

    # Validate EOFs against mjoindices own reference (results should be equal)
    mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename)
    for idx, target_eof in enumerate(eofs.eof_list):
        if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
            errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

def test_post_process_eofs_original_kiladis_approach_with_kw_dict():

        errors = []

        raw_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_raw)

        kw_params = {"sign_doy1reference": True,
                     "interpolate_eofs": True,
                     "interpolation_start_doy":293,
                     "interpolation_end_doy": 316}

        eofs = pp_kil2014.post_process_eofs_original_kiladis_approach(raw_eofs, **kw_params)

        # Validate EOFs against mjoindices own reference (results should be equal)
        mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename)
        for idx, target_eof in enumerate(eofs.eof_list):
            if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
                errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

        assert not errors, "errors occurred:\n{}".format("\n".join(errors))

# ToDo: (Sarah): Add a similar test_file for your pp_script

