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
import mjoindices.omi.postprocessing_rotation_approach as omir
import mjoindices.principal_components as pc
import numpy as np


mjoindices_reference_eofs_filename_raw = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_raw.npz"
mjoindices_reference_eofs_filename_rotated = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_pp_rot.npz"

# TODO: (Sarah) write these unit tests
def test_compute_angle_between_vectors():
    pass

def test_calculate_rotation_angle():
    pass

def test_rotation_direction():
    pass

def test_normalize_eofs():
    pass

def test_post_process_eofs_rotation_approach():

    errors = []

    raw_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_raw)

    eofs=omir.post_process_eofs_rotation(raw_eofs)

    # Validate rotated EOFs against mjoindices own reference (results should be equal)
    mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_rotated)
    for idx, target_eof in enumerate(eofs.eof_list):
        if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
            errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

def test_post_process_eofs_rotation_approach_with_kw_dict():

        errors = []

        raw_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_raw)

        kw_params = {"sign_doy1reference": True}

        eofs = omir.post_process_eofs_rotation(raw_eofs, **kw_params)

        # Validate EOFs against mjoindices own reference (results should be equal)
        mjoindices_reference_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename_rotated)
        for idx, target_eof in enumerate(eofs.eof_list):
            if not mjoindices_reference_eofs.eof_list[idx].close(target_eof):
                errors.append("mjoindices-reference-validation: EOF data at index %i is incorrect" % idx)

        assert not errors, "errors occurred:\n{}".format("\n".join(errors))

