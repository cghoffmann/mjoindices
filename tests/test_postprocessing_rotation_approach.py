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

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.omi.postprocessing_rotation_approach as omir
import mjoindices.principal_components as pc
import numpy as np


mjoindices_reference_eofs_filename_raw = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_raw.npz"
mjoindices_reference_original_omi = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs.npz" 
mjoindices_reference_eofs_filename_rotated = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs_pp_rot.npz"

setups_angle = [(np.array([1,2,3]), np.array([3,-2,1]), 1.4274487578895312), (np.array([1,0,0]), np.array([0,1,0]), np.pi/2)]
@pytest.mark.parametrize("a,b,result", setups_angle)
def test_compute_angle_between_vectors(a,b,result):
    
    errors = []
    
    vect_angle = omir.angle_btwn_vectors(a,b)

    if not np.isclose(vect_angle, result):
       errors.append("error with rotating by small rotation matrix") 

    assert not errors, "errors occurred:\n{}".format("\n".join(errors)) 

def test_calculate_rotation_angle():
    # test rotation direction also
    errors = []

    orig_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_original_omi)
    result_raw = -0.0001639582543790491

    if not np.isclose(result_raw, omir.calculate_angle_from_discontinuity(orig_eofs)):
        errors.append("error with calculating rotation angle")
    
    # tests rotation direction as well
    flip_eofs = []
    for t_eof in orig_eofs.eof_list:    
        flip_eofs.append(eof.EOFData(t_eof.lat, t_eof.long, 
                                -1*t_eof.eof1vector, 
                                t_eof.eof2vector,
                                explained_variances=t_eof.explained_variances,
                                eigenvalues=t_eof.eigenvalues,
                                no_observations=t_eof.no_observations))
    
    f_eofs = eof.EOFDataForAllDOYs(flip_eofs, orig_eofs.no_leap_years)

    if not np.isclose(-result_raw, omir.calculate_angle_from_discontinuity(f_eofs)):
        errors.append("error with calculating rotation angle")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))  


setups_rotmat = [(np.pi/2, np.array([0,1])), (-np.pi/3, np.array([np.cos(-np.pi/3), np.sin(-np.pi/3)]))]
@pytest.mark.parametrize("delta, result", setups_rotmat)
def test_rotation_matrix(delta, result):

    errors = []

    M = np.array([1, 0])
    R = omir.rotation_matrix(delta)

    if not np.all(np.isclose(np.matmul(R,M),result)):
        errors.append("error with rotating by small rotation matrix")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

def test_normalize_eofs():
    
    errors = []

    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    eofs = eof.EOFDataForAllDOYs(eofs, no_leap_years=False)

    norm_eofs = omir.normalize_eofs(eofs)

    for idx, target_eof in enumerate(norm_eofs.eof_list):
        if (not np.isclose(np.linalg.norm(target_eof.eof1vector), 1)) or (not np.isclose(np.linalg.norm(target_eof.eof2vector), 1)):
            errors.append("error with normalizing EOFs after rotation")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors)) 

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

