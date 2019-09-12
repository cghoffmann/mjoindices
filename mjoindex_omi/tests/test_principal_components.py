# -*- coding: utf-8 -*-

""" """

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindex_omi

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

import numpy as np
import pytest

import mjoindex_omi.principal_components as pc

originalOMIDataDirname = Path(__file__).parent / "testdata" / "OriginalOMI"
origOMIPCsFilename = originalOMIDataDirname / "omi.1x.txt"


def test_basic_properties():
    test_pc1 = np.array([0.12345678, 0.33333333, 0.555555555])
    test_pc2 = np.array([0.38462392, 0.44444444, 0.666666666])
    test_dates = np.array([np.datetime64("2019-06-10"), np.datetime64("2019-06-11"), np.datetime64("2019-06-12")])
    target = pc.PCData(test_dates, test_pc1, test_pc2)

    errors = []
    if not np.all(target.pc1 == test_pc1):
        errors.append("PC1 property not correct")
    if not np.all(target.pc2 == test_pc2):
        errors.append("PC2 property not correct")
    if not np.all(target.time == test_dates):
        errors.append("Time property not correct")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_save_pcs_to_txt_file_and_load_pcs_from_txt_file(tmp_path):
    filename = tmp_path / "test_save_pcs_to_txt_file.txt"
    test_pc1 = np.array([0.12345678, 0.33333333, 0.555555555])
    test_pc2 = np.array([0.38462392, 0.44444444, 0.666666666])
    test_dates= np.array([np.datetime64("2019-06-10"),np.datetime64("2019-06-11"),np.datetime64("2019-06-12")])
    testPC = pc.PCData(test_dates, test_pc1, test_pc2)
    testPC.save_pcs_to_txt_file(filename)

    target = pc.load_pcs_from_txt_file(filename)

    errors = []
    if not np.all(test_dates == target.time):
        errors.append("Dates do not match.")
    if not np.allclose(np.array([0.12346, 0.33333, 0.55556]), target.pc1):
        errors.append("PC1 values do not match.")
    if not np.allclose(np.array([0.38462, 0.44444, 0.66667]), target.pc2):
        errors.append("PC2 values do not match.")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not origOMIPCsFilename.is_file(), reason="Original OMI PCs not available for comparison")
def test_load_original_pcs_from_txt_file():
    # works with original data file that ends on August 28, 2018.
    target = pc.load_original_pcs_from_txt_file(origOMIPCsFilename)

    errors = []
    if not target.pc1[0] == 0.16630:
        errors.append("First Entry of PC1 wrong!")
    if not target.pc2[0] == 0.76455:
        errors.append("First Entry of PC2 wrong!")
    if not target.time[0] == np.datetime64("1979-01-01"):
        errors.append("First Entry of Dates wrong!")

    index_somewhere = 10
    if not target.pc1[index_somewhere] == -1.49757:
        errors.append("Some entry of PC1 wrong!")
    if not target.pc2[index_somewhere] == 0.30697:
        errors.append("Some entry of PC2 wrong!")
    if not target.time[index_somewhere] == np.datetime64("1979-01-11"):
        errors.append("Some entry of Dates wrong!")

    if not target.pc1[-1] == 0.23704:
        errors.append("Last Entry of PC1 wrong!")
    if not target.pc2[-1] == 0.17256:
        errors.append("Last Entry of PC2 wrong!")
    if not target.time[-1] == np.datetime64("2018-08-28"):
        errors.append("Last Entry of Dates wrong!")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))