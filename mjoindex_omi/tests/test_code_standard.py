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

import os
import pycodestyle
import pytest

import mjoindex_omi


@pytest.mark.skip
def test_codingstandard_pep8():
    style = pycodestyle.StyleGuide()
    dirs = [os.path.dirname(mjoindex_omi.__file__)]
    print(mjoindex_omi.__file__)
    result = style.check_files(dirs)
    assert result.total_errors == 0
