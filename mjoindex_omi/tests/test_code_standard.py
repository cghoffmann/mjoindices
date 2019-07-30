# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:13:34 2019

@author: ch
"""
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
