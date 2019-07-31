# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import os
import typing

import numpy as np


def load_eofs_for_doy(path: str, doy: int, prefix: str = "eof", suffix: str = ".txt") -> typing.Tuple[np.ndarray, np.ndarray]:
    """Loads the EOF values for the first 2 EOFs
    Note that as in the original treatment, the EOFs are represented as vectors,
    which means that a connection to the individual locations on a world map is not obvious without any further
    knowledge.

    :param path: Path the where the EOFs for all doys are stored. This path should contain the sub directories "eof1"
                 and "eof2", in which the 366 files each are located: One file per day of the year.
                 The original EOFs are found here:
                 ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
                 ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/
    :param doy: The day of the year (DOY) for which the 2 EOFs are loaded (number between 1 and 366)
    :param prefix: might be removed
    :param suffix: might be removed
    :return: Tuple of 2 numpy array, each containing a 1D array with the EOF values for either EOF1 or EOF2.
    """
    # FIXME: Remove prefix and suffix?
    eof1filename = path + os.path.sep + "eof1" + os.path.sep + prefix + str(doy).zfill(3) + suffix
    eof1 = np.genfromtxt(eof1filename)
    eof2filename = path + os.path.sep + "eof2" + os.path.sep + prefix + str(doy).zfill(3) + suffix
    eof2 = np.genfromtxt(eof2filename)
    return eof1, eof2
