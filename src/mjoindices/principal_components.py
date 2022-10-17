# -*- coding: utf-8 -*-

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

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
This module provides basic functionality to handle PC data, which is a basic output of the OMI calculation.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class PCData:
    """
    Class as a container for the principal component (PC) data.

    The basic PC computation :func:`mjoindices.omi.omi_calculator.calculate_pcs_from_olr` will return an object of this
    class as a major result of this package.

    :param time: Array containing the :class:`numpy.datetime64` dates.
	:param period: Array containing the :class:`pandas.Period` dates.
    :param pc1: Array containing the values of PC1 (has to be of same length as the time array).
    :param pc2: Array containing the values of PC2 (has to be of same length as the time array).
    """

    # ToDo: Check if we need some deepcopy or writable=false here
    def __init__(self, time: np.ndarray, pc1: np.ndarray, pc2: np.ndarray) -> None:
        """
        Initialization with all necessary variables.
        """
        if pc1.size != time.size:
            raise ValueError('Length of the first PC time series does not fit to the length of the '
                             'time grid')
        if pc2.size != time.size:
            raise ValueError('Length of the second PC time series does not fit to the length of the '
                             'time grid')
        self._time = time.copy()
        self._pc1 = pc1.copy()
        self._pc2 = pc2.copy()

    @property
    def time(self) -> np.ndarray:
        """
        The time grid of the PC time series as array of :class:`numpy.datetime64` elements.
        """
        return self._time

    @property
    def pc1(self) -> np.ndarray:
        """
        The time series of the PC1 values.
        """
        return self._pc1

    @property
    def pc2(self) -> np.ndarray:
        """
        The time series of the PC2 values.
        """
        return self._pc2

    def save_pcs_to_txt_file(self, filename: Path) -> None:
        """
        Saves the computed PCs to a text file.

        Please note that the file format is not exactly that of the original data files. However, a
        suitable reader is available in this module for both formats
        (:func:`mjoindices.principal_components.load_pcs_from_txt_file` and
        :func:`mjoindices.principal_components.load_original_pcs_from_txt_file`).

        :param filename: The full filename.
        """
        df = pd.DataFrame({"Date": self._time, "PC1": self._pc1, "PC2": self._pc2})
        df.to_csv(filename, index=False, float_format="%.5f")


def load_pcs_from_txt_file(filename: Path) -> PCData:
    """
    Loads the PCs of OMI, which were previously saved with this package
    (:func:`mjoindices.principal_components.PCData.save_pcs_to_txt_file`).

    :param filename: Path to the PC file.

    :return: The PC data.
    """
    df = pd.read_csv(filename, sep=',', parse_dates=[0], header=0)
    dates = df.Date.values
    pc1 = df.PC1.values
    pc2 = df.PC2.values
    return PCData(dates, pc1, pc2)


def load_original_pcs_from_txt_file(filename: Path) -> PCData:
    """
    Loads the PCs of OMI, which are stored in the original file format.
    For example, the following file can be loaded: https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt

    Note that the present software package stores the PCs slightly different. Those files can be loaded with
    :func:`load_pcs_from_txt_file`.

    :param filename: Path to the PC file.

    :return:  The original PC data.
    """
    my_data = np.genfromtxt(filename)
    dates_temp = []
    for i in range(0, my_data.shape[0]):
        dates_temp.append(
            datetime.datetime(my_data[i, 0].astype(int), my_data[i, 1].astype(int), my_data[i, 2].astype(int)))
    dates = np.array(dates_temp, dtype='datetime64')
    pc1 = np.array(my_data[:, 4])
    pc2 = np.array(my_data[:, 5])
    return PCData(dates, pc1, pc2)

