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

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCData:
    """
    Class as a container for the principal component (PC) data.
    """

    # FIXME: Check if we need some deepcopy or writable=false here
    def __init__(self, time: np.ndarray, pc1: np.ndarray, pc2: np.ndarray) -> None:
        """Initialization with all necessary variables.

        :param time: The numpy array containing the np.datetime64 dates.
        :param pc1: The numpy array containing the values of PC1 (has to be of same length as the time array).
        :param pc2: The numpy array containing the values of PC2 (has to be of same length as the time array).
        """
        if pc1.size == time.size and pc2.size == time.size:
            self._time = time.copy()
            self._pc1 = pc1.copy()
            self._pc2 = pc2.copy()
        else:
            raise ValueError('Length of at least one principal component time series does not fit to length of the '
                             'time grid')

    @property
    def time(self) -> np.ndarray:
        """ The time grid of the PC time series as numpy array of numpy.datetime64 elements.
        :return: The grid
        """
        return self._time

    @property
    def pc1(self) -> np.ndarray:
        """The time series of the PC1 values
        :return: The PC1 values
        """
        return self._pc1

    @property
    def pc2(self) -> np.ndarray:
        """The time series of the PC2 values
        :return: The PC2 values
        """
        return self._pc2

    # FIXME: implement also storing to npz file
    def save_pcs_to_txt_file(self, filename: Path) -> None:
        """Saves the computed principal components to a .txt file.

        Please note that the file format is not exactly that of the original data files. However, a
        suitable reader is available in this module for both formats.

        :param filename: The full filename.
        """
        df = pd.DataFrame({"Date": self._time, "PC1": self._pc1, "PC2": self._pc2})
        df.to_csv(filename, index=False, float_format="%.5f")


def load_pcs_from_txt_file(filename: Path) -> PCData:
    """Loads the principal components (PCs) of OMI, which were previously saved with this package.

    :param filename: Path to local principal component file
    :return: A PCData instance containing the values
    """
    df = pd.read_csv(filename, sep=',', parse_dates=[0], header=0)
    dates = df.Date.values
    pc1 = df.PC1.values
    pc2 = df.PC2.values
    return PCData(dates, pc1, pc2)


def load_original_pcs_from_txt_file(filename: Path) -> PCData:
    """Loads the principal components (PCs) of OMI, which are stored in the original file format.
       Particularly the following file can be loaded:
       https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt

       Note that the present software package stores the PCs slightly different (see load_pcs_from_txt_file)

       :param filename: Path to local principal component file
       :return:  A PCData instance containing the values
    """
    my_data = np.genfromtxt(filename)
    dates_temp = []
    for i in range(0, my_data.shape[0]):
        dates_temp.append(
            datetime.datetime(my_data[i, 0].astype(np.int), my_data[i, 1].astype(np.int), my_data[i, 2].astype(np.int)))
    dates = np.array(dates_temp, dtype='datetime64')
    pc1 = np.array(my_data[:, 4])
    pc2 = np.array(my_data[:, 5])
    return PCData(dates, pc1, pc2)


def plot_comparison_orig_calc_pcs(calc_pcs: PCData, orig_pcs: PCData, startDate=None, endDate=None):
    fig, axs = plt.subplots(2, 1, num="ReproduceOriginalOMIPCs_PCs", clear=True, figsize=(8, 6), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("PC Recalculation")

    ax = axs[0]
    ax.set_title("PC1")
    p1, = ax.plot(orig_pcs.time, orig_pcs.pc1, label="Original")
    p2, = ax.plot(calc_pcs.time, calc_pcs.pc1, label="Recalculation")
    if startDate != None and endDate != None:
        ax.set_xlim((startDate, endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    ax.legend(handles=(p1, p2))

    #corr = (np.corrcoef(orig_pcs.pc1, calc_pcs.pc1))[0, 1]
    # FIXME: Calculate correlation only for wanted period
    # FIXME: Check that periods covered by orig_omi and recalc_omi are actually the same
    #plt.text(0.1, 0.1, "Correlation over complete period: %.3f" % corr, transform=ax.transAxes)

    ax = axs[1]
    ax.set_title("PC2")
    p3, = ax.plot(orig_pcs.time, orig_pcs.pc2, label="Original")
    p4, = ax.plot(calc_pcs.time, calc_pcs.pc2, label="Recalculation")
    if startDate is not None and endDate is not None:
        ax.set_xlim((startDate, endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment="right")
    ax.legend(handles=(p3, p4))

    #corr = (np.corrcoef(orig_pcs.pc2, calc_pcs.pc2))[0, 1]
    #plt.text(0.1, 0.1, "Correlation over complete period: %.3f" % corr, transform=ax.transAxes)

    return fig
