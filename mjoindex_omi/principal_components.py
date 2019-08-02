# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import datetime

import numpy as np
import pandas as pd


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
            self._time = time
            self._pc1 = pc1
            self._pc2 = pc2
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

    def save_pcs_to_txt_file(self, filename: str) -> None:
        """Saves the computed principal components to a .txt file.

        Please note that the file format is not exactly that of the original data files. However, a
        suitable reader is available in this module for both formats.

        :param filename: The full filename.
        """
        df = pd.DataFrame({"Date": self._time, "PC1": self._pc1, "PC2": self._pc2})
        df.to_csv(filename, index=False, float_format="%.5f")


def load_pcs_from_txt_file(filename: str) -> PCData:
    """Loads the principal components (PCs) of OMI, which were previously saved with this package.

    :param filename: Path to local principal component file
    :return: A PCData instance containing the values
    """
    df = pd.read_csv(filename, sep=',', parse_dates=[0], header=0)
    dates = df.Date.values
    pc1 = df.PC1.values
    pc2 = df.PC2.values
    return PCData(dates, pc1, pc2)


def load_original_pcs_from_txt_file(filename: str) -> PCData:
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