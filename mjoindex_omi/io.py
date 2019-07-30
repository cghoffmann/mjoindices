# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import os
import datetime
import numpy as np
import pandas as pd
import typing


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


def save_pcs_to_txt_file(dates: np.ndarray, pc1: np.ndarray, pc2: np.ndarray, filename: str) -> None:
    """Saves the computed principal components to a .txt file.

    Please note that the file format is not exactly that of the original data files. However, a
    suitable reader is available in this module for both formats.

    :param dates: The numpy array containing the np.datetime64 dates.
    :param pc1: The numpy array containing the values of PC1 (has to be of same length as the dates array).
    :param pc2: The numpy array containing the values of PC2 (has to be of same length as the dates array).
    :param filename: The filename.
    """
    df = pd.DataFrame({"Date": dates, "PC1": pc1, "PC2": pc2})
    df.to_csv(filename, index=False, float_format="%.5f")


def load_pcs_from_txt_file(filename: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the principal components (PCs) of OMI, which were previously saved with this package.

    :param filename: Path to local principal component file
    :return: A tuple containing three array with 1) the dates as numpy.datetime64, 2 PC1 and 3) PC2
    """
    df = pd.read_csv(filename, sep=',', parse_dates=[0], header=0)
    dates = df.Date.values
    pc1 = df.PC1.values
    pc2 = df.PC2.values
    return dates, pc1, pc2


def load_original_pcs_from_txt_file(filename: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the principal components (PCs) of OMI, which are stored in the original file format.
       Particularly the following file can be loaded:
       https://www.esrl.noaa.gov/psd/mjo/mjoindex/omi.1x.txt

       Note that the present software package stores the PCs slightly different (see load_pcs_from_txt_file)

       :param filename: Path to local principal component file
       :return: A tuple containing three array with 1) the dates as numpy.datetime64, 2 PC1 and 3) PC2
    """
    my_data = np.genfromtxt(filename)
    dates_temp = []
    for i in range(0, my_data.shape[0]):
        dates_temp.append(
            datetime.datetime(my_data[i, 0].astype(np.int), my_data[i, 1].astype(np.int), my_data[i, 2].astype(np.int)))
    dates = np.array(dates_temp, dtype='datetime64')
    pc1 = np.array(my_data[:, 4])
    pc2 = np.array(my_data[:, 5])
    return dates, pc1, pc2
