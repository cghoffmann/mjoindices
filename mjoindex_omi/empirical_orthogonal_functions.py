# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import copy
import os
import typing
from pathlib import Path

import numpy as np
import pandas as pd


class EOFData:
    """
    Class as a container for the EOF data of one pair of EOFs.
    """
    #FIXME: Check if we need some deepcopy or writable=false here
    def __init__(self, lat: np.ndarray, long: np.ndarray, eof1: np.ndarray, eof2:np.ndarray) -> None:
        """Initialization with all necessary variables.

        :param lat: The latitude grid, which represents the EOF data
        :param long: The longitude grid,
        which represents the EOF data
        :param eof1: Values of the first EOF. Can either be a 1-dim vector with
        lat.size*long.size elements (Start with all values of the first latitude, then all values of the second
        latitude, etc.) or a 2-dim map with the first index representing the latitude axis and the second index
        representing longitude.
        :param eof2: Values of the second EOF. Structure similar to the first EOF
        """
        if not eof1.shape == eof2.shape:
            raise AttributeError("EOF1 and EOF2 must have the same shape")
        expected_n = lat.size * long.size

        if eof1.size != expected_n  or eof2.size != expected_n :
            raise AttributeError("Number of elements of EOF1 and EOF2 must be identical to lat.size*long.size")

        self._lat = lat
        self._long = long
        if eof1.ndim == 1 and eof2.ndim == 1:
            self._eof1 = eof1
            self._eof2 = eof2
        elif eof1.ndim == 2 and eof2.ndim == 2:
            if not (eof1.shape[0] == lat.size and eof1.shape[1] == long.size and eof2.shape[0] == lat.size and eof2.shape[1]):
                raise AttributeError("Length of first axis of EOS 1 and 2 must correspond to latitude axis, length of "
                                     "second axis to the longitude axis")
            self._eof1 = self.reshape_to_vector(eof1)
            self._eof2 = self.reshape_to_vector(eof2)
        else:
            raise AttributeError("EOF1 and EOF2 must have a dimension of 1 or 2.")

    # FIXME: Unittest for operator
    # FIXME: Typing
    def __eq__(self, other):
        """Override the default Equals behavior
        """
        return (np.all(self.lat == other.lat)
                and np.all(self.long == other.long)
                and np.all(self.eof1vector == other.eof1vector)
                and np.all( self.eof2vector == other.eof2vector))

    @property
    def lat(self) -> np.ndarray:
        """ The latitude grid of the EOF.
        :return: The grid
        """
        return self._lat

    @property
    def long(self) -> np.ndarray:
        """ The longitude grid of the EOF.
        :return: The grid
        """
        return self._long

    @property
    def eof1vector(self) -> np.ndarray:
        """ EOF1 as a vector
        :return: The EOF values
        """
        return self._eof1

    @property
    def eof2vector(self) -> np.ndarray:
        """ EOF2 as a vector
        :return: The EOF values
        """
        return self._eof2

    @property
    def eof1map(self) -> np.ndarray:
        """ EOF1 as a 2-dimensional map
        :return: The EOF values
        """
        return self.reshape_to_map(self._eof1)

    @property
    def eof2map(self) -> np.ndarray:
        """ EOF2 as a 2-dimensional map
        :return: The EOF values
        """
        return self.reshape_to_map(self._eof2)

    def reshape_to_vector(self, map: np.ndarray) -> np.ndarray:
        """ Reshapes the horizontally distributed data to fit into a vector The vector elements will contain the
        values of all longitudes for the first latitude, and then the values of all logituted for the second latitude
        etc.
        :param map: The 2-dim data. first dimension must correspond to the latitude grid, the second to the
        longitude grid
        :return: the vector
        """
        if not map.ndim == 2:
            raise AttributeError("eof_map must have 2 dimensions")
        if not (map.shape[0] == self.lat.size and map.shape[1] == self.long.size):
            raise AttributeError("Length of first dimension of eof_map must correspond to the latitude grid, length of second dimension to the longitude grid")
        return np.reshape(map, self.lat.size * self.long.size)

    def reshape_to_map(self, vector: np.ndarray) -> np.ndarray:
        """ Reshapes data in a vector to fit to the present lat-long-grid.

        :param vector: the vector with the data. Must have the length lat.size*long.size
        :return: The map. The first index corresponds to the latitude grid, the second to longitude.
        """
        if not vector.ndim == 1:
            raise AttributeError("Vector must have only 1 dimension.")
        if not vector.size == self.lat.size * self.long.size:
            raise AttributeError("Vector must have lat.size*long.size elements.")
        # The following transformation has been double-checked graphically by comparing resulting plot to
        # https://www.esrl.noaa.gov/psd/mjo/mjoindex/animation/
        return np.reshape(vector, [self.lat.size, self.long.size])

    def save_eofs_to_txt_file(self, filename: str) -> None:
        """Saves both EOFs to a .txt file.

        Please note that the file format is not exactly that of the original data files. However, a suitable reader
        is available in this module for both formats. Particularly, both EOFs are written into the same file instead
        of 2 separate files. Furthermore, the lat/long-grids are also explicitly saved.

        :param filename: The full filename.
        """
        lat_full = np.empty(self.eof1vector.size)
        long_full = np.empty(self.eof1vector.size)
        for i_vec in range(0,self.eof1vector.size):
            # find out lat/long corresponding to the vector position by evaluating the index transformation from map
            # to vector
            (i_lat, i_long) = np.unravel_index(i_vec,self.eof1map.shape)
            lat_full[i_vec] = self.lat[i_lat]
            long_full[i_vec] = self.long[i_long]
        df = pd.DataFrame({"Lat": lat_full, "Long": long_full, "EOF1": self.eof1vector, "EOF2": self.eof2vector}).astype(float)
        df.to_csv(filename, index=False, float_format="%13.7f")


class EOFDataForAllDOYs:

    def __init__(self, eof_list: typing.List[EOFData]) -> None:
        if len(eof_list) != 366:
            raise AttributeError("List of EOFs must contain 366 entries")
        reference_lat = eof_list[0].lat
        reference_long = eof_list[0].long
        for i in range(0,366):
            if not np.all(eof_list[i].lat == reference_lat):
                raise AttributeError("All EOFs must have the same latitude grid. Problematic is DOY %i" % i)
            if not np.all(eof_list[i].long == reference_long):
                raise AttributeError("All EOFs must have the same longitude grid. Problematic is DOY %i" % i)
        # deepcopy eofs so that they cannot be modified accidently from outside after the consistency checks
        self._eof_list = copy.deepcopy(eof_list)

    # FIXME: Could this property be used to modify the EOFData objects because they are mutual?
    @property
    def eof_list(self) -> typing.List[EOFData]:
        """EOF data for all DOYs as a List.

        Remember that DOY 1 corresponds to list entry 0
        :return: The List of EOFData objects
        """
        return self._eof_list

    @property
    def lat(self) -> np.ndarray:
        """The latitude grid common to the EOFs of all DOYs.

        :return: The grid.
        """
        return self.eof_list[0].lat

    @property
    def long(self) -> np.ndarray:
        """The longitude grid common to the EOFs of all DOYs.

        :return: The grid.
        """
        return self.eof_list[0].long

    def eofdata_for_doy(self, doy: int) -> EOFData:
        """
        Returns the EOFData object for a particular DOY
        :param doy: The DOY
        :return: The EOFData object
        """
        return self.eof_list[doy-1]

    def eof1vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF1 vector of a paricular day
        :param doy: The DOY
        :return: The vector
        """
        return self.eof_list[doy-1].eof1vector

    def eof2vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF1 vector of a paricular day
        :param doy: The DOY
        :return: The vector
        """
        return self.eof_list[doy-1].eof2vector

    def save_all_eofs_to_dir(self,dirname: Path, createDir=True):
        if not dirname.exists() and createDir == True:
            dirname.mkdir(parents=False, exist_ok=False)
        for i in range(0,366):
            filename = dirname / Path("eof%s.txt" %  format(i+1, '03'))
            self.eof_list[i].save_eofs_to_txt_file(filename)




def load_single_eofs_from_txt_file(filename: str) -> EOFData:
    """Loads the Empirical Orthogonal Functions (EOFs) of OMI, which were previously saved with this package.

    :param filename: Path to local principal component file
    :return: A PCData instance containing the values
    """
    df = pd.read_csv(filename, sep=',', header=0)
    full_lat = df.Lat.values
    full_long = df.Long.values
    eof1 = df.EOF1.values
    eof2 = df.EOF2.values

    # retrieve unique lat/long grids
    # FIXME: Does probably not work for a file with only one lat
    lat, repetition_idx, rep_counts = np.unique(full_lat,return_index=True,return_counts=True)
    long = full_long[repetition_idx[0]:repetition_idx[1]]

    # Apply some heuristic consistency checks
    if not np.unique(rep_counts).size == 1:
        # All latitudes must have the same number of longitudes
        raise AttributeError("Lat/Long grid in input file seems to be corrupted 1")
    if not np.unique(repetition_idx[0:-1] - repetition_idx[1:]).size == 1:
        # All beginnings of a new lat have to be equally spaced
        raise AttributeError("Lat/Long grid in input file seems to be corrupted 2")
    if not lat.size * long.size == eof1.size:
        raise AttributeError("Lat/Long grid in input file seems to be corrupted 3")
    if not np.all(np.tile(long,lat.size) == full_long):
        # The longitude grid has to be the same for all latitudes
        raise AttributeError("Lat/Long grid in input file seems to be corrupted 4")

    return EOFData(lat, long, eof1, eof2)


def load_original_eofs_for_doy(path: str, doy: int) -> EOFData:
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
    orig_lat = np.arange(-20., 20.1, 2.5)
    orig_long = np.arange(0., 359.9, 2.5)
    eof1filename = path + os.path.sep + "eof1" + os.path.sep + "eof" + str(doy).zfill(3) + ".txt"
    eof1 = np.genfromtxt(eof1filename)
    eof2filename = path + os.path.sep + "eof2" + os.path.sep + "eof" + str(doy).zfill(3) + ".txt"
    eof2 = np.genfromtxt(eof2filename)
    return EOFData(orig_lat, orig_long, eof1, eof2)


def load_all_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    eofs = []
    for doy in range(1, 367):
        filename = dirname / Path("eof%s.txt" % format(doy, '03'))
        eof = load_single_eofs_from_txt_file(filename)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def load_all_original_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    eofs = []
    for doy in range(1, 367):
        eof = load_original_eofs_for_doy(str(dirname), doy)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)