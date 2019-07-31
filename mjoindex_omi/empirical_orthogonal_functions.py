# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import os

import numpy as np


class EOFData:
    """
    Class as a container for the EOF data of one pair of EOFs.
    """

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
