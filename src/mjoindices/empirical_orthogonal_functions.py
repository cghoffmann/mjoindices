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

""" This module provides basic functionality to handle EOF data, which is a basic output of the OMI calculation. """

import copy
import typing
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm

from mjoindices.tools import doy_list


class EOFData:
    """
    This class serves as a container for the EOF data of one pair of EOFs.

    :param lat: The latitude grid of the EOF data.
    :param long: The longitude grid of the EOF data.
    :param eof1: Values of the first EOF. Can either be a 1-dim vector with
        lat.size*long.size elements (Start with all values of the first latitude, then all values of the second
        latitude, etc.) or a 2-dim map with the first index representing the latitude axis and the second index
        representing longitude axis.
    :param eof2: Values of the second EOF. Structure similar to *eof1*
    :param explained_variances: Fraction of data variance that is explained by each EOF (for all EOFs and not only the
        first two EOFs). May be None.
    :param eigenvalues: Eigenvalue corresponding to each EOF. May be None.
    :param no_observations: The number of observations that went into the EOF calculation. May be None.

    Note that the explained variances are not independent from the eigenvalues. However, this class is meant to only
    store the data. Hence, we store redundant data here intentionally to be able to have all computation rules together
    in another location.
    """

    def __init__(self, lat: np.ndarray, long: np.ndarray, eof1: np.ndarray, eof2: np.ndarray,
                 explained_variances: np.ndarray = None, eigenvalues: np.ndarray = None,
                 no_observations: int = None) -> None:
        """
        Initialization with all necessary variables.
        """
        if not eof1.shape == eof2.shape:
            raise ValueError("EOF1 and EOF2 must have the same shape")
        expected_n = lat.size * long.size

        if eof1.size != expected_n or eof2.size != expected_n:
            raise ValueError("Number of elements of EOF1 and EOF2 must be identical to lat.size*long.size")

        self._lat = lat.copy()
        self._long = long.copy()
        if eof1.ndim == 1 and eof2.ndim == 1:
            self._eof1 = eof1.copy()
            self._eof2 = eof2.copy()
        elif eof1.ndim == 2 and eof2.ndim == 2:
            if not (eof1.shape[0] == lat.size and eof1.shape[1] == long.size and eof2.shape[0] == lat.size and
                    eof2.shape[1]):
                raise ValueError("Length of first axis of EOS 1 and 2 must correspond to latitude axis, length of "
                                 "second axis to the longitude axis")
            self._eof1 = self.reshape_to_vector(eof1.copy())
            self._eof2 = self.reshape_to_vector(eof2.copy())
        else:
            raise ValueError("EOF1 and EOF2 must have a dimension of 1 or 2.")

        if eigenvalues is not None:
            if eigenvalues.size != self.eof1vector.size:
                raise ValueError("Eigenvalues (if not None) must have same length as the second axis of the EOFs")
            self._eigenvalues = eigenvalues.copy()
        else:
            self._eigenvalues = None
        if explained_variances is not None:
            if explained_variances.size != self.eof1vector.size:
                raise ValueError("Explained variances (if not None) must have same length as the second axis of "
                                 "the EOFs")
            self._explained_variances = explained_variances.copy()
        else:
            self._explained_variances = None
        self._no_observations = no_observations

    def __eq__(self, other: "EOFData") -> bool:
        """
        Override the default equals behavior
        """
        return (np.all(self.lat == other.lat)
                and np.all(self.long == other.long)
                and np.all(self.eof1vector == other.eof1vector)
                and np.all(self.eof2vector == other.eof2vector)
                and np.all(self._explained_variances == other.explained_variances)
                and np.all(self._eigenvalues == other.eigenvalues)
                and self._no_observations == other.no_observations)

    def close(self, other: "EOFData") -> bool:
        """
        Checks equality of two :class:`EOFData` objects, but allows for numerical tolerances.

        :param other: The object to compare with.

        :return: Equality of all members considering the default tolerances of :func:`numpy.allclose`
        """
        return (np.allclose(self.lat, other.lat)
                and np.allclose(self.long, other.long)
                and np.allclose(self.eof1vector, other.eof1vector)
                and np.allclose(self.eof2vector, other.eof2vector)
                and np.allclose(self._explained_variances, other.explained_variances)
                and np.allclose(self._eigenvalues, other.eigenvalues)
                and self._no_observations == other.no_observations)

    @property
    def lat(self) -> np.ndarray:
        """
        The latitude grid of the EOFs.
        """
        return self._lat

    @property
    def long(self) -> np.ndarray:
        """
        The longitude grid of the EOFs.
        """
        return self._long

    @property
    def eof1vector(self) -> np.ndarray:
        """
        EOF1 as a vector (and not a matrix).
        """
        return self._eof1

    @property
    def eof2vector(self) -> np.ndarray:
        """
        EOF2 as a vector (and not a matrix).
        """
        return self._eof2

    @property
    def eof1map(self) -> np.ndarray:
        """
        EOF1 as a 2-dimensional map.
        """
        return self.reshape_to_map(self._eof1)

    @property
    def eof2map(self) -> np.ndarray:
        """
        EOF2 as a 2-dimensional map.
        """
        return self.reshape_to_map(self._eof2)

    @property
    def explained_variances(self) -> np.ndarray:
        """
        The explained variances of all EOFs (not only the first two ones).

        :return: Explained Variances. Might be None.
        """
        return self._explained_variances

    @property
    def explained_variance_eof1(self) -> float:
        """
        The explained variance of EOF1 as fraction between 0 and 1.

        :return: The variance. Might be None.
        """
        if self._explained_variances is not None:
            return self._explained_variances[0]
        else:
            return None

    @property
    def explained_variance_eof2(self) -> float:
        """
        The explained variance of EOF2 as fraction between 0 and 1.

        :return: The variance. Might be None.
        """
        if self._explained_variances is not None:
            return self._explained_variances[1]
        else:
            return None

    @property
    def sum_of_explained_variances(self) -> float:
        """
        Returns the total variance explained by all EOFs.

        This should be close to 1 if the calculation was successful.

        :return: The total explained variance. Might be None.
        """
        if self._explained_variances is not None:
            return np.sum(self._explained_variances)
        else:
            return None

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        The eigenvalues of all EOFs (not only the first two ones).

        :return: The eigenvalues. Might be None.
        """
        return self._eigenvalues

    @property
    def eigenvalue_eof1(self) -> float:
        """
        The eigenvalue of EOF1.

        :return: The eigenvalue. Might be None.
        """
        if self.eigenvalues is not None:
            return self._eigenvalues[0]
        else:
            return None

    @property
    def eigenvalue_eof2(self) -> float:
        """
        The eigenvalue of EOF2.

        :return: The eigenvalue. Might be None.
        """
        if self.eigenvalues is not None:
            return self._eigenvalues[1]
        else:
            return None

    @property
    def no_observations(self) -> int:
        """
        The number of observations that went into the calculation of the EOFs.
        """
        return self._no_observations

    def reshape_to_vector(self, map: np.ndarray) -> np.ndarray:
        """
        Reshapes the horizontally distributed data to fit into a vector.

        The vector elements will contain the values of all longitudes for the first latitude,
        and then the values of all longitudes for the second latitude, etc.

        :param map: The 2-dim data. The first dimension must correspond to the latitude grid, the second to the
            longitude grid.

        :return: The data as a vector.
        """
        if not map.ndim == 2:
            raise ValueError("eof_map must have 2 dimensions")
        if not (map.shape[0] == self.lat.size and map.shape[1] == self.long.size):
            raise ValueError(
                "Length of first dimension of eof_map must correspond to the latitude grid, length of second dimension to the longitude grid")
        return np.reshape(map, self.lat.size * self.long.size)

    def reshape_to_map(self, vector: np.ndarray) -> np.ndarray:
        """
        Reshapes data in a vector to fit into a matrix, which corresponds to the present latitude/longitude grid.

        :param vector: The vector with the data. Must have the length lat.size*long.size.

        :return: The map. The first index corresponds to the latitude grid, the second to longitude grid.
        """
        if not vector.ndim == 1:
            raise ValueError("Vector must have only 1 dimension.")
        if not vector.size == self.lat.size * self.long.size:
            raise ValueError("Vector must have lat.size*long.size elements.")
        # The following transformation has been double-checked graphically by comparing resulting plot to
        # https://www.esrl.noaa.gov/psd/mjo/mjoindex/animation/
        return np.reshape(vector, [self.lat.size, self.long.size])

    def save_eofs_to_txt_file(self, filename: Path) -> None:
        """
        Saves both EOFs to a .txt file.

        Note that the file format is not exactly that of the original data files. Particularly, both EOFs are written
        into the same file instead of 2 separate files. Furthermore, the lat/long-grids are also explicitly saved.

        A suitable reader is provided by the function
        :func:`mjoindices.empirical_orthogonal_functions.load_single_eofs_from_txt_file`, whereas a reader for the
        original files is provided by :func:`mjoindices.empirical_orthogonal_functions.load_original_eofs_for_doy`

        :param filename: The full path- and filename.
        """
        lat_full = np.empty(self.eof1vector.size)
        long_full = np.empty(self.eof1vector.size)
        for i_vec in range(0, self.eof1vector.size):
            # find out lat/long corresponding to the vector position by evaluating the index transformation from map
            # to vector
            (i_lat, i_long) = np.unravel_index(i_vec, self.eof1map.shape)
            lat_full[i_vec] = self.lat[i_lat]
            long_full[i_vec] = self.long[i_long]
        df = pd.DataFrame(
            {"Lat": lat_full, "Long": long_full, "EOF1": self.eof1vector, "EOF2": self.eof2vector}).astype(float)
        df.to_csv(filename, index=False, float_format="%13.7f")


class EOFDataForAllDOYs:
    """
    This class serves as a container for a series of EOF pairs, which covers all 366 DOYs and provides some overall
    statistical quantities.

    The basic EOF computation :func:`mjoindices.omi.omi_calculator.calc_eofs_from_olr` will return an object of this
    class as a major result of this package.

    The individual EOF pairs are represented by :class:`EOFData` objects.

    :param eof_list: A list with the 366 :class:`EOFData` objects.
    """

    def __init__(self, eof_list: typing.List[EOFData]) -> None:
        """

        :param eof_list:
        """
        if len(eof_list) != 366:
            raise ValueError("List of EOFs must contain 366 entries")
        reference_lat = eof_list[0].lat
        reference_long = eof_list[0].long
        for i in range(0, 366):
            if not np.all(eof_list[i].lat == reference_lat):
                raise ValueError("All EOFs must have the same latitude grid. Problematic is DOY %i" % i)
            if not np.all(eof_list[i].long == reference_long):
                raise ValueError("All EOFs must have the same longitude grid. Problematic is DOY %i" % i)
        # deepcopy eofs so that they cannot be modified accidentally from outside after the consistency checks
        self._eof_list = copy.deepcopy(eof_list)

    # FIXME: Could this property be used to modify the EOFData objects because they are mutual?
    @property
    def eof_list(self) -> typing.List[EOFData]:
        """
        EOF data for all DOYs as a list.

        Remember that DOY 1 corresponds to list entry 0.
        """
        return self._eof_list

    @property
    def lat(self) -> np.ndarray:
        """
        The latitude grid common to the EOFs of all DOYs.

        """
        return self.eof_list[0].lat

    @property
    def long(self) -> np.ndarray:
        """
        The longitude grid common to the EOFs of all DOYs.
        """
        return self.eof_list[0].long

    def eofdata_for_doy(self, doy: int) -> EOFData:
        """
        Returns the :class:`EOFData` object for a particular DOY.

        :param doy: The DOY

        :return: The :class:`EOFData` object
        """
        return self.eof_list[doy - 1]

    def eof1vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF1 vector of a particular DOY.

        :param doy: The DOY.

        :return: The vector.
        """
        return self.eof_list[doy - 1].eof1vector

    def eof2vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF2 vector of a particular DOY.

        :param doy: The DOY.

        :return: The vector.
        """
        return self.eof_list[doy - 1].eof2vector

    def explained_variance1_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the explained variance of EOF1 for each DOY.

        :return: The variance vector.

        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).explained_variance_eof1)
        return np.array(result)

    def explained_variance2_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the explained variance of EOF2 for each DOY.

        :return: The variance vector.
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).explained_variance_eof2)
        return np.array(result)

    def total_explained_variance_for_all_doys(self):
        """
        Returns a vector with 366 elements containing for each DOY the sum of the explained variance over all EOFs.

        :return: The variance vector. Should by close to 1 for each DOY if computation was successful.
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).sum_of_explained_variances)
        return np.array(result)

    def no_observations_for_all_doys(self):
        """
        Returns a vector with 366 elements containing for each DOY the number of observations that went into the
        computation of the EOFs.

        :return: The number of observations vector.
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).no_observations)
        return np.array(result)

    def eigenvalue1_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the eigenvalues of EOF1 for each DOY.

        :return: The eigenvalue vector.
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).eigenvalue_eof1)
        return np.array(result)

    def eigenvalue2_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the eigenvalues of EOF2 for each DOY.

        :return: The eigenvalue vector.
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).eigenvalue_eof2)
        return np.array(result)

    def save_all_eofs_to_dir(self, dirname: Path, create_dir=True) -> None:
        """
        Saves the EOF1 and EOF2 functions for each of the DOYs in the given directory.

        For each DOY, one text file will be created, which contains both EOF functions.
        Note that the text files do not contain the eigenvalues and explained variance values. To save also those
        values, use the function :func:`save_all_eofs_to_npzfile`.

        :param dirname: The directory, where the files will be saved into.
        :param create_dir: If True, the directory (and parent directories) will be created, if not existing.
        """
        if not dirname.exists() and create_dir:
            dirname.mkdir(parents=True, exist_ok=False)
        for doy in doy_list():
            filename = dirname / Path("eof%s.txt" % format(doy, '03'))
            self.eofdata_for_doy(doy).save_eofs_to_txt_file(filename)

    def save_all_eofs_to_npzfile(self, filename: Path) -> None:
        """
        Saves the complete EOF data to a numpy file.

        :param filename: The filename.
        """
        doys = doy_list()
        eof1 = np.empty((doys.size, self.lat.size * self.long.size))
        eof2 = np.empty((doys.size, self.lat.size * self.long.size))
        eigenvalues = np.empty((doys.size, self.lat.size * self.long.size))
        explained_variances = np.empty((doys.size, self.lat.size * self.long.size))
        no_observations = np.empty(doys.size)
        for i in range(0, doys.size):
            eof = self.eof_list[i]
            eof1[i, :] = eof.eof1vector
            eof2[i, :] = eof.eof2vector
            eigenvalues[i, :] = eof.eigenvalues
            explained_variances[i, :] = eof.explained_variances
            no_observations[i] = eof.no_observations
        np.savez(filename,
                 eof1=eof1,
                 eof2=eof2,
                 explained_variances=explained_variances,
                 eigenvalues=eigenvalues,
                 no_observations=no_observations,
                 lat=self.lat,
                 long=self.long)


def load_single_eofs_from_txt_file(filename: Path) -> EOFData:
    """
    Loads a pair of EOFs, which was previously saved with this package (function :func:`EOFData.save_eofs_to_txt_file`).

    :param filename: Path to the  EOF file.

    :return: The pair of EOFs.
    """
    df = pd.read_csv(filename, sep=',', header=0)
    full_lat = df.Lat.values
    full_long = df.Long.values
    eof1 = df.EOF1.values
    eof2 = df.EOF2.values

    # retrieve unique lat/long grids
    # FIXME: Does probably not work for a file with only one lat
    lat, repetition_idx, rep_counts = np.unique(full_lat, return_index=True, return_counts=True)
    long = full_long[repetition_idx[0]:repetition_idx[1]]

    # Apply some heuristic consistency checks
    if not np.unique(rep_counts).size == 1:
        # All latitudes must have the same number of longitudes
        raise ValueError("Lat/Long grid in input file seems to be corrupted 1")
    if not np.unique(repetition_idx[0:-1] - repetition_idx[1:]).size == 1:
        # All beginnings of a new lat have to be equally spaced
        raise ValueError("Lat/Long grid in input file seems to be corrupted 2")
    if not lat.size * long.size == eof1.size:
        raise ValueError("Lat/Long grid in input file seems to be corrupted 3")
    if not np.all(np.tile(long, lat.size) == full_long):
        # The longitude grid has to be the same for all latitudes
        raise ValueError("Lat/Long grid in input file seems to be corrupted 4")

    return EOFData(lat, long, eof1, eof2)


def load_original_eofs_for_doy(dirname: Path, doy: int) -> EOFData:
    """
    Loads the EOF values for the first 2 EOFs from the original file format.

    Note that the EOFs are represented as pure vectors in the original treatment, so that a connection to the
    individual locations on a world map is not obvious without any further knowledge. The corresponding grid is here
    inserted hardcodedly.

    The original EOFs are found here: ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
    ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/

    :param dirname: Path to the directory, in which the EOFs for all DOYs are stored.
        This path should contain the sub directories *eof1* and *eof2*, in which the 366 files each are located:
        One file per day of the year.
    :param doy: DOY for which the 2 EOFs are loaded (number between 1 and 366).

    :return: The pair of EOFs.
    """
    orig_lat = np.arange(-20., 20.1, 2.5)
    orig_long = np.arange(0., 359.9, 2.5)
    eof1filename = dirname / "eof1" / ("eof" + str(doy).zfill(3) + ".txt")
    eof1 = np.genfromtxt(eof1filename)
    eof2filename = dirname / "eof2" / ("eof" + str(doy).zfill(3) + ".txt")
    eof2 = np.genfromtxt(eof2filename)
    return EOFData(orig_lat, orig_long, eof1, eof2)


def load_all_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    """
    Loads the EOF functions (created with the function :func:`EOFDataForAllDOYs.save_all_eofs_to_dir`)
    for all DOYs from the given directory

    :param dirname: The directory in which the files are stored.

    :return: The EOFs for all DOYs.
    """
    eofs = []
    for doy in doy_list():
        filename = dirname / Path("eof%s.txt" % format(doy, '03'))
        eof = load_single_eofs_from_txt_file(filename)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def load_all_original_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    """
    Loads the EOF functions for all DOYs from the original file format.

    The original EOFs are found here: ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and
    ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/

    Note that the EOFs are represented as pure vectors in the original treatment, so that a connection to the
    individual locations on a world map is not obvious without any further knowledge. The corresponding grid is here
    inserted hardcodedly.

    :param dirname: Path to the directory, in which the EOFs for all DOYs are stored.
        This path should contain the sub directories *eof1* and *eof2*, in which the 366 files each are located:
        One file per day of the year.

    :return: The original EOFs for all DOYs.
    """
    eofs = []
    for doy in doy_list():
        eof = load_original_eofs_for_doy(dirname, doy)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def restore_all_eofs_from_npzfile(filename: Path) -> EOFDataForAllDOYs:
    """
    Loads all EOF data from a numpy file, which was written with :func:`EOFDataForAllDOYs.save_all_eofs_to_npzfile`.

    :param filename: The filename.

    :return: The EOFs for all DOYs.
    """
    with np.load(filename) as data:
        eof1 = data["eof1"]
        eof2 = data["eof2"]
        lat = data["lat"]
        long = data["long"]
        eigenvalues = data["eigenvalues"]
        explained_variances = data["explained_variances"]
        no_observations = data["no_observations"]
    eofs = []
    for i in range(0, doy_list().size):
        eof = EOFData(lat, long, np.squeeze(eof1[i, :]), np.squeeze(eof2[i, :]),
                      eigenvalues=np.squeeze(eigenvalues[i, :]),
                      explained_variances=np.squeeze(explained_variances[i, :]),
                      no_observations=no_observations[i])
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def plot_explained_variance_for_all_doys(eofs: EOFDataForAllDOYs, include_total_variance: bool = False,
                                         include_no_observations: bool = False) -> Figure:
    """
    Plots the explained variance values for EOF1 and EOF2 for all DOYs.

    Comparable to Kiladis (2014), Fig. 1 (although the values there are to high by a factor of 2).

    :param eofs: The EOF data to plot.

    :return: Handle to the figure.
    """
    doygrid = doy_list()
    fig = plt.figure("plot_explained_variance_for_all_doys", clear=True, figsize=(6, 4), dpi=150)
    ax1 = fig.add_subplot(111)
    handles = []
    p1, = ax1.plot(doygrid, eofs.explained_variance1_for_all_doys(), color="blue", label="EOF1")
    handles.append(p1)
    p2, = ax1.plot(doygrid, eofs.explained_variance2_for_all_doys(), color="red", label="EOF2")
    handles.append(p2)
    if include_total_variance:
        p3, = ax1.plot(doygrid, eofs.total_explained_variance_for_all_doys(), color="green", label="Total")
        handles.append(p3)
    ax1.set_xlabel("DOY")
    ax1.set_ylabel("Fraction of explained variance")
    ax1.set_xlim((0, 366))

    if include_no_observations:
        ax2 = ax1.twinx()
        p4, = ax2.plot(doygrid, eofs.no_observations_for_all_doys(), color="black", label="Number of observations", linestyle="--")
        handles.append(p4)
        ax2.set_ylabel("Number of observations")
        ymin = np.min(eofs.no_observations_for_all_doys()) - np.min(eofs.no_observations_for_all_doys()) * 0.1
        ymax = np.max(eofs.no_observations_for_all_doys()) + np.max(eofs.no_observations_for_all_doys()) * 0.1
        ax2.set_ylim([ymin, ymax])
    plt.title("Explained variance")
    plt.legend(handles=tuple(handles))
    return fig


def plot_eigenvalues_for_all_doys(eofs: EOFDataForAllDOYs) -> Figure:
    """
    Plots the Eigenvalues for EOF1 and EOF2 for all DOYs.

    :param eofs: The EOF data to plot.

    :return: Handle to the figure.
    """
    doygrid = doy_list()
    fig = plt.figure("plot_eigenvalues_for_all_doys", clear=True, figsize=(6, 4), dpi=150)
    p1, = plt.plot(doygrid, eofs.eigenvalue1_for_all_doys(), color="blue", label="EOF1")
    p2, = plt.plot(doygrid, eofs.eigenvalue2_for_all_doys(), color="red", label="EOF2")
    plt.xlabel("DOY")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues")
    plt.legend(handles=(p1, p2))
    return fig


def plot_original_individual_eof_map(path, doy: int) -> Figure:
    """
    Plots a pair of original EOFs, which are loaded from a directory, in two maps.

    :param path: The directory with the EOF data (see :func:`load_original_eofs_for_doy` for details).
    :param doy: The corresponding DOY. Only used to display it in the title.

    :return: Handle to the figure.
    """
    eofdata = load_original_eofs_for_doy(path, doy)
    return plot_individual_eof_map(eofdata, doy=doy)


def plot_individual_eof_map_from_file(filename, doy: int) -> Figure:
    """
    Plots a pair of EOFs, which are loaded from a file, in two maps.

    :param filename: The file with the EOF data.
    :param doy: The corresponding DOY. Only used to display it in the title.

    :return: Handle to the figure.
    """
    eofdata = load_single_eofs_from_txt_file(filename, doy=doy)
    return plot_individual_eof_map(eofdata)


def plot_individual_eof_map(eofdata: EOFData, doy: int = None) -> Figure:
    """
    Plots a pair of EOFs for a particular DOY in two maps.

    :param eofdata: The EOF data to plot.
    :param doy: The corresponding DOY. Only used to display it in the title.

    :return: Handle to the figure.
    """
    # TODO: Plot underlying map
    fig, axs = plt.subplots(2, 1, num="plotting.plot_eof_for_doy", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0]

    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof1map, levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF1")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1]
    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof2map, levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    return fig


def plot_individual_explained_variance_all_eofs(eof: EOFData, doy: int = None, max_eof_number: int = None) -> Figure:
    """
    Plots the explained variances for each EOF function, but only for EOF data of one DOY.

    This is useful to confirm that the first 2 EOFs cover actually most of the variance.

    :param eof: The EOF data.
    :param doy: The corresponding DOY. Only used to display it in the title.
    :param max_eof_number: The limit of the x-axis.

    :return: Handle to the figure.
    """
    eof_number_grid = np.arange(0, eof.explained_variances.size, 1) + 1
    fig = plt.figure("plot_individual_explained_variance_all_eofs", clear=True, figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(111)
    p1 = plt.plot(eof_number_grid, eof.explained_variances, color="blue", label="Explained variance")
    if max_eof_number is not None:
        plt.xlim(0.5, max_eof_number)
    plt.xlabel("EOF Number")
    plt.ylabel("Fraction of explained variance")
    plt.text(0.6, 0.9, "Sum over all EOFs: %1.2f" % eof.sum_of_explained_variances, transform=ax.transAxes)

    if doy is not None:
        plt.title("Explained variance for DOY %i" % doy)
    else:
        plt.title("Explained variance")
    return fig
