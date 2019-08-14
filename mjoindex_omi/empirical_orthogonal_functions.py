# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import copy
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class EOFData:
    """
    Class as a container for the EOF data of one pair of EOFs.
    """

    # FIXME: Enable saving of statictical values (to csv or nz?) and force setting them?
    def __init__(self, lat: np.ndarray, long: np.ndarray, eof1: np.ndarray, eof2: np.ndarray,
                 explained_variances: np.ndarray = None,  eigenvalues: np.ndarray = None,
                 no_observations: int = None) -> None:
        """Initialization with all necessary variables.

        :param lat: The latitude grid, which represents the EOF data
        :param long: The longitude grid,
        which represents the EOF data
        :param eof1: Values of the first EOF. Can either be a 1-dim vector with
        lat.size*long.size elements (Start with all values of the first latitude, then all values of the second
        latitude, etc.) or a 2-dim map with the first index representing the latitude axis and the second index
        representing longitude.
        :param eof2: Values of the second EOF. Structure similar to the first EOF
        :param explained_variances: Fraction of data variance that is explained by each EOF for all EOFs. Can be None
        :param eigenvalues: Eigenvalue corresponding to each EOF. Can be None
        :param no_observations: The number of observations that went into the EOF calculation

        Note that the explained variances are not independent from the Eigenvalues. However, this class is meant to store
        the data only, so that the computation logic is somewhere else and we keep redundant data here intentionally.
        """
        if not eof1.shape == eof2.shape:
            raise AttributeError("EOF1 and EOF2 must have the same shape")
        expected_n = lat.size * long.size

        if eof1.size != expected_n or eof2.size != expected_n:
            raise AttributeError("Number of elements of EOF1 and EOF2 must be identical to lat.size*long.size")

        self._lat = lat.copy()
        self._long = long.copy()
        if eof1.ndim == 1 and eof2.ndim == 1:
            self._eof1 = eof1.copy()
            self._eof2 = eof2.copy()
        elif eof1.ndim == 2 and eof2.ndim == 2:
            if not (eof1.shape[0] == lat.size and eof1.shape[1] == long.size and eof2.shape[0] == lat.size and
                    eof2.shape[1]):
                raise AttributeError("Length of first axis of EOS 1 and 2 must correspond to latitude axis, length of "
                                     "second axis to the longitude axis")
            self._eof1 = self.reshape_to_vector(eof1.copy())
            self._eof2 = self.reshape_to_vector(eof2.copy())
        else:
            raise AttributeError("EOF1 and EOF2 must have a dimension of 1 or 2.")

        if eigenvalues is not None:
            if eigenvalues.size != self.eof1vector.size:
                raise AttributeError("Eigenvalues (if not None) must have same length as the second axis of the EOFs")
            self._eigenvalues = eigenvalues.copy()
        else:
            self._eigenvalues = None
        if explained_variances is not None:
            if explained_variances.size != self.eof1vector.size:
                raise AttributeError("Explained variances (if not None) must have same length as the second axis of "
                                     "the EOFs")
            self._explained_variances = explained_variances.copy()
        else:
            self._explained_variances = None
        self._no_observations = no_observations

    # FIXME: Unittest for operator
    # FIXME: Typing
    def __eq__(self, other):
        """Override the default Equals behavior
        """
        return (np.all(self.lat == other.lat)
                and np.all(self.long == other.long)
                and np.all(self.eof1vector == other.eof1vector)
                and np.all(self.eof2vector == other.eof2vector)
                and np.all(self._explained_variances == other.explained_variances)
                and np.all(self._eigenvalues == other.eigenvalues)
                and self._no_observations == other.no_observations)

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

    @property
    def explained_variances(self) -> np.ndarray:
        """
        All explained variances
        :return: Explained Variances. Might be None.
        """
        return self._explained_variances

    @property
    def explained_variance_eof1(self) -> float:
        """
        Explained variance of EOF1 as fraction between 0 and 1
        :return: Variance. Might be None.
        """
        if self._explained_variances is not None:
            return self._explained_variances[0]
        else:
            return None

    @property
    def explained_variance_eof2(self) -> float:
        """
        Explained variance of EOF1 as fraction between 0 and 1
        :return: Variance. Might be None.
        """
        if self._explained_variances is not None:
            return self._explained_variances[1]
        else:
            return None

    @property
    def sum_of_explained_variances(self) -> float:
        """
        Returns the total variance explained by all EOFs. This should be close to 1., if the calculation was successful
        :return: The total explained variance
        """
        return np.sum(self._explained_variances)

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        All Eigenvalues
        :return: Eigenvalues. Might be None.
        """
        return self._eigenvalues

    @property
    def eigenvalue_eof1(self) -> float:
        """
        Eigenvalue of EOF1
        :return: Eigenvalue. Might be None.
        """
        if self.eigenvalues is not None:
            return self._eigenvalues[0]
        else:
            return None

    @property
    def eigenvalue_eof2(self) -> float:
        """
        Eigenvalue of EOF2
        :return: Eigenvalue. Might be None.
        """
        if self.eigenvalues is not None:
            return self._eigenvalues[1]
        else:
            return None

    @property
    def no_observations(self) -> int:
        """
        The number of observations that went into the calculation of the EOF
        :return: The number
        """
        return self._no_observations

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
            raise AttributeError(
                "Length of first dimension of eof_map must correspond to the latitude grid, length of second dimension to the longitude grid")
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

    def save_eofs_to_txt_file(self, filename: Path) -> None:
        """Saves both EOFs to a .txt file.

        Please note that the file format is not exactly that of the original data files. However, a suitable reader
        is available in this module for both formats. Particularly, both EOFs are written into the same file instead
        of 2 separate files. Furthermore, the lat/long-grids are also explicitly saved.

        :param filename: The full filename.
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

    def __init__(self, eof_list: typing.List[EOFData]) -> None:
        if len(eof_list) != 366:
            raise AttributeError("List of EOFs must contain 366 entries")
        reference_lat = eof_list[0].lat
        reference_long = eof_list[0].long
        for i in range(0, 366):
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
        return self.eof_list[doy - 1]

    def eof1vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF1 vector of a paricular day
        :param doy: The DOY
        :return: The vector
        """
        return self.eof_list[doy - 1].eof1vector

    def eof2vector_for_doy(self, doy: int) -> np.ndarray:
        """
        Shortcut to the EOF1 vector of a paricular day
        :param doy: The DOY
        :return: The vector
        """
        return self.eof_list[doy - 1].eof2vector

    def explained_variance1_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the explained variance of EOF1 for each DOY.
        :return: The variance vector
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).explained_variance_eof1)
        return result

    def explained_variance2_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the explained variance of EOF2 for each DOY.
        :return: The variance vector
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).explained_variance_eof2)
        return result

    def total_explained_variance_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the sum of the explained variance over all EOFs.
        :return: The variance vector. Should by clode to 1 for each DOY if computation was successful
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).sum_of_explained_variances)
        return result

    def no_observations_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the number of observations that went into the computation of the
        EOFs.
        :return: The number of observations vector
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).no_observations)
        return result

    def eigenvalue1_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the Eigenvalue of EOF1 for each DOY.
        :return: The Eigenvalue vector
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).eigenvalue_eof1)
        return result

    def eigenvalue2_for_all_doys(self):
        """
        Returns a vector with 366 elements containing the Eigenvalue of EOF2 for each DOY.
        :return: The Eigenvalue vector
        """
        doys = doy_list()
        result = []
        for doy in doys:
            result.append(self.eofdata_for_doy(doy).eigenvalue_eof2)
        return result


    def save_all_eofs_to_dir(self, dirname: Path, create_dir=True) -> None:
        """
        Saves the EOF1 and EOF2 functions for each of the DOYs in the given directory
        For each DOY, one text file will be created, which contains both EOF functions
        Note that the textfiles do not contain the Eigenvalues and explained variance values.
        :param dirname: The directory, where the files will be saved
        :param create_dir: If True, the directory (and parent directories) will be created, if is does not exist.
        """
        if not dirname.exists() and create_dir == True:
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
    lat, repetition_idx, rep_counts = np.unique(full_lat, return_index=True, return_counts=True)
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
    if not np.all(np.tile(long, lat.size) == full_long):
        # The longitude grid has to be the same for all latitudes
        raise AttributeError("Lat/Long grid in input file seems to be corrupted 4")

    return EOFData(lat, long, eof1, eof2)


def load_original_eofs_for_doy(dirname: Path, doy: int) -> EOFData:
    """Loads the EOF values for the first 2 EOFs
    Note that as in the original treatment, the EOFs are represented as vectors,
    which means that a connection to the individual locations on a world map is not obvious without any further
    knowledge.

    :param dirname: Path the where the EOFs for all doys are stored. This path should contain the sub directories "eof1"
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
    eof1filename = dirname / "eof1" / ("eof" + str(doy).zfill(3) + ".txt")
    eof1 = np.genfromtxt(eof1filename)
    eof2filename = dirname / "eof2" / ("eof" + str(doy).zfill(3) + ".txt")
    eof2 = np.genfromtxt(eof2filename)
    return EOFData(orig_lat, orig_long, eof1, eof2)


def load_all_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    """
    Loads the EOF functions for all DOYs from the given directory The directory content should have been created with
    EOFDataForAllDOYs.save_all_eofs_to_dir or have a similar structure to be compatible.
    :param dirname: The directory to search for the files
    :return: The EOFs for all DOYs as EOFDataForAllDOYs object.
    """
    eofs = []
    for doy in doy_list():
        filename = dirname / Path("eof%s.txt" % format(doy, '03'))
        eof = load_single_eofs_from_txt_file(filename)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def load_all_original_eofs_from_directory(dirname: Path) -> EOFDataForAllDOYs:
    """
    Loads the EOF functions for all DOYs from the given directory, which are saved in the original format.
    The directory shpuld contain the sub directories "eof1" and "eof2", which contain the downloaded files from
    ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/
    :param dirname: The directory to search for the files.
    :return: The original EOFs for all DOYs as EOFDataForAllDOYs object.
    """
    eofs = []
    for doy in doy_list():
        eof = load_original_eofs_for_doy(dirname, doy)
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def restore_all_eofs_from_npzfile(filename: Path) -> EOFDataForAllDOYs:
    """
    Loads all EOF data from a numpy file, which was written with EOFDataForAllDOYs.save_all_eofs_to_npzfile(...)
    :param filename: The filename
    :return: The data object
    """
    with np.load(filename) as data:
        eof1 = data["eof1"]
        eof2 = data["eof2"]
        lat = data["lat"]
        long = data["long"]
        eigenvalues = data["eigenvalues"]
        explained_variances = data["explained_variances"]
        no_observations=data["no_observations"]
    eofs = []
    for i in range(0, doy_list().size):
        eof = EOFData(lat, long, np.squeeze(eof1[i, :]), np.squeeze(eof2[i, :]),
                      eigenvalues=np.squeeze(eigenvalues[i, :]),
                      explained_variances=np.squeeze(explained_variances[i, :]),
                      no_observations=no_observations[i])
        eofs.append(eof)
    return EOFDataForAllDOYs(eofs)


def doy_list() -> np.array:
    """
    Returns an array of all DOYs in a year, hence simply the numbers from 1 to 366.
    Useful for, e.g., as axis for plotting

    :return: The doy array
    """
    return np.arange(1, 367, 1)


def plot_correlation_with_original_eofs(recalc_eof: EOFDataForAllDOYs, orig_eof: EOFDataForAllDOYs) -> Figure:
    """
    Creates a diagnosis plot showing the correlations for all DOYs of between the original EOFs and newly
    calculated EOF for both, EOF1 and EOF2
    :param recalc_eof: The object containing the calculated EOFs
    :param orig_eof: The object containing the ortiginal EOFs
    :return: Handle to the figure
    """
    doys = doy_list()
    corr1 = np.zeros(doys.size)
    corr2 = np.zeros(doys.size)
    for idx, doy in enumerate(doys):
        corr1[idx] = \
            (np.corrcoef(orig_eof.eofdata_for_doy(doy).eof1vector, recalc_eof.eofdata_for_doy(doy).eof1vector))[0, 1]
        corr2[idx] = \
            (np.corrcoef(orig_eof.eofdata_for_doy(doy).eof2vector, recalc_eof.eofdata_for_doy(doy).eof2vector))[0, 1]
    fig = plt.figure("plot_correlation_with_original_eofs", clear=True, figsize=(6, 4), dpi=150)
    plt.ylim([0, 1.05])
    plt.xlabel("DOY")
    plt.ylabel("Correlation")
    plt.title("Correlation Original-Recalculated EOF")
    p1, = plt.plot(doys, corr1, label="EOF1")
    p2, = plt.plot(doys, corr2, label="EOF2")
    plt.legend(handles=(p1, p2))
    return fig


def plot_explained_variance_for_all_doys(eofs: EOFDataForAllDOYs, include_total_variance:bool = False,
                                         include_no_observations: bool = False) -> Figure:
    """
    Plots the explained variance values for EOF1 and EOF2 for all doys.
    Comparable to Kiladis (2014), Fig. 1
    :param eofs: The EOF data to plot
    :return: Handle to the figure,.
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
    Plots the Eigenvalues for EOF1 and EOF2 for all doys.
    :param eofs: The EOF data to plot
    :return: Handle to the figure,.
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
    eofdata = load_original_eofs_for_doy(path, doy)
    return plot_individual_eof_map(eofdata, doy=doy)


def plot_individual_eof_map_from_file(filename, doy: int) -> Figure:
    eofdata = load_single_eofs_from_txt_file(filename, doy=doy)
    return plot_individual_eof_map(eofdata)


def plot_individual_eof_map(eofdata: EOFData, doy: int = None) -> Figure:
    # TODO: Plot underlying map
    fig, axs = plt.subplots(2, 1, num="plotting.plot_eof_for_doy", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0]

    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF1")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1]
    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    return fig


def plot_individual_eof_map_comparison(orig_eof: EOFData, compare_eof: EOFData, doy: int=None):

    # TODO: Print correlation values into figure
    print(np.corrcoef(orig_eof.eof1vector, compare_eof.eof1vector))
    print(np.corrcoef(orig_eof.eof2vector, compare_eof.eof2vector))

    fig, axs = plt.subplots(2, 3, num="ReproduceOriginalOMIPCs_ExplainedVariance_EOF_Comparison", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF1")
    ax.set_ylabel("Latitude [°]")

    ax = axs[0, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF1")

    # FIXME: Check that grids are equal
    ax = axs[0, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map - compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 1")

    ax = axs[1, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF2")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map - compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 2")
    ax.set_xlabel("Longitude [°]")

    return fig


def plot_individual_explained_variance_all_eofs(eof:EOFData, doy: int = None, max_eof_number: int = None) -> Figure:
    """
    Plots the explained variances for each EOF function, but only for EOF data of one DOY.
    This is useful to confirm that the first 2 EOFs covers actually most of the variance.
    :param eof: The EOFData object
    :param doy: The corresponding DOY. Only used to display it in the title.
    :param max_eof_number: The limit of the x-axis.
    :return: Handle to the figure
    """
    eof_number_grid = np.arange(0, eof.explained_variances.size , 1) + 1
    fig = plt.figure("plot_individual_explained_variance_all_eofs", clear=True, figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(111)
    p1 = plt.plot(eof_number_grid, eof.explained_variances, color="blue", label="Explained variance" )
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


