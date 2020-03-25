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
This module provides basic functionality to handle OLR data, which is the basic input for the OMI calculation.
"""

from pathlib import Path

import numpy as np
import scipy
import scipy.interpolate
from matplotlib.figure import Figure
from scipy.io import netcdf
import matplotlib.pyplot as plt

import mjoindices.tools as tools


class OLRData:
    """
    This class serves as a container for spatially distributed and temporally resolved OLR data.

    A filled object of this class has to be provided by the user in order to start the OMI calculation.

    :param olr: The OLR data as a 3-dim array. The three dimensions correspond to time, latitude, and longitude, in this
        order.
    :param time: The temporal grid as 1-dim array of :class:`numpy.datetime64` dates.
    :param lat: The latitude grid as 1-dim array.
    :param long: The longitude grid as 1-dim array.
    """

    def __init__(self, olr: np.ndarray, time: np.ndarray, lat: np.ndarray, long: np.ndarray) -> None:
        """
        Initialization of basic variables.
        """
        if olr.shape[0] != time.size:
            raise ValueError('Length of time grid does not fit to first dimension of OLR data cube')
        if olr.shape[1] != lat.size:
            raise ValueError('Length of lat grid does not fit to second dimension of OLR data cube')
        if olr.shape[2] != long.size:
            raise ValueError('Length of long grid does not fit to third dimension of OLR data cube')
        self._olr = olr.copy()
        self._time = time.copy()
        self._lat = lat.copy()
        self._long = long.copy()

    @property
    def olr(self):
        """
        The OLR data as a 3-dim array. The three dimensions correspond to time, latitude, and longitude, in this
        order.
        """
        return self._olr

    @property
    def time(self):
        """
        The temporal grid as 1-dim array of :class:`numpy.datetime64` dates.
        """
        return self._time

    @property
    def lat(self):
        """
        The latitude grid as 1-dim array.
        """
        return self._lat

    @property
    def long(self):
        """
        The longitude grid as 1-dim array.
        """
        return self._long

    def __eq__(self, other: "OLRData") -> bool:
        """
        Override the default Equals behavior
        """
        return (np.all(self.lat == other.lat)
                and np.all(self.long == other.long)
                and np.all(self.time == other.time)
                and np.all(self.olr == other.olr))

    def close(self, other: "OLRData") -> bool:
        """
         Checks equality of two :class:`OLRData` objects, but allows for numerical tolerances.

        :param other: The object to compare with.

        :return: Equality of all members considering the default tolerances of :func:`numpy.allclose`
        """
        return (np.allclose(self.lat, other.lat)
                and np.allclose(self.long, other.long)
                and np.allclose(self.time.astype("float"), other.time.astype("float"))  # allclose does not work with datetime64
                and np.allclose(self.olr, other.olr))

    def get_olr_for_date(self, date: np.datetime64) -> np.ndarray:
        """
        Returns the spatially distributed OLR map for a particular date.

        :param date: The date, which hat to be exactly matched by one of the dates in the OLR time grid.

        :return: The excerpt of the OLR data as a 2-dim array. The two dimensions correspond to
            latitude, and longitude, in this order. Returns None if the date is not contained in the OLR time series.
        """
        cand = self.time == date
        if not np.all(cand == False):  # noqa: E712
            return np.squeeze(self.olr[cand, :, :])
        else:
            return None

    def extract_olr_matrix_for_doy_range(self, center_doy: int, window_length: int = 0,
                                         strict_leap_year_treatment: bool = False) -> np.ndarray:
        """
        Extracts the OLR data, which belongs to all DOYs around one center (center_doy +/- windowlength).

        Keep in mind that the OLR time series might span several years. In this case the center DOY is found more than
        once and the respective window in considered for each year.
        Example: 3 full years of data, centerdoy = 20, and window_length = 4 results in 3*(2*4+1) = 27 entries in the
        time axis

        :param center_doy: The center DOY of the window.
        :param window_length: The window length in DOYs on both sides of the center DOY. Hence, if the window is fully
            covered by the data, one gets 2*window_length + 1 entries per year in the result.
        :param strict_leap_year_treatment: see description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.

        :return: The excerpt of the OLR data as a 3-dim array. The three dimensions correspond to
            time, latitude, and longitude, in this order.
        """
        inds, doys = tools.find_doy_ranges_in_dates(self.time, center_doy, window_length=window_length,
                                                    strict_leap_year_treatment=strict_leap_year_treatment)
        return self.olr[inds, :, :]

    def save_to_npzfile(self, filename: Path) -> None:
        """
        Saves the data arrays contained in the OLRData object to a numpy file.

        :param filename: The full filename.
        """
        np.savez(filename, olr=self.olr, time=self.time, lat=self.lat, long=self.long)


def interpolate_spatial_grid_to_original(olr: OLRData) -> OLRData:
    """
    Interpolates the OLR data in an :class:`OLRData` object spatially onto the spatial grid, which was used for the
    original OMI calculation by Kiladis (2014).

    This original grid has the following properties:

    * Latitude: 2.5 deg-sampling in the tropics from -20 to 20 deg (20 S to 20 N).
    * Longitude: Whole globe with 2.5 deg-sampling.

    :param olr: The OLR data

    :return:  A new :class:`OLRData` object with the interpolated data.
    """
    # FIXME Combine with definition in empirical_or....py
    orig_lat = np.arange(-20., 20.1, 2.5)
    orig_long = np.arange(0., 359.9, 2.5)
    return interpolate_spatial_grid(olr, orig_lat, orig_long)


def interpolate_spatial_grid(olr: OLRData, target_lat: np.ndarray, target_long: np.ndarray) -> OLRData:
    """
    Interpolates the OLR data linearly onto the given grids.

    No extrapolation will be done. Instead a :py:class:`ValueError` is raised if the data does not cover the target
    grid.

    Note that no sophisticated resampling is provided here. So, if some kind of averaging, etc., is needed, it should
    be performed by the user himself before injecting the data into the OMI calculation.

    :param olr: The OLR data to resample.
    :param target_lat: The new latitude grid.
    :param target_long: The new longitude grid.

    :return: A new :class:`OLRData` object containing the resampled OLR data.
    """
    no_days = olr.time.size
    olr_interpol = np.empty((no_days, target_lat.size, target_long.size))
    for idx in range(0, no_days):
        f = scipy.interpolate.interp2d(olr.long, olr.lat, np.squeeze(olr.olr[idx, :, :]), kind='linear', bounds_error=True)
        olr_interpol[idx, :, :] = f(target_long, target_lat)
    return OLRData(olr_interpol, olr.time, target_lat, target_long)


def restrict_time_coverage(olr: OLRData, start: np.datetime64, stop: np.datetime64) -> OLRData:
    """
    Cuts the OLR time series at the given dates (given dates are included).

    This is useful when the OLR data should be restricted for the EOF calculation, which is based on a subset.
    Of course, it can also be used to limit the PC calculation to a specific period.

    Note that a temporal resampling method is not provided here, since the possible resampling methods,
    which the user might want to apply are too diverse. Hence, it is assumed that the temporal spacing ist already
    correct (daily averages recommended) and only a restriction of the period is needed before calculation.

    :param olr: The OLR data to restrict.
    :param start: The beginning of the wanted period of OLR data (included).
    :param stop: The ending of the wanted period (included).

    :return: A new :class:`OLRData` object with restricted temporal coverage.

    :raises: :py:class:`ValueError` if no OLR Data is found for the specified period
    """
    window_inds = (olr.time >= start) & (olr.time <= stop)
    if np.all(window_inds == False):  # noqa: E712
        raise ValueError("No OLR data within specified period found. Data covers the period from %s to %s."
                         % (str(olr.time[0]), str(olr.time[-1])))
    else:
        return OLRData(olr.olr[window_inds, :, :], olr.time[window_inds], olr.lat, olr.long)


def load_noaa_interpolated_olr(filename: Path) -> OLRData:
    """
    Loads the standard OLR data product provided by NOAA.

    The dataset can be obtained from
    ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc

    A description is found at
    https://www.esrl.noaa.gov/psd/data/gridded/data.interp_OLR.html

    :param filename: Full filename of a local copy of OLR data file.

    :return: The OLR data.
    """
    f = netcdf.netcdf_file(str(filename), 'r')
    lat = f.variables['lat'].data.copy()
    lon = f.variables['lon'].data.copy()
    # scaling and offset as given in meta data of nc file
    olr = f.variables['olr'].data.copy() / 100. + 327.65
    hours_since1800 = f.variables['time'].data.copy()
    f.close()

    temptime = []
    for item in hours_since1800:
        delta = np.timedelta64(int(item / 24), 'D')
        day = np.datetime64('1800-01-01') + delta
        temptime.append(day)
    time = np.array(temptime, dtype=np.datetime64)
    result = OLRData(np.squeeze(olr), time, lat, lon)

    return result


def restore_from_npzfile(filename: Path) -> OLRData:
    """
    Loads an :class:`OLRData` object from a numpy file, which has been saved with the function
    :func:`mjoindices.olr_handling.OLRData.save_to_npzfile`

    :param filename: The filename to the .npz file.

    :return: The OLR data.
    """
    with np.load(filename) as data:
        olr = data["olr"]
        time = data["time"]
        lat = data["lat"]
        long = data["long"]
    return OLRData(olr, time, lat, long)


def plot_olr_map_for_date(olr: OLRData, date: np.datetime64) -> Figure:
    """
    Plots a map pf the OLR data for a specific date.

    :param olr: The complete OLR data.
    :param date: The date for which da OLR data should be plotted
        (has to be exactly matched by a date of the OLR time grid).

    :return: The handle to the figure.
    """
    # TODO: Plot underlying map

    mapdata = olr.get_olr_for_date(date)

    if mapdata is not None:
        fig, axs = plt.subplots(1, 1, num="plot_olr_map_for_date", clear=True,
                                figsize=(10, 5), dpi=150, sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        ax = axs

        c = ax.contourf(olr.long, olr.lat, mapdata)
        fig.colorbar(c, ax=ax, label="OLR [W/m²]")
        ax.set_title("OLR")
        ax.set_ylabel("Latitude [°]")
        ax.set_xlabel("Longitude [°]")
    else:
        raise ValueError("No OLR data found for given date.")

    return fig
