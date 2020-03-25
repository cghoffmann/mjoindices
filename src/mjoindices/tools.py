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
This module provides basic helper routines for the OMI calculation
"""

import datetime as dt
import typing
import numpy as np


def calc_day_of_year(date: typing.Union[np.datetime64, np.ndarray]) -> typing.Union[int, np.ndarray]:
    """
    Calculates the days of the year (DOYs) for an individual date or an array of dates.

    :param date: The date (or the dates), given as (NumPy array of) :class:`numpy.datetime64` value(s).

    :return: the DOY (or the DOYs) as (NumPy array of) int value(s).
    """
    if np.isscalar(date):
        if date.dtype == "<M8[ns]":
            # work around a bug, which prevents datetime64[ns] from being converted correctly to dt.datetime.
            date = date.astype("datetime64[us]")
        temp = date.astype(dt.datetime)
        time_fragments = temp.timetuple()
        result = time_fragments[7]
    else:
        result = np.empty(date.size)
        for i, d in enumerate(date):
            result[i] = calc_day_of_year(d)
    return result


def find_doy_ranges_in_dates(dates: np.ndarray, center_doy: int, window_length: int,
                             strict_leap_year_treatment: bool=False) -> typing.Tuple:
    """
    Finds the indices in a given array of dates that fit into a particular window of DOYs.

    This task sounds trivial, but is a little bit complicated by the appearance of leap years.

    :param dates: The array of dates as :class:`numpy.datetime64` values.
    :param center_doy: The center of the wanted window.
    :param window_length: the length of the window to both sides  of the center in days.
        The window spans 2*window_length+1 days in total (for exceptions see below).
    :param strict_leap_year_treatmenmt: distinguishes between 2 different methods of constructing the DOY range window.
        1: Setting the switch to False will use a pragmatic implementation in which the start and end of the DOY window
        is directly computed as distance in units of DOYs. 2: Setting the switch to True will transfer the DOYs to
        actual calender dates and will calculate the start and end of the window also as calender dates using
        built-in numpy datetime functions. In the context of the EOF calculation, the setting has major implications
        only for the EOFs calculated for DOY 366 and causes only minor differences for the other DOYs. The results for
        the setting False are closer to the original values, and approximately the same total number of calender days
        covered by the window is found for each center DOY including 366. However, the length of the window is not
        guaranteed to be 2*window_length+1, but can also be 2*window_length+2 if the window crosses the ending of a leap
        year. The setting True is somewhat more stringently implemented. The window length is always 2*window_length+1,
        however, the number of calender days covered by a window is reduced by approximately a factor of 4 for
        center_doy=366, since a window is found only during leap years at all, which might cause the EOF to differ quite
        a lot from those of DOY 365 and DOY 1, which is not wanted. Because of that the recommended setting
        is the default value False.

    :return: Tuple with, first, the array of indices and, second, the resulting DOYs for comparison.
    """
    doys = calc_day_of_year(dates)

    if strict_leap_year_treatment:
        center_inds = np.nonzero(doys == center_doy)

        # switch from DOYs to real dates to use built-in leap year functionality
        startdates = dates[center_inds] - np.timedelta64(window_length, 'D')
        too_early_inds = np.nonzero(startdates < dates[0])
        startdates[too_early_inds] = dates[0]

        enddates = dates[center_inds] + np.timedelta64(window_length, 'D')
        too_late_inds = np.nonzero(enddates > dates[-1])
        enddates[too_late_inds] = dates[-1]

        resulting_idxlist = np.array([], dtype="int")
        for ind, startdate in enumerate(startdates):
            one_window_indices = np.nonzero((dates >= startdates[ind]) & (dates <= enddates[ind]))[0]
            resulting_idxlist = np.concatenate((resulting_idxlist, one_window_indices))
    else:
        lower_limit = center_doy - window_length
        if lower_limit < 1:
            lower_limit = lower_limit + 365
        upper_limit = center_doy + window_length
        if upper_limit > 365:
            upper_limit = upper_limit - 365

        if lower_limit <= upper_limit:
            inds_consider = ((doys >= lower_limit) & (doys <= upper_limit))
        else:
            inds_consider = ((doys >= lower_limit) | (doys <= upper_limit))
        resulting_idxlist = np.nonzero(inds_consider)[0]

    return np.asarray(resulting_idxlist), doys[resulting_idxlist]


def doy_list() -> np.array:
    """
    Returns an array of all DOYs in a year, hence simply the numbers from 1 to 366.
    Useful, e.g., as axis for plotting.

    :return: The doy array.
    """
    return np.arange(1, 367, 1)
