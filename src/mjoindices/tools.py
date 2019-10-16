# -*- coding: utf-8 -*-

""" """

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

import datetime as dt
import typing
import numpy as np

def calc_day_of_year(date: typing.Union[np.datetime64, np.ndarray]) -> typing.Union[int, np.ndarray]:
    """Calculates the days of the year (DOYs) for an individual date or an array of dates


    :param date: The date or the dates, given as numpy.datetime64 values.
    :return: the DOY or the DOYs as int.
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
                             strict_leap_year_treatment: bool=True) -> typing.Tuple:
    """
    Finds the indices in a given array of dates that fit into a particular window of DOYs (days in the year).
    This task sounds trivial, but is a little bit complicated by the appearance of leap years.

    :param dates: The array of dates
    :param center_doy: the center of the wanted window
    :param window_length: the length of the window to both sides in days. The window spans 2*window_length+1 days in
     total.
    :param strict_leap_year_treatmenmt:
    :return: Tuple with, first, the array of indices and, second, the resulting DOYs for comparison.
    """
    #ToDO: Add description of param strict_leap_year_treatmenmt...
    #ToDo: Add unit test for strict_leap_year_treatmenmt=FALSE
    doys = calc_day_of_year(dates)

    if strict_leap_year_treatment:
        center_inds = np.nonzero(doys == center_doy)

        # switch from DOYs to real dates to use built-in leap year functionality
        startdates = dates[center_inds] - np.timedelta64(window_length,'D')
        too_early_inds = np.nonzero(startdates < dates[0])
        startdates[too_early_inds] = dates[0]

        enddates=dates[center_inds] + np.timedelta64(window_length,'D')
        too_late_inds = np.nonzero(enddates > dates[-1])
        enddates[too_late_inds] = dates[-1]

        resulting_idxlist = np.array([],dtype="int")
        for ind,startdate in enumerate(startdates):
            one_window_indices = np.nonzero((dates >= startdates[ind]) & (dates <= enddates[ind]))[0]
            resulting_idxlist = np.concatenate((resulting_idxlist, one_window_indices))
    else:
        lower_limit = center_doy - window_length
        if lower_limit < 1:
            lower_limit = lower_limit + 366
        upper_limit = center_doy + window_length
        if upper_limit > 366:
            upper_limit = upper_limit - 366

        if lower_limit <= upper_limit:
            inds_consider = ((doys >= lower_limit) & (doys <= upper_limit))
        else:
            inds_consider = ((doys >= lower_limit) | (doys <= upper_limit))
        resulting_idxlist = np.nonzero(inds_consider)[0]

    return np.asarray(resulting_idxlist), doys[resulting_idxlist]
