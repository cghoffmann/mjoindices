# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:55:00 2019

@author: ch
"""

import datetime as dt
import typing

import numpy as np


def calc_day_of_year(date: typing.Union[np.datetime64, np.ndarray]) -> typing.Union[int, np.ndarray]:
    """Calculates the days of the year (DOYs) for an individual date or an array of dates


    :param date: The date or the dates, given as numpy.datetime64 values.
    :return: the DOY or the DOYs as int.
    """
    if np.isscalar(date):
        temp = date.astype(dt.datetime)
        time_fragments = temp.timetuple()
        result = time_fragments[7]
    else:
        result = np.empty(date.size)
        for i, d in enumerate(date):
            result[i] = calc_day_of_year(d)
    return result


def find_doy_ranges_in_dates(dates: np.ndarray, center_doy: int, window_length: int) -> typing.Tuple:
    """
    Finds the indices in a given array of dates that fit into a particular window of DOYs (days in the year).
    This task sounds trivial, but is a little bit complicated by the appearance of leap years.

    :param dates: The array of dates
    :param center_doy: the center of the wanted window
    :param window_length: the length of the window to both sides in days. The window spans 2*window_length+1 days in
     total.
    :return: Tuple with, first, the array of indices and, second, the resulting DOYs for comparison.
    """
    doys = calc_day_of_year(dates)
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
    return np.asarray(resulting_idxlist), doys[resulting_idxlist]
