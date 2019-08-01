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
