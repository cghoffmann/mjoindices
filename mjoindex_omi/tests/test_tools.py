# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""

import numpy as np

import mjoindex_omi.tools as tools


def test_calc_day_of_year_scalar():

    errors = []
    date = np.datetime64("2019-01-01")
    target = tools.calc_day_of_year(date)
    if not target == 1:
        errors.append("Error in DOY calc for %s" % str(date))

    date = np.datetime64("2019-01-11")
    target = tools.calc_day_of_year(date)
    if not target == 11:
        errors.append("Error in DOY calc for %s" % str(date))

    date = np.datetime64("2019-02-28")
    target = tools.calc_day_of_year(date)
    if not target == 31+28:
        errors.append("Error in DOY calc for %s" % str(date))

    date = np.datetime64("2019-12-31")
    target = tools.calc_day_of_year(date)
    if not target == 365:
        errors.append("Error in DOY calc for %s" % str(date))

    #test leap year
    date = np.datetime64("2020-02-29")
    target = tools.calc_day_of_year(date)
    if not target == 31 + 29:
        errors.append("Error in DOY calc for %s" % str(date))

        # test leap year
    date = np.datetime64("2020-12-31")
    target = tools.calc_day_of_year(date)
    if not target == 366:
        errors.append("Error in DOY calc for %s" % str(date))

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_calc_day_of_year_array():

    errors = []
    dates = np.array([np.datetime64("2019-01-01"),np.datetime64("2019-01-02"),np.datetime64("2019-01-03")])
    target = tools.calc_day_of_year(dates)
    if not np.all(target == np.array([1, 2, 3])):
        errors.append("Error in DOY calc for array")

    dates = np.array([np.datetime64("2020-12-30"), np.datetime64("2020-12-31"), np.datetime64("2021-01-01"),
                      np.datetime64("2021-01-02")])
    target = tools.calc_day_of_year(dates)
    if not np.all(target == np.array([365, 366, 1, 2])):
        errors.append("Error in DOY calc for array")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_find_doy_ranges_in_dates():
    dates = np.arange("2018-01-01", "2019-12-31", dtype='datetime64[D]')

    errors = []

    target = tools.find_doy_ranges_in_dates(dates, 50, 20)
    control = np.concatenate((np.arange(30, 71, 1)-1, np.arange(30, 71, 1) + 365-1))
    if not np.all(target == control):
        errors.append("DOY range in the middle of the year is wrong")

    dates = np.arange("2018-06-01", "2019-06-30", dtype='datetime64[D]')
    doys = tools.calc_day_of_year(dates)
    target = doys[tools.find_doy_ranges_in_dates(dates, 10, 20)]
    if not np.all(target == control):
        errors.append("DOY range ranging in the ending of the previous year is wrong")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

test_find_doy_ranges_in_dates()