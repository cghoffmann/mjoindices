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