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

import numpy as np

import mjoindices.tools as tools


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

    errors = []

    # Non leap years
    dates = np.arange("2018-01-01", "2019-12-31", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 50, 20)
    if not target_inds.size == (20*2 + 1) * 2:
        errors.append("Length of DOY range covering the middle of the year is wrong")
    control_inds = np.concatenate((np.arange(30, 71, 1)-1, np.arange(30, 71, 1) + 365 - 1))
    if not np.all(target_inds == control_inds):
        errors.append("Indices of DOY range covering the middle of the year are wrong")
    control_doys = np.concatenate((np.arange(30, 71, 1), np.arange(30, 71, 1)))
    if not np.all(target_doys == control_doys):
        errors.append("DOY values of DOY range covering the middle of the year are wrong")

    dates = np.arange("2018-06-01", "2019-06-30", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 10, 20)
    if not target_inds.size == 20*2 + 1:
        errors.append("Length of DOY range covering the ending of the previous year is wrong")
    control = np.concatenate((np.arange(355, 366, 1), np.arange(1, 31, 1)))
    if not np.all(target_doys == control):
        errors.append("DOY range covering the ending of the previous year is wrong")

    dates = np.arange("2018-06-01", "2019-06-30", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 350, 25)
    if not target_inds.size == 25 * 2 + 1:
        errors.append("Length of DOY range covering the beginning of the next year is wrong")
    control = np.concatenate((np.arange(325, 366, 1), np.arange(1, 11, 1)))
    if not np.all(target_doys == control):
        errors.append("DOY range covering the the beginning of the next year is wrong")

    # Leap years
    dates = np.arange("2016-01-01", "2017-12-31", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 50, 20)
    if not target_inds.size == (20 * 2 + 1) * 2:
        errors.append("Length of DOY range covering the middle of the year is wrong")
    control_inds = np.concatenate((np.arange(30, 71, 1) - 1, np.arange(30, 71, 1) + 366 - 1))
    if not np.all(target_inds == control_inds):
        errors.append("Indices of DOY range covering the middle of the year are wrong")
    control_doys = np.concatenate((np.arange(30, 71, 1), np.arange(30, 71, 1)))
    if not np.all(target_doys == control_doys):
        errors.append("DOY values of DOY range covering the middle of the year are wrong (Leap year test)")

    dates = np.arange("2016-06-01", "2017-06-30", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 10, 20)
    if not target_inds.size == 20 * 2 + 1:
        errors.append("Length of DOY range covering the ending of the previous year is wrong")
    control = np.concatenate((np.arange(356, 367, 1), np.arange(1, 31, 1)))
    if not np.all(target_doys == control):
        errors.append("DOY range covering the ending of the previous year is wrong (Leap year test)")

    dates = np.arange("2016-06-01", "2017-06-30", dtype='datetime64[D]')
    target_inds, target_doys = tools.find_doy_ranges_in_dates(dates, 350, 25)
    if not target_inds.size == 25 * 2 + 1:
        errors.append("Length of DOY range covering the beginning of the next year is wrong")
    control = np.concatenate((np.arange(325, 367, 1), np.arange(1, 10, 1)))
    if not np.all(target_doys == control):
        errors.append("DOY range covering the the beginning of the next year is wrong (Leap year test)")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))