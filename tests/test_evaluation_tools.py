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
import numpy.random
import pytest

import mjoindices.evaluation_tools as evalt
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.tools as tools


def test_compute_vector_difference_quantity():
    n = 1000
    signal = np.ones(n)
    numpy.random.seed(1000)
    noise = numpy.random.randn(n)
    data = signal + noise

    errors = []

    target = evalt.compute_vector_difference_quantity(signal, data, percentage=False)
    if not np.allclose(target, noise):
        errors.append("Absolute differences incorrect.")

    target = evalt.compute_vector_difference_quantity(signal, data, percentage=True)
    if not np.allclose(target, noise * 100):
        errors.append("Absolute differences incorrect.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_calc_vector_agreement():

    n = 100000
    signal = np.ones(n)

    errors = []

    # start with a normal distribution and absolute deviations.
    numpy.random.seed(1000)
    noise = numpy.random.randn(n)
    data = signal + noise

    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_vector_agreement(signal, data, percentage=False, do_print=False)

    if not np.allclose(diff_vec, noise):
        errors.append("Vector of differences not correct (absolute values).")

    if not np.isclose(diff_mean, -0.0035294031521152136):
        errors.append("Mean value of differences not correct (absolute values).")

    if not np.isclose(diff_std, 1.0031438836641624):
        errors.append("Mean value of differences not correct (absolute values).")

    if not np.isclose(diff_abs_percent68, 0.9990298365839766):  # approximately like stddev
        errors.append("68% percentile not correct (absolute values).")

    if not np.isclose(diff_abs_percent95, 1.961966296714916):  # approximately like 2*stddev
        errors.append("95% percentile not correct (absolute values).")

    if not np.isclose(diff_abs_percent99, 2.5952045271622053):  # smaller than 3*stddev, since 3*stddev corresponds to 99.9%
        errors.append("99% percentile not correct (absolute values).")

    # now deviations normalized with mean of reference (which is exactly 1) and multiplied by 100.
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 \
        = evalt.calc_vector_agreement(signal, data, percentage=True, do_print=False)

    if not np.allclose(diff_vec, noise * 100):
        errors.append("Vector of differences not correct (relative values).")

    if not np.isclose(diff_mean, -0.3529403152115213):
        errors.append("Mean value of differences not correct (relative values).")

    if not np.isclose(diff_std, 100.31438836641625):
        errors.append("Mean value of differences not correct (relative values).")

    if not np.isclose(diff_abs_percent68, 99.90298365839766):  # approximately like stddev
        errors.append("68% percentile not correct (relative values).")

    if not np.isclose(diff_abs_percent95, 196.19662967149162):  # approximately like 2*stddev
        errors.append("95% percentile not correct (relative values).")

    if not np.isclose(diff_abs_percent99, 259.5204527162205):  # smaller than 3*stddev, since 3*stddev corresponds to 99.9%
        errors.append("99% percentile not correct (relative values).")

    numpy.random.seed(1000)
    noise = numpy.random.rand(n)
    data = signal + noise

    # now using a uniform distribution and absolute deviations
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_vector_agreement(
        signal, data, percentage=False, do_print=False)

    errors = []

    if not np.allclose(diff_vec, noise):
        errors.append("Vector of differences not correct (uniform distribution).")

    if not np.isclose(diff_mean, 0.500672176427001):
        errors.append("Mean value of differences not correct (uniform distribution).")

    if not np.isclose(diff_std, 0.2889275424927285):  # std dev of standard uniform distribution = sqrt(1/12*(1-0)) = sqrt(1/12)
        errors.append("Mean value of differences not correct (uniform distribution).")

    if not np.isclose(diff_abs_percent68, 0.681562679012755):  # approximately like stddev
        errors.append("68% percentile not correct (uniform distribution).")

    if not np.isclose(diff_abs_percent95, 0.9510118583576715):  # approximately like 2*stddev
        errors.append("95% percentile not correct (uniform distribution).")

    if not np.isclose(diff_abs_percent99, 0.9903721351476673):  # smaller than 3*stddev, since 3*stddev corresponds to 99.9%
        errors.append("99% percentile not correct (uniform distribution).")

    n = 1000
    signal = np.arange(n)
    data = signal
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = \
        evalt.calc_vector_agreement(signal, data, percentage=False, do_print=False)
    if not corr == 1:
        errors.append("Correlation incorrect.")

    # Check if vector length test
    n = 1000
    signal = np.ones(n)
    numpy.random.seed(1000)
    noise = numpy.random.randn(n)
    data = signal[:-2] + noise[:-2]

    with pytest.raises(ValueError):
        corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_vector_agreement(
            signal, data, percentage=False, do_print=False)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_calc_comparison_stats_for_eofs_all_doys():
    no_leap_years = False # using all 366 days

    doys = tools.doy_list(no_leap_years)
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])

    errors = []

    eofs_reference = []
    eofs_data = []
    for doy in doys:
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs_reference.append(eof.EOFData(lat, long, eof1, eof2))
        if doy == 3:
            eof1 = -1 * eof1
        if doy == 4:
            eof2 = -1 * eof2
        eofs_data.append(eof.EOFData(lat, long, eof1, eof2))
    reference = eof.EOFDataForAllDOYs(eofs_reference, no_leap_years)
    data = eof.EOFDataForAllDOYs(eofs_data, no_leap_years)

    corr, diff_mean, diff_std, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = \
        evalt.calc_comparison_stats_for_eofs_all_doys(reference, data, 1, exclude_doy366=False, percentage=False,
                                                      do_print=False)

    if not (np.allclose(corr[:2], 1) and np.allclose(corr[3:], 1)):
        errors.append("EOF1: Correlations wrong")
    if not np.isclose(corr[2], -1.):
        errors.append("EOF1: Correlation for DOY 3 wrong")
    if not (np.allclose(diff_mean[:2], 0) and np.allclose(diff_mean[3:], 0)):
        errors.append("EOF1: Mean wrong")
    if not diff_mean[2] < 0:
        errors.append("EOF1: Mean for DOY 3 wrong")
    if not (np.allclose(diff_std[:2], 0) and np.allclose(diff_std[3:], 0)):
        errors.append("EOF1: StdDev wrong")
    if not diff_std[2] > 0:
        errors.append("EOF1: StdDev for DOY 3 wrong")
    if not (np.allclose(diff_abs_percent68[:2], 0) and np.allclose(diff_abs_percent68[3:], 0)):
        errors.append("EOF1: 68% Percentile wrong")
    if not diff_abs_percent68[2] > 0:
        errors.append("EOF1: 68% Percentile for DOY 3 wrong")
    if not (np.allclose(diff_abs_percent95[:2], 0) and np.allclose(diff_abs_percent95[3:], 0)):
        errors.append("EOF1: 95% Percentile wrong")
    if not diff_abs_percent95[2] > 0:
        errors.append("EOF1: 95% Percentile for DOY 3 wrong")
    if not (np.allclose(diff_abs_percent99[:2], 0) and np.allclose(diff_abs_percent99[3:], 0)):
        errors.append("EOF1: 99% Percentile wrong")
    if not diff_abs_percent99[2] > 0:
        errors.append("EOF1: 99% Percentile for DOY 3 wrong")

    corr, diff_mean, diff_std, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_comparison_stats_for_eofs_all_doys(
        reference, data,
        2, exclude_doy366=False, percentage=False,
        do_print=False)

    if not (np.allclose(corr[:3], 1) and np.allclose(corr[4:], 1)):
        errors.append("EOF2: Correlations wrong")
    if not np.isclose(corr[3], -1.):
        errors.append("EOF2: Correlation for DOY 4 wrong")
    if not (np.allclose(diff_mean[:3], 0) and np.allclose(diff_mean[4:], 0)):
        errors.append("EOF2: Mean wrong")
    if not diff_mean[3] < 0:
        errors.append("EOF2: Mean for DOY 4 wrong")
    if not (np.allclose(diff_std[:3], 0) and np.allclose(diff_std[4:], 0)):
        errors.append("EOF2: StdDev wrong")
    if not diff_std[3] > 0:
        errors.append("EOF2: StdDev for DOY 4 wrong")
    if not (np.allclose(diff_abs_percent68[:3], 0) and np.allclose(diff_abs_percent68[4:], 0)):
        errors.append("EOF2: 68% Percentile wrong")
    if not diff_abs_percent68[3] > 0:
        errors.append("EOF2: 68% Percentile for DOY 4 wrong")
    if not (np.allclose(diff_abs_percent95[:3], 0) and np.allclose(diff_abs_percent95[4:], 0)):
        errors.append("EOF2: 95% Percentile wrong")
    if not diff_abs_percent95[3] > 0:
        errors.append("EOF2: 95% Percentile for DOY 4 wrong")
    if not (np.allclose(diff_abs_percent99[:3], 0) and np.allclose(diff_abs_percent99[4:], 0)):
        errors.append("EOF2: 99% Percentile wrong")
    if not diff_abs_percent99[3] > 0:
        errors.append("EOF2: 99% Percentile for DOY 4 wrong")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_calc_timeseries_agreement():

    signal_time = np.arange("2010-01-01", "2020-12-31", dtype='datetime64[D]')
    data_time = np.arange("2010-01-01", "2020-12-31", dtype='datetime64[D]')

    n = signal_time.size

    signal = np.ones(n)
    numpy.random.seed(1000)
    noise = numpy.random.randn(n)
    data = signal + noise

    inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_timeseries_agreement(signal, signal_time, data, data_time, exclude_doy366=False, do_print=False)

    errors = []

    if not np.allclose(diff_ts, noise):
        errors.append("Vector of differences not correct.")

    if not np.isclose(diff_mean, -0.016998885708167304):
        errors.append("Mean value of differences not correct.")

    if not np.isclose(diff_std, 1.011669296908716):  # std dev of standard uniform distribution = sqrt(1/12*(1-0)) = sqrt(1/12)
        errors.append("Mean value of differences not correct.")

    if not np.isclose(diff_abs_percent68, 1.0021767602961196):  # approximately like stddev
        errors.append("68% percentile not correct.")

    if not np.isclose(diff_abs_percent95, 1.9723863150018495):  # approximately like 2*stddev
        errors.append("95% percentile not correct.")

    if not np.isclose(diff_abs_percent99, 2.6244012732344646):  # smaller than 3*stddev, since 3*stddev corresponds to 99.9%
        errors.append("99% percentile not correct.")

    noise = np.ones(n) * 2
    no_leap_years = False
    doys = tools.calc_day_of_year(signal_time, no_leap_years)
    doy366_inds = np.nonzero(doys == 366)
    noise[doy366_inds] = 1000000
    data = signal + noise

    inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_timeseries_agreement(
        signal, signal_time, data, data_time, exclude_doy366=True, do_print=False)

    if not np.all(inds_not_used[0] == doy366_inds[0]):
        errors.append("Wrong indices excluded.")
    if not diff_mean == 2.0:
        errors.append("Mean of difference is influenced by DOY 366 value.")

    inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_timeseries_agreement(
        signal, signal_time, data, data_time, exclude_doy366=False, do_print=False)

    if inds_not_used[0].size != 0:
        errors.append("Indices excluded, which should not be the case.")
    if not diff_mean > 2.0:
        errors.append("Mean of difference is not influenced by DOY 366 value, which should be the case.")

    signal_time = np.arange("2010-01-01", "2020-12-31", dtype='datetime64[D]')
    data_time = np.arange("2010-01-02", "2021-01-01", dtype='datetime64[D]')
    with pytest.raises(ValueError) as e:
        inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = evalt.calc_timeseries_agreement(
            signal, signal_time, data, data_time, exclude_doy366=False, do_print=False)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))
