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

import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.tools as tools



def calc_correlations_of_eofs_all_doys(eofs1: eof.EOFDataForAllDOYs, eofs2: eof.EOFDataForAllDOYs) -> typing.Tuple:
    doys = eof.doy_list()
    corr1 = np.empty(doys.size)
    corr2 = np.empty(doys.size)
    for idx, doy in enumerate(doys):
        corr1[idx] = \
            (np.corrcoef(eofs1.eofdata_for_doy(doy).eof1vector, eofs2.eofdata_for_doy(doy).eof1vector))[0, 1]
        corr2[idx] = \
            (np.corrcoef(eofs1.eofdata_for_doy(doy).eof2vector, eofs2.eofdata_for_doy(doy).eof2vector))[0, 1]
    return corr1, corr2



def calc_maxdifference_of_eofs_all_doys(eofs_ref: eof.EOFDataForAllDOYs, eofs: eof.EOFDataForAllDOYs) -> typing.Tuple:
    doys = eof.doy_list()
    maxdiff_abs_1 = np.empty(doys.size)
    maxdiff_abs_2 = np.empty(doys.size)
    maxdiff_rel_1 = np.empty(doys.size)
    maxdiff_rel_2 = np.empty(doys.size)
    for idx, doy in enumerate(doys):
        maxdiff_abs_1[idx] = \
            np.max(np.abs(eofs_ref.eofdata_for_doy(doy).eof1vector - eofs.eofdata_for_doy(doy).eof1vector))
        maxdiff_abs_2[idx] = \
            np.max(np.abs(eofs_ref.eofdata_for_doy(doy).eof2vector - eofs.eofdata_for_doy(doy).eof2vector))
        maxdiff_rel_1[idx] = maxdiff_abs_1[idx] / np.mean(np.abs(eofs_ref.eofdata_for_doy(doy).eof1vector))
        maxdiff_rel_2[idx] = maxdiff_abs_2[idx] / np.mean(np.abs(eofs_ref.eofdata_for_doy(doy).eof2vector))
    return maxdiff_abs_1, maxdiff_abs_2, maxdiff_rel_1, maxdiff_rel_2


def plot_maxdifference_of_eofs_all_doys(recalc_eof: eof.EOFDataForAllDOYs, orig_eof: eof.EOFDataForAllDOYs) -> Figure:
    """
    Creates a diagnosis plot showing the correlations for all DOYs of between the original EOFs and newly
    calculated EOF for both, EOF1 and EOF2
    :param recalc_eof: The object containing the calculated EOFs
    :param orig_eof: The object containing the ortiginal EOFs
    :return: Handle to the figure
    """
    doys = eof.doy_list()
    maxdiff_abs_1, maxdiff_abs_2, maxdiff_rel_1, maxdiff_rel_2 = calc_maxdifference_of_eofs_all_doys(orig_eof, recalc_eof)
    fig = plt.figure("plot_correlation_with_original_eofs", clear=True, figsize=(6, 4), dpi=150)
    plt.ylim([0, 1.05])
    plt.xlabel("DOY")
    plt.ylabel("Relative Difference")
    plt.title(" Relative absolute Difference Original-Recalculated EOF")
    p1, = plt.plot(doys, maxdiff_rel_1, label="EOF1")
    p2, = plt.plot(doys, maxdiff_rel_2, label="EOF2")
    plt.legend(handles=(p1, p2))
    return fig


def compute_vector_difference_quantity(ref_vec, vec, percentage=True):
    result = vec - ref_vec
    if percentage is True:
        result = result / np.mean(np.abs(ref_vec)) *100
    return result

def calc_vector_agreement(ref_vec, vec, percentage=True):
    corr = np.corrcoef(ref_vec, vec)[0, 1]

    diff_ts = compute_vector_difference_quantity(ref_vec, vec, percentage=percentage)
    diff_mean = np.mean(diff_ts)
    diff_std = np.std(diff_ts)

    return corr, diff_mean, diff_std, diff_ts


def calc_timeseries_agreement(ref_data, ref_time, data, time):
    if not np.all(ref_time == time):
        raise AttributeError("Time series do not cover the same lengths.")
    doys = tools.calc_day_of_year(ref_time)
    doy_not_366_inds = np.nonzero(doys != 366)
    doy366_inds = np.nonzero(doys == 366)

    corr_complete, diff_mean_complete, diff_std_complete, diff_ts_abs_complete = calc_vector_agreement(ref_data, data, percentage=False)

    ref_data_not366 = ref_data[doy_not_366_inds]
    data_not366 = data[doy_not_366_inds]
    corr_not366, diff_mean_not366, diff_std_not366, diff_ts_abs_not366 = calc_vector_agreement(ref_data_not366, data_not366, percentage=False)

    return doy_not_366_inds, doy366_inds, corr_complete, diff_mean_complete, diff_std_complete, diff_ts_abs_complete, corr_not366, diff_mean_not366, diff_std_not366, diff_ts_abs_not366

def plot_timeseries_agreement(ref_data, ref_time, data, time, title = None, do_print=False):

    doy_not_366_inds, doy366_inds, corr_complete, diff_mean_complete, diff_std_complete, diff_ts_abs_complete, corr_not366, diff_mean_not366, diff_std_not366, diff_ts_abs_not366 = calc_timeseries_agreement(ref_data, ref_time, data, time)

    ref_data_not366 = ref_data[doy_not_366_inds]
    data_not366 = data[doy_not_366_inds]

    fig_id = "evaluate_timeseries_agreement"
    if title is not None:
        fig_id = fig_id + "_" + title
    fig, axs = plt.subplots(2, 2, num=fig_id, clear=True, figsize=(12, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    if title is not None:
        fig.suptitle(title)

    ax = axs[0, 0]
    ax.set_title("Time series")
    p11, = ax.plot(time, ref_data, label="Reference")
    p12, = ax.plot(time, data, label="Recalculation")

    ax = axs[1, 0]
    ax.set_title("Difference")
    p21, = ax.plot(time, diff_ts_abs_complete, color="red")
    p22, = ax.plot(time[doy_not_366_inds], diff_ts_abs_not366, color="blue")

    ax = axs[0, 1]
    ax.set_title("Scatterplot")
    p31, = ax.plot(ref_data_not366, data_not366, linestyle='None', marker="x", color="blue")
    p32, = ax.plot(ref_data[doy366_inds], data[doy366_inds], linestyle='None', marker="x", color="red")

    ax = axs[1, 1]
    (n2, bins2, patches) = ax.hist(diff_ts_abs_not366, bins=100, density=True)

    sigma = diff_std_not366
    mu = diff_mean_not366
    ax.plot(bins2, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu) ** 2 / (2 * sigma ** 2)), linewidth = 2, color = 'r')

    if do_print:
        if title is not None:
            print(title)
        print("Mean of difference (without DOY 366): %1.2f" % diff_mean_not366)
        print("Stddev. of difference (without DOY 366): %1.2f" % diff_std_not366)
        print("68%% Percentile (abs. value of differences; without DOY 366): %1.2f" % np.percentile(np.abs(diff_ts_abs_not366), 68))
        print(np.percentile(np.abs(diff_ts_abs_not366), 99))
        print(np.percentile(np.abs(diff_ts_abs_complete), 68))
        print(np.percentile(np.abs(diff_ts_abs_complete), 99))
    return fig

