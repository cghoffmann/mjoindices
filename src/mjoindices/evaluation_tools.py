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
This module provides a bunch of methods that help to evaluate the agreement of the OMI calculation by this package
and the original calculation by Kiladis (2014).
It is probably not of major relevance for the user of this package.
"""

import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm
import re
import warnings

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import mjoindices.tools as tools
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def compute_vector_difference_quantity(ref_vec: np.ndarray, vec: np.ndarray, percentage: bool = True) -> np.ndarray:
    """
    Calculates a standardized difference between two vectors.

    :param ref_vec: The reference vector.
    :param vec: The vector to validate.
    :param percentage: If true: calculation results will be in percent of the mean of the absolute reference values.

    :return: A vector containing the difference.
    """
    result = vec - ref_vec
    if percentage is True:
        result = result / np.mean(np.abs(ref_vec)) * 100
    return result


def calc_vector_agreement(ref_vec: np.ndarray, vec: np.ndarray, percentage: bool = True, do_print: bool = False) -> typing.Tuple:
    """
    Calculates extended comparison statistics between two vectors.

    :param ref_vec: The reference vector.
    :param vec: The vector to validate.
    :param percentage: If true: calculation results will be in percent of the mean of the absolute reference values.
    :param do_print: If true, some statistical values will we printed to the console.

    :return: A tuple containing values for the following quantities: correlation, mean of the differences, standard
        deviation of the differences, and percentiles of the absolute differences for 68%, 95%, and 99%.
    """
    if not np.all(ref_vec.size == vec.size):
        raise ValueError("Vectors do not have the same lengths.")
    corr = np.corrcoef(ref_vec, vec)[0, 1]

    diff_vec = compute_vector_difference_quantity(ref_vec, vec, percentage=percentage)
    diff_mean = np.mean(diff_vec)
    diff_std = np.std(diff_vec)
    diff_abs_percent68 = np.percentile(np.abs(diff_vec), 68.)
    diff_abs_percent95 = np.percentile(np.abs(diff_vec), 95.)
    diff_abs_percent99 = np.percentile(np.abs(diff_vec), 99)

    if do_print:
        print("CorrelationCoefficient: %1.4f" % corr)
        print("Mean of difference: %1.4f" % diff_mean)
        print("Stddev. of difference: %1.4f" % diff_std)
        print("68%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent68)
        print("95%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent95)
        print("99%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent99)

    return corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99


def calc_comparison_stats_for_eofs_all_doys(eofs_ref: eof.EOFDataForAllDOYs,
                                            eofs: eof.EOFDataForAllDOYs,
                                            eof_number,
                                            exclude_doy366: bool = False,
                                            percentage: bool=False,
                                            do_print: bool = False) -> typing.Tuple:
    """
    Calculates extended comparison statistics between calculated EOFs and reference EOFs.

    :param recalc_eof: The EOFs to validate.
    :param orig_eof: The reference EOFs.
    :param exclude_doy366: If True, DOY 366 will be included in the calculation (sometimes worse agreement depending on
        the leap year treatment mode).
    :param percentage: If True: calculation results will be in percent of the mean of the absolute reference values.
    :param do_print: If True, some characteristic values will we printed to the console.

    :return: A tuple containing arrays with 366 or 365 elements each for the following quantities: correlation, mean
        of the differences, standard deviation of the differences, and percentiles of the absolute differences
        for 68%, 95%, and 99%.
    """
    if eof_number != 1 and eof_number != 2:
        raise ValueError("Argument eof_number must be 1 or 2.")
    doys = tools.doy_list()
    if exclude_doy366:
        doys = doys[:-1]
    corr = np.empty(doys.size)
    diff_mean = np.empty(doys.size)
    diff_std = np.empty(doys.size)
    diff_abs_percent68 = np.empty(doys.size)
    diff_abs_percent95 = np.empty(doys.size)
    diff_abs_percent99 = np.empty(doys.size)

    for idx, doy in enumerate(doys):
        if eof_number == 1:
            eof_ref = eofs_ref.eof1vector_for_doy(doy)
            eof_test = eofs.eof1vector_for_doy(doy)
        elif eof_number == 2:
            eof_ref = eofs_ref.eof2vector_for_doy(doy)
            eof_test = eofs.eof2vector_for_doy(doy)
        (corr_single, diff_mean_single, diff_std_single, diff_vec_single, diff_abs_percent68_single,
         diff_abs_percent95_single, diff_abs_percent99_single) \
            = calc_vector_agreement(eof_ref, eof_test, percentage=percentage, do_print=False)
        corr[idx] = corr_single
        diff_mean[idx] = diff_mean_single
        diff_std[idx] = diff_std_single
        diff_abs_percent68[idx] = diff_abs_percent68_single
        diff_abs_percent95[idx] = diff_abs_percent95_single
        diff_abs_percent99[idx] = diff_abs_percent99_single

    if do_print:
        print("########## Summary of EOF comparison for all DOYs (EOF %i)" % eof_number)
        print("Worst Correlation (at DOY %i): %1.4f" % (doys[np.argmin(corr)], np.amin(corr)))
        print("Worst 99%% percentile (at DOY %i): %1.4f" % (doys[np.argmax(diff_abs_percent99)],
                                                            np.amax(diff_abs_percent99)))
        print("Worst 68%% percentile (at DOY %i): %1.4f" % (doys[np.argmax(diff_abs_percent68)],
                                                            np.amax(diff_abs_percent68)))

    return corr, diff_mean, diff_std, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99


def plot_comparison_stats_for_eofs_all_doys(recalc_eof: eof.EOFDataForAllDOYs,
                                            orig_eof: eof.EOFDataForAllDOYs,
                                            exclude_doy366: bool = False,
                                            do_print: bool = False) -> Figure:
    """
    Plots extended comparison statistics between calculated EOFs and references EOFs.

    Statistics of the differences will be shown for each DOY (DOY on the abscissa) and with one line each for EOF1 and
    EOF2.

    :param recalc_eof: The EOFs to compare.
    :param orig_eof: The reference eofs.
    :param exclude_doy366: If True, DOY 366 will be included in the plot (sometimes worse agreement depending on the
        leap year treatment mode).
    :param do_print: If True, some characteristic values will we printed to the console.

    :return: The figure handle
    """

    doys = tools.doy_list()
    if exclude_doy366:
        doys = doys[:-1]
    xlim = (0, doys[-1])
    (corr_1, diff_mean_1, diff_std_1, diff_abs_percent68_1, diff_abs_percent95_1, diff_abs_percent99_1)\
        = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=1,
                                                  percentage=False, do_print=do_print)
    (corr_2, diff_mean_2, diff_std_2, diff_abs_percent68_2, diff_abs_percent95_2, diff_abs_percent99_2)\
        = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=2,
                                                  percentage=False, do_print=do_print)

    fig_id = "plot_comparison_stats_for_eofs_all_doys"
    fig, axs = plt.subplots(4, 1, num=fig_id, clear=True, figsize=(9, 9), dpi=150)
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle("Comparison of EOFs for all DOYs")

    ax = axs[0]
    ax.set_title("Correlation")
    p11, = ax.plot(doys, corr_1, label="EOF1", color="blue")
    p12, = ax.plot(doys, corr_2, label="EOF2", color="green")
    ax.set_xlim(xlim)
    ax.set_ylabel("Correlation")
    fig.legend(handles=[p11, p12])

    ax = axs[1]
    ax.set_title("Mean of differences of EOF vector elements")
    p21, = ax.plot(doys, diff_mean_1, label="EOF1", color="blue")
    p22, = ax.plot(doys, diff_mean_2, label="EOF2", color="green")
    ax.set_xlim(xlim)
    ax.set_ylabel(r"Mean [$\mathrm{W/m^2}$]")

    ax = axs[2]
    ax.set_title("Standard deviation of differences of EOF vector elements")
    p31, = ax.plot(doys, diff_std_1, label="EOF1", color="blue")
    p32, = ax.plot(doys, diff_std_2, label="EOF2", color="green")
    ax.set_xlim(xlim)
    ax.set_ylabel(r"Std.Dev. [$\mathrm{W/m^2}$]")

    ax = axs[3]
    ax.set_title("Percentiles of absolute differences of EOF vector elements")
    p31, = ax.plot(doys, diff_abs_percent99_1, label="99% EOF1", color="blue")
    p32, = ax.plot(doys, diff_abs_percent99_2, label="99% EOF2", color="green")
    p33, = ax.plot(doys, diff_abs_percent95_1, label="EOF1", color="blue", linestyle="--")
    p34, = ax.plot(doys, diff_abs_percent95_2, label="EOF2", color="green", linestyle="--")
    p35, = ax.plot(doys, diff_abs_percent68_1, label="EOF1", color="blue", linestyle=":")
    p36, = ax.plot(doys, diff_abs_percent68_2, label="EOF2", color="green", linestyle=":")
    ax.set_xlim(xlim)
    ax.set_ylabel(r"Percentiles [$\mathrm{W/m^2}$]")
    ax.legend(labels=["99%", "95%", "68%"], handles=[p31, p33, p35], loc="upper right")
    ax.set_xlabel("Day of year")

    return fig


def plot_correlation_for_eofs_all_doys(recalc_eof: eof.EOFDataForAllDOYs,
                                       orig_eof: eof.EOFDataForAllDOYs,
                                       exclude_doy366: bool = False,
                                       do_print: bool = False,
                                       full_value_range: bool = True) -> Figure:
    """
    Plots the correlations between calculated EOFs and reference EOFs.

    Correlations will be shown for each DOY (DOY on the abscissa) and with one line each for EOF1 and EOF2.

    :param recalc_eof: The EOFs to validate.
    :param orig_eof: The reference EOFs.
    :param exclude_doy366: If False, DOY 366 will be included in the plot (sometimes worse correlation depending on the
        leap year treatment mode).
    :param do_print: If True, some characteristic values will we printed to the console.
    :param full_value_range: If True, the ordinate spans the range from 0 to 1 instead of the used value range only.
    :return: A handle to the figure.
    """

    doys = tools.doy_list()
    if exclude_doy366:
        doys = doys[:-1]
    xlim = (0, doys[-1])
    (corr_1, diff_mean_1, diff_std_1, diff_abs_percent68_1, diff_abs_percent95_1, diff_abs_percent99_1)\
        = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=1,
                                                  percentage=False, do_print=do_print)
    (corr_2, diff_mean_2, diff_std_2, diff_abs_percent68_2, diff_abs_percent95_2, diff_abs_percent99_2)\
        = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=2,
                                                  percentage=False, do_print=do_print)

    fig_id = "plot_correlation_for_eofs_all_doys"
    fig, axs = plt.subplots(1, 1, num=fig_id, clear=True, figsize=(6, 4.5), dpi=150)
    plt.subplots_adjust(hspace=0.3)

    fig.suptitle("Correlation of EOFs for all DOYs")

    ax = axs
    p11, = ax.plot(doys, corr_1, label="EOF1", color="blue")
    p12, = ax.plot(doys, corr_2, label="EOF2", color="green")
    ax.set_xlim(xlim)
    ax.legend(handles=[p11, p12])
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Correlation coefficient")
    if full_value_range:
        ax.set_ylim((0, 1.1))

    return fig


def plot_individual_eof_map_comparison(orig_eof: eof.EOFData, compare_eof: eof.EOFData, doy: int = None) -> Figure:
    """
    Shows the maps of EOFs 1 and 2 and a respective reference together with a map of differences between both.

    :param orig_eof: The reference EOF data.
    :param compare_eof: The EOF data to validate.
    :param doy: The DOY, which is evaluated (only used for the figure title).

    :return: The figure handle.
    """

    fig, axs = plt.subplots(2, 3, num="ReproduceOriginalOMIPCs_ExplainedVariance_EOF_Comparison", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map,
                    levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF1")
    ax.set_ylabel("Latitude [°]")

    ax = axs[0, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof1map,
                    levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF1")

    # FIXME: Check that grids are equal
    ax = axs[0, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map - compare_eof.eof1map,
                    levels=100, cmap=matplotlib.cm.get_cmap("Purples"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 1")

    ax = axs[1, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map,
                    levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof2map,
                    levels=np.arange(-0.1, 0.11, 0.01), cmap=matplotlib.cm.get_cmap("bwr"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF2")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map - compare_eof.eof2map,
                    levels=100, cmap=matplotlib.cm.get_cmap("Purples"))
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 2")
    ax.set_xlabel("Longitude [°]")

    return fig


def calc_comparison_stats_for_explained_variance(ref_var: np.ndarray,
                                                 calc_var: np.ndarray,
                                                 do_print: bool = False,
                                                 exclude_doy366: bool = False) -> typing.Tuple:
    """
    Calculates the comparison statistics of the explained variances for one EOF and all DOYs.

    :param ref_var: The reference variances.
    :param calc_var: The variances to compare.
    :param do_print: If True, some statistical values will also be shown in the console.
    :param exclude_doy366: If True, the data for DOY 366 will not be considered in the statistics.

    :return: A tuple containing values for the following quantities: correlation, mean of the differences,
        standard deviation of the differences, and percentiles of the absolute differences for 68%, 95%, and 99%.
    """
    ref_data = ref_var.copy()
    calc_data = calc_var.copy()
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        ref_data = ref_data[:-1]
        calc_data = calc_data[:-1]
    (corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99)\
        = calc_vector_agreement(ref_data, calc_data, percentage=False, do_print=do_print)
    return corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99


def plot_comparison_stats_for_explained_variance(ref_var: np.ndarray,
                                                 calc_var: np.ndarray,
                                                 title: str = None,
                                                 do_print: bool = False,
                                                 exclude_doy366: bool = False) -> Figure:
    """
    Plots the comparison of the explained variances for one EOF and for all DOYs.

    :param ref_var: The reference variances.
    :param calc_var: The variances to validate.
    :param title: A title for the figure.
    :param do_print: If True, some statistical values will also be shown in the console.
    :param exclude_doy366: If True, the data for DOY 366 will not be considered in the statistics.

    :return: The figure handle
    """
    if do_print:
        print("##########")
        if title is not None:
            print(title)
    ref_data = ref_var.copy()
    calc_data = calc_var.copy()
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        ref_data = ref_data[:-1]
        calc_data = calc_data[:-1]
    fig = plot_vector_agreement(ref_data, calc_data, title=title, do_print=do_print)
    return fig


def calc_timeseries_agreement(ref_data: np.ndarray, ref_time: np.ndarray, data: np.ndarray, time: np.ndarray,
                              exclude_doy366: bool = False, do_print: bool = False) -> typing.Tuple:
    """
    Calculates comparison values of two time series.

    :param ref_data: The reference time series vector.
    :param ref_time: The time grid of the reference.
    :param data: The time series vector to validate.
    :param time: The time grid of the time series to validate. It will be checked if this is similar to the time grid
        of the reference.
    :param exclude_doy366: If True, the data for DOY 366 will not be considered in the statistics.
    :param do_print: If True, some statistical values will also be shown in the console.

    :return: A tuple containing values for the following quantities: correlation, mean of the differences,
        standard deviation of the differences, and percentiles of the absolute differences for 68%, 95%, and 99%.
    """
    if not np.all(ref_time == time):
        raise ValueError("Time series do not cover the same periods.")
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        doys = tools.calc_day_of_year(ref_time)
        inds_used = np.nonzero(doys != 366)
        inds_not_used = np.nonzero(doys == 366)
        calc_ref_data = ref_data[inds_used]
        calc_data = data[inds_used]
    else:
        if do_print:
            print("##### Complete time series")
        calc_ref_data = ref_data
        calc_data = data
        inds_used = (np.arange(0, ref_time.size, 1),)
        inds_not_used = (np.array([], dtype="int64"),)

    (corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99)\
        = calc_vector_agreement(calc_ref_data, calc_data, percentage=False, do_print=do_print)

    return (inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95,
            diff_abs_percent99)


def plot_timeseries_agreement(ref_data: np.ndarray,
                              ref_time: np.ndarray,
                              data: np.ndarray,
                              time: np.ndarray,
                              title: str = None,
                              do_print: bool = False) -> Figure:
    """
    Plots a graphical comparison of 2 time series.

    Shows 4 subplots with 1) the data, 2) the difference, 3) a scatterplot of the data, and 4) a histogram of the
    differences.

    :param ref_data: The time series reference vector.
    :param ref_time: The time grid of the reference.
    :param data: The time series vector to validate.
    :param time: The time grid of the data to validate. It will be checked if this is similar to the time grid of the reference.
    :param title: A title for the plot. Use this to explain the quantity which is compared evaluated with the plot.
    :param do_print: If True, some statistical values will also be shown in the console.

    :return: The figure handle.
    """
    if do_print:
        print("##########")
        if title is not None:
            print(title)

    if not (time[0] == ref_time[0]) or not (time[-1] == ref_time[-1]):
        (time_intersect, data_ind, ref_data_ind) = np.intersect1d(time, ref_time, return_indices=True)
        if time_intersect.size == 0:
            raise ValueError("Provided time series to plot do not overlap.")
        if not np.all(np.diff(time_intersect) == np.diff(time_intersect)[0]):
            raise ValueError("Combined time grid of both time series to plot is not equidistant.")
        time = time_intersect
        data = data[data_ind]
        ref_time = time_intersect
        ref_data = ref_data[ref_data_ind]
        message = ("The provided time series do not cover the same period.\nThe compared range was restricted to the "
                   "overlapping period from {0} to {1}.\nStronger differences may occur at least at the end of the "
                   "compared period, since the temporal filtering worked on different samples."
                   .format(str(time[0]), str(time[-1])))
        warnings.warn(message)

    (tempa, tempb, corr_complete, diff_mean_complete, diff_std_complete, diff_ts_abs_complete,
     diff_abs_percent68_complete, diff_abs_percent95_complete, diff_abs_percent99_complete) \
        = calc_timeseries_agreement(ref_data, ref_time, data, time, exclude_doy366=False, do_print=do_print)

    (doy_not_366_inds, doy366_inds, corr_not366, diff_mean_not366, diff_std_not366, diff_ts_abs_not366,
     diff_abs_percent68_not366, diff_abs_percent95_not366, diff_abs_percent99_not366) \
        = calc_timeseries_agreement(ref_data, ref_time, data, time, exclude_doy366=True, do_print=do_print)

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
    ax.set_xlabel("Time")
    ax.set_ylabel("Data [units of quantity]")

    ax = axs[1, 0]
    ax.set_title("Difference")
    p21, = ax.plot(time, diff_ts_abs_complete, color="red")
    p22, = ax.plot(time[doy_not_366_inds], diff_ts_abs_not366, color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Difference [units of quantity]")

    ax = axs[0, 1]
    ax.set_title("Scatterplot")
    p31, = ax.plot(ref_data_not366, data_not366, linestyle='None', marker="x", color="blue")
    p32, = ax.plot(ref_data[doy366_inds], data[doy366_inds], linestyle='None', marker="x", color="red")
    ax.set_xlabel("Reference data [units of quantity]")
    ax.set_ylabel("Data [units of quantity]")

    ax = axs[1, 1]
    (n2, bins2, patches) = ax.hist(diff_ts_abs_not366, bins=100, density=True)
    ax.set_xlabel("Difference [units of quantity]")
    ax.set_ylabel("Number of occurrences")
    sigma = diff_std_not366
    mu = diff_mean_not366
    ax.plot(bins2, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu) ** 2 / (2 * sigma ** 2)),
            linewidth=2, color='r')

    return fig


def plot_vector_agreement(ref_data: np.ndarray, data: np.ndarray, title: str = None, do_print: bool = False) -> Figure:
    """
    Plot a graphical comparison of 2 vectors.

    Shows 4 subplots with 1) the data, 2) the difference, 3) a scatterplot of the data, and 4) a histogram of
    the differences.

    :param ref_data: The reference vector.
    :param data: The vector to validate.
    :param title: A title for the plot. Use this to explain the quantity which is evaluated with the plot.
    :param do_print: If True, some statistics values will also be shown in the console.

    :return: The figure handle.
    """
    if do_print:
        print("##########")
        if title is not None:
            print(title)
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = \
        calc_vector_agreement(ref_data, data, percentage=False, do_print=do_print)

    fig_id = "evaluate_vector_agreement"
    if title is not None:
        fig_id = fig_id + "_" + title
    fig, axs = plt.subplots(2, 2, num=fig_id, clear=True, figsize=(12, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    if title is not None:
        fig.suptitle(title)

    ax = axs[0, 0]
    ax.set_title("Data")
    p11, = ax.plot(ref_data, label="Reference")
    p12, = ax.plot(data, label="Recalculation")
    ax.set_xlabel("Position in vector")
    ax.set_ylabel("Data [units of quantity]")

    ax = axs[1, 0]
    ax.set_title("Difference")
    p21, = ax.plot(diff_vec, color="red")
    ax.set_xlabel("Position in vector")
    ax.set_ylabel("Difference [units of quantity]")

    ax = axs[0, 1]
    ax.set_title("Scatterplot")
    p31, = ax.plot(ref_data, data, linestyle='None', marker="x", color="blue")
    ax.set_xlabel("Reference data [units of quantity]")
    ax.set_ylabel("Data [units of quantity]")

    ax = axs[1, 1]
    (n2, bins2, patches) = ax.hist(diff_vec, bins=100, density=True)
    ax.set_xlabel("Difference [units of quantity]")
    ax.set_ylabel("Number of occurrences")
    sigma = diff_std
    mu = diff_mean
    ax.plot(bins2, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu) ** 2 / (2 * sigma ** 2)),
            linewidth=2, color='r')

    return fig


def plot_comparison_orig_calc_pcs(calc_pcs: pc.PCData,
                                  orig_pcs: pc.PCData,
                                  start_date: np.datetime64 = None,
                                  end_date: np.datetime64 = None):
    """
    Plots both PC time series (one in a subplot each) of the recalculation and a reference.

    The period to plot can be adjusted.

    :param calc_pcs: The recalculated PC time series.
    :param orig_pcs: The reference PC time series.
    :param start_date: Start of the period to plot. If None, the whole period will be plotted.
    :param end_date: End of the period to plot. If None, the whole period will be plotted.

    :return: The figure handle.
    """
    fig, axs = plt.subplots(2, 1, num="plot_comparison_orig_calc_pcs", clear=True, figsize=(8, 6), dpi=150)
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle("PC Recalculation")

    ax = axs[0]
    ax.set_title("Principal Component 1")
    p1, = ax.plot(orig_pcs.time, orig_pcs.pc1, label="Original")
    p2, = ax.plot(calc_pcs.time, calc_pcs.pc1, label="Recalculation")
    if start_date is not None and end_date is not None:
        ax.set_xlim((start_date, end_date))
    ax.set_ylabel("PC1 [1]")
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    fig.legend(handles=(p1, p2))

    ax = axs[1]
    ax.set_title("Principal Component 2")
    p3, = ax.plot(orig_pcs.time, orig_pcs.pc2, label="Original")
    p4, = ax.plot(calc_pcs.time, calc_pcs.pc2, label="Recalculation")
    if start_date is not None and end_date is not None:
        ax.set_xlim((start_date, end_date))
    ax.set_ylabel("PC2 [1]")
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment="right")

    return fig


def _explained_variance_file_converter(s: str) -> float:
    return float(re.sub(r"[ \[;\]]", "", s.decode("utf-8")))


def load_omi_explained_variance(filename: str) -> typing.Tuple:
    """
    Loads original explained variance files provided by Juliana Dias.

    :param filename: The file to load.

    :return: A tuple containing two arrays, for EOF1 and EOF2, respectively.
    """
    data = np.genfromtxt(filename,
                         converters={0: _explained_variance_file_converter, 1: _explained_variance_file_converter},
                         skip_header=7)
    var1 = data[:, 0]
    var2 = data[:, 1]
    return var1, var2
