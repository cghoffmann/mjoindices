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

import mjoindices.empirical_orthogonal_functions as eof


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
            np.mean(np.abs(eofs_ref.eofdata_for_doy(doy).eof1vector - eofs.eofdata_for_doy(doy).eof1vector))
        maxdiff_abs_2[idx] = \
            np.mean(np.abs(eofs_ref.eofdata_for_doy(doy).eof2vector - eofs.eofdata_for_doy(doy).eof2vector))
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