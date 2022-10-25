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
This module provides a simple 1-dim filtering algorithm, which can be used during the PC calculation instead of the
full 2-dim Wheeler-Kiladis-Filter.

This module is not intended to be used stand-alone outside the OMI context, as it has only been extensively 
tested for the specific OMI filtering conditions.

Hence, there is usually no need for the users of the mjoindices package to call functions of this module themselves.
Instead, they probably want to use the module :py:mod:`mjoindices.omi.omi_calculator` directly.
"""

import numpy as np
import scipy
import scipy.fftpack

import mjoindices.olr_handling as olr


def filter_olr_for_mjo_pc_calculation_1d_spectral_smoothing(olrdata: olr.OLRData) -> olr.OLRData:
    """
    Filters OLR data temporally using a 1d Fourier transform filter.

    The temporal filtering constants are chosen to meet the values in the description by :ref:`refKiladis2014`.

    :param olrdata: The original OLR data

    :return: The filtered OLR.
    """
    return filter_olr_temporally_1d_spectral_smoothing(olrdata, 20., 96.)


def filter_olr_temporally_1d_spectral_smoothing(olrdata: olr.OLRData, period_min: float, period_max: float) -> olr.OLRData:
    """
    Filters OLR data temporally using a 1d Fourier transform filter.

    :param olrdata: The original OLR data
    :param period_min: Temporal filter constant: Only greater periods (in days) remain in the data.
    :param period_max: Temporal filter constant: Only lower periods (in days) remain in the data.

    :return: The filtered OLR.
    """
    filteredOLR = np.empty(olrdata.olr.shape)
    time_spacing = (olrdata.time[1] - olrdata.time[0]).astype('timedelta64[s]') / np.timedelta64(1, 'D')  # time spacing in days
    for idx_lat in range(0, olrdata.olr.shape[1]):
        for idx_lon in range(0, olrdata.olr.shape[2]):
            tempolr = np.squeeze(olrdata.olr[:, idx_lat, idx_lon])
            filteredOLR[:, idx_lat, idx_lon] = _perform_spectral_smoothing(tempolr, time_spacing, period_min, period_max)
    return olr.OLRData(filteredOLR, olrdata.time, olrdata.lat, olrdata.long)


def _perform_spectral_smoothing(y, dt, lower_cutoff, higher_cutoff):
    """
    Applies a 1d Fourier Transform filter to the vector y.

    :param y: The data to filter
    :param dt: The spacing of the data
    :param lower_cutoff: Filter constant: Only greater periods (same units as dt) remain in the data.
    :param higher_cutoff: Filter constant: Only lower periods (same units as dt) remain in the data.

    :return: The filtered vector
    """
    N = y.size
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, dt)
    P = 1 / f
    w2 = w.copy()
    w2[P < lower_cutoff] = 0
    w2[P > higher_cutoff] = 0
    y2 = scipy.fftpack.irfft(w2)
    return y2
