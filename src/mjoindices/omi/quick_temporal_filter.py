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
import scipy
import scipy.fftpack

import mjoindices.olr_handling as olr

# FIXME: Codestyle
# FIXME: Constants

def filterOLRForMJO_PC_CalculationWith1DSpectralSmoothing(olrdata: olr.OLRData) -> olr.OLRData:
    """
    Filters OLR data temporally using a 1d Fourier transform filter.
    The temporal filtering constants are chosen to meet the values in the description by Kiladis 2014.
    :param olrdata: The OLRData object containing the original OLR data
    :return: An OLRData object containing the filtered OLR.
    """
    return filterOLRTemporallyWith1DSpectralSmoothing(olrdata, 20., 96.)


def filterOLRTemporallyWith1DSpectralSmoothing(olrdata: olr.OLRData, period_min: float, period_max: float) -> olr.OLRData:
    """
    Filters OLR data temporally using a 1d Fourier transform filter.
    The temporal filtering constants are chosen to meet the values in the description by Kiladis 2014.
    :param olrdata: The OLRData object containing the original OLR data
    :param period_min: Temporal filter constant: Only greater periods remain in the data.
    :param period_max: Temporal filter constant: Only lower periods remain in the data.
    :return: An OLRData object containing the filtered OLR.
    """
    print("Smooth data temporally...")
    filteredOLR = np.empty(olrdata.olr.shape)
    time_spacing = (olrdata.time[1] - olrdata.time[0]).astype('timedelta64[s]') / np.timedelta64(1, 'D')  # time spacing in days
    for idx_lat in range(0, olrdata.olr.shape[1]):
        for idx_lon in range(0, olrdata.olr.shape[2]):
            tempolr = np.squeeze(olrdata.olr[:, idx_lat, idx_lon])
            filteredOLR[:, idx_lat, idx_lon] = _performSpectralSmoothing(tempolr, time_spacing, period_min, period_max)
    return olr.OLRData(filteredOLR, olrdata.time, olrdata.lat, olrdata.long)


def _performSpectralSmoothing(y, dt, lowerCutOff, HigherCutOff):
    """
    Applies a 1d Fourier Transform filter to the vector y.
    :param y: The data to filter
    :param dt: The spacing of the data
    :param lowerCutOff: Filter constant: Only greater periods (same units as dt) remain in the data.
    :param HigherCutOff: Filter constant: Only lower periods (same units as dt) remain in the data.
    :return: The filtered vector
    """
    N = y.size
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, dt)
    P = 1 / f
    w2 = w.copy()
    w2[P < lowerCutOff] = 0
    w2[P > HigherCutOff] = 0
    y2 = scipy.fftpack.irfft(w2)
    return y2

