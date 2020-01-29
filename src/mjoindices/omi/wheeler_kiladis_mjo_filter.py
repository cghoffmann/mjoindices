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
This module provides the 2-dim (temporal and longitudinal) filtering algorithm used by the OMI approach.

Although implemented generically, it is not intended to use this module stand-alone outside of the OMI context,
since it is only extensively tested for the specific OMI filtering conditions.

Hence, there is usually no need for the users of the mjoindices package to call functions of this module themselves.
Instead, they might probably want to use the module :py:mod:`mjoindices.omi.omi_calculator` directly.

The complete algorithm is described in Kiladis, G.N., J. Dias, K.H. Straub, M.C. Wheeler, S.N. Tulich, K. Kikuchi, K.M.
Weickmann, and M.J. Ventrice, 2014: A Comparison of OLR and Circulation-Based Indices for Tracking the MJO.
Mon. Wea. Rev., 142, 1697–1715, https://doi.org/10.1175/MWR-D-13-00301.1
"""

import matplotlib.pyplot as plt
import numpy as np
import mjoindices.olr_handling as olr


def filter_olr_for_mjo_pc_calculation(olrdata: olr.OLRData, do_plot: bool = False):
    """
    Filters OLR data temporally.

    The filter algorithm is the same as for the combined temporal and longitudinal filtering,
    but the longitudinal bandpass filter constants are defined so broad that effectively no longitudinal filtering is
    applied.
    The temporal filtering constants are chosen to meet the values in the description by Kiladis (2014).

    :param olrdata: The original OLR data.
    :param do_plot: If True, diagnosis plots will be generated.

    :return: The filtered OLR.
    """
    return filter_olr_temporally(olrdata, 20., 96., do_plot=do_plot)


# Implicitly tested for special conditions with specific caller functions
def filter_olr_temporally(olrdata: olr.OLRData, period_min: float, period_max: float, do_plot: bool = False):
    """
    Filters OLR data temporally.

    The filter algorithm is the same as for the combined temporal and longitudinal filtering,
    but the longitudinal bandpass filter constants are defined so broad that effectively no longitudinal filtering is
    applied.

    Note that this function has strictly only been tested for filtering constants used by the OMI algorithm.

    :param olrdata: The original OLR data
    :param period_min: Temporal filter constant: Only greater periods (in days) remain in the data.
    :param period_max: Temporal filter constant: Only lower periods (in days) remain in the data.
    :param do_plot: If True, diagnosis plots will be generated.

    :return: The filtered OLR.
    """
    return filter_olr_temporally_and_longitudinally(olrdata, period_min, period_max, -720., 720, do_plot=do_plot)


def filter_olr_for_mjo_eof_calculation(olrdata: olr.OLRData, do_plot: bool = False) -> olr.OLRData:
    """
    Filters OLR data temporally and longitudinally.

    The filter setup meets the description of Kiladis (2014) for the EOF calculation.

    :param olrdata: The original OLR data
    :param do_plot: If True, diagnosis plots will be generated.

    :return: The filtered OLR data.
    """
    return filter_olr_temporally_and_longitudinally(olrdata, 30., 96., 0., 720, do_plot=do_plot)


# Implicitly tested for special conditions with specific caller functions
def filter_olr_temporally_and_longitudinally(olrdata: olr.OLRData,
                                             period_min: float,
                                             period_max: float,
                                             wn_min: float,
                                             wn_max: float,
                                             do_plot: bool=False) -> olr.OLRData:
    """
    Performs a temporal and longitudinal bandpass filtering of the OLR data with configurable filtering thresholds.

    Note that this function has strictly only been tested for filtering constants used by the OMI algorithm.

    :param olrdata: The original OLR data.
    :param period_min: Temporal filter constant: Only greater periods (in days) remain in the data.
    :param period_max: Temporal filter constant: Only lower periods (in days) remain in the data.
    :param wn_min: Longitudinal filter constant: Only greater wave numbers (in cycles per globe) remain in the data.
    :param wn_max:  Longitudinal filter constant: Only lower wave numbers (in cycles per globe) remain in the data.
    :param do_plot: If True, diagnosis plots will be generated.

    :return: The filtered OLR.
    """
    print("Smooth data temporally and longitudinally...")
    filtered_olr = np.empty(olrdata.olr.shape)

    for ilat, lat in enumerate(olrdata.lat):
        print("Filtering for latitude: ", lat)
        time_spacing = (olrdata.time[1] - olrdata.time[0]).astype('timedelta64[s]') / np.timedelta64(1, 'D')  # time spacing in days
        dataslice = np.squeeze(olrdata.olr[:, ilat, :])
        wkfilter = WKFilter()
        filtered_data = wkfilter.perform_2dim_spectral_filtering(dataslice, time_spacing, period_min, period_max, wn_min,
                                                                 wn_max, do_plot=do_plot, save_debug=False)
        filtered_olr[:, ilat, :] = filtered_data

    return olr.OLRData(filtered_olr, olrdata.time, olrdata.lat, olrdata.long)


def detrend_vector(data: np.ndarray) -> np.ndarray:
    """
    Removes the trend from the given vector.

    :param data: The vector to detrend

    :return: The data with removed trend.
    """
    x = np.arange(0, data.size, 1)
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, data, rcond=-1)[0]

    result = data - (m * x + b)
    #    plt.plot(ts)
    #    plt.plot(m*x+b)
    #    plt.plot(result)
    return result


def taper_vector_to_zero(data: np.ndarray, window_length: int) -> np.ndarray:
    """
    Taper the data in the given vector to zero at both the beginning and the ending.

    :param data: The data to taper.
    :param window_length: The length of the window (measured in vector indices),
        in which the tapering is applied for the beginning and the ending independently

    :return: The tapered data.
    """
    startinds = np.arange(0, window_length, 1)
    endinds = np.arange(-window_length - 1, -1, 1) + 2

    result = data
    result[0:window_length] = result[0:window_length] * 0.5 * (1 - np.cos(startinds * np.pi / window_length))
    result[data.size - window_length:data.size] = \
        result[data.size - window_length:data.size] * 0.5 * (1 - np.cos(endinds * np.pi / window_length))
    return result


class WKFilter:
    """
    This class contains the major Wheeler-Kiladis-Filtering functionality.
    The functionality is encapsulated in a class because values of intermediate processing steps
    are saved as class members for debugging purposes.
    To run the filtering, only the method :func:`perform_2dim_spectral_filtering` has to be executed.
    """
    def __init__(self):
        self.DebugInputOLR = []
        self.DebugFilterOLR = []
        self.DebugDetrendedOLR = []
        self.DebugPreprocessedOLR = []
        self.DebugFreqAxis = []
        self.DebugWNAxis = []
        self.DebugOriginalFourierSpectrum = []
        self.DebugFilteredFourierSpectrum = []
        self.DebugNoElementsInFilteredSpectrum = []

    def perform_2dim_spectral_filtering(self,
                                        data: np.ndarray,
                                        time_spacing: float,
                                        period_min: float,
                                        period_max: float,
                                        wn_min: float,
                                        wn_max: float,
                                        do_plot: bool = False,
                                        save_debug: bool = False) -> np.ndarray:
        """
        Bandpass-filters OLR data in time- and longitude-direction according to
        the original Kiladis algorithm.

        Note that the temporal and longitudinal dimension have in principle
        different characteristics, so that they are in detail treated a bit
        differently:
        While the time is evolving into infinity (so that the number of data
        points and the time_spacing variable are needed to calculate the
        full temporal coverage), the longitude is periodic with the periodicity
        of one globe (so that it is assumed that the data covers exactly one
        globe and only passing the number of longitudes provides already the complete information).

        :param data: The OLR data as 2-dim array: first dimension time, second
            dimension longitude, both equally spaced.
            The longitudinal dimension has to cover the full globe.
            The time dimension is further described by the variable
            `time_spacing`.
        :param time_spacing: Temporal resolution of the data in days (often 1 or 0.5 (if two
            data points exist per day)).
        :param period_min: Minimal period (in days) that remains in the dataset.
        :param period_max: Maximal period (in days) that remains in the dataset.
        :param wn_min: Minimal wavenumber (in cycles per globe) that remains in the dataset.
        :param wn_max: Maximal wavenumber (in cycles per globe) that remains in the dataset.
        :param do_plot: If True, diagnosis plots will be generated.
        :param save_debug: If true, some variables will be filled with values of intermediate processing steps
            for debugging purposes.

        :return: The filtered data.
        """

        # ###################### Process input data #######################
        if save_debug:
            self.DebugInputOLR = np.copy(data)

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_DataIn", clear=True)
            plt.contourf(data)
            plt.colorbar()
            plt.title("Original Data")

        dataperday = 1 / time_spacing
        freq_min = 1 / period_max
        freq_max = 1 / period_min

        # ######################## Detrend #################################
        # "orig" refers to the original size in the time dimension in the following, i.e. not the zero-padded version.
        orig_data = data
        orig_nt, nl = orig_data.shape

        for idx_l in range(0, nl):
            orig_data[:, idx_l] = detrend_vector(orig_data[:, idx_l])

        if save_debug:
            self.DebugDetrendedOLR = np.copy(orig_data)
        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_Detrended", clear=True)
            plt.contourf(orig_data)
            plt.colorbar()
            plt.title("Detrended Data")

        # ######################## Zero Padding ############################
        nt = 2 ** 17  # Zero padding for performance and resolution optimization, as well as consistency with origininal Kiladis code

        if orig_nt > nt:
            raise ValueError('Time series is longer than hard-coded value for zero-padding!')

        data = np.zeros([nt, nl])
        data[0:orig_nt, :] = orig_data

        # ######################## Tapering to zero ########################
        # 10 days tapering according ot Kiladis Code
        # only relevant at beginning of time series as it is zero-padded in the end
        for idx_l in range(0, nl):
            data[:, idx_l] = taper_vector_to_zero(data[:, idx_l], int(10 * dataperday))

        if save_debug:
            self.DebugPreprocessedOLR = np.copy(data)

        # ########################## Forward Fourier transform ############
        fourier_fft = np.fft.fft2(data)
        # reordering of spectrum is done to be consistent with the original kiladis ordering.
        fourier_fft = np.fft.fftshift(fourier_fft, axes=(0, 1))
        fourier_fft = np.roll(fourier_fft, int(nt / 2), axis=0)
        fourier_fft = np.roll(fourier_fft, int(nl / 2), axis=1)

        # ## Calculation of the frequency grid in accordance with Kiladis code
        freq_axis = np.zeros(nt)
        for i_f in range(0, nt):
            if (i_f <= nt / 2):
                freq_axis[i_f] = i_f * dataperday / nt
            else:
                freq_axis[i_f] = -1 * (nt - i_f) * dataperday / nt
        # the following code based on scipy function produces qualitatively the same grid.
        # However, numerical differences seem to have larger effect for the filtering step.
        # freq_axis = np.fft.fftfreq(nt, d=time_spacing)
        # freq_axis = np.fft.fftshift(freq_axis)
        # freq_axis = np.roll(freq_axis, int(nt/2))

        # ## Calculation of the wavenumber grid in accordance with Kiladis code
        wn_axis = np.zeros(nl)
        for i_wn in range(0, nl):
            if i_wn <= nl / 2:
                wn_axis[i_wn] = -1 * i_wn
                # note: to have this consistent with the time-dimension, one could write wn_axis[i_wn]= -1*i_wn*dataperglobe/nl
                # However, since data is required to cover always one globe nl will always be equal to dataperglobe
                # The sign is not consistent with the time dimension, which is for reasons of consitency with the original Kiladis implementation
            else:
                wn_axis[i_wn] = nl - i_wn
        # the following code based on scipy function produces qualitatively the same grid.
        # However, numerical differences seem to have larger effect for the filtering step.
        # wn_axis = np.fft.fftfreq(nl, d=dy)
        # wn_axis = np.fft.fftshift(wn_axis)  #identical with  wn_axis=np.arange(-int(nlong/2), int(nlong/2),1.)
        # wn_axis = -1 *wn_axis
        # wn_axis = np.roll(wn_axis, int(nl/2))

        if save_debug:
            self.DebugFreqAxis = np.copy(freq_axis)
            self.DebugWNAxis = np.copy(wn_axis)
            self.DebugOriginalFourierSpectrum = np.copy(fourier_fft)

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_freqAxis", clear=True)
            plt.plot(freq_axis)
            plt.title("Calc freq axis")

            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_WNAxis", clear=True)
            plt.plot(wn_axis)
            plt.title("Calc wn axis")

            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_Spectrum", clear=True)
            plt.contourf(wn_axis, freq_axis, np.squeeze(fourier_fft))
            plt.colorbar()
            plt.title("Fourier Transformation")

        # ################### Filtering of the Fourier Spectrum #############
        # ## name filter boundaries like in Kiladis Fortran Code
        f1 = freq_min
        f2 = freq_min
        f3 = freq_max
        f4 = freq_max
        s1 = wn_min
        s2 = wn_max
        s3 = wn_min
        s4 = wn_max

        # ### Very similar to original Kiladis Code
        fourier_fft_filtered = fourier_fft
        count = 0
        for i_f in range(0, int(nt / 2) + 1):
            for i_wn in range(0, nl):
                ff = freq_axis[i_f]
                ss = wn_axis[i_wn]
                if ((ff >= ((ss * (f1 - f2) + f2 * s1 - f1 * s2) / (s1 - s2))) and
                        (ff <= ((ss * (f3 - f4) + f4 * s3 - f3 * s4) / (s3 - s4))) and
                        (ss >= ((ff * (s3 - s1) - f1 * s3 + f3 * s1) / (f3 - f1))) and
                        (ss <= ((ff * (s4 - s2) - f2 * s4 + f4 * s2) / (f4 - f2)))):
                    count = count + 1
                else:
                    fourier_fft_filtered[i_f, i_wn] = 0
                    if i_wn == 0 and i_f == 0:
                        pass
                    elif i_wn == 0:
                        ind_f = nt - i_f
                        if (ind_f < nt):
                            fourier_fft_filtered[ind_f, i_wn] = 0
                    elif i_f == 0:
                        ind_wn = nl - i_wn
                        if ind_wn < nl:
                            fourier_fft_filtered[i_f, ind_wn] = 0
                    else:
                        ind_f = nt - i_f
                        ind_wn = nl - i_wn
                        if ind_f < nt and ind_wn < nl:
                            fourier_fft_filtered[ind_f, ind_wn] = 0
        if save_debug:
            self.DebugFilteredFourierSpectrum = np.copy(fourier_fft_filtered)
            self.DebugNoElementsInFilteredSpectrum = count

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_FilteredSpectrum", clear=True)
            plt.contourf(wn_axis, freq_axis, np.squeeze(fourier_fft_filtered))
            plt.colorbar()
            plt.title("Filtered Fourier Transformation")
            print("Number of elements in filtered spectrum: ", count)

        # ############################ FFT Backward transformation ############
        # #reorder spectrum back from kiladis ordering to python ordering
        fourier_fft_filtered = np.roll(fourier_fft_filtered, -int(nt / 2), axis=0)
        fourier_fft_filtered = np.roll(fourier_fft_filtered, -int(nl / 2), axis=1)
        fourier_fft_filtered = np.fft.ifftshift(fourier_fft_filtered, axes=(0, 1))
        filtered_olr = np.fft.ifft2(fourier_fft_filtered)
        filtered_olr = np.real(filtered_olr)

        # ############################# remove zero padding elements ##########
        result = filtered_olr[0:orig_nt, :]

        if save_debug:
            self.DebugFilterOLR = np.copy(result)

        if do_plot:
            fig = plt.figure("perform2dimSpectralSmoothing_4", clear=True)
            plt.contourf(result)
            plt.colorbar()
            plt.title("Filtered Data")
        # ToDo: Make sure that result is real
        return result
