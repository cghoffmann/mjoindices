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
This major module provides the execution of the algorithm of the OMI calculation.

First, the algorithm can be used to compute the empirical orthogonal functions (EOFs), which serve as a new basis for
the OLR data. Second, the time-dependent coefficients of the OLR maps w.r.t the EOFs are computed. These coefficients
are called the principal components (PCs).

According to the OMI algorithm, the EOFs have to be computed for each day of the year (DOY).

Basically, only OLR data on a suitable spatial grid is needed. With that, the the OMI EOFs and afterwards the PCs are
computed using the functions :func:`calc_eofs_from_olr` and :func:`calculate_pcs_from_olr`, respectively.

The complete algorithm is described in Kiladis, G.N., J. Dias, K.H. Straub, M.C. Wheeler, S.N. Tulich, K. Kikuchi, K.M.
Weickmann, and M.J. Ventrice, 2014: A Comparison of OLR and Circulation-Based Indices for Tracking the MJO.
Mon. Wea. Rev., 142, 1697–1715, https://doi.org/10.1175/MWR-D-13-00301.1

An additional post-processing step can be run after the EOFs have been calculated that reduces noise and potential 
degeneracy issues. This algorithm is described in  Weidman, S., Kleiner, N., & Kuang, Z. (2022). A rotation procedure 
to improve seasonally varying empirical orthogonal function bases for MJO indices. Geophysical Research Letters, 
49, e2022GL099998. https://doi.org/10.1029/2022GL099998 

"""

from pathlib import Path
from typing import Tuple

import numpy as np
import warnings
import importlib

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.olr_handling as olr
import mjoindices.principal_components as pc
import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindices.omi.quick_temporal_filter as qfilter
import mjoindices.omi.postprocessing_original_kiladis2014 as pp_kil2014
import mjoindices.omi.postprocessing_rotation_approach as pp_rotation
import mjoindices.tools as tools

eofs_spec = importlib.util.find_spec("eofs")
eofs_package_available = eofs_spec is not None
if eofs_package_available:
    import eofs.standard as eofs_package


# #################EOF calculation

def calc_eofs_from_olr(olrdata: olr.OLRData,
                       implementation: str = "internal",
                       leap_year_treatment: str = "original",
                       eofs_postprocessing_type:str ="kiladis2014",
                       eofs_postprocessing_params:dict=None) -> eof.EOFDataForAllDOYs:
    """
    One major function of this module. It performs the complete OMI EOF computation.

    This function executes consistently the preprocessing (filtering), the actual EOF analysis, and the postprocessing.

    :param olrdata: The OLR dataset, from which OMI should be calculated. Note that OLR values are assumed to be given
        in positive values. The spatial grid of the OLR datasets defines also the spatial grid of the complete OMI
        calculation.
    :param implementation: See :meth:`calc_eofs_from_preprocessed_olr`.
    :param leap_year_treatment: Either "original", "strict" or "no_leap_years".
        "original" will be as close to the original version of Kiladis (2014) as possible.
        "strict" (not recommended) will treat leap years somewhat more strictly, which might, however, cause the results to deviate from the original. 
        See also description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.
        "no_leap_years" will act as if there are no leap years in the dataset (365 days consistently), which might be useful for modeled data.
    :param eofs_postprocessing_type: Different approaches of the post-processing of the EOFs are available: "kiladis2014"
    for the original post-processing described in Kiladis, 2014. "eof_rotation" for the post-processing rotation algorithm 
    described in Weidman, 2022; and None for no post-processing.
    :param eofs_postprocessing_params: dict of specific parameters, which will be passed as keyword parameters to the
    respective post-processing function (:meth:`mjoindices.omi.postprocessing_original_kiladis2014.post_process_eofs_original_kiladis_approach` 
    or :meth:`mjoindices.omi.postprocessing_rotation_approach.post_process_eofs_rotation`)
    :param eofs_postprocessing_params: dict of specific parameters, which will be passed as keyword parameters to the respective post-processing function
    :return: The computed EOFs.
    """

    preprocessed_olr = preprocess_olr(olrdata)
    raw_eofs = calc_eofs_from_preprocessed_olr(preprocessed_olr, implementation=implementation,
                                               leap_year_treatment=leap_year_treatment)
    result = initiate_eof_post_processing(raw_eofs, eofs_postprocessing_type, eofs_postprocessing_params)
    return result


def initiate_eof_post_processing(raw_eofs: eof.EOFDataForAllDOYs,
                                 eofs_postprocessing_type: str = "kiladis2014",
                                 eofs_postprocessing_params: dict = None) -> eof.EOFDataForAllDOYs:
    if eofs_postprocessing_type is None:
        result = raw_eofs
    elif eofs_postprocessing_type == "kiladis2014":
        if eofs_postprocessing_params is None:
            eofs_postprocessing_params = {"sign_doy1reference": True,
                                          "interpolate_eofs": False,
                                          "interpolation_start_doy": 293,
                                          "interpolation_end_doy": 316}
        result = pp_kil2014.post_process_eofs_original_kiladis_approach(raw_eofs, **eofs_postprocessing_params)
    elif eofs_postprocessing_type == "eof_rotation":
        if eofs_postprocessing_params is None:
            eofs_postprocessing_params = {"sign_doy1reference": True}
        result = pp_rotation.post_process_eofs_rotation(raw_eofs, **eofs_postprocessing_params)

    else:
        raise ValueError("EOF post-processing type unknown.")
    return result

def preprocess_olr(olrdata: olr.OLRData) -> olr.OLRData:
    """
    Performs the preprocessing of an OLR dataset to make it suitable for the EOF analysis.

    This is actually a major step of the OMI algorithm and includes the Wheeler-Kiladis-Filtering.

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: The OLR dataset, which should be preprocessed. Note that OLR values are assumed to be given in
        positive values.

    :return: The filtered OLR dataset.
    """
    if np.mean(olrdata.olr) < 0:
        warnings.warn("OLR data apparently given in negative numbers. Here it is assumed that OLR is positive.")
    olrdata_filtered = wkfilter.filter_olr_for_mjo_eof_calculation(olrdata)
    return olrdata_filtered


def calc_eofs_from_preprocessed_olr(olrdata: olr.OLRData, implementation: str = "internal",
                                    leap_year_treatment: str = "original") -> eof.EOFDataForAllDOYs:
    """
    Calculates a series of EOF pairs: one pair for each DOY.

    This is based on already preprocessed OLR. Note that it is recommended to use the function
    :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: the preprocessed OLR data, from which the EOFs are calculated.
    :param implementation: Two options are available: First, "internal": uses the internal implementation of the EOF
        approach. Second, "eofs_package": Uses the implementation of the external package :py:mod:`eofs`.
    :param leap_year_treatment: Either "original", "strict" or "no_leap_years".
        "original" will be as close to the original version of Kiladis (2014) as possible.
        "strict" (not recommended) will treat leap years somewhat more strictly, which might, however, cause the results to deviate from the original. 
        See also description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.
        "no_leap_years" will act as if there are no leap years in the dataset (365 days consistently), which might be useful for modeled data.

    :return: A pair of EOFs for each DOY. This series of EOFs has probably still to be postprocessed.
    """
    if implementation == "eofs_package" and not eofs_package_available:
        raise ValueError("Selected calculation with external eofs package, but package not available. Use "
                         "internal implementation or install eofs package")
    no_leap_years = False
    if leap_year_treatment == "no_leap_years":
        no_leap_years = True
    doys = tools.doy_list(no_leap_years)
    eofs = []
    for doy in doys:
        print("Calculating EOFs for DOY %i" % doy)
        if (implementation == "eofs_package"):
            singleeof = calc_eofs_for_doy_using_eofs_package(olrdata, doy, leap_year_treatment=leap_year_treatment)
        else:
            singleeof = calc_eofs_for_doy(olrdata, doy, leap_year_treatment=leap_year_treatment)
        eofs.append(singleeof)
    return eof.EOFDataForAllDOYs(eofs, no_leap_years)


def calc_eofs_for_doy(olrdata: olr.OLRData, doy: int, leap_year_treatment: str = "original") -> eof.EOFData:
    """
    Calculates a pair of EOFs for a particular DOY.

    An explicit internal implementation of the EOF approach is used.

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: The filtered OLR data to calculate the EOFs from.
    :param doy: The DOY for which the EOFs are calculated.
    :param leap_year_treatment: Either "original", "strict" or "no_leap_years".
        "original" will be as close to the original version of Kiladis (2014) as possible.
        "strict" (not recommended) will treat leap years somewhat more strictly, which might, however, cause the results to deviate from the original. 
        See also description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.
        "no_leap_years" will act as if there are no leap years in the dataset (365 days consistently), which might be useful for modeled data.

    :return: An object containing the pair of EOFs together with diagnostic values.

    .. seealso:: :meth:`calc_eofs_for_doy_using_eofs_package`

    """
    nlat = olrdata.lat.size
    nlong = olrdata.long.size
    olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60,
                                                                leap_year_treatment=leap_year_treatment)
    N = olr_maps_for_doy.shape[0]
    M = nlat * nlong
    F = np.reshape(olr_maps_for_doy, [N, M]).T  # vector: only one dimension. Length given by original longitude and latitude bins
    R = np.matmul(F, F.T) / N  # in some references, it is divided by (N-1), however, we follow Kutzbach (1967), in which it is only divided by N. In any case, the result should not differ much.
    if not np.allclose(R, R.T):
        warnings.warn("Covariance matrix is not symmetric within defined tolerance")
    L, E = np.linalg.eig(R)

    if not np.allclose(np.imag(L), 0.):
        warnings.warn("Imaginary part of at least one Eigenvalue greater than expected. Neglecting it anyway")
    L = np.real(L)
    order = (np.flip(L.argsort(), axis=None))
    L = L[order]
    total_var = np.sum(L)
    explainedVariances = L / total_var  # See Kutzbach (1967), Eq 12

    E = E[:, order]
    if not np.allclose(np.imag(E[:, 0:2]), 0.):
        warnings.warn("Imaginary part of one of the first two Eigenvectors greater than expected. Neglecting it anyway")
    E = np.real(E)
    eof1_vec = np.squeeze(E[:, 0])
    eof2_vec = np.squeeze(E[:, 1])

    return eof.EOFData(olrdata.lat, olrdata.long, eof1_vec, eof2_vec,
                       eigenvalues=L, explained_variances=explainedVariances, no_observations=N)


def calc_eofs_for_doy_using_eofs_package(olrdata: olr.OLRData, doy: int,
                                         leap_year_treatment: str = "original") -> eof.EOFData:
    """
    Calculates a pair of EOFs for a particular DOY.

    The external package :py:mod:`eofs` is used for the core calculation

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: The filtered OLR data to calculate the EOFs from.
    :param doy: The DOY for which the EOFs are calculated.
    :param leap_year_treatment: Either "original", "strict" or "no_leap_years".
        "original" will be as close to the original version of Kiladis (2014) as possible.
        "strict" (not recommended) will treat leap years somewhat more strictly, which might, however, cause the results to deviate from the original. 
        See also description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.
        "no_leap_years" will act as if there are no leap years in the dataset (365 days consistently), which might be useful for modeled data.
    :return: An object containing the pair of EOFs together with diagnostic values.

    .. seealso:: :meth:`calc_eofs_for_doy`

    """
    if eofs_package_available:
        nlat = olrdata.lat.size
        nlong = olrdata.long.size
        olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60,
                                                                    leap_year_treatment=leap_year_treatment)

        ntime = olr_maps_for_doy.shape[0]
        N = ntime
        M = nlat * nlong
        F = np.reshape(olr_maps_for_doy,
                       [N, M]).T  # vector: only one dimension. Length given by original longitude and latitude bins
        solver = eofs_package.Eof(F.T)
        # Todo: Should we care about complex values here?
        eofs = solver.eofs(neofs=2)
        explainedVariances = solver.varianceFraction()
        L = solver.eigenvalues()

        if L.size < M:
            # This usually happens if the covariance matrix did not have full rank (e.g. N<M). Missing Eigenvalues
            # are 0 and can be simply padded here
            L = np.pad(L, (0, M - L.size), 'constant', constant_values=(0, 0))
            explainedVariances = np.pad(explainedVariances, (0, M - explainedVariances.size), 'constant',
                                        constant_values=(0, 0))
        return eof.EOFData(olrdata.lat, olrdata.long, np.squeeze(eofs[0, :]), np.squeeze(eofs[1, :]),
                           eigenvalues=L, explained_variances=explainedVariances, no_observations=N)
    else:
        raise ModuleNotFoundError("eofs")

# #################PC Calculation

def calculate_pcs_from_olr(olrdata: olr.OLRData,
                           eofdata: eof.EOFDataForAllDOYs,
                           period_start: np.datetime64,
                           period_end: np.datetime64,
                           use_quick_temporal_filter=False) -> pc.PCData:
    """
    This major function computes PCs according to the OMI algorithm based on given OLR data and previously calculated
    EOFs.

    :param olrdata: The OLR dataset. The spatial grid must fit to that of the EOFs
    :param eofdata: The previously calculated DOY-dependent EOFs.
    :param period_start: the beginning of the period, for which the PCs should be calculated.
    :param period_end: the ending of the period, for which the PCs should be calculated.
    :param use_quick_temporal_filter: There are two implementations of the temporal filtering: First, the original
        Wheeler-Kiladis-Filter, which is closer to the original implementation while being slower (because it is based
        on a 2-dim FFT) or a 1-dim FFT Filter. Setting this parameter to True uses the quicker 1-dim implementation. The
        results are quite similar.

    :return: The PC time series. Normalized by the full PC time series
    """
    resticted_olr_data = olr.restrict_time_coverage(olrdata, period_start, period_end)
    resampled_olr_data = olr.interpolate_spatial_grid(resticted_olr_data, eofdata.lat, eofdata.long)
    if use_quick_temporal_filter:
        filtered_olr_data = qfilter.filter_olr_for_mjo_pc_calculation_1d_spectral_smoothing(resampled_olr_data)
    else:
        filtered_olr_data = wkfilter.filter_olr_for_mjo_pc_calculation(resampled_olr_data)
    raw_pcs = regress_3dim_data_onto_eofs(filtered_olr_data, eofdata)
    normalization_factor = 1 / np.std(raw_pcs.pc1)
    pc1 = np.multiply(raw_pcs.pc1, normalization_factor)
    pc2 = np.multiply(raw_pcs.pc2, normalization_factor)
    return pc.PCData(raw_pcs.time, pc1, pc2)


def calculate_pcs_from_olr_original_conditions(olrdata: olr.OLRData,
                                               original_eof_dirname: Path,
                                               use_quick_temporal_filter=False) -> pc.PCData:
    """
    Calculates the OMI PCs for the original period using the original dataset (which has, however, to be provided
    by the user himself).

    :param olrdata: The original OLR data, which can be downloaded from
        ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc
    :param original_eof_dirname: Path to the original EOFs, which can be downloaded
        from ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof1/ and ftp://ftp.cdc.noaa.gov/Datasets.other/MJO/eof2/ .
        The contents of both remote directories should again be placed into sub directories *eof1* and *eof2*
    :param use_quick_temporal_filter: see :func:`calculate_pcs_from_olr`

    :return: The PCs, which should be similar to the original ones.
    """
    period_start = np.datetime64("1979-01-01")
    period_end = np.datetime64("2018-08-28")
    eofs = eof.load_all_original_eofs_from_directory(original_eof_dirname)
    return calculate_pcs_from_olr(olrdata,
                                  eofs,
                                  period_start,
                                  period_end,
                                  use_quick_temporal_filter)


def regress_3dim_data_onto_eofs(data: object, eofdata: eof.EOFDataForAllDOYs) -> pc.PCData:
    """
    Finds time-dependent coefficients w.r.t the DOY-dependent EOF basis for time-dependent spatially resolved data.

    I.e. it finds the PCs for temporally resolved OLR data. But the function can also be used for other datasets,
    as long as those datasets have the same structure like the the class :class:`mjoindices.olr_handling.OLRData`.

    :param data: The data, for which the coefficients are sought. Should be an object of class
        :class:`mjoindices.olr_handling.OLRData` or of similar structure.
    :param eofdata: The DOY-dependent pairs of EOFs, like computed by, e.g., :func:`calc_eofs_from_olr`

    :return: The time-dependent PCs as :class:`mjoindices.principal_components.PCData`
    """
    if not np.all(data.lat == eofdata.lat):
        raise ValueError("Latitude grid of EOFs and OLR is not equal.")
    if not np.all(data.long == eofdata.long):
        raise ValueError("Longitude grid of EOFs and OLR is not equal.")
    pc1 = np.empty(data.time.size)
    pc2 = np.empty(data.time.size)

    for idx, val in enumerate(data.time):
        day = val
        olr_singleday = data.get_olr_for_date(day)
        doy = tools.calc_day_of_year(day, eofdata.no_leap_years)
        (pc1_single, pc2_single) = regress_vector_onto_eofs(
            eofdata.eofdata_for_doy(doy).reshape_to_vector(olr_singleday),
            eofdata.eof1vector_for_doy(doy),
            eofdata.eof2vector_for_doy(doy))
        pc1[idx] = pc1_single
        pc2[idx] = pc2_single
    return pc.PCData(data.time, pc1, pc2)


def regress_vector_onto_eofs(vector: np.ndarray, eof1: np.ndarray, eof2: np.ndarray) -> Tuple[float, float]:
    """
    Helper method that finds the coefficients of the given vector with respect to the given basis of 2 EOFs.

    The computed coefficients are the PCs in the terminology of the EOF analysis.

    :param vector: The vector for which the coefficients in the EOF basis should be found.
    :param eof1: EOF basis vector 1.
    :param eof2: EOF basis vector 2.

    :return: The two PCs.
    """
    eof_mat = np.array([eof1, eof2]).T

    # Alternative implementation 1:
    x = np.linalg.lstsq(eof_mat, vector, rcond=-1)
    pc1, pc2 = x[0]
    return pc1, pc2

    # Alternative implementation 2:
    # pseudo_inverse = np.linalg.pinv(eof_mat)
    # pcs = np.matmul(pseudo_inverse, vector)
    # return pcs[0], pcs[1]

