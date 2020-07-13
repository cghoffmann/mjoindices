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

"""

from pathlib import Path
from typing import Tuple
import os.path
import inspect

import numpy as np
import warnings
import importlib
import scipy.interpolate

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.olr_handling as olr
import mjoindices.principal_components as pc
import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindices.omi.quick_temporal_filter as qfilter
import mjoindices.tools as tools

eofs_spec = importlib.util.find_spec("eofs")
eofs_package_available = eofs_spec is not None
if eofs_package_available:
    import eofs.standard as eofs_package


# #################EOF calculation

def calc_eofs_from_olr(olrdata: olr.OLRData, implementation: str = "internal", sign_doy1reference: bool = True,
                       interpolate_eofs: bool = False, interpolation_start_doy: int = 293,
                       interpolation_end_doy: int = 316, strict_leap_year_treatment: bool = False) -> eof.EOFDataForAllDOYs:
    """
    One major function of this module. It performs the complete OMI EOF computation.

    This function executes consistently the preprocessing (filtering), the actual EOF analysis, and the postprocessing.

    :param olrdata: The OLR dataset, from which OMI should be calculated. Note that OLR values are assumed to be given
        in positive values. The spatial grid of the OLR datasets defines also the spatial grid of the complete OMI
        calculation.
    :param implementation: See :meth:`calc_eofs_from_preprocessed_olr`.
    :param sign_doy1reference: See :meth:`correct_spontaneous_sign_changes_in_eof_series`.
    :param interpolate_eofs: If true, the EOF sub-series between the given DOYs will be interpolated.
    :param interpolation_start_doy: See description of :meth:`interpolate_eofs_between_doys`.
    :param interpolation_end_doy: See description of :meth:`interpolate_eofs_between_doys`.
    :param strict_leap_year_treatment: See description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.
    :return:
    """
    preprocessed_olr = preprocess_olr(olrdata)
    raw_eofs = calc_eofs_from_preprocessed_olr(preprocessed_olr, implementation=implementation, strict_leap_year_treatment=strict_leap_year_treatment)
    result = post_process_eofs(raw_eofs, sign_doy1reference=sign_doy1reference, interpolate_eofs=interpolate_eofs,
                               interpolation_start_doy=interpolation_start_doy,
                               interpolation_end_doy=interpolation_end_doy)
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
                                    strict_leap_year_treatment: bool = False) -> eof.EOFDataForAllDOYs:
    """
    Calculates a series of EOF pairs: one pair for each DOY.

    This is based on already preprocessed OLR. Note that it is recommended to use the function
    :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: the preprocessed OLR data, from which the EOFs are calculated.
    :param implementation: Two options are available: First, "internal": uses the internal implementation of the EOF
        approach. Second, "eofs_package": Uses the implementation of the external package :py:mod:`eofs`.
    :param strict_leap_year_treatment: see description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.

    :return: A pair of EOFs for each DOY. This series of EOFs has probably still to be postprocessed.
    """
    if implementation == "eofs_package" and not eofs_package_available:
        raise ValueError("Selected calculation with external eofs package, but package not available. Use "
                         "internal implementation or install eofs package")
    doys = tools.doy_list()
    eofs = []
    for doy in doys:
        print("Calculating EOFs for DOY %i" % doy)
        if (implementation == "eofs_package"):
            singleeof = calc_eofs_for_doy_using_eofs_package(olrdata, doy,
                                                             strict_leap_year_treatment=strict_leap_year_treatment)
        else:
            singleeof = calc_eofs_for_doy(olrdata, doy, strict_leap_year_treatment=strict_leap_year_treatment)
        eofs.append(singleeof)
    return eof.EOFDataForAllDOYs(eofs)


def post_process_eofs(eofdata: eof.EOFDataForAllDOYs, sign_doy1reference: bool = True,
                      interpolate_eofs: bool = False, interpolation_start_doy: int = 293,
                      interpolation_end_doy: int = 316) -> eof.EOFDataForAllDOYs:
    """
    Post processes a series of EOF pairs for all DOYs.

    Postprocessing includes an alignment of EOF signs and an interpolation of the EOF functions in a given DOY
    window. Both steps are part of the original OMI algorithm described by Kiladis (2014).

    See documentation of  the methods :meth:`correct_spontaneous_sign_changes_in_eof_series` and
    :meth:`interpolate_eofs_between_doys` for further information.

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param eofdata: The EOF series, which should be post processed.
    :param sign_doy1reference: See description of :meth:`correct_spontaneous_sign_changes_in_eof_series`.
    :param interpolate_eofs: If true, the EOF sub-series between the given DOYs will be interpolated.
    :param interpolation_start_doy: See description of :meth:`interpolate_eofs_between_doys`.
    :param interpolation_end_doy: See description of :meth:`interpolate_eofs_between_doys`.

    :return: the postprocessed series of EOFs
    """
    pp_eofs = correct_spontaneous_sign_changes_in_eof_series(eofdata, doy1reference=sign_doy1reference)
    if interpolate_eofs:
        pp_eofs = interpolate_eofs_between_doys(pp_eofs, start_doy=interpolation_start_doy,
                                                end_doy=interpolation_end_doy)
    return pp_eofs


def calc_eofs_for_doy(olrdata: olr.OLRData, doy: int, strict_leap_year_treatment: bool = False) -> eof.EOFData:
    """
    Calculates a pair of EOFs for a particular DOY.

    An explicit internal implementation of the EOF approach is used.

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: The filtered OLR data to calculate the EOFs from.
    :param doy: The DOY for which the EOFs are calculated.
    :param strict_leap_year_treatment: see description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.

    :return: An object containing the pair of EOFs together with diagnostic values.

    .. seealso:: :meth:`calc_eofs_for_doy_using_eofs_package`

    """
    nlat = olrdata.lat.size
    nlong = olrdata.long.size
    olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60,
                                                                strict_leap_year_treatment=strict_leap_year_treatment)
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
                                         strict_leap_year_treatment: bool = False) -> eof.EOFData:
    """
    Calculates a pair of EOFs for a particular DOY.

    The external package :py:mod:`eofs` is used for the core calculation

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr` to cover the complete algorithm.

    :param olrdata: The filtered OLR data to calculate the EOFs from.
    :param doy: The DOY for which the EOFs are calculated.
    :param strict_leap_year_treatment: see description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.

    :return: An object containing the pair of EOFs together with diagnostic values.

    .. seealso:: :meth:`calc_eofs_for_doy`

    """
    if eofs_package_available:
        nlat = olrdata.lat.size
        nlong = olrdata.long.size
        olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60,
                                                                    strict_leap_year_treatment=strict_leap_year_treatment)

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


def correct_spontaneous_sign_changes_in_eof_series(eofs: eof.EOFDataForAllDOYs,
                                                   doy1reference: bool = False) -> eof.EOFDataForAllDOYs:
    """
    Switches the signs of all pairs of EOFs (for all DOYs) if necessary, so that the signs are consistent for all DOYs.

    Note that the sign of the EOFs is not uniquely defined by the PCA. Hence, the sign may jump from one DOY to another,
    which can be improved using this function. As long as this step is performed before computing the PCs, it will not
    change the overall result.

    Generally, the sign of the EOFs for a specific DOY is changed if it differs from the sign of the EOF for the previous
    DOY. The EOFs for DOY 1 are by default aligned with the original calculation by Kiladis (2014), resulting in a
    an EOF series, which is totally comparable to the original Kiladis (2014) calculation. This can be switched off.

    :param eofs: The EOF series for which the signs should be aligned.
    :param doy1reference: If true, the EOFs of DOY 1 are aligned w.r.t to the original Kiladis (2014) calculation.

    :return: The EOFs with aligned signs.
    """
    switched_eofs = []
    if doy1reference is True:
        reference_path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) / "sign_reference"
        reference_eofs = eof.load_original_eofs_for_doy(reference_path, 1)
        if not reference_eofs.lat.size == eofs.lat.size \
                or not reference_eofs.long.size == eofs.long.size \
                or not np.all(reference_eofs.lat == eofs.lat) \
                or not np.all(reference_eofs.long == eofs.long):
            warnings.warn("References for the sign of the EOFs for DOY1 have to be interpolated to spatial grid of the"
                          " target EOFs. Treat results with caution.")
            f1 = scipy.interpolate.interp2d(reference_eofs.long, reference_eofs.lat, reference_eofs.eof1map,
                                            kind='linear')
            eof1map_interpol = f1(eofs.long, eofs.lat)
            f2 = scipy.interpolate.interp2d(reference_eofs.long, reference_eofs.lat, reference_eofs.eof2map,
                                            kind='linear')
            eof2map_interpol = f2(eofs.long, eofs.lat)
            reference_eofs = eof.EOFData(eofs.lat, eofs.long, eof1map_interpol, eof2map_interpol)
        corrected_doy1 = _correct_spontaneous_sign_change_of_individual_eof(reference_eofs, eofs.eofdata_for_doy(1))
    else:
        corrected_doy1 = eofs.eofdata_for_doy(1)
    switched_eofs.append(corrected_doy1)
    previous_eof = corrected_doy1
    for doy in tools.doy_list()[1:]:
        corrected_eof = _correct_spontaneous_sign_change_of_individual_eof(previous_eof, eofs.eofdata_for_doy(doy))
        switched_eofs.append(corrected_eof)
        previous_eof = corrected_eof
    return eof.EOFDataForAllDOYs(switched_eofs)


def _correct_spontaneous_sign_change_of_individual_eof(reference: eof.EOFData, target=eof.EOFData) -> eof.EOFData:
    """
    Switches the sign of a particular pair of EOFs (for a particular DOY) if necessary, so that is aligned with the
    reference.

    Note that the signs of the EOFs is not uniquely defined by the PCA. Hence, the sign may jump from one DOY to another,
    which can be improved using this function. As long as this step is performed before computing the PCs, it will not
    change the overall result.

    :param reference: The reference-EOFs. This is usually the EOF pair of the previous DOY.
    :param target: The EOFs of which the signs should be switched

    :return: The target EOFs with aligned signs.
    """
    if (np.mean(np.abs(target.eof1vector + reference.eof1vector))
            < np.mean(np.abs(
                target.eof1vector - reference.eof1vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof1_switched = -1 * target.eof1vector
    else:
        eof1_switched = target.eof1vector
    if (np.mean(np.abs(target.eof2vector + reference.eof2vector))
            < np.mean(np.abs(
                target.eof2vector - reference.eof2vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof2_switched = -1 * target.eof2vector
    else:
        eof2_switched = target.eof2vector
    return eof.EOFData(target.lat,
                       target.long,
                       eof1_switched,
                       eof2_switched,
                       eigenvalues=target.eigenvalues,
                       explained_variances=target.explained_variances,
                       no_observations=target.no_observations)


def interpolate_eofs_between_doys(eofs: eof.EOFDataForAllDOYs, start_doy: int = 293,
                                  end_doy: int = 316) -> eof.EOFDataForAllDOYs:
    """
    Replaces the EOF1 and EOF2 functions between 2 DOYs by a linear interpolation between these 2 DOYs.

    This should only rarely be used and has only been implemented to closely reproduce the original OMI values. There,
    the EOFs have also been replaced by an interpolation according to Kiladis (2014). However, the period stated in
    Kiladis (2014) from 1 November to 8 November is too short. The authors have confirmed that the right
    interpolation period is from DOY 294 to DOY 315, which is used here as default value.

    ATTENTION: The corresponding statistical values (e.g., the explained variances) are not changed by this routine.
    So these values further on represent the original results of the PCA also for the interpolated EOFs.

    :param eofs: The complete EOF series, in which the interpolation takes place.
    :param start_doy: The DOY, which is used as the first point of the interpolation (i.e. start_doy + 1 is the first
        element, which will be replaced by the interpolation.
    :param end_doy:  The DOY, which is used as the last point of the interpolation (i.e. end_doy - 1 is the last
        element, which will be replaced by the interpolation.

    :return: The complete EOF series with the interpolated values.
    """
    doys = tools.doy_list()
    start_idx = start_doy - 1
    end_idx = end_doy - 1
    eof_len = eofs.lat.size * eofs.long.size
    eofs1 = np.empty((doys.size, eof_len))
    eofs2 = np.empty((doys.size, eof_len))
    # Todo: Maybe this could be solved more efficiently
    # by using internal numpy functions for multidimenasional operations
    for (idx, doy) in enumerate(doys):
        eofs1[idx, :] = eofs.eof1vector_for_doy(doy)
        eofs2[idx, :] = eofs.eof2vector_for_doy(doy)

    for i in range(0, eof_len):
        eofs1[start_idx + 1:end_idx - 1, i] = np.interp(doys[start_idx + 1:end_idx - 1],
                                                        [doys[start_idx], doys[end_idx]],
                                                        [eofs1[start_idx, i], eofs1[end_idx, i]])
        eofs2[start_idx + 1:end_idx - 1, i] = np.interp(doys[start_idx + 1:end_idx - 1],
                                                        [doys[start_idx], doys[end_idx]],
                                                        [eofs2[start_idx, i], eofs2[end_idx, i]])
    interpolated_eofs = []
    for (idx, doy) in enumerate(doys):
        orig_eof = eofs.eofdata_for_doy(doy)
        interpolated_eofs.append(eof.EOFData(orig_eof.lat, orig_eof.long, np.squeeze(eofs1[idx, :]),
                                             np.squeeze(eofs2[idx, :]),
                                             explained_variances=orig_eof.explained_variances,
                                             eigenvalues=orig_eof.eigenvalues, no_observations=orig_eof.no_observations)
                                 )
    return eof.EOFDataForAllDOYs(interpolated_eofs)


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

    :return: The PC time series.
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
        doy = tools.calc_day_of_year(day)
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
