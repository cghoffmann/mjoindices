# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:39:02 2019

@author: ch
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import warnings
import importlib
eofs_spec = importlib.util.find_spec("eofs")
eofs_package_available = eofs_spec is not None
if eofs_package_available:
    import eofs.standard as eofs_package
import mjoindex_omi.empirical_orthogonal_functions as eof
import mjoindex_omi.olr_handling as olr
import mjoindex_omi.principal_components as pc
import mjoindex_omi.tools as tools
import mjoindex_omi.wheeler_kiladis_mjo_filter as wkfilter



# #################EOF calculation

def calc_eofs_from_olr(olrdata: olr.OLRData) -> eof.EOFDataForAllDOYs:
    preprocessed_olr = preprocess_olr(olrdata)
    raw_eofs = calc_eofs_from_preprocessed_olr(preprocessed_olr)
    result = post_process_eofs(raw_eofs)
    return result


def preprocess_olr(olrdata: olr.OLRData) -> olr.OLRData:
    if np.mean(olrdata.olr) < 0:
        warnings.warn("OLR data apparently given in negative numbers. Here it is assumed that OLR is positive.")
    # interpolate on particular spatial grid?
    olrdata_filtered = wkfilter.filterOLRForMJO_EOF_Calculation(olrdata)
    return olrdata_filtered


def calc_eofs_from_preprocessed_olr(olrdata: olr.OLRData, implementation="internal") -> eof.EOFDataForAllDOYs:
    if implementation == "eofs_package" and not eofs_package_available:
        raise AttributeError("Selected calculation with external eofs package, but package not available. Use "
                             "internal implementation or install eofs package")
    doys = eof.doy_list()
    eofs = []
    for doy in doys:
        print("Calculationg for DOY %i" % doy)
        if (implementation == "eofs_package"):
            singleeof = calc_eofs_for_doy_using_eofs_package(olrdata, doy)
        else:
            singleeof = calc_eofs_for_doy(olrdata, doy)
        eofs.append(singleeof)
    return eof.EOFDataForAllDOYs(eofs)


def post_process_eofs(eofdata: eof.EOFDataForAllDOYs, sign_doy1reference: eof.EOFData = None, interpolate_eofs = False, interpolation_start_doy: int = 304, interpolation_end_doy: int = 313) -> eof.EOFDataForAllDOYs:
    pp_eofs = correct_spontaneous_sign_changes_in_eof_series(eofdata, doy1reference=sign_doy1reference)
    if interpolate_eofs:
        pp_eofs = interpolate_eofs_between_doys(pp_eofs, start_doy=interpolation_start_doy, end_doy=interpolation_end_doy)
    return pp_eofs


def calc_eofs_for_doy(olrdata: olr.OLRData, doy: int) -> eof.EOFData:
    nlat = olrdata.lat.size
    nlong = olrdata.long.size
    olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60)
    ntime = olr_maps_for_doy.shape[0]
    # doyOLR_3dim = self.__olrdata_filtered.returnAverageOLRForIndividualDOY(doy,window_length=60)
    # doyOLR_3dim = np.mean(doyOLR_3dim,axis=0)
    # FIXME: remove weights code if not needed
    # weights= np.empty_like(olr_maps_for_doy)
    # for i_t in range(0,ntime):
    #    for i_lon in range (0, nlong):
    #        for i_lat in range (0,nlat):
    #            weights[i_t, i_lat, i_lon] = np.cos(np.deg2rad(olrdata.lat[i_lat]))
    #olr_maps_for_doy = olr_maps_for_doy * weights


    N = ntime
    M = nlat * nlong
    # FIXME: The sollowing reshape iincluding the transpose etc. Is this correct? Rule: If original Kiladis Grid is used, the EOFs must be compatible to the saved ones.
    F = np.reshape(olr_maps_for_doy,
                   [N, M]).T  # vector: only one dimension. Length given by original longitude and latitude bins
    R = np.matmul(F, F.T) / N  # FIXME ( has to  N-1 ? See own PCA presentation
    if not np.allclose(R, R.T):
        warnings.warn("Covariance matrix is not symmetric within defined tolerance")
    L, E = np.linalg.eig(R)
    if not np.allclose(np.imag(L), 0.):
        warnings.warn("Imaginary part of at least one Eigenvalue greater than expected. Neglecting it anyway")
    L = np.real(L)
    order = (np.flip(L.argsort()))
    L = L[order]
    E = E[:, order]
    if not np.allclose(np.imag(E[:, 0:2]), 0.):
        warnings.warn("Imaginary part of one of the first two Eigenvectors greater than expected. Neglecting it anyway")
    E = np.real(E)

    total_var = np.sum(L)
    explainedVariances = L / total_var  # See Kutzbach Eq 12
    # print(order)
    #order = order[:2]
    #L = L[order]

    # print("total var:", total_var)
    # print("L",L)
    #E = E[:, order]  # is that right?
    eof1_vec = np.squeeze(E[:, 0])
    eof2_vec = np.squeeze(E[:, 1])
    return eof.EOFData(olrdata.lat, olrdata.long, eof1_vec, eof2_vec,
                       eigenvalues=L, explained_variances=explainedVariances, no_observations=N)


def calc_eofs_for_doy_using_eofs_package(olrdata: olr.OLRData, doy: int) -> eof.EOFData:
    if eofs_package_available:
        nlat = olrdata.lat.size
        nlong = olrdata.long.size
        olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60)

        ntime = olr_maps_for_doy.shape[0]
        N = ntime
        M = nlat * nlong
        F = np.reshape(olr_maps_for_doy,
                       [N, M]).T  # vector: only one dimension. Length given by original longitude and latitude bins
        solver = eofs_package.Eof(F.T)
        # FIXME Do we have to think about complex values here?
        eofs = solver.eofs(neofs=2)
        explainedVariances = solver.varianceFraction()
        L=solver.eigenvalues()

        if L.size < M:
            # This usually happens if the covariance matrix did not have full rank (e.g. N<M). Missing Eigenvalues
            # are 0 and can be simply padded here
            L = np.pad(L, (0, M-L.size), 'constant', constant_values=(0, 0))
            explainedVariances = np.pad(explainedVariances, (0, M-explainedVariances.size), 'constant', constant_values=(0, 0))
        return eof.EOFData(olrdata.lat, olrdata.long, np.squeeze(eofs[0, :]), np.squeeze(eofs[1, :]),
                           eigenvalues=L, explained_variances=explainedVariances, no_observations=N)
    else:
        raise ModuleNotFoundError("eofs")


def correct_spontaneous_sign_changes_in_eof_series(eofs: eof.EOFDataForAllDOYs,
                                                   doy1reference: eof.EOFData = None) -> eof.EOFDataForAllDOYs:
    switched_eofs = []
    if doy1reference is not None:
        corrected_doy1 = _correct_spontaneous_sign_change_of_individual_eof(doy1reference, eofs.eofdata_for_doy(1))
    else:
        corrected_doy1 = eofs.eofdata_for_doy(1)
    switched_eofs.append(corrected_doy1)
    previous_eof = corrected_doy1
    for doy in eof.doy_list()[1:]:
        print(doy)
        corrected_eof = _correct_spontaneous_sign_change_of_individual_eof(previous_eof, eofs.eofdata_for_doy(doy))
        switched_eofs.append(corrected_eof)
        previous_eof = corrected_eof
    return eof.EOFDataForAllDOYs(switched_eofs)


def _correct_spontaneous_sign_change_of_individual_eof(reference: eof.EOFData, target=eof.EOFData) -> eof.EOFData:
    if (np.mean(np.abs(target.eof1vector + reference.eof1vector))
            < np.mean(np.abs(target.eof1vector - reference.eof1vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof1_switched = -1 * target.eof1vector
        print("Sign of EOF1 switched")
    else:
        eof1_switched = target.eof1vector
    if (np.mean(np.abs(target.eof2vector + reference.eof2vector))
            < np.mean(np.abs(target.eof2vector - reference.eof2vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof2_switched = -1 * target.eof2vector
        print("Sign of EOF2 switched")
    else:
        eof2_switched = target.eof2vector
    return eof.EOFData(target.lat,
                       target.long,
                       eof1_switched,
                       eof2_switched,
                       eigenvalues=target.eigenvalues,
                       explained_variances=target.explained_variances,
                       no_observations=target.no_observations)


def interpolate_eofs_between_doys(eofs: eof.EOFDataForAllDOYs, start_doy: int = 304, end_doy: int = 313) -> eof.EOFDataForAllDOYs:
    """
    Replaces the EOF1 and EOF2 functions between 2 DOYs by a linear interpolation between these 2 DOYs.
    This should only rarely be used and has only been implemented to closely reproduce the original OMI values. Here,
    the EOFs of 1 November to 8 November have been replaced by a interpolation according to Kiladis (2014). However, we
    find that the original EOFs are better reproduced if the replacement takes place during DOY 294 and DOY 315.
    ATTENTION: The statistical values like the explained variance are not changed by this routine. So they further on
    represent the original results of the PCA also for the interpolated EOFs.
    :param eofs: The complete EOF series to interpolate
    :param start_doy: The DOY, which is used as the first point of the interpolation (i.e. start_doy + 1 is the first
    element, which will be replaced by the interpolation. Default value corresponds to 31 October.
    :param end_doy:  The DOY, which is used as the last point of the interpolation (i.e. end_doy - 1 is the last
    element, which will be replaced by the interpolation. Default value corresponds to 9 November.
    :return: The complere EOF series with the interpolated values
    """
    # FIXME: Why does correlation not maximize with original Kiladis dates (see and change comment)
    doys = eof.doy_list()
    start_idx = start_doy - 1
    end_idx = end_doy - 1
    eof_len = eofs.lat.size * eofs.long.size
    eofs1 = np.empty((doys.size, eof_len))
    eofs2 = np.empty((doys.size, eof_len))
    # FIXME: Maybe this could be solved more efficiently by using internal numpy functions for multidimenasional operations
    for (idx, doy) in enumerate(doys):
        eofs1[idx,:] = eofs.eof1vector_for_doy(doy)
        eofs2[idx,:] = eofs.eof2vector_for_doy(doy)

    for i in range (0, eof_len):
        eofs1[start_idx+1:end_idx-1, i] = np.interp(doys[start_idx+1:end_idx-1], [doys[start_idx], doys[end_idx]],
                                                    [eofs1[start_idx, i], eofs1[end_idx, i]])
        eofs2[start_idx+1:end_idx-1, i] = np.interp(doys[start_idx+1:end_idx-1], [doys[start_idx], doys[end_idx]],
                                                    [eofs2[start_idx, i], eofs2[end_idx, i]])
    interpolated_eofs = []
    for (idx, doy) in enumerate(doys):
        orig_eof = eofs.eofdata_for_doy(doy)
        interpolated_eofs.append(eof.EOFData(orig_eof.lat, orig_eof.long, np.squeeze(eofs1[idx,:]),
                                             np.squeeze(eofs2[idx,:]),explained_variances=orig_eof.explained_variances,
                                             eigenvalues=orig_eof.eigenvalues, no_observations=orig_eof.no_observations)
                                 )
    return eof.EOFDataForAllDOYs(interpolated_eofs)





# def switchSignOfEOFs(inputDir, outputdir, file_prefix, eof_number=0):
#     for doy in range(1, 367):
#         (eof1_orig_vec, eof2_orig_vec) = MJO.RecalculateOMI.load_OMI_EOFs(inputDir, doy, prefix=file_prefix)
#         eof1_switched = eof1_orig_vec
#         eof2_switched = eof2_orig_vec
#         if eof_number == 1 or eof_number == 0:
#             eof1_switched = -1 * eof1_orig_vec
#         if eof_number == 2 or eof_number == 0:
#             eof2_switched = -1 * eof2_orig_vec
#         MJO_OMI_EOF_Recalculated.saveSingeEOFVecInKiladisStyle(eof1_switched, 1, outputdir, file_prefix, doy)
#         MJO_OMI_EOF_Recalculated.saveSingeEOFVecInKiladisStyle(eof2_switched, 2, outputdir, file_prefix, doy)


# #################PC Calculation
def calculatePCsFromOLRWithOriginalConditions(olrData: olr.OLRData,
                                              original_eof_dirname: Path,
                                              useQuickTemporalFilter=False):
    period_start = np.datetime64("1979-01-01")
    period_end = np.datetime64("2018-08-28")
    eofs = eof.load_all_original_eofs_from_directory(original_eof_dirname)
    return calculatePCsFromOLR(olrData,
                               eofs,
                               period_start,
                               period_end,
                               useQuickTemporalFilter)


# Calculate The index values (principal components)
# based on a set of already computed EOFs
def calculatePCsFromOLR(olrData: olr.OLRData,
                        eofdata: eof.EOFDataForAllDOYs,
                        period_start: np.datetime64,
                        period_end: np.datetime64,
                        useQuickTemporalFilter=False) -> pc.PCData:
    restictedOLRData = olr.restrict_time_coverage(olrData, period_start, period_end)
    resampledOLRData = olr.resample_spatial_grid(restictedOLRData, eofdata.lat, eofdata.long)
    if useQuickTemporalFilter:
        filtered_olr_data = wkfilter.filterOLRForMJO_PC_CalculationWith1DSpectralSmoothing(resampledOLRData)
    else:
        filtered_olr_data = wkfilter.filterOLRForMJO_PC_Calculation(resampledOLRData)
    raw_pcs = regress_3dim_data_onto_eofs(filtered_olr_data, eofdata)
    normalization_factor = 1 / np.std(raw_pcs.pc1)
    pc1 = np.multiply(raw_pcs.pc1, normalization_factor)
    pc2 = np.multiply(raw_pcs.pc2, normalization_factor)
    return pc.PCData(raw_pcs.time, pc1, pc2)


def regress_3dim_data_onto_eofs(data: object, eofdata: eof.EOFDataForAllDOYs) -> pc.PCData:
    if not np.all(data.lat == eofdata.lat):
        raise AttributeError("Latitude grid of EOFs and OLR is not equal.")
    if not np.all(data.long == eofdata.long):
        raise AttributeError("Longitude grid of EOFs and OLR is not equal.")
    # FIXME: Don't use zeros
    pc1 = np.zeros(data.time.size)
    pc2 = np.zeros(data.time.size)

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


def regress_vector_onto_eofs(vector: np.ndarray, eof1: np.ndarray, eof2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # FIXME: Swap rows and columns to fit to standard lin alg ?!
    eof_mat = np.array([eof1, eof2]).T

    # Alternative implementation 1:
    x = np.linalg.lstsq(eof_mat, vector, rcond=-1)
    pc1, pc2 = x[0]
    return pc1, pc2

    # Alternative implementation 2:
    # pseudo_inverse = np.linalg.pinv(eof_mat)
    # pcs = np.matmul(pseudo_inverse, vector)
    # return pcs[0], pcs[1]
