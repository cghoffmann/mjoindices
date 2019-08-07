# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:39:02 2019

@author: ch
"""

from pathlib import Path
from typing import Tuple

import numpy as np

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
    # interpolate on particular spatial grid?
    olrdata_filtered = wkfilter.filterOLRForMJO_EOF_Calculation(olrdata)
    return olrdata_filtered


def calc_eofs_from_preprocessed_olr(olrdata: olr.OLRData, implementation="Kutzbach1967") -> eof.EOFDataForAllDOYs:
    doys = eof.doy_list()
    eofs = []
    for doy in doys:
        print("Calculationg for DOY %i" % doy)
        if (implementation == "SVD_eofs_package"):
            singleeof = calc_eofs_for_doy_using_eofs_package(olrdata, doy)
        else:
            singleeof = calc_eofs_for_doy(olrdata, doy)
        eofs.append(singleeof)
    return eof.EOFDataForAllDOYs(eofs)


def post_process_eofs(eofdata: eof.EOFDataForAllDOYs) -> eof.EOFDataForAllDOYs:
    return correct_spontaneous_sign_changes_in_eof_series(eofdata)


def calc_eofs_for_doy(olrdata: olr.OLRData, doy: int) -> eof.EOFData:
    nlat = olrdata.lat.size
    nlong = olrdata.long.size
    olr_maps_for_doy = olrdata.extract_olr_matrix_for_doy_range(doy, window_length=60)
    # doyOLR_3dim = self.__olrdata_filtered.returnAverageOLRForIndividualDOY(doy,window_length=60)
    # doyOLR_3dim = np.mean(doyOLR_3dim,axis=0)

    ntime = olr_maps_for_doy.shape[0]
    N = ntime
    M = nlat * nlong
    # FIXME: The sollowing reshape iincluding the transpose etc. Is this correct? Rule: If original Kiladis Grid is used, the EOFs must be compatible to the saved ones.
    F = np.reshape(olr_maps_for_doy,
                   [N, M]).T  # vector: only one dimension. Length given by original longitude and latitude bins
    R = np.matmul(F, F.T) / N
    L, E = np.linalg.eig(R)
    total_var = np.sum(L)
    order = (np.flip(L.argsort()))
    # print(order)
    order = order[:2]
    L = L[order]
    varianceExplained = L / total_var  # See Kutzbach Eq 12
    # print("total var:", total_var)
    # print("L",L)
    E = E[:, order]  # is that right?
    eof1_vec = np.squeeze(E[:, 0])
    eof2_vec = np.squeeze(E[:, 1])
    return eof.EOFData(olrdata.lat, olrdata.long, eof1_vec, eof2_vec,
                       explained_variance_eof1=varianceExplained[0], explained_variance_eof2=varianceExplained[1],
                       eigenvalue_eof1=L[0], eigenvalue_eof2=L[1])


def calc_eofs_for_doy_using_eofs_package(olrdata: olr.OLRData, doy: int) -> eof.EOFData:
    raise NotImplementedError()


#
#     nlat = self.__olrdata_filtered.LatGrid.size
#     nlong = self.__olrdata_filtered.LongGrid.size
#     doyOLR_3dim = self.__olrdata_filtered.returnOLRForDOY(doy,window_length=60)
#     #doyOLR_3dim = self.__olrdata_filtered.returnAverageOLRForIndividualDOY(doy,window_length=60)
#
#     N=doyOLR_3dim.shape[0]
#     print("N", N)
#     M =nlat*nlong
#     F = np.reshape(doyOLR_3dim,[N,M]).T  #vector: only one dimension. Length given by original longitude and latitude bins
#
#     print("F:", F.shape)
#     solver = Eof(F.T)
#     eofs = solver.eofs(neofs=2)
#     print("VarianceFraction",solver.varianceFraction())
#     plt.plot(solver.varianceFraction())
#     print(np.sum(solver.varianceFraction()))
#
#     varianceExplained = solver.varianceFraction(neigs=2)
#
#     L=solver.eigenvalues(neigs=2)
#
#     print("L", L)
#
#     print("EOFs:", eofs.shape)
#     eof1_map = np.reshape(eofs[0,:],[nlat,nlong])
#     eof2_map = np.reshape(eofs[1,:],[nlat,nlong])
#
#     print("EOF1: ", eof1_map.shape)
#
#
#     #        R = np.matmul(F,F.T)/N
#     #        L,E =  numpy.linalg.eig(R)
#     #        total_var = np.sum(L)
#     #        order = (np.flip(L.argsort()))
#     #        #print(order)
#     #        order=order[:2]
#     #        L=L[order] / total_var #See Kutzbach Eq 12
#     #        #print("total var:", total_var)
#     #        #print("L",L)
#     #        E=E[:,order] #is that right?
#     #        eof1_vec = np.squeeze(E[:,0])
#     #        eof2_vec = np.squeeze(E[:,1])
#     #        eof1_map = np.reshape(eof1_vec,[nlat,nlong])
#     #        eof2_map = np.reshape(eof2_vec,[nlat,nlong])
#     return eof1_map, eof2_map, L, varianceExplained


def __correctSpontaneousSignChangesofEOFs(self, eof1_raw, eof2_raw):
    eof1 = eof1_raw
    eof2 = eof2_raw
    for idx in range(0, eof1_raw.shape[2]):
        print(idx)
        if (
                idx > 0):  # Account for spoantaneous sign changes in the EOFS from one day to another. This is metioned in Kiladis 2014 and has been confirmed by G.Kiladis via Mail.
            if (np.mean(np.abs(eof1[:, :, idx] + eof1[:, :, idx - 1])) < np.mean(np.abs(eof1[:, :, idx] - eof1[:, :,
                                                                                                          idx - 1]))):  # if abs(sum) is lower than abs(diff), than the signs are different...
                eof1[:, :, idx] = -1 * eof1[:, :, idx]
                print("Sign of EOF1 switched")
            if (np.mean(np.abs(eof2[:, :, idx] + eof2[:, :, idx - 1])) < np.mean(
                    np.abs(eof2[:, :, idx] - eof2[:, :, idx - 1]))):
                eof2[:, :, idx] = -1 * eof2[:, :, idx]
                print("Sign of EOF2 switched")
        else:  # to adjust the signs of the EOFs of the first day, the original Kiladis selection is used.
            (eof1_orig_vec, eof2_orig_vec) = MJO.RecalculateOMI.load_OMI_EOFs(
                os.path.dirname(os.path.abspath(__file__)) + '/', 1)
            eof1_orig_map = np.reshape(eof1_orig_vec, [17, 144])
            eof2_orig_map = np.reshape(eof2_orig_vec, [17, 144])
            if (np.mean(np.abs(eof1[:, :, idx] + eof1_orig_map)) < np.abs(np.mean(eof1[:, :,
                                                                                  idx] - eof1_orig_map))):  # if abs(sum) is lower than abs(diff), than the signs are different...
                eof1[:, :, idx] = -1 * eof1[:, :, idx]
                print("Sign of EOF1 switched")
            if (np.mean(np.abs(eof2[:, :, idx] + eof2_orig_map)) < np.mean(np.abs(eof2[:, :, idx] - eof2_orig_map))):
                eof2[:, :, idx] = -1 * eof2[:, :, idx]
                print("Sign of EOF2 switched")
    return (eof1, eof2)


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
                       explained_variance_eof1=target.explained_variance_eof1,
                       explained_variance_eof2=target.explained_variance_eof2,
                       eigenvalue_eof1=target.eigenvalue_eof1,
                       eigenvalue_eof2=target.eigenvalue_eof2)


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
    restictedOLRData = olr.restrictOLRDataToTimeRange(olrData, period_start, period_end)
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
        olr_singleday = data.extractDayFromOLRData(day)
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
