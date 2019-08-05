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


def calculatePCsFromOLRWithOriginalConditions(olrData: olr.OLRData,
                                              original_eof_dirname: Path,
                                              useQuickTemporalFilter = False):

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
                        useQuickTemporalFilter = False) -> pc.PCData:

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
    if not  np.all(data.long == eofdata.long):
        raise AttributeError("Longitude grid of EOFs and OLR is not equal.")
    # FIXME: Don't use zeros
    pc1 = np.zeros(data.time.size)
    pc2 = np.zeros(data.time.size)

    for idx, val in enumerate(data.time):
        day = val
        olr_singleday = data.extractDayFromOLRData(day)
        doy = tools.calc_day_of_year(day)
        (pc1_single, pc2_single) = regress_vector_onto_eofs(eofdata.eofdata_for_doy(doy).reshape_to_vector(olr_singleday),
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
    #pseudo_inverse = np.linalg.pinv(eof_mat)
    #pcs = np.matmul(pseudo_inverse, vector)
    #return (pcs[0], pcs[1])
