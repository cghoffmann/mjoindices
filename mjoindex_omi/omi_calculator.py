# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:39:02 2019

@author: ch
"""

import os
import numpy as np
import mjoindex_omi.olr_handling as olr
import mjoindex_omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindex_omi.io as omiio
import mjoindex_omi.tools as tools

def calculatePCsFromOLRWithOriginalConditions(olrData,
                                              eof_dir,
                                              period_start,
                                              period_end,
                                              result_file,
                                              useQuickTemporalFilter = False):
    restictedOLRData = olr.restrictOLRDataToTimeRange(olrData, period_start, period_end)
    resampledOLRData = olr. resampleOLRToOriginalSpatialGrid(restictedOLRData)
    return calculatePCsFromOLR(resampledOLRData,
                               eof_dir,
                               period_start,
                               period_end,
                               result_file,
                               useQuickTemporalFilter)


# Calculate The index values (principal components)
# based on a set of already computed EOFs
def calculatePCsFromOLR(olrData,
                        eof_dir,
                        period_start,
                        period_end,
                        result_file,
                        useQuickTemporalFilter = False):
    return __calculatePCs(olrData,
                          eof_dir,
                          period_start,
                          period_end,
                          result_file,
                          useQuickTemporalFilter)


def __calculatePCs(olrData,
                   eof_dir,
                   period_start,
                   period_end,
                   result_file,
                   useQuickTemporalFilter = False):
    print(olrData.lat)
    print(olrData.long)
    if useQuickTemporalFilter:
        olrDataFiltered = wkfilter.filterOLRForMJO_PC_CalculationWith1DSpectralSmoothing(olrData)
    else:
        olrDataFiltered = wkfilter.filterOLRForMJO_PC_Calculation(olrData)
    # FIXME: Don't use zeros
    pc1 = np.zeros(olrDataFiltered.time.size)
    pc2 = np.zeros(olrDataFiltered.time.size)
    for idx, val in enumerate(olrDataFiltered.time):
        # print(idx)
        day = val
        olr_singleday = olrDataFiltered.extractDayFromOLRData(day)
        doy = tools.calcDayOfYear(day)
        # FIXME: Don't load EOFs from disk every time
        (eof1, eof2) = omiio.load_EOFs(eof_dir, doy)
        (pc1_single, pc2_single) = __regressOLROnEOFs(olr_singleday,
                                                       eof1,
                                                       eof2)
        pc1[idx] = pc1_single
        pc2[idx] = pc2_single
    nomalization_factor = 1/np.std(pc1)
    pc1 = np.multiply(pc1, nomalization_factor)
    pc2 = np.multiply(pc2, nomalization_factor)
    omiio.savePCsToTxt(olrDataFiltered.time, pc1, pc2, result_file )
    return (pc1, pc2)

def __regressOLROnEOFs(olrData, eof1, eof2):
        #FIXME: Swap rows and columns to fit ti standard lin alg ?!
    #    print("reg", olr.shape)
    olr_vec = np.reshape(olrData, eof1.shape)
        #olr_vec = np.subtract(olr_vec,np.mean(olr_vec))
        #olr_vec = np.divide(olr_vec,np.std(olr_vec))
    eof = np.array([eof1, eof2]).T
        #print(eof.shape)
    x =np.linalg.lstsq(eof,olr_vec, rcond=-1)
    (pc1, pc2) = x[0]
        #print(pc1, pc2)
    return (pc1, pc2)

        #print("pseudo inverse")
        #test = numpy.linalg.pinv(eof)
        #print (test.shape)
        #testpc = numpy.matmul(test, olr_vec)
        #testpc = numpy.matmul(eof.T, olr_vec)
        #print(testpc)
        #print(testpc[0]-pc1)
        #return (pc1, pc2)
        #return (testpc[0], testpc[1])
