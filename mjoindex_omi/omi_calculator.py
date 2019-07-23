# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:39:02 2019

@author: ch
"""

import numpy as np
import mjoindex_omi.olr_handling as olr
import mjoindex_omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindex_omi.io as omiio


# Calculate The index values (principal components)
# based on a set of already computed EOFs
def calculatePCsFromOLR(olrData,
                        eof_dir,
                        period_start,
                        period_end,
                        result_dir):
    pass


def __calculatePCs():
    # FIXME This is a big difference to initial implementation. There, self.__olrdata.filterDataTemporally(20, 96) was used. Test if this here is appropriate
    olrDataFiltered = wkfilter.filterOLRForMJO_PC_Calculation(olrData)
    # FIXME: Don't use zeros
    pc1 = np.zeros(olrDataFiltered.time.size)
    pc2 = np.zeros(olrDataFiltered.time.size)
    for idx, val in enumerate(olrDataFiltered.time):
        # print(idx)
        day = val
        olr_singleday = olrDataFiltered.extractDayFromOLRData(day)
        doy = Tools.DateTime.calcDayOfYear(day)
        # FIXME: Don't load EOFs from disk every time
        (eof1, eof2) = omiio.load_EOFs(eof_dir,
                                     doy, prefix=self.__EOFFilePrefix,
                                     suffix=self.__EOFFileSuffix)
        (pc1_single, pc2_single) = self._perform_regression(olr_singleday,
                                                            eof1,
                                                            eof2)
        pc1[idx] = pc1_single
        pc2[idx] = pc2_single
    factor = 1/np.std(pc1)
    # print(1/factor)varexp
    pc1 = np.multiply(pc1, factor)
    pc2 = np.multiply(pc2, factor)
    return (pc1, pc2)
