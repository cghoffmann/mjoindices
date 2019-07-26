# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:45:11 2019

@author: ch
"""

import numpy as np
import matplotlib.pyplot as plt
import mjoindex_omi.io as omiio


def plotComparisonOrigRecalcPCs(filenameRecalcOMIPCs, filenameOrigOMIPCs, startDate=None, endDate=None):
    (origDates, origPC1, origPC2) = omiio.loadOriginalPCsFromTxt(filenameOrigOMIPCs)
    (recalcDates, recalcPC1, recalcPC2) = omiio.loadPCsFromTxt(filenameRecalcOMIPCs)

    fig, axs = plt.subplots(2,1,num="ReproduceOriginalOMIPCs_PCs",clear=True,  figsize=(8, 6), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("PC Recalculation")

    ax = axs[0]
    ax.set_title("PC1")
    p1,=ax.plot(origDates,origPC1,label="Original")
    p2,=ax.plot(recalcDates,recalcPC1,label="Recalculation")
    if(startDate != None and endDate != None):
        ax.set_xlim((startDate,endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    ax.legend(handles=(p1,p2))

    corr = (np.corrcoef(origPC1,recalcPC1))[0,1]
    # FIXME: Calculate correlation only for wanted period
    plt.text(0.1,0.1,"Correlation over complete period: %.3f"%corr,transform=ax.transAxes)

    ax = axs[1]
    ax.set_title("PC2")
    p3,=ax.plot(origDates,origPC2,label="Original")
    p4,=ax.plot(recalcDates,recalcPC2,label="Recalculation")
    if(startDate != None and endDate != None):
        ax.set_xlim((startDate,endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    ax.legend(handles=(p3,p4))

    corr = (np.corrcoef(origPC2,recalcPC2))[0,1]
    plt.text(0.1,0.1,"Correlation over complete period: %.3f"%corr,transform=ax.transAxes)

    return fig

