# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:45:11 2019

@author: ch
"""

import matplotlib.pyplot as plt
import numpy as np

import mjoindex_omi.empirical_orthogonal_functions as eof
import mjoindex_omi.principal_components as pc


def plotComparisonOrigRecalcPCs(filenameRecalcOMIPCs, filenameOrigOMIPCs, startDate=None, endDate=None):
    orig_omi = pc.load_original_pcs_from_txt_file(filenameOrigOMIPCs)
    recalc_omi = pc.load_pcs_from_txt_file(filenameRecalcOMIPCs)

    fig, axs = plt.subplots(2,1,num="ReproduceOriginalOMIPCs_PCs",clear=True,  figsize=(8, 6), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("PC Recalculation")

    ax = axs[0]
    ax.set_title("PC1")
    p1,=ax.plot(orig_omi.time, orig_omi.pc1, label="Original")
    p2,=ax.plot(recalc_omi.time, recalc_omi.pc1, label="Recalculation")
    if startDate != None and endDate != None:
        ax.set_xlim((startDate,endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    ax.legend(handles=(p1, p2))

    corr = (np.corrcoef(orig_omi.pc1, recalc_omi.pc1))[0,1]
    # FIXME: Calculate correlation only for wanted period
    # FIXME: Check that periods covered by orig_omi and recalc_omi are actually the same
    plt.text(0.1,0.1,"Correlation over complete period: %.3f"%corr,transform=ax.transAxes)

    ax = axs[1]
    ax.set_title("PC2")
    p3,=ax.plot(orig_omi.time, orig_omi.pc2, label="Original")
    p4,=ax.plot(recalc_omi.time, recalc_omi.pc2, label="Recalculation")
    if startDate != None and endDate != None:
        ax.set_xlim((startDate, endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment="right")
    ax.legend(handles=(p3, p4))

    corr = (np.corrcoef(orig_omi.pc2,recalc_omi.pc2))[0,1]
    plt.text(0.1, 0.1, "Correlation over complete period: %.3f" % corr, transform=ax.transAxes)

    return fig


def plot_original_eof_for_doy(path, doy):
    eofdata = eof.load_original_eofs_for_doy(path, doy)

    return plot_eof(eofdata, doy)

def plot_eof_from_file(filename):
    eofdata = eof.load_single_eofs_from_txt_file(filename)

    return plot_eof(eofdata)

def plot_eof(eofdata: eof.EOFData, doy: int = None):
    # TODO: Plot underlying map


    fig, axs = plt.subplots(2, 1, num="plotting.plot_eof_for_doy", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0]

    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF1")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1]
    c = ax.contourf(eofdata.long, eofdata.lat, eofdata.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    return fig


def plot_eof_comparison(orig_eof: eof.EOFData, compare_eof: eof.EOFData, doy=None):

    #nlat = orig_eof.lat.size()
    #nlong = orig_eof.long.size()

    print(np.corrcoef(orig_eof.eof1vector, compare_eof.eof1vector))
    print(np.corrcoef(orig_eof.eof2vector, compare_eof.eof2vector))

    #eof1_orig = np.reshape(eof1_orig, [nlat, nlong])
    #eof2_orig = np.reshape(eof2_orig, [nlat, nlong])
    #eof1_recalc = np.reshape(eof1_recalc, [nlat, nlong])
    #eof2_recalc = np.reshape(eof2_recalc, [nlat, nlong])

    #print(np.corrcoef(eof1_orig, eof1_recalc))

    fig, axs = plt.subplots(2, 3, num="ReproduceOriginalOMIPCs_ExplainedVariance_EOF_Comparison", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF1")
    ax.set_ylabel("Latitude [°]")

    ax = axs[0, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF1")

    # FIXME: Check that grids are equal
    ax = axs[0, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map - compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 1")

    ax = axs[1, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF2")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map - compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 2")
    ax.set_xlabel("Longitude [°]")

    return fig


