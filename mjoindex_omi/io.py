# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""
import os
import datetime
import numpy as np
import pandas as pd

def load_EOFs(path, doy, prefix="eof", suffix=".txt"):
    eof1filename = path + os.path.sep + "eof1" + os.path.sep + prefix +  str(doy).zfill(3) + suffix
    eof1=np.genfromtxt(eof1filename)
    eof2filename = path + os.path.sep + "eof2" + os.path.sep + prefix +  str(doy).zfill(3) + suffix
    eof2=np.genfromtxt(eof2filename)
    return (eof1, eof2)

def savePCsToTxt(dates, pc1, pc2, filename):
    df = pd.DataFrame({"Date":dates, "PC1" : pc1, "PC2" : pc2 })
    df.to_csv(filename, index=False)

def loadPCsFromTxt(filename):
    df = pd.read_csv(filename, sep=',',parse_dates=[0], header=0)
    dates=df.Date.values
    pc1 = df.PC1.values
    pc2 = df.PC2.values
    return (dates, pc1, pc2)

def loadOriginalPCsFromTxt(filename):
    my_data = np.genfromtxt(filename)
    dates_temp=[]
    for  i in range(0, my_data.shape[0]):
        dates_temp.append(datetime.datetime(my_data[i,0].astype(np.int), my_data[i,1].astype(np.int), my_data[i,2].astype(np.int)))
    dates=np.array(dates_temp,dtype='datetime64')
    pc1 = np.array(my_data[:,4])
    pc2 = np.array(my_data[:,5])
    return (dates, pc1, pc2)