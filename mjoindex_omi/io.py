# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:44:36 2019

@author: ch
"""

import numpy as np

def load_EOFs(path, doy, prefix="eof", suffix=".txt"):
    eof1filename = path + 'eof1/'+ prefix+  str(doy).zfill(3) + suffix
    eof1=np.genfromtxt(eof1filename)
    eof2filename = path + 'eof2/'+ prefix +  str(doy).zfill(3) + suffix
    eof2=np.genfromtxt(eof2filename)
    return (eof1, eof2)