# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:55:00 2019

@author: ch
"""

import datetime as dt
import numpy as np

def calcDayOfYear(date):
    temp = date.astype(dt.datetime)
    test = temp.timetuple()
    return test[7]

def calcDaysOfYear(dates):
    #FIXME: don't use zeros here
    doys = np.zeros(dates.size)
    for idx,date in enumerate(dates):
        doys[idx]=calcDayOfYear(date)
    return doys