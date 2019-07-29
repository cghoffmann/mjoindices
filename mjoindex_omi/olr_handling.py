# -*- coding: utf-8 -*-
from scipy.io import netcdf
import numpy as np
#import Manipulation.Filters
#import Tools.DateTime
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate

"""
Created on Tue Dec  4 14:05:34 2018

@author: ch
"""

class OLRData:
    def __init__(self, olr, time, lat, long):
        if( olr.shape[0] == time.size):
            self._olr =olr
            self._time= time
            self._lat = lat
            self._long = long
        else:
            raise ValueError('Length of time grid does not fit to first dimension of OLR data cube')


    @property
    def olr(self):
        return self._olr

    @property
    def time(self):
        return self._time

    @property
    def lat(self):
        return self._lat

    @property
    def long(self):
        return self._long

    def extractDayFromOLRData(self, date):
        #FIXME: Check if date is in time range
        cand = self.time == date
        #print(cand)
        #print(self.__olr_data_cube.shape)
        return np.squeeze(self.olr[cand,:,:])

    def returnOLRForDOY(self, doy, window_length=0):
        doys = Tools.DateTime.calcDaysOfYear(self.TimeGrid)
        print("doys", doys)


        lower_limit = doy - window_length
        if(lower_limit < 1):
            lower_limit = lower_limit + 366
        upper_limit = doy + window_length
        if(upper_limit > 366):
            upper_limit = upper_limit - 366

        if(lower_limit <= upper_limit):
            dayIndsToConsider = ((doys >= lower_limit) & (doys <= upper_limit) )
        else:
            dayIndsToConsider = ((doys >= lower_limit) | (doys <= upper_limit) )


        print(np.where(dayIndsToConsider==True))
        print("Days To Consider", dayIndsToConsider)
        result=self.OLRData[dayIndsToConsider,:,:]
        return result

def resampleOLRToOriginalSpatialGrid(olr):
    """Resamples the data in an OLRData object spatially.

    Afterwards, it corresponds to the original spatial calculation grid:
    Latitude: 2.5 deg sampling in the tropics (20S to 20 N)
    Longitude: Whole globe with 2.5 deg sampling

    **Arguments**

    *olr*
    The OLRData object

    **Returns**
    A new OLRData object with the resampled data

    """

    orig_lat = np.arange(-20., 20.1, 2.5)

    orig_long = np.arange(0., 359.9, 2.5)

    no_days = olr.time.size

    olr_interpol = np.empty((no_days, orig_lat.size, orig_long.size))
    for idx in range(0,no_days):
        f = scipy.interpolate.interp2d(olr.long, olr.lat, np.squeeze(olr.olr[idx,:,:]), kind='linear')
        olr_interpol[idx,:,:] = f(orig_long, orig_lat)
    return OLRData(olr_interpol, olr.time, orig_lat, orig_long)

# FIXME: Implement temporal interpolation to original 1day-spacing

def restrictOLRDataToTimeRange(olr, startDate, stopDate):
    print("Restricting time range...")
    windowInds = (olr.time >= startDate) & (olr.time <= stopDate)
    print(windowInds)
    return OLRData(olr.olr[windowInds,:,:], olr.time[windowInds], olr.lat, olr.long)

def loadNOAAInterpolatedOLR(filename):
    """Loads the standard OLR data product provided by NOAA

    The dataset can be obtained from
    ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc

    A description is found at
    https://www.esrl.noaa.gov/psd/data/gridded/data.interp_OLR.html

    **Arguments**

    *filename*
    full filename of local copy of OLR data file

    """
    # FIXME: NETCDF Package throws warning
    f = netcdf.netcdf_file(filename, 'r')
    lat = f.variables['lat'].data
    lon = f.variables['lon'].data
    # scaling and offset as given in meta data of nc file
    olr = f.variables['olr'].data/100+327.65
    # reverse latitude axis to be consistent with kiladis OLR and,
    # therefore with OMI EOFs
    olr = np.flip(olr, 1)
    lat = np.flip(lat, 0)
    hours_since1800 = f.variables['time'].data
    temptime = []
    for item in hours_since1800:
        delta = np.timedelta64(int(item/24), 'D')
        day = np.datetime64('1800-01-01') + delta
        temptime.append(day)
    time = np.array(temptime, dtype=np.datetime64)
    result = OLRData(np.squeeze(olr), time, lat, lon)
    f.close()
    return result