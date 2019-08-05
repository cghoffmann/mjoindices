# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import scipy
import scipy.interpolate
from scipy.io import netcdf

import mjoindex_omi.tools as tools

"""
Created on Tue Dec  4 14:05:34 2018

@author: ch
"""


class OLRData:
    def __init__(self, olr, time, lat, long):
        if( olr.shape[0] == time.size):
            self._olr = olr.copy()
            self._time= time.copy()
            self._lat = lat.copy()
            self._long = long.copy()
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
        doys = tools.calc_day_of_year(self.TimeGrid)
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
        result=self.olr[dayIndsToConsider, :, :]
        return result


def resample_spatial_grid_to_original(olr: OLRData) -> OLRData:
    """Resamples the data in an OLRData object spatially according to the original OMI EOF grid.

    Afterwards, the data corresponds to the original spatial calculation grid:
    Latitude: 2.5 deg sampling in the tropics from -20 to 20 deg (20S to 20 N)
    Longitude: Whole globe with 2.5 deg sampling

    :param olr: The OLRData object
    :return:  A new OLRData object with the resampled data
    """
    orig_lat = np.arange(-20., 20.1, 2.5)
    orig_long = np.arange(0., 359.9, 2.5)
    return resample_spatial_grid(olr, orig_lat, orig_long)


def resample_spatial_grid(olr: OLRData, target_lat: np.array, target_long: np.array) -> OLRData:
    """ Resamples the OLR data according to the given grids and returns a new OLRData object

    :param olr: The OLR data to resample
    :param target_lat: The new latitude grid
    :param target_long: the new longitude grid
    :return: an OLRData object containing the resampled OLR data
    """
    no_days = olr.time.size
    olr_interpol = np.empty((no_days, target_lat.size, target_long.size))
    for idx in range(0,no_days):
        f = scipy.interpolate.interp2d(olr.long, olr.lat, np.squeeze(olr.olr[idx,:,:]), kind='linear')
        olr_interpol[idx,:,:] = f(target_long, target_lat)
    return OLRData(olr_interpol, olr.time, target_lat, target_long)

def restrictOLRDataToTimeRange(olr, startDate, stopDate):
    print("Restricting time range...")
    windowInds = (olr.time >= startDate) & (olr.time <= stopDate)
    print(windowInds)
    return OLRData(olr.olr[windowInds,:,:], olr.time[windowInds], olr.lat, olr.long)


def load_noaa_interpolated_olr(filename: Path) -> OLRData:
    """Loads the standard OLR data product provided by NOAA

    The dataset can be obtained from
    ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc

    A description is found at
    https://www.esrl.noaa.gov/psd/data/gridded/data.interp_OLR.html

    :param filename: full filename of local copy of OLR data file
    :return: The OLR data
    """
    f = netcdf.netcdf_file(str(filename), 'r')
    lat = f.variables['lat'].data.copy()
    lon = f.variables['lon'].data.copy()
    # scaling and offset as given in meta data of nc file
    olr = f.variables['olr'].data.copy()/100.+327.65
    hours_since1800 = f.variables['time'].data.copy()
    f.close()

    temptime = []
    for item in hours_since1800:
        delta = np.timedelta64(int(item/24), 'D')
        day = np.datetime64('1800-01-01') + delta
        temptime.append(day)
    time = np.array(temptime, dtype=np.datetime64)
    result = OLRData(np.squeeze(olr), time, lat, lon)

    return result
