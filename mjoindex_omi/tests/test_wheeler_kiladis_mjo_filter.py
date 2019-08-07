import numpy as np
from scipy.io import FortranFile

import subprocess
import mjoindex_omi.olr_handling as olr
import subprocess

import numpy as np
from scipy.io import FortranFile

import mjoindex_omi.olr_handling as olr


def configure_and_run_fortran_code(lat_index: int):
    fortranfile = "/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/stfilt_CHDebugOutput_MJOConditions_Automatic.f"
    scriptfile = "/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/compileAndStartFilter_CHDebugOutputMJOCond_Automatic.sh"
    with open(fortranfile, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[112] = "      parameter (soutcalc=%i,noutcalc=%i)  ! Region of output 90ns AUTOMATIC CHANGE!\n" %(lat_index,lat_index)

    # and write everything back
    with open(fortranfile, 'w') as file:
        file.writelines(data)
    out = subprocess.call([scriptfile],cwd="/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/")
    print(out)

def check_test_input_OLRData():
    data_exchange_dir = "/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering/"
    kiladis_olr = loadKiladisBinaryOLRDataTwicePerDay(data_exchange_dir + "/olr.2x.7918.b")
    k_inputOLR = loadKiladisOriginalOLR(data_exchange_dir + "/OLROriginal.b")

    found = None
    for i in range(0, kiladis_olr.olr.shape[1]):
        if np.all(np.isclose(np.squeeze(kiladis_olr.olr[:,i,:]), k_inputOLR)):
            found = i
    print(found)
    testdata = np.squeeze(kiladis_olr.olr[:, found, :])  # select one latitude
    print(np.mean(testdata - k_inputOLR))


def loadKiladisOriginalOLR(filename):
    nl = 144
    nt = 28970
    f = FortranFile(filename, 'r')
    olr=np.zeros([nt,nl])
    for i_l in range(0,nl):
        record1 = np.squeeze(f.read_record('(1,28970)<f4'))
        olr[:,i_l] = record1
    return olr

def loadKiladisBinaryOLRDataTwicePerDay(filename):
    nt=28970 #known from execution of kiladis fortran code

    time=np.zeros(nt,dtype='datetime64[m]')

    lat = np.arange(-90,90.1,2.5)
    nlat= lat.size
    long = np.arange(0,360, 2.5)
    nlong= long.size

    olrdata = np.zeros([nt, nlat, nlong],dtype='f4')
    f = FortranFile(filename, 'r')
    for i_t in range(0,nt):
        record1 = np.squeeze(f.read_record('(1,7)<i4'))
        year = str(record1[0])
        month = str(record1[1]).zfill(2)
        day=str(record1[2]).zfill(2)
        hour=str(record1[3]).zfill(2)
        date = np.datetime64(year + '-' + month + '-' + day + 'T' + hour + ':00')
        time[i_t]=date
        #print(record1)
        olr_record = f.read_record('(145,73)<f4').reshape(nlat,145)
        #((xx(lon,lat),lon=1,NLON),lat=soutcalc,noutcalc)
#        if(i_t == 0):
#            print(olr_record.shape)
#            print(record1[0])
#            print(olr_record[0,:])
#            print(olr_record[69,:])
        olrdata[i_t,:,:] = olr_record[:,0:nlong] #the first longitude is repeated at the end in this file, so skip last value
        #print(i_t, ':', date)
    f.close()
    #print(time.shape)
    result=olr.OLRData(olrdata, time, lat, long[0:nlong])
    return result

configure_and_run_fortran_code(73)
check_test_input_OLRData()