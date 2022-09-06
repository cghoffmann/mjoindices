"""
Contains the post-processing of the EOFs as described in Weidman, S., Kleiner, N., & Kuang, Z. (2022). 
A rotation procedure to improve seasonally varying empirical orthogonal function bases for MJO indices. 
Geophysical Research Letters, 49, e2022GL099998. https://doi.org/10.1029/2022GL099998. The new method 
includes a projection and rotation postprocessing step that reduces noise in the original EOF calculation. 

The post-processing procedure follows the below steps:
    1. Corrects spontaneous sign changes in the EOFs (same as the original procedure)
    2. Projects EOFs at DOY = n-1 onto EOF space for DOY = n. This is done to reduce spurious oscillations
    between EOFs on sequential days
    3. Rotate the projected EOFs by 1/366 (or 1/365) per day to ensure continuity across January to December
    4. Renormalize the EOFs to have a length of 1 (this is a small adjustment to account for small numerical
    errors).

"""

from typing import Tuple
import inspect

import numpy as np
import warnings
import importlib

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.omi.postprocessing_original_kiladis2014 as pp_kil2014
import mjoindices.tools as tools


def post_process_eofs_rotation(eofdata: eof.EOFDataForAllDOYs, sign_doy1reference: bool = True) -> eof.EOFDataForAllDOYs:
    """
    Post processes a series of EOF pairs for all DOYs.

    Postprocessing includes an alignment of EOF signs and a rotation algorithm that rotates the EOFs
    in three steps:
    1. Projects EOFs at DOY = n-1 onto EOF space for DOY = n. This is done to reduce spurious oscillations
    between EOFs on sequential days
    2. Rotate the projected EOFs by 1/366 (or 1/365) per day to ensure continuity across January to December
    3. Renormalize the EOFs to have a length of 1 (this is a small adjustment to account for small numerical
    errors).

    See documentation of the methods :meth:`correct_spontaneous_sign_changes_in_eof_series` in postprocessing_original_kiladis2014.py
    for EOF sign flipping

    Note that it is recommended to use the function :meth:`calc_eofs_from_olr_with_rotation` to cover the complete algorithm.

    :param eofdata: The EOF series, which should be post processed.
    :param sign_doy1reference: See description of :meth:`correct_spontaneous_sign_changes_in_eof_series` 
    in omi_calculatory.py.

    :return: the postprocessed series of EOFs
    """
    
    pp_eofs = pp_kil2014.correct_spontaneous_sign_changes_in_eof_series(eofdata, doy1reference=sign_doy1reference)
    rot_eofs = rotate_eofs(pp_eofs)
    norm_eofs = normalize_eofs(rot_eofs)
    
    return norm_eofs


def rotate_eofs(orig_eofs: eof.EOFDataForAllDOYs) -> eof.EOFDataForAllDOYs:
    """
    Rotate EOFs at each DOY to 1) align with the EOFs of the previous day and 2) be continuous across December to
    January boundary. Described more in detail in :meth:'post_process_rotation'

    :param orig_eofs: calculated EOFs, signs have been changed via spontaneous_sign_changes 

    :return: set of rotated EOFs
    """

    delta = calculate_angle_from_discontinuity(orig_eofs)

    print('Rotating by ', delta)

    return rotate_each_eof_by_delta(orig_eofs, delta)


def rotation_matrix(delta):
    """
    Return 2d rotation matrix for corresponding delta
    """
    return np.array([[np.cos(delta), -np.sin(delta)],[np.sin(delta), np.cos(delta)]])


def calculate_angle_from_discontinuity(orig_eofs: eof.EOFDataForAllDOYs):
    """
    Project the matrix to align with previous day's EOFs and calculate the resulting
    discontinuity between January 1 and December 31. Divide by number of days in year to 
    result in delta for rotation matrix. 

    :param orig_eofs: calculated EOFs, signs have been changed via spontaneous_sign_changes

    :return: float of (negative) average angular discontinuity between EOF1 and EOF2 on the 
    first and last day of year, divided by the length of the year.
    """

    list_of_doys = tools.doy_list(orig_eofs.no_leap)
    doy1 = orig_eofs.eofdata_for_doy(1)

    ndoys = len(list_of_doys)
    
    # set DOY1 initialization
    rots = np.array([doy1.eof1vector, doy1.eof2vector])

    # project onto previous day
    for d in list_of_doys:
        if d+1 > ndoys: # for last day in cycle, return to January 1
            doyn = orig_eofs.eofdata_for_doy(1)
        else:
            doyn = orig_eofs.eofdata_for_doy(d+1)

        B = np.array([doyn.eof1vector, doyn.eof2vector]).T 
        A = np.array([rots[0,:], rots[1,:]]).T
    
        rots = np.matmul(np.matmul(B, B.T),A).T
    
    # calculate discontinuity between Jan 1 and Jan 1 at end of rotation cycle
    discont = angle_btwn_vectors(doy1.eof1vector, rots[0,:])

    return -discont/ndoys


def rotate_each_eof_by_delta(orig_eofs: eof.EOFDataForAllDOYs, delta: float) -> eof.EOFDataForAllDOYs:
    """
    Use delta calculated by optimization function to rotate original EOFs by delta.
    First projects EOFs from DOY n-1 onto EOF space for DOY n, then rotates projected
    EOFs by small angle delta. 

    :param orig_eofs: calculated EOFs, signs have been changed via spontaneous_sign_changes
    :param delta: scalar by which to rotate EOFs calculated from discontinuity

    :returns: new EOFdata with rotated EOFs.  
    """

    R = rotation_matrix(delta)

    doy1 = orig_eofs.eofdata_for_doy(1)   
    list_of_doys = tools.doy_list(orig_eofs.no_leap)
    
    eofdata_rotated = []
    eofdata_rotated.append(doy1) # first doy is unchanged

    # set DOY1 initialization
    rots = np.array([doy1.eof1vector, doy1.eof2vector])

    # project onto previous day and rotate 
    for d in list_of_doys[1:]:
        doyn = orig_eofs.eofdata_for_doy(d)

        B = np.array([doyn.eof1vector, doyn.eof2vector]).T 
        A = np.array([rots[0,:], rots[1,:]]).T
    
        rots = np.matmul(np.matmul(np.matmul(B, B.T),A),R).T

        # create new EOFData variable for rotated EOFs
        eofdata_rotated.append(eof.EOFData(doyn.lat, doyn.long, 
                                np.squeeze(rots[0,:]), 
                                np.squeeze(rots[1,:]),
                                explained_variances=doyn.explained_variances,
                                eigenvalues=doyn.eigenvalues,
                                no_observations=doyn.no_observations))

    return eof.EOFDataForAllDOYs(eofdata_rotated, orig_eofs.no_leap)


def normalize_eofs(orig_eofs: eof.EOFDataForAllDOYs) -> eof.EOFDataForAllDOYs:
    """
    Normalize all EOFs to have a magnitude of 1

    :param eofdata: The rotated EOF series

    :return: normalized EOFdata for all days
    """

    list_of_doys = tools.doy_list(orig_eofs.no_leap)

    eofdata_normalized = []

    for d in list_of_doys:

        doyn = orig_eofs.eofdata_for_doy(d)
        eof1_norm = doyn.eof1vector/np.linalg.norm(doyn.eof1vector)
        eof2_norm = doyn.eof2vector/np.linalg.norm(doyn.eof2vector) 

       # create new EOFData variable for rotated EOFs
        eofdata_normalized.append(eof.EOFData(doyn.lat, doyn.long, 
                                            np.squeeze(eof1_norm), 
                                            np.squeeze(eof2_norm),
                                            explained_variances=doyn.explained_variances,
                                            eigenvalues=doyn.eigenvalues,
                                            no_observations=doyn.no_observations)) 

    return eof.EOFDataForAllDOYs(eofdata_normalized, orig_eofs.no_leap) 


def angle_between_eofs(reference: eof.EOFData, target=eof.EOFData):
    """
    Calculates angle between two EOF vectors to determine their "closeness."
    theta = arccos(t . r / (||r||*||t||)), 

    :param reference: The reference-EOFs. This is usually the EOF pair of the previous or "first" DOY.
    :param target: The EOF that you want to find the angle with

    :return: A tuple of the  the angles between the reference and target EOFs for both EOF1 and EOF2
    """

    angle1 = angle_btwn_vectors(reference.eof1vector, target.eof1vector)
    angle2 = angle_btwn_vectors(reference.eof2vector, target.eof2vector)

    return (angle1, angle2)


def angle_btwn_vectors(vector1, vector2):
    """
    Calculates the angle between vectors, theta = arccos(t . r / (||r||*||t||))

    Returns angle in radians
    """

    return np.arccos(np.clip(np.dot(vector1, vector2)
                             /(np.linalg.norm(vector1)*np.linalg.norm(vector2)),-1.,1.))
