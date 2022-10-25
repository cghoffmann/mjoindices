"""
Contains the post-processing of the EOFs as described in :ref:`refWeidman2022`. This is an alternative post-processing
approach and does not lead to the same results as shown in :ref:`refKiladis2014`. It reduces noise and avoids
potential degeneracy issues.

The post-processing procedure follows the below steps:

#. Corrects spontaneous sign changes in the EOFs (same as the original procedure)
#. Projects EOFs at DOY = n-1 onto EOF space for DOY = n. This is done to reduce spurious oscillations between EOFs on sequential days
#. Rotate the projected EOFs by 1/366 (or 1/365) per day to ensure continuity across January to December
#. Renormalize the EOFs to have a length of 1 (this is a small adjustment to account for small numerical errors).

.. seealso:: :py:mod:`mjoindices.omi.postprocessing_original_kiladis2014`

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
    Executes the complete post-processing of a series of EOF pairs for all DOYs according to the alterntive procedure
    described by :ref:`refWeidman2022`.

    Postprocessing includes an alignment of EOF signs and a rotation algorithm that rotates the EOFs
    in three steps:

    #. Projects EOFs at DOY = n-1 onto EOF space for DOY = n. This is done to reduce spurious oscillations between EOFs
       on sequential days
    #. Rotate the projected EOFs by 1/366 (or 1/365) of the December --> January discontinuity each day to ensure 
       continuity across December to January
    #. Renormalize the EOFs to have a length of 1 (this is a small adjustment to account for small numerical errors).

    See documentation of the methods :py:func:`~mjoindices.omi.postprocessing_original_kiladis2014.correct_spontaneous_sign_changes_in_eof_series`
    for EOF sign flipping

    Note that it is recommended to use the function :py:func:`~mjoindices.omi.omi_calculator.calc_eofs_from_olr` to cover
    the complete algorithm.

    :param eofdata: The EOF series, which should be post processed.
    :param sign_doy1reference: See :py:func:`~mjoindices.omi.postprocessing_original_kiladis2014.correct_spontaneous_sign_changes_in_eof_series`.

    :return: The postprocessed series of EOFs

    """
    
    pp_eofs = pp_kil2014.correct_spontaneous_sign_changes_in_eof_series(eofdata, doy1reference=sign_doy1reference)
    rot_eofs = rotate_eofs(pp_eofs)
    norm_eofs = normalize_eofs(rot_eofs)
    
    return norm_eofs


def rotate_eofs(orig_eofs: eof.EOFDataForAllDOYs) -> eof.EOFDataForAllDOYs:
    """
    Rotate EOFs at each DOY to

    #. align with the EOFs of the previous day and
    #. be continuous across December to January boundary.

    Described more in detail in :py:func:'post_process_eofs_rotation'

    :param orig_eofs: Calculated EOFs, which already has algined signs between neighboring DOYs.

    :return: set of rotated EOFs.
    """

    delta = calculate_angle_from_discontinuity(orig_eofs)

    return rotate_each_eof_by_delta(orig_eofs, delta)


def rotation_matrix(delta):
    """
    Creates a 2d rotation matrix for corresponding delta.

    :param delta: Scalar angle, in radians, of desired rotation.

    :return: 2x2 rotation matrix that can be used to rotate an Nx2 matrix in the x-y plane counterclockwise 
        by delta.

    """
    return np.array([[np.cos(delta), -np.sin(delta)],[np.sin(delta), np.cos(delta)]])


def calculate_angle_from_discontinuity(orig_eofs: eof.EOFDataForAllDOYs):
    """
    Project the matrix to align with the EOFs from the previous DOY and calculate the resulting
    discontinuity between January 1 and January 1 after one year of projections. Divide by the number of days 
    in one year to determine the discontinuity used for rotation. 

    :param orig_eofs: calculated EOFs, after signs have been changed via spontaneous_sign_changes

    :return: float of angular discontinuity between EOF1 on DOY1 and EOF1 on DOY1 after a full year of 
        projection, divided by the length of the year.

    """

    list_of_doys = tools.doy_list(orig_eofs.no_leap_years)
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

    # determine whether to rotate clockwise or counterclockwise, based on angle of E1 from projected
    # E2 and angle of E2 from projected E1
    cross_angle = np.dot(doy1.eof1vector, rots[1,:])/(np.linalg.norm(doy1.eof1vector)*np.linalg.norm(rots[1,:]))
    if cross_angle <= 0:
        return -discont/ndoys
    else: 
        return discont/ndoys


def rotate_each_eof_by_delta(orig_eofs: eof.EOFDataForAllDOYs, delta: float) -> eof.EOFDataForAllDOYs:
    """
    Use delta calculated by optimization function to rotate original EOFs by delta.
    First projects EOFs from DOY n-1 onto EOF space for DOY n, then rotates projected
    EOFs by small angle delta. 

    :param orig_eofs: calculated EOFs, signs have been changed via spontaneous_sign_changes
    :param delta: scalar by which to rotate EOFs. Calculated as the angular discontinuity between EOF1 on DOY1 
        and EOF1 on DOY1 after a full year of projection, divided by the length of the year.

    :return: new EOFdata with projected and rotated EOFs.

    """

    R = rotation_matrix(delta)

    doy1 = orig_eofs.eofdata_for_doy(1)   
    list_of_doys = tools.doy_list(orig_eofs.no_leap_years)
    
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

    return eof.EOFDataForAllDOYs(eofdata_rotated, orig_eofs.no_leap_years)


def normalize_eofs(orig_eofs: eof.EOFDataForAllDOYs) -> eof.EOFDataForAllDOYs:
    """
    Normalize all EOFs to have a magnitude of 1.

    :param eofdata: The rotated EOF series.

    :return: normalized EOFdata for all days.
    """

    list_of_doys = tools.doy_list(orig_eofs.no_leap_years)

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

    return eof.EOFDataForAllDOYs(eofdata_normalized, orig_eofs.no_leap_years) 


def angle_between_eofs(reference: eof.EOFData, target=eof.EOFData):
    """
    Calculates angle between two EOF vectors to determine their "closeness" :math:`theta = arccos(t . r / (||r||*||t||))`.

    :param reference: The reference-EOFs. This is usually the EOF pair of the previous or "first" DOY.
    :param target: The EOF data from the target DOY.

    :return: A tuple of the angles between the reference and target EOFs for both EOF1 and EOF2
    """

    angle1 = angle_btwn_vectors(reference.eof1vector, target.eof1vector)
    angle2 = angle_btwn_vectors(reference.eof2vector, target.eof2vector)

    return (angle1, angle2)


def angle_btwn_vectors(vector1, vector2):
    """
    Calculates the angle between vectors, :math:`theta = arccos(t . r / (||r||*||t||))`.

    :param vector1: 1d vector, generally corresponding to EOF1 or 2 from some DOY
    :param vector2: 1d vector, generally corresponding to EOF1 or 2 from a different DOY

    :return: scalar angle between vectors 1 and 2, in radians. 
    """

    return np.arccos(np.clip(np.dot(vector1, vector2)
                             /(np.linalg.norm(vector1)*np.linalg.norm(vector2)),-1.,1.))
