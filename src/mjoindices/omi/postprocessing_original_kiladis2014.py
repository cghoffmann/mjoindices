# -*- coding: utf-8 -*-

# Copyright (C) 2022 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindices

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Contact: christoph.hoffmann@uni-greifswald.de

"""
Contains the post-processing routines for the EOFs as described in the original OMI paper by :ref:`refKiladis2014`.

The original post-processing consists of two steps:

#. The signs of the EOFs for subsequent DOYs are aligned.
#. The EOFs for a period between the DOYs 293 and 316 are discarded and replaced by an interpolation between the EOFs
   of the mentioned DOYs 293 and 316.

.. seealso:: :py:mod:`mjoindices.omi.postprocessing_rotation_approach`

"""

from pathlib import Path
from typing import Tuple
import os.path
import inspect

import numpy as np
import warnings
import importlib
import scipy.interpolate

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.olr_handling as olr
import mjoindices.principal_components as pc
import mjoindices.omi.wheeler_kiladis_mjo_filter as wkfilter
import mjoindices.omi.quick_temporal_filter as qfilter
import mjoindices.tools as tools

def post_process_eofs_original_kiladis_approach(eofdata: eof.EOFDataForAllDOYs, sign_doy1reference: bool = True,
                      interpolate_eofs: bool = False, interpolation_start_doy: int = 293,
                      interpolation_end_doy: int = 316) -> eof.EOFDataForAllDOYs:
    """
    Executes the complete post-processing of a series of EOF pairs for all DOYs according to the original approach
    by :ref:`refKiladis2014`. This includes an alignment of EOF signs and an interpolation of the EOF functions in a given DOY
    window.

    See documentation of the functions :py:func:`correct_spontaneous_sign_changes_in_eof_series` and
    :py:func:`interpolate_eofs_between_doys` for further information.

    Note that it is recommended to use the function :py:func:`~mjoindices.omi.omi_calculator.calc_eofs_from_olr`
    to cover the complete algorithm.

    :param eofdata: The EOF series, which should be post-processed.
    :param sign_doy1reference: See :py:func:`correct_spontaneous_sign_changes_in_eof_series`.
    :param interpolate_eofs: If ``True``, the EOF sub-series between the given DOYs will be interpolated.
    :param interpolation_start_doy: See  :py:func:`interpolate_eofs_between_doys`.
    :param interpolation_end_doy: See :py:func:`interpolate_eofs_between_doys`.

    :return: The postprocessed series of EOFs
    """
    pp_eofs = correct_spontaneous_sign_changes_in_eof_series(eofdata, doy1reference=sign_doy1reference)
    if interpolate_eofs:
        pp_eofs = interpolate_eofs_between_doys(pp_eofs, start_doy=interpolation_start_doy,
                                                end_doy=interpolation_end_doy)
    return pp_eofs

def correct_spontaneous_sign_changes_in_eof_series(eofs: eof.EOFDataForAllDOYs,
                                                   doy1reference: bool = False) -> eof.EOFDataForAllDOYs:
    """
    Switches the signs of all pairs of EOFs (for all DOYs) if necessary, so that the signs are consistent for all DOYs.

    Note that the signs of the EOFs are not uniquely defined by the PCA. Hence, the sign may jump from one DOY to another,
    which can be improved using this function. As long as this step is performed before computing the PCs, it will not
    change the overall result.

    Generally, the sign of the EOFs for a specific DOY is changed if it differs from the sign of the EOF for the previous
    DOY. The EOFs for DOY 1 are by default aligned with the original calculation by :ref:`refKiladis2014`, resulting in
    an EOF series which is totally comparable to the original Kiladis (2014) calculation. This can be switched off, so
    that only the EOFs for the DOYs beginning with DOY 2 are aligned according to the sign of the EOFS for DOY 1.

    :param eofs: The EOF series for which the signs should be aligned.
    :param doy1reference: If ``True``, the EOFs of DOY 1 are aligned w.r.t to the original :ref:`refKiladis2014` calculation.

    :return: The EOFs with aligned signs.
    """
    switched_eofs = []
    if doy1reference is True:
        reference_path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) / "sign_reference"
        reference_eofs = eof.load_original_eofs_for_doy(reference_path, 1)
        if not reference_eofs.lat.size == eofs.lat.size \
                or not reference_eofs.long.size == eofs.long.size \
                or not np.all(reference_eofs.lat == eofs.lat) \
                or not np.all(reference_eofs.long == eofs.long):
            warnings.warn("References for the sign of the EOFs for DOY1 have to be interpolated to spatial grid of the"
                          " target EOFs. Treat results with caution.")
            f1 = scipy.interpolate.interp2d(reference_eofs.long, reference_eofs.lat, reference_eofs.eof1map,
                                            kind='linear')
            eof1map_interpol = f1(eofs.long, eofs.lat)
            f2 = scipy.interpolate.interp2d(reference_eofs.long, reference_eofs.lat, reference_eofs.eof2map,
                                            kind='linear')
            eof2map_interpol = f2(eofs.long, eofs.lat)
            reference_eofs = eof.EOFData(eofs.lat, eofs.long, eof1map_interpol, eof2map_interpol)
        corrected_doy1 = _correct_spontaneous_sign_change_of_individual_eof(reference_eofs, eofs.eofdata_for_doy(1))
    else:
        corrected_doy1 = eofs.eofdata_for_doy(1)
    switched_eofs.append(corrected_doy1)
    previous_eof = corrected_doy1
    for doy in tools.doy_list(eofs.no_leap_years)[1:]:
        corrected_eof = _correct_spontaneous_sign_change_of_individual_eof(previous_eof, eofs.eofdata_for_doy(doy))
        switched_eofs.append(corrected_eof)
        previous_eof = corrected_eof
    return eof.EOFDataForAllDOYs(switched_eofs, eofs.no_leap_years)



def _correct_spontaneous_sign_change_of_individual_eof(reference: eof.EOFData, target=eof.EOFData) -> eof.EOFData:
    """
    Switches the sign of a particular pair of EOFs (for a particular DOY) if necessary, so that is aligned with the
    reference.

    Note that the signs of the EOFs are not uniquely defined by the PCA. Hence, the sign may jump from one DOY to another,
    which can be improved using this function. As long as this step is performed before computing the PCs, it will not
    change the overall result.

    :param reference: The reference-EOFs. This is usually the EOF pair of the previous DOY.
    :param target: The EOFs of which the signs should be switched

    :return: The target EOFs with aligned signs.
    """
    if (np.mean(np.abs(target.eof1vector + reference.eof1vector))
            < np.mean(np.abs(
                target.eof1vector - reference.eof1vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof1_switched = -1 * target.eof1vector
    else:
        eof1_switched = target.eof1vector
    if (np.mean(np.abs(target.eof2vector + reference.eof2vector))
            < np.mean(np.abs(
                target.eof2vector - reference.eof2vector))):  # if abs(sum) is lower than abs(diff), than the signs are different...
        eof2_switched = -1 * target.eof2vector
    else:
        eof2_switched = target.eof2vector
    return eof.EOFData(target.lat,
                       target.long,
                       eof1_switched,
                       eof2_switched,
                       eigenvalues=target.eigenvalues,
                       explained_variances=target.explained_variances,
                       no_observations=target.no_observations)


def interpolate_eofs_between_doys(eofs: eof.EOFDataForAllDOYs, start_doy: int = 293,
                                  end_doy: int = 316) -> eof.EOFDataForAllDOYs:
    """
    Replaces the EOF1 and EOF2 functions for the range of DOYS between 2 given DOYs using a linear interpolation 
    between the 2 boundary DOYs.

    This should only rarely be used and has only been implemented to closely reproduce the original OMI values. There,
    the EOFs have also been replaced by an interpolation according to :ref:`refKiladis2014`. However, the period stated in
    :ref:`refKiladis2014` from 1 November to 8 November is too short. The authors have confirmed that the right
    interpolation period is from DOY 294 to DOY 315, which is used here as default value.

    ATTENTION: The corresponding statistical values (e.g., the explained variances) are not changed by this routine.
    So these values further on represent the original results of the PCA also for the interpolated EOFs.

    :param eofs: The complete EOF series, in which the interpolation takes place.
    :param start_doy: The DOY, which is used as the first point of the interpolation (i.e. start_doy + 1 is the first
        element that will be replaced by the interpolation.
    :param end_doy:  The DOY, which is used as the last point of the interpolation (i.e. end_doy - 1 is the last
        element that will be replaced by the interpolation.

    :return: The complete EOF series with the interpolated values.
    """
    doys = tools.doy_list(eofs.no_leap_years)
    start_idx = start_doy - 1
    end_idx = end_doy - 1
    eof_len = eofs.lat.size * eofs.long.size
    eofs1 = np.empty((doys.size, eof_len))
    eofs2 = np.empty((doys.size, eof_len))
    # Todo: Maybe this could be solved more efficiently
    # by using internal numpy functions for multidimenasional operations
    for (idx, doy) in enumerate(doys):
        eofs1[idx, :] = eofs.eof1vector_for_doy(doy)
        eofs2[idx, :] = eofs.eof2vector_for_doy(doy)

    for i in range(0, eof_len):
        eofs1[start_idx + 1:end_idx - 1, i] = np.interp(doys[start_idx + 1:end_idx - 1],
                                                        [doys[start_idx], doys[end_idx]],
                                                        [eofs1[start_idx, i], eofs1[end_idx, i]])
        eofs2[start_idx + 1:end_idx - 1, i] = np.interp(doys[start_idx + 1:end_idx - 1],
                                                        [doys[start_idx], doys[end_idx]],
                                                        [eofs2[start_idx, i], eofs2[end_idx, i]])
    interpolated_eofs = []
    for (idx, doy) in enumerate(doys):
        orig_eof = eofs.eofdata_for_doy(doy)
        interpolated_eofs.append(eof.EOFData(orig_eof.lat, orig_eof.long, np.squeeze(eofs1[idx, :]),
                                             np.squeeze(eofs2[idx, :]),
                                             explained_variances=orig_eof.explained_variances,
                                             eigenvalues=orig_eof.eigenvalues, no_observations=orig_eof.no_observations)
                                 )
    return eof.EOFDataForAllDOYs(interpolated_eofs, no_leap_years=eofs.no_leap_years)




