"""
HSDA implementation (Hail Size Discrimination Algorthim)
This algorthim was developed by Ortega et al. 2016 doi:10.1175/JAMC-D-15-0203.1 
and Ryzhkov et al. 2013 doi:10.1175/JAMC-D-13-074.1

Joshua Soderholm - 15 June 2018
"""

from pyhail import hsda_mf, common
from numba import jit
import numpy as np
import copy


def pyart(
    radar,
    reflectivity_fname,
    differential_reflectivity_fname,
    cross_correlation_ratio_fname,
    radar_classification_fname,
    levels,
    hca_hail_idx,
    hsda_fname="hsda",
    dzdr=0,
    q=None,
):
    """

    wrapper function using pyart object

    Parameters:
    ===========
    radar: class
        pyart radar object
    reflectivity_fname: string
        name of reflectivity field
    differential_reflectivity_fname: string
        name of differential reflectivity field
    cross_correlation_ratio_fname: string
        name of the cross correlation ratio field
    radar_classification_fname: string
        name of the radar classification field
    levels: list of length 2
        altitude above radar level of the 0C and -20C levels in meters
    hca_hail_idx: list of integers
        index of hail related fields in classification to apply HSDA
    hsda_fname: string
        name of HSDA field
    dzdr: float
        calibration value for ZDR
    q: array
        quality array of length 3 for [ZH, ZDR, RHV] defined by section 3 in 
        https://journals.ametsoc.org/view/journals/wefo/24/3/2008waf2222205_1.xml
    Returns:
    ========
    radar: pyart radar class
        updated with hsda fieldIAG

    """
    # init radar fields
    empty_radar_field = {
        "data": np.zeros((radar.nrays, radar.ngates)),
        "units": "",
        "long_name": "",
        "description": "",
        "comments": "",
    }
    radar.add_field(hsda_fname, copy.deepcopy(empty_radar_field))

    # for each sweep
    radar_altitude = radar.altitude["data"][0]
    for sweep in range(radar.nsweeps):
        # load sweep metadata
        _, _, gate_z = radar.get_gate_x_y_z(sweep)
        # run hsda
        hsda_meta = main(
            radar.get_field(sweep, reflectivity_fname, copy=True).filled(np.nan),
            radar.get_field(sweep, differential_reflectivity_fname, copy=True).filled(
                np.nan
            ),
            radar.get_field(sweep, cross_correlation_ratio_fname, copy=True).filled(np.nan),
            radar.get_field(sweep, radar_classification_fname, copy=True),
            gate_z + radar_altitude,
            levels,
            hca_hail_idx,
            dzdr,
            q,
        )
        # add data back into pyart object
        radar.fields[hsda_fname]["data"][radar.get_slice(sweep)] = hsda_meta["data"]

    # add metadata
    radar.fields[hsda_fname]["data"][radar.get_slice(sweep)] = hsda_meta["data"]

    return radar


def pyodim(
    datasets,
    reflectivity_fname,
    differential_reflectivity_fname,
    cross_correlation_ratio_fname,
    radar_classification_fname,
    levels,
    hca_hail_idx,
    z_fname="z",
    hsda_fname="hsda",
    dzdr=0,
    q=None,
):
    """

    wrapper function using pyodim dict

    Parameters:
    ===========
    datasets: list of dicts
        pyodim dataset
    filename: string
        full path to source
    reflectivity_fname: string
        name of reflectivity field
    differential_reflectivity_fname: string
        name of differential reflectivity field
    cross_correlation_ratio_fname: string
        name of the cross correlation ratio field
    radar_classification_fname: string
        name of the radar classification field
    levels: list of length 2
        altitude above radar level of the 0C and -20C levels in meters
    hca_hail_idx: list of integers
        index of hail related fields in classification to apply HSDA
    hsda_fname: string
        name of HSDA field
    dzdr: float
        calibration value for ZDR
    q: array
        quality array of length 3 for [ZH, ZDR, RHV] defined by section 3 in 
        https://journals.ametsoc.org/view/journals/wefo/24/3/2008waf2222205_1.xml
    Returns:
    ========
    datasets: list of dicts
        updated with hsda field

    """
    for sweep_idx, _ in enumerate(datasets):
        # run hsda
        hsda_meta = main(
            datasets[sweep_idx][reflectivity_fname].values,
            datasets[sweep_idx][differential_reflectivity_fname].values,
            datasets[sweep_idx][cross_correlation_ratio_fname].values,
            datasets[sweep_idx][radar_classification_fname].values,
            datasets[sweep_idx][z_fname].values + datasets[0].attrs["height"],
            levels,
            hca_hail_idx,
            dzdr,
            q,
        )

        # add new fields
        datasets[sweep_idx] = datasets[sweep_idx].merge(
            {hsda_fname: (("azimuth", "range"), hsda_meta["data"])}
        )
        # metadata
        datasets[sweep_idx][hsda_fname] = common.add_pyodim_metadata(
            datasets[sweep_idx][hsda_fname], hsda_meta
        )


    return datasets


def _smooth_ppi_rays(ppi_data, n):
    """
    Apply a smoothing average filter of size n over ppi_data
    (rays are columns).

    Parameters
    ----------
    ppi_data : ndarray
        PPI data.
    n : float
        Smoothing kernel size (must be odd).

    Returns
    -------
    out : ndarray
        Ray smoothed ppi.

    """
    # calculate offset from edges
    offset = int((n - 1) / 2)
    # init ppi cumulative sum with zero values in first row
    zero_mat = np.zeros((ppi_data.shape[0], 1))
    ppi_cs = np.hstack((zero_mat, ppi_data))
    # calculate cumulative sum
    ppi_cs = np.nancumsum(ppi_cs, axis=1)
    # calculate simple moving average
    ppi_sma = (ppi_cs[:, n:] - ppi_cs[:, :-n]) / float(n)
    # stack data in output with zeros
    out = np.hstack((ppi_data[:, :offset], ppi_sma, ppi_data[:, -offset:]))

    return out


def main(
    reflectivity_sweep,
    differential_reflectivity_sweep,
    cross_correlation_sweep,
    classification_sweep,
    gate_z_sweep,
    levels,
    hca_hail_idx,
    dzdr=0,
    q=None,
):
    """
    Wrapper function for HSDA processing

    Parameters:
    ===========
    reflectivity_sweep: 2d ndarray
        reflectivity data in an array with dimensions (azimuth, range)
    differential_reflectivity_sweep: 2d ndarray
        differential reflectivity data in an array with dimensions (azimuth, range)
    cross_correlation_sweep: 2d ndarray
        cross correlation data in an array with dimensions (azimuth, range)
    classification_sweep: 2d ndarray
        classification data in an array with dimensions (azimuth, range)
    gate_z_sweep: 2d ndarray
        altitude above sea level (m) of each gate
    levels : list of length 2
        height above sea level (m) of the wet bulb freezing level and -25C level (in any order)
    hca_hail_idx: list
        index of hail related fields in classification to apply HSDA
    dzdr:
        offset for differential reflectivity
    q:
        quality array of length 3 for [ZH, ZDR, RHV] defined by section 3 in 
        https://journals.ametsoc.org/view/journals/wefo/24/3/2008waf2222205_1.xml

    Returns:
    ========
    hsda: 2d ndarray
        hsda classe array (1 = small < 25, 2 = large 25-50, 
        3 = giant > 50 with dimensions (azimuth, range)

    """
    # metadata
    classes = (
        "1: Small Hail (< 25 mm); 2: Large Hail (25 - 50 mm); 3: Giant Hail (> 50 mm)"
    )

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 25 deg C isotherm
    wbt_minus25c = max(levels)
    wbt_0c = min(levels)

    # check for any valid data
    hail_mask = np.isin(classification_sweep, hca_hail_idx)
    hsda_data = np.zeros(classification_sweep.shape)
    hsda_data[:] = np.nan  # set all to nan, which is masked
    # skip processing if there's no valid hail pixels
    if not np.any(hail_mask):
        return {
            "data": hsda_data,
            "units": "NA",
            "long_name": "Hail Size Discrimination Algorithm",
            "description:": ("Hail Size Discrimination Algorithm developed by Ryzhkov et al. (2013)"
                             " doi:10.1175/JAMC-D-13-074.1 and Ortega et al. (2016)"
                             " doi:10.1175/JAMC-D-15-0203.1"),
            "comments": classes,
        }

    # smooth radar data
    reflectivity_sweep_smooth = _smooth_ppi_rays(reflectivity_sweep, 5)
    differential_reflectivity_sweep_smooth = _smooth_ppi_rays(
        differential_reflectivity_sweep, 5
    )
    cross_correlation_sweep_smooth = _smooth_ppi_rays(cross_correlation_sweep, 5)

    # generate quality vector if none exists
    if q is None:
        q = {
            "zh": np.ones(hsda_data.shape),
            "zdr": np.ones(hsda_data.shape),
            "rhv": np.ones(hsda_data.shape),
        }

    # find all pixels in hca which match the hail classes
    # for each pixel, apply transform
    hail_idx = np.where(hail_mask)

    # loop through every pixel
    # check for valid hail pixels
    try:
        # loop through every hail pixel
        for i in np.nditer(hail_idx):

            # extract altitude
            tmp_alt = gate_z_sweep[i]

            # extract radar data
            tmp_zh = reflectivity_sweep_smooth[i]
            tmp_zdr = differential_reflectivity_sweep_smooth[i]
            tmp_rhv = cross_correlation_sweep_smooth[i]

            # extract quality
            tmp_q_zh = q["zh"][i]
            tmp_q_zdr = q["zdr"][i]
            tmp_q_rhv = q["rhv"][i]
            tmp_q = np.array([tmp_q_zh, tmp_q_zdr, tmp_q_rhv])

            # check for valid values
            if np.isnan(tmp_zh) or np.isnan(tmp_zdr) or np.isnan(tmp_rhv):
                continue

            # allocate alt field
            if tmp_alt >= wbt_minus25c:
                alt_index = 0
            elif tmp_alt >= wbt_0c:
                alt_index = 1
            elif tmp_alt >= (wbt_0c - 1000):
                alt_index = 2
            elif tmp_alt >= (wbt_0c - 2000):
                alt_index = 3
            elif tmp_alt >= (wbt_0c - 3000):
                alt_index = 4
            else:
                alt_index = 5

            # build membership functions
            w, mf_h1, mf_h2, mf_h3 = hsda_mf.build_mf(alt_index, tmp_zh, dzdr)
            w = np.array(w)
            mf_h1 = np.array(mf_h1)
            mf_h2 = np.array(mf_h2)
            mf_h3 = np.array(mf_h3)

            # calculate hsda value
            pixel_hsda = h_sz(
                tmp_zh, tmp_zdr, tmp_rhv, mf_h1, mf_h2, mf_h3, tmp_q, w
            )
            hsda_data[i] = pixel_hsda

    except Exception as e:
        print("error in HSDA: ", e)

    # propagate nan as needed
    hsda_data[np.isnan(reflectivity_sweep)] = np.nan
    hsda_data[np.isnan(differential_reflectivity_sweep)] = np.nan
    hsda_data[np.isnan(cross_correlation_sweep)] = np.nan

    # generate meta
    hsda_meta = {
        "data": hsda_data,
        "units": "NA",
        "long_name": "Hail Size Discrimination Algorithm",
        "comments": classes,
    }
    # return radar object
    return hsda_meta


@jit(nopython=True)
def h_sz(zh, zdr, rhv, mf_h1, mf_h2, mf_h3, q, w):
    """
    calculates the hail size class for a radar voxel

    Parameters:
    ===========
    zh: float
        zh value for voxel (dbz)
    zdr: float
        zdr value for voxel (db)
    rhv: float
        CC value for voxel
    mf_h1: array of floats of length 4
        membership function shape for small hail
    mf_h2: array of floats of length 4
        membership function shape for large hail
    mf_h3: array of floats of length 4
        membership function shape for giant hail
    q: array of floats of length 3
        [zh quality, zdr quality, rhv quality]
    w: array of floats of length 3
        [zh weight in HSDA for altitude, zdr weight in HSDA for altitude, 
        rhv weight in HSDA for altitude]

    Returns:
    ========
    out: ndarray
        hail size class for every valid element (1: <25mm, 2: 25-50mm, 3: >50mm)

    """

    # small hail
    h1_ag = calc_ag(mf_h1, zh, zdr, rhv, q, w)
    # large hail
    h2_ag = calc_ag(mf_h2, zh, zdr, rhv, q, w)
    # giant hail
    h3_ag = calc_ag(mf_h3, zh, zdr, rhv, q, w)

    # find last (largest) max ag
    ag_vec = np.array([h1_ag, h2_ag, h3_ag])
    max_ag = np.nanmax(ag_vec)
    out = np.where(ag_vec == max_ag)
    if len(out) == 0:
        out = 0  # entirely nan/invalid data, so no hail assignment
    else:
        out = out[0][-1] + 1  # last item, using 1,2,3 indexing
        # rule 2
        if max_ag < 0.6:
            out = 1
        # rule 3
        if out > 1 and zdr >= 2:
            out = 1

    return out


@jit(nopython=True)
def calc_ag(mf, zh, zdr, rhv, q, w):
    """
    calculates the polarmetic aggregates for a hail size class

    Parameters:
    ===========
    h_field: string
        hail size field name (h1,h2 or h3)
    alt_field: string
        alt field name (alt1,...alt6)
    zh: float
        zh value for voxel (dbz)
    zdr: float
        zdr value for voxel (db)
    rhv: float
        CC value for voxel
    mf: array of floats of length 4
        membership function shape
    q: array of floats of length 3
        [zh quality, zdr quality, rhv quality]
    w: array of floats of length 3
        [zh weight in HSDA for altitude, zdr weight in HSDA for altitude, 
        rhv weight in HSDA for altitude]


    Returns:
    ========
    out: ndarray
        aggregate value for hail size class

    """
    q_zh = q[0]
    q_zdr = q[1]
    q_rhv = q[2]
    w_zh = w[0]
    w_zdr = w[1]
    w_rhv = w[2]
    zh_mf_coeff = mf[0, :]
    zdr_mf_coeff = mf[1, :]
    rhv_mf_coeff = mf[2, :]

    zh_mf = trapmf(zh, zh_mf_coeff[0], zh_mf_coeff[1], zh_mf_coeff[2], zh_mf_coeff[3])
    zdr_mf = trapmf(
        zdr, zdr_mf_coeff[0], zdr_mf_coeff[1], zdr_mf_coeff[2], zdr_mf_coeff[3]
    )
    rhv_mf = trapmf(
        rhv, rhv_mf_coeff[0], rhv_mf_coeff[1], rhv_mf_coeff[2], rhv_mf_coeff[3]
    )

    # rule 1
    if np.min(np.array([zh_mf, zdr_mf, rhv_mf])) < 0.2:
        out = 0
    else:
        # calc h_ag
        out = (
            (w_zh * q_zh * zh_mf) + (w_zdr * q_zdr * zdr_mf) + (w_rhv * q_rhv * rhv_mf)
        ) / (w_zh * q_zh + w_zdr * q_zdr + w_rhv * q_rhv)

    return out


@jit(nopython=True)
def trapmf(x, a, b, c, d):
    """
    Trapezoidal membership function generator.
    Parameters
    ========
    x : single element array like
    abcd : 1d array, length 4
        Four-element vector.  Ensure a <= b <= c <= d.
    Returns
    ========
    y : 1d array
        Trapezoidal membership function.
    """

    assert (
        a <= b and b <= c and c <= d
    ), "abcd requires the four elements \
                                          a <= b <= c <= d."
    # Compute y1
    if x > a and x < b:
        y = (x - a) / (b - a)
    elif x >= b and x <= c:
        y = 1.0
    elif x > c and x < d:
        y = (d - x) / (d - c)
    else:
        y = 0.0

    return y
