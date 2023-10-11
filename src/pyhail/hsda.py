"""
HSDA implementation (Hail Size Discrimination Algorthim)
This algorthim was developed by Ortega et al. 2016 doi:10.1175/JAMC-D-15-0203.1 and Ryzhkov et al. 2013 doi:10.1175/JAMC-D-13-074.1

Joshua Soderholm - 15 June 2018
"""

from pyhail import common, hsda_mf

from numba import jit
import numpy as np
import netCDF4

def main(
    radar,
    levels,
    hca_hail_idx,
    dzdr=0,
    q=None,
    zh_name="corrected_reflectivity",
    zdr_name="corrected_differential_reflectivity",
    rhv_name="cross_correlation_ratio",
    hca_name="radar_echo_classification",
    heights_fieldname='gate_z'
):

    """
    Wrapper function for HSDA processing

    Parameters:
    ===========
    radar: struct
        Py-ART radar object.
    levels : list of length 2
        height above sea level (m) of the wet bulb freezing level and -25C level (in any order)
    hca_hail_idx: list
        index of hail related fields in classification to apply HSDA
    dzdr:
        offset for differential reflectivity
    q:
        quality array of length 3 for [ZH, ZDR, RHV] defined by section 3 in https://journals.ametsoc.org/view/journals/wefo/24/3/2008waf2222205_1.xml
    ####_name: string
        field name from radar object

    Returns:
    ========
    hsda: ndarray
        hsda classe array (1 = small < 25, 2 = large 25-50, 3 = giant > 50

    """
    # metadata
    classes = (
        "1: Small Hail (< 25 mm); 2: Large Hail (25 - 50 mm); 3: Giant Hail (> 50 mm)"
    )

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 25 deg C isotherm
    wbt_minus25C = max(levels)
    wbt_0C = min(levels)

    # load data
    zh_cf = radar.fields[zh_name]["data"]
    zdr_cf = radar.fields[zdr_name]["data"]
    rhv_cf = radar.fields[rhv_name]["data"]
    hca = radar.fields[hca_name]["data"]

    # check for any valid data
    hail_mask = np.isin(hca, hca_hail_idx)
    hsda = np.zeros(hca.shape, dtype=int)
    # skip processing if there's no valid hail pixels
    if not np.any(hail_mask):
        return {
            "data": np.ma.masked_array(hsda, True, dtype=int),
            "units": "NA",
            "long_name": "Hail Size Discrimination Algorithm",
            "description:": "Hail Size Discrimination Algorithm developed by Ryzhkov et al. (2013) doi:10.1175/JAMC-D-13-074.1 and Ortega et al. (2016) doi:10.1175/JAMC-D-15-0203.1",
            "comments": classes,
        }

    # smooth radar data
    zh_cf_smooth = common.smooth_ppi_rays(zh_cf, 5)
    zdr_cf_smooth = common.smooth_ppi_rays(zdr_cf, 5)
    rhv_cf_smooth = common.smooth_ppi_rays(rhv_cf, 5)
    
    # generate quality vector if none exists
    if q is None:
        q = {"zh":np.ones(hca.shape), "zdr":np.ones(hca.shape), "rhv":np.ones(hca.shape)}
    
    # calc pixel alt
    try:
        alt = radar.fields[heights_fieldname]['data']
    except:
        rg, azg = np.meshgrid(radar.range["data"], radar.azimuth["data"])
        _, eleg = np.meshgrid(radar.range["data"], radar.elevation["data"])
        _, _, alt_arl = common.antenna_to_cartesian(rg / 1000, azg, eleg)
        # convert from ARL to ASL (required when using NWP products)
        alt = alt_arl + radar.altitude['data'][0]
    
    # find all pixels in hca which match the hail classes
    # for each pixel, apply transform
    hail_idx = np.where(hail_mask)
    
    # loop through every pixel
    # check for valid hail pixels
    try:
        # loop through every hail pixel
        for i in np.nditer(hail_idx):
            
            #extract altitude
            tmp_alt = alt[i]
            
            #extract radar data
            tmp_zh = zh_cf_smooth[i]
            tmp_zdr = zdr_cf_smooth[i]
            tmp_rhv = rhv_cf_smooth[i]
            
            #extract quality
            tmp_q_zh = q["zh"][i]
            tmp_q_zdr = q["zdr"][i]
            tmp_q_rhv = q["rhv"][i]
            tmp_q = np.array([tmp_q_zh, tmp_q_zdr, tmp_q_rhv])
            
            #check for valid values
            if (
                np.ma.is_masked(tmp_zh)
                or np.ma.is_masked(tmp_zdr)
                or np.ma.is_masked(tmp_rhv)
            ):
                continue
            
            # allocate alt field
            if tmp_alt >= wbt_minus25C:
                alt_index = 0
            elif tmp_alt >= wbt_0C:
                alt_index = 1
            elif tmp_alt >= (wbt_0C - 1000):
                alt_index = 2
            elif tmp_alt >= (wbt_0C - 2000):
                alt_index = 3
            elif tmp_alt >= (wbt_0C - 3000):
                alt_index = 4
            else:
                alt_index = 5

            # build membership functions
            w, mf_h1, mf_h2, mf_h3 = hsda_mf.build_mf(alt_index, tmp_zh, dzdr)
            w = np.array(w)
            mf_h1 = np.array(mf_h1)
            mf_h2 = np.array(mf_h2)
            mf_h3 = np.array(mf_h3)
            
            #calculate hsda value
            pixel_hsda = h_sz(tmp_alt, tmp_zh, tmp_zdr, tmp_rhv, mf_h1, mf_h2, mf_h3, tmp_q, w, dzdr)
            hsda[i] = pixel_hsda
        
    except Exception as e:
        print("error in HSDA: ", e, 'time:', radar.time['units'])
    
        
    # combine data masks
    combined_mask = np.ones((radar.nrays, radar.ngates)).astype("bool")
    for field in [zh_name, zdr_name, rhv_name]:
        combined_mask *= ~radar.fields[field]["data"].mask
    hsda_masked = np.ma.masked_array(hsda, ~combined_mask, dtype=int)

    # generate meta
    hsda_meta = {
        "data": hsda_masked,
        "units": "NA",
        "long_name": "Hail Size Discrimination Algorithm",
        "comments": classes,
    }
    # return radar object
    return hsda_meta

@jit(nopython=True)
def h_sz(alt, zh, zdr, rhv, mf_h1, mf_h2, mf_h3, q, w, dzdr):

    """
    calculates the hail size class for a radar voxel

    Parameters:
    ===========
    alt: float
        altitude of voxel (m) ASL
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
        [zh weight in HSDA for altitude, zdr weight in HSDA for altitude, rhv weight in HSDA for altitude]
    dzdr: float
        offset for differential reflectivity

    Returns:
    ========
    out: ndarray
        hail size class for every valid element (1: <25mm, 2: 25-50mm, 3: >50mm)

    """

    # small hail
    h1_ag = calc_ag(mf_h1, zh, zdr, rhv, q, w, dzdr)
    # large hail
    h2_ag = calc_ag(mf_h2, zh, zdr, rhv, q, w, dzdr)
    # giant hail
    h3_ag = calc_ag(mf_h3, zh, zdr, rhv, q, w, dzdr)

    # find last (largest) max ag
    ag_vec = np.array([h1_ag, h2_ag, h3_ag])
    max_ag = np.nanmax(ag_vec)
    out = np.where(ag_vec == max_ag)
    if len(out) == 0:
        out = 0 #entirely nan/invalid data, so no hail assignment
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
def calc_ag(mf, zh, zdr, rhv, q, w, dzdr):

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
        [zh weight in HSDA for altitude, zdr weight in HSDA for altitude, rhv weight in HSDA for altitude]
    dzdr: float
        zdr offset
        

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
    zh_mf_coeff = mf[0,:]
    zdr_mf_coeff = mf[1,:]
    rhv_mf_coeff = mf[2,:]
    
    zh_mf = trapmf(zh, zh_mf_coeff[0], zh_mf_coeff[1], zh_mf_coeff[2], zh_mf_coeff[3])
    zdr_mf = trapmf(zdr, zdr_mf_coeff[0], zdr_mf_coeff[1], zdr_mf_coeff[2], zdr_mf_coeff[3])
    rhv_mf = trapmf(rhv, rhv_mf_coeff[0], rhv_mf_coeff[1], rhv_mf_coeff[2], rhv_mf_coeff[3])

    # rule 1
    if np.min(np.array([zh_mf, zdr_mf, rhv_mf])) < 0.2:
        out = 0
    else:
        # calc h_ag
        out = ((w_zh * q_zh * zh_mf) +
                (w_zdr * q_zdr * zdr_mf) +
                (w_rhv * q_rhv * rhv_mf)) / (w_zh * q_zh + w_zdr * q_zdr + w_rhv * q_rhv)

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
