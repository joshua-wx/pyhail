"""
HSDA sub-module of pyhail

Contains the dual pol hail size discrimination algorithm
requires temperature, reflectivity, differential reflectivity and cross correlation

Joshua Soderholm - 15 June 2018
"""

from pyhail import common, hsda_mf
import numpy as np
import netCDF4
import pyart

def main(radar,snd_input,fieldnames,hca_hail_idx,dzdr):

    """
    Wrapper function for HSDA processing

    Parameters:
    ===========
    radar: struct
        Py-ART radar object.
    fieldnames: dict
        map to pyart field names
    snd_input: string
        sounding full filename (inc path)
    hca_hail_idx: list
        index of hail related fields in classification to apply HSDA
    dzdr:
        offset for differential reflectivity

    Returns:
    ========
    hsda: ndarray
        hsda classe array (1 = small < 25, 2 = large 25-50, 3 = giant > 50

    """

    #build sounding data
    snd_data = netCDF4.Dataset(snd_input)
    snd_temp = snd_data.variables["temp"][:]
    snd_geop = snd_data.variables["height"][:]
    snd_rh   = snd_data.variables["rh"][:]
    #calc wbt
    snd_wbt  = common.wbt(snd_temp,snd_rh)
    #run interpolation
    wbt_minus25C = common.sounding_interp(snd_wbt,snd_geop,-25)/1000
    wbt_0C       = common.sounding_interp(snd_wbt,snd_geop,0)/1000

    #building consts
    const  = {'wbt_minus25C' : wbt_minus25C, 'wbt_0C' : wbt_0C, 'dzdr' : dzdr, 'hca_hail_idx':hca_hail_idx}

    #load data
    zh_cf  = radar.fields[fieldnames['dbzh_corr']]['data']
    zdr_cf = radar.fields[fieldnames['zdr_corr']]['data']
    rhv_cf = radar.fields[fieldnames['rhv_corr']]['data']
    hca    = radar.fields[fieldnames['hca']]['data']

    #smooth radar data
    zh_cf_smooth  = common.smooth_ppi_rays(zh_cf,5)
    zdr_cf_smooth = common.smooth_ppi_rays(zdr_cf,5)
    rhv_cf_smooth = common.smooth_ppi_rays(rhv_cf,5)

    #build membership functions
    w, q, mf   = hsda_mf.build_mf()
    
    #calc pixel alt
    rg, azg   = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg  = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, alt = pyart.core.antenna_to_cartesian(rg / 1000.0, azg, eleg)  
    
    #find all pixels in hca which match the hail classes
    #for each pixel, apply transform
    hail_mask = np.isin(hca, const['hca_hail_idx'])
    hail_idx  = np.where(hail_mask)

    #loop through every pixel
    hsda = np.zeros(hca.shape)
    if hail_idx:
        for i in np.nditer(hail_idx):
            tmp_alt = alt[i]
            tmp_zh  = zh_cf_smooth[i]
            tmp_zdr = zdr_cf_smooth[i]
            tmp_rhv = rhv_cf_smooth[i]
            if np.ma.is_masked(tmp_zh) or np.ma.is_masked(tmp_zdr) or np.ma.is_masked(tmp_rhv):
                continue
            pixel_hsda = h_sz(tmp_alt,tmp_zh,tmp_zdr,tmp_rhv,mf,q,w,const)
            hsda[i]    = pixel_hsda

    #return radar object
    return hsda

def h_sz(alt,zh,zdr,rhv,mf,q,w,const):

    """
    calculates the hail size class for a radar voxel

    Parameters:
    ===========
    alt: ndarray
        altitude of voxel (km)
    zh: ndarray
        zh value for voxel (dbz)
    zdr: ndarray
        zdr value for voxel (db)
    rhv: ndarray
        CC value for voxel
    mf: dict
        hsda memebership functions
    q: float
        confidence constant
    w: dict
        field height interval weights
    const: dict
        containing dzdr and height constantssses in exisiting classification field
    dzdr: float
        offset for differential reflectivity

    Returns:
    ========
    out: ndarray
        hail size class for every valid element (1: <25mm, 2: 25-50mm, 3: >50mm)

    """

    #allocate alt field
    if alt >= const['wbt_minus25C']:
        alt_field = 'a1'
    elif alt >= const['wbt_0C']:
        alt_field = 'a2'
    elif alt >= (const['wbt_0C'] - 1):
        alt_field = 'a3'
    elif alt >= (const['wbt_0C'] - 2):
        alt_field = 'a4'
    elif alt >= (const['wbt_0C'] - 3):
        alt_field = 'a5'
    else:
        alt_field = 'a6'

    #small hail
    h1_ag = calc_ag('h1',alt_field,zh,zdr,rhv,mf,q,w,const)
    #large hail
    h2_ag = calc_ag('h2',alt_field,zh,zdr,rhv,mf,q,w,const)
    #giant hail
    h3_ag = calc_ag('h3',alt_field,zh,zdr,rhv,mf,q,w,const)

    #find last (largest) max ag
    ag_vec = np.array([h1_ag,h2_ag,h3_ag])
    max_ag = np.max(ag_vec)
    out    = np.where(ag_vec == max_ag)
    out    = out[0][-1] + 1 #last item, using 1,2,3 indexing

    #rule 2
    if max_ag < 0.6:
        out = 1
    #rule 3
    if out > 1 and zdr >= 2:
        out = 1

    return out

def calc_ag(h_field,alt_field,zh,zdr,rhv,mf,q,w,const):

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
    mf: dict
        hsda memebership functions
    q: float
        confidence constant
    w: dict
        field height interval weights
    const: dict
        containing dzdr and height constantssses in exisiting classification field

    Returns:
    ========
    out: ndarray
        aggregate value for hail size class

    """

    #weight
    w_zh  = w[''.join([alt_field,'.zh'])]
    w_zdr = w[''.join([alt_field,'.zdr'])]
    w_rhv = w[''.join([alt_field,'.rhv'])]

    #mf
    zh_mf  = h_mf(zh,zh, mf[''.join([alt_field,'.',h_field,'.zh'])], const)
    zdr_mf = h_mf(zdr,zh,mf[''.join([alt_field,'.',h_field,'.zdr'])],const)
    rhv_mf = h_mf(rhv,zh,mf[''.join([alt_field,'.',h_field,'.rhv'])],const)

    #rule 1
    if np.min([zh_mf,zdr_mf,rhv_mf]) < 0.2:
        out = 0
    else:
        #calc h_ag
        out = np.sum([w_zh*q*zh_mf,w_zdr*q*zdr_mf,w_rhv*q*rhv_mf])/np.sum([w_zh*q,w_zdr*q,w_rhv*q])

    return out

def h_mf(var,zh,mf_const,const):

    """
    calculates the membership function values for a polarmetic variable

    Parameters:
    ===========
    var: float
        variable value to apply to memebership functions
    h_sz: string
        hail size field name (h1,h2 or h3)
    zh: float
       zh value for voxel (dbz)
    const: dict
        containing dzdr and height constants
    mf_const: dict
        trap membership function values

    Returns:
    ========
    out: float
        membership function value for input var

    """

    #if cell, apply functions to get trap values
    if type(mf_const[0]) is list:
        fun0   = mf_const[0][0]
        offset = mf_const[0][1]
        x0     = fun0(offset,zh,const['dzdr'])

        fun1   = mf_const[1][0]
        offset = mf_const[1][1]
        x1     = fun1(offset,zh,const['dzdr'])

        fun2   = mf_const[2][0]
        offset = mf_const[2][1]
        x2     = fun2(offset,zh,const['dzdr'])

        fun3   = mf_const[3][0]
        offset = mf_const[3][1]
        x3     = fun3(offset,zh,const['dzdr'])

    else:
        x0 = mf_const[0]
        x1 = mf_const[1]
        x2 = mf_const[2]
        x3 = mf_const[3]

    #apply trap values for var to get membership function output  
    out = trapmf(np.array([var]),np.array([x0, x1, x2, x3]))

    return out


def trapmf(x, abcd):
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

    assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
    a, b, c, d = np.r_[abcd]

    assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                          a <= b <= c <= d.'

    y = 0

    # Compute y1
    if x > a and x < b:
        y = (x-a)/(b-a)
    elif x >= b and x <= c:
        y = 1
    elif x > c and x < d:
        y = (d-x)/(d-c)

    return y
