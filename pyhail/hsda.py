"""
HSDA sub-module of pyhail

Contains the dual pol hail size discrimination algorithm
requires temperature, reflectivity, differential reflectivity and cross correlation

Joshua Soderholm - 15 June 2018
"""

from pyhail import common, hsda_mf
import numpy as np
import netCDF4

def main(radar, snd_dict, hca_hail_idx,
        dzdr=0,
        zh_name='corrected_reflectivity',
        zdr_name='corrected_differential_reflectivity',
        rhv_name='cross_correlation_ratio',
        phi_name='corrected_differential_phase',
        snr_name='signal_to_noise_ratio',
        cbb_name='cbb',
        hca_name='radar_echo_classification'):

    """
    Wrapper function for HSDA processing

    Parameters:
    ===========
    radar: struct
        Py-ART radar object.
    snd_input: string
        sounding full filename (inc path)
    hca_hail_idx: list
        index of hail related fields in classification to apply HSDA
    dzdr:
        offset for differential reflectivity
    ####_name: string
        field name from radar object

    Returns:
    ========
    hsda: ndarray
        hsda classe array (1 = small < 25, 2 = large 25-50, 3 = giant > 50

    """

    #load sounding data
    snd_temp = snd_dict["t"]
    snd_geop = snd_dict["z"]
    snd_rh   = snd_dict["r"]
    #calc wbt
    snd_wbt  = common.wbt(snd_temp,snd_rh)
    #run interpolation
    wbt_minus25C = common.sounding_interp(snd_wbt,snd_geop,-25)/1000
    wbt_0C       = common.sounding_interp(snd_wbt,snd_geop,0)/1000

    #building consts
    const  = {'wbt_minus25C' : wbt_minus25C, 'wbt_0C' : wbt_0C, 'dzdr' : dzdr, 'hca_hail_idx':hca_hail_idx}

    #load data
    zh_cf  = radar.fields[zh_name]['data']
    zdr_cf = radar.fields[zdr_name]['data']
    rhv_cf = radar.fields[rhv_name]['data']
    phi_cf = radar.fields[phi_name]['data']
    snr_cf = radar.fields[snr_name]['data']
    cbb_cf = radar.fields[cbb_name]['data']
    hca    = radar.fields[hca_name]['data']

    #smooth radar data
    zh_cf_smooth  = common.smooth_ppi_rays(zh_cf,5)
    zdr_cf_smooth = common.smooth_ppi_rays(zdr_cf,5)
    rhv_cf_smooth = common.smooth_ppi_rays(rhv_cf,5)

    #build membership functions
    w, mf   = hsda_mf.build_mf()
    
    #generate quality vector
    q = hsda_q(zh_cf_smooth, phi_cf, rhv_cf_smooth, snr_cf, cbb_cf, cbb_threshold=0.5)
    
    #calc pixel alt
    rg, azg   = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg  = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, alt = common.antenna_to_cartesian(rg / 1000.0, azg, eleg)  
    
    #find all pixels in hca which match the hail classes
    #for each pixel, apply transform
    hail_mask = np.isin(hca, const['hca_hail_idx'])
    hail_idx  = np.where(hail_mask)
    #loop through every pixel
    hsda = np.zeros(hca.shape)
    #check for valid hail pixels
    try:
        #loop through every hail pixel
        for i in np.nditer(hail_idx):
            tmp_alt   = alt[i]
            tmp_zh    = zh_cf_smooth[i]
            tmp_zdr   = zdr_cf_smooth[i]
            tmp_rhv   = rhv_cf_smooth[i]
            tmp_q_zh  = q['zh'][i]
            tmp_q_zdr = q['zdr'][i]
            tmp_q_rhv = q['rhv'][i]     
            tmp_q     = {'zh':tmp_q_zh, 'zdr':tmp_q_zdr, 'rhv':tmp_q_rhv}
            if np.ma.is_masked(tmp_zh) or np.ma.is_masked(tmp_zdr) or np.ma.is_masked(tmp_rhv):
                continue
            if np.ma.is_masked(tmp_q_zh) or np.ma.is_masked(tmp_q_zdr) or np.ma.is_masked(tmp_q_rhv):
                continue        
            pixel_hsda = h_sz(tmp_alt, tmp_zh, tmp_zdr, tmp_rhv, mf, tmp_q, w, const)
            hsda[i]    = pixel_hsda
    except Exception as e:
        print('error in HSDA: ', e)
        

    #generate meta        
    the_comments = "1: Small Hail (< 25 mm); 2: Large Hail (25 - 50 mm); 3: Giant Hail (> 50 mm)"
    hsda_meta    = {'data': hsda, 'units': ' ', 'long_name': 'Hail Size Discrimination Algorithm',
                  'standard_name': 'HSDA', 'comments': the_comments}
    
    #return radar object
    return hsda_meta

def h_sz(alt,zh,zdr,rhv,mf,q,w,const):

    """
    calculates the hail size class for a radar voxel

    Parameters:
    ===========
    alt: float
        altitude of voxel (km)
    zh: float
        zh value for voxel (dbz)
    zdr: float
        zdr value for voxel (db)
    rhv: float
        CC value for voxel
    mf: dict
        hsda memebership functions
    q: dict
        confidence vectors
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
    q: dict
        confidence vectors
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
    w_zh  = w[''.join([alt_field, '.zh'])]
    w_zdr = w[''.join([alt_field, '.zdr'])]
    w_rhv = w[''.join([alt_field, '.rhv'])]

    #mf
    zh_mf  = h_mf(zh,  zh, mf[''.join([alt_field,'.',h_field, '.zh'])], const)
    zdr_mf = h_mf(zdr, zh, mf[''.join([alt_field,'.',h_field, '.zdr'])],const)
    rhv_mf = h_mf(rhv, zh, mf[''.join([alt_field,'.',h_field, '.rhv'])],const)

    #rule 1
    if np.min([zh_mf,zdr_mf,rhv_mf]) < 0.2:
        out = 0
    else:
        #calc h_ag
        out = np.sum([w_zh*q['zh']*zh_mf, w_zdr*q['zdr']*zdr_mf, w_rhv*q['rhv']*rhv_mf])  /np.sum([w_zh*q['zh'], w_zdr*q['zdr'], w_rhv*q['rhv']])

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
        x0     = fun0(offset, zh, const['dzdr'])

        fun1   = mf_const[1][0]
        offset = mf_const[1][1]
        x1     = fun1(offset, zh, const['dzdr'])

        fun2   = mf_const[2][0]
        offset = mf_const[2][1]
        x2     = fun2(offset, zh, const['dzdr'])

        fun3   = mf_const[3][0]
        offset = mf_const[3][1]
        x3     = fun3(offset, zh, const['dzdr'])

    else:
        x0 = mf_const[0]
        x1 = mf_const[1]
        x2 = mf_const[2]
        x3 = mf_const[3]

    #apply trap values for var to get membership function output  
    out = trapmf(np.array([var]), np.array([x0, x1, x2, x3]))

    return out

def hsda_q(dbzh, phi, rhv, snr, cbb, cbb_threshold):

    """
    calculates the membership function values for a polarmetic variable

    Parameters:
    ===========
    dbzh: array
        reflectivity (dBZ)
    phi: array
        differential phase
    rhv: array
       correlation coefficent
    snr: array
        signal to noise
    cbb: array
        cumulative beam blocking
    cbb_threshold: float
        fraction for cbb (typically 0.5)

    Returns:
    ========
    q: dict
        zh: array
            reflectivity quality array
        zdr: array
            differential reflectivity quality array
        rhv: array
            correlation coefficent quality array
    """
    
    #fill missing phidp values with 0
    phi = phi.filled(0)
    
    #parameters
    Ac = phi / 600
    Bc = 1.0 / ( 10 ** (0.1 * snr))
    Cc = phi / 300
    Dc = (1 - rhv) / 0.5
    Ec = 3.16228 / (10 ** (0.1 * snr))
    Fc = phi / 100
    Gc = Dc 
    Hc = 1 / (10 ** (0.1 * snr))

    #mask for low quality/dbzh
    Dc[rhv < 0.8] = 0
    Gc[rhv < 0.8] = 0
    Dc[dbzh < 25] = 0
    Gc[dbzh < 25] = 0

    #apply beam blocking percentage
    T = cbb / cbb_threshold

    #generate quality vectors
    q_zh  = np.exp( -0.69 * (Ac**2 + Bc**2 + T**2))
    q_zdr = np.exp( -0.69 * (Cc**2 + Dc**2 + T**2))
    q_rhv = np.exp( -0.69 * (Fc**2 + Gc**2 + Hc**2))

    #apply limits
    q_zh[q_zh<0]   = 0
    q_zh[q_zh>1]   = 1
    q_zdr[q_zdr<0] = 0
    q_zdr[q_zdr>1] = 1
    q_rhv[q_rhv<0] = 0
    q_rhv[q_rhv>1] = 1
    
    #pack in dict
    q = {'zh':q_zh, 'zdr':q_zdr, 'rhv':q_rhv}
    return q

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
