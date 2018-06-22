"""
common sub-module of pyhail

Contains commonly used functions.

Joshua Soderholm - 15 June 2018
"""

import numpy as np
from scipy.interpolate import interp1d
import wradlib as wrl

def wbt(temp,rh):
    """
    calculate wet bulb temperature from temperature and relative humidity. 

    Parameters:
    ===========
    temp: ndarray
        temperature data (degrees C)
    rh: ndarray
        relative humidity data (%)

    Returns:
    ========
    wb_temp: ndarray
        wet bulb temperature (degrees C)
    """
    wb_temp = temp * np.arctan(0.151977*(rh+8.313659)**0.5) + np.arctan(temp+rh) - np.arctan(rh-1.676331) + 0.00391838*(rh**1.5)*np.arctan(0.023101*rh) - 4.686035
    return wb_temp

def sounding_interp(snd_temp,snd_height,target_temp):
    """
    Provides an linear interpolated height for a target temperature using a sounding vertical profile. 
    Looks for first instance of temperature below target_temp from surface upward.

    Parameters:
    ===========
    snd_temp: ndarray
        temperature data (degrees C)
    snd_height: ndarray
        relative height data (m)
    target_temp: float
        target temperature to find height at (m)

    Returns:
    ========
    intp_h: float
        interpolated height of target_temp (m)
    """

    intp_h = np.nan

    #find index above and below freezing level
    mask      = np.where(snd_temp<target_temp)
    above_ind = mask[0][0]
    #check to ensure operation is possible
    if above_ind > 0:
        #index below 
        below_ind  = above_ind-1
        #apply linear interplation to points above and below target_temp
        set_interp = interp1d(snd_temp[below_ind:above_ind+1], snd_height[below_ind:above_ind+1], kind='linear')
        #apply interpolant
        intp_h     = set_interp(target_temp)   
        return intp_h
    else:
        return target_temp[0]

def smooth_ppi_rays(ppi_data,n):
    """
    Apply a smoothing average filter of size n over ppi_data (rays are columns)

    Parameters:
    ===========
    ppi_data: ndarray
        PPI data
    n: float
        smoothing kernel size (must be odd)

    Returns:
    ========
    out: ndarray
        ray smoothed ppi
    """
    #calculate offset from edges
    offset   = int((n-1)/2)
    #init ppi cumulative sum with zero values in first row
    zero_mat = np.ma.zeros((ppi_data.shape[0],1))
    ppi_cs   = np.ma.hstack((zero_mat,ppi_data))
    #calculate cumulative sum
    ppi_cs   = ppi_cs.cumsum(axis=1)
    #calculate simple moving average
    ppi_sma  = (ppi_cs[:,n:] - ppi_cs[:,:-n]) / float(n)
    #stack data in output with zeros
    out      = np.ma.hstack((ppi_data[:,:offset],ppi_sma,ppi_data[:,-offset:]))

    return out

def beam_blocking(radar,srtm_ffn):
    """
    Apply the wradlib beam blocking library for the target volume

    Parameters:
    ===========
    radar: pyart radar object
        
    srtm_ffn: string
        full path to srtm geotiff file

    Returns:
    ========
    radar_ccb: ndarray
        cumulative beam blocking for every pixel
    """
    #site parameters
    radar_lat  = radar.latitude['data'][0]
    radar_lon  = radar.longitude['data'][0]
    radar_alt  = radar.altitude['data'][0]
    sitecoords = (radar_lon, radar_lat, radar_alt)
    nsweeps    = radar.nsweeps
    nrays      = int(radar.nrays / nsweeps)
    nbins      = int(radar.ngates)
    el_list    = radar.fixed_angle['data']
    bw         = radar.instrument_parameters['radar_beam_width_h']['data'][0]
    range_res  = radar.range['meters_between_gates']

    #grid arrays
    r = np.arange(nbins) * range_res
    beamradius = wrl.util.half_power_radius(r, bw)

    #init output cumulative beam blocking
    radar_ccb = np.zeros((radar.nrays, radar.ngates))

    for tilt, el in enumerate(el_list):
        #indexcurrent slice
        sweep_idx = radar.get_slice(tilt)

        #calculate lon, lat and alt
        coord  = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
        coords = wrl.georef.spherical_to_proj(coord[..., 0],
                                              np.degrees(coord[..., 1]),
                                              coord[..., 2], sitecoords)
        lon       = coords[..., 0]
        lat       = coords[..., 1]
        alt       = coords[..., 2]
        polcoords = coords[..., :2]
        rlimits   = (lon.min(), lat.min(), lon.max(), lat.max())

        #read geotiff
        ds = wrl.io.open_raster(srtm_ffn)
        rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)

        # Clip the region inside our bounding box
        ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
        rastercoords = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
        rastervalues = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]

        # Map rastervalues to polar grid points
        polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoords, rastervalues,
                                                     polcoords, order=3,
                                                     prefilter=False)

        #calculate beam blocking for each bin
        PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
        PBB = np.ma.masked_invalid(PBB)

        #calculate beam blocking along each ray
        CBB = wrl.qual.cum_beam_block_frac(PBB)

        #allocate to output array
        radar_ccb[sweep_idx] = CBB
    
    #generate meta        
    the_comments = "wradlib cumulative beam blocking"
    cbb_meta    = {'data': radar_ccb, 'units': '%', 'long_name': 'cumulative beam blocking percentage',
                  'standard_name': 'CBB', 'comments': the_comments}
        
    return cbb_meta