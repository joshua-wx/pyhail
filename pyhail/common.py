"""
common sub-module of pyhail

Contains commonly used functions.

Joshua Soderholm - 15 June 2018
"""

import numpy as np
from scipy.interpolate import interp1d

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

def calc_pixel_alt(radar_rng,radar_elv,data_shape):
    """
    calculate a altitude array for the radar volume. NEED TO REMOVE AND SOURCE FROM RADAR OBJECT

    Parameters:
    ===========
    radar_rng:
        ndarray vector of radar range (m)
    radar_elv:
        ndarray vector of radar elevation (deg)
    data_shape:
        size of radar data array (i,j)

    Returns:
    ========
    out:
        altitude ndarray (km)
    """
    #calc radar voxel heights in km
    ra         = np.tile(radar_rng,(data_shape[0],1))
    elev       = np.rot90(np.tile(radar_elv,(data_shape[1],1)),k=3)
    ke         = 4/3
    a          = 6371.*1000
    alt        = np.sqrt(ra**2+(ke*a)**2 + 2*ra*ke*a*np.sin(elev*np.pi/180))-ke*a
    out        = alt/1000

    return out