import numpy as np
from scipy.interpolate import interp1d

def wbt(temp,rh):
    wb_temp = temp * np.arctan(0.151977*(rh+8.313659)**0.5) + np.arctan(temp+rh) - np.arctan(rh-1.676331) + 0.00391838*(rh**1.5)*np.arctan(0.023101*rh) - 4.686035
    return wb_temp
    
def sounding_interp(snd_temp,snd_height,target_temp):
    #WHAT: Provides an interpolated height for a target temperature using a
    #sounding vertical profile

    intp_h = np.nan
    #find index above and below freezing level
    mask      = np.where(snd_temp<target_temp)
    above_ind = mask[0][0]
    if above_ind > 1:
        below_ind = above_ind-1
        set_interp = interp1d(snd_temp[below_ind:above_ind+1], snd_height[below_ind:above_ind+1], kind='linear')
        intp_h     = set_interp(target_temp)   
        return intp_h

def smooth_ppi_rays(ppi_data,n):
    #apply a smoothing average filter
    #n must be an odd interger (>=3)
    #padding is done using input values
    offset   = int((n-1)/2)
    zero_mat = np.ma.zeros((ppi_data.shape[0],1))

    ppi_cs   = np.ma.hstack((zero_mat,ppi_data))
    ppi_cs   = ppi_cs.cumsum(axis=1)
    ppi_sma  = (ppi_cs[:,n:] - ppi_cs[:,:-n]) / float(n)
    out      = np.ma.hstack((ppi_data[:,:offset],ppi_sma,ppi_data[:,-offset:]))

    return out

def calc_pixel_alt(radar_rng,radar_elv,data_shape):
    #calc radar voxel heights in km
    ra         = np.tile(radar_rng,(data_shape[0],1))
    elev       = np.rot90(np.tile(radar_elv,(data_shape[1],1)),k=3)
    ke         = 4/3
    a          = 6371.*1000
    alt        = np.sqrt(ra**2+(ke*a)**2 + 2*ra*ke*a*np.sin(elev*np.pi/180))-ke*a
    out        = alt/1000

    return out