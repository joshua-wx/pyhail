"""
MESH sub-module of pyhail

Contains the single pol MESH retrieval for gridded radar data.
Required reflectivity and temperature data

Joshua Soderholm - 15 June 2018
"""

import pyart
from  pyhail import common
import netCDF4
import numpy as np

def _get_latlon(radgrid, ref_name):
    """
    Generates lattitude and longitude arrays.
    Parameters:
    ===========
    radgrid: struct
        Py-ART grid object.
    Returns:
    ========
    longitude: ndarray
        Array of coordinates for all points.
    latitude: ndarray
        Array of coordinates for all points.
	
	From cpol_processing: https://github.com/vlouf/cpol_processing
    """
    # Declare array, filled 0 in order to not have a masked array.
    lontot = np.zeros_like(radgrid.fields[ref_name]['data'].filled(0))
    lattot = np.zeros_like(radgrid.fields[ref_name]['data'].filled(0))

    for lvl in range(radgrid.nz):
        lontot[lvl, :, :], lattot[lvl, :, :] = radgrid.get_point_longitude_latitude(lvl)

    longitude = pyart.config.get_metadata('longitude')
    latitude  = pyart.config.get_metadata('latitude')

    longitude['data'] = lontot
    latitude['data']  = lattot

    return longitude, latitude

def main(grid, out_ffn, snd_input, ref_name):

    """
 	Hail grids adapted fromWitt et al. 1998 and Cintineo et al. 2012.
    Exapnded to grids (adapted from wdss-ii)

	Gridding set to 1x1x1km on a 20,145x145km domain

    Parameters:
    ===========
    radgrid: struct
        Py-ART grid object.
	out_ffn: string
		output full filename (inc path)
	snd_input: string
		sounding full filename (inc path)
    ref_name: string
        name of reflectivity field in radar object

    Returns:
    ========
    None, write to file
	
    """

    #MESH constants
    z_lower_bound = 40
    z_upper_bound = 50
    
    #build sounding data
    snd_data = netCDF4.Dataset(snd_input)
    snd_temp = snd_data.variables["temp"][:]
    snd_geop = snd_data.variables["height"][:]
    snd_rh   = snd_data.variables["rh"][:]
    
    #run interpolation
    snd_t_minus20C = common.sounding_interp(snd_temp,snd_geop,-20)/1000
    snd_t_0C       = common.sounding_interp(snd_temp,snd_geop,0)/1000

    # Latitude Longitude field for each point.
    longitude, latitude = _get_latlon(grid, ref_name)
    grid.add_field('longitude', longitude)
    grid.add_field('latitude', latitude)
    
    # extract grids
    refl_grid = grid.fields[ref_name]['data']
    grid_sz   = np.shape(refl_grid)
    alt_vec   = grid.z['data']
    alt_grid  = np.tile(alt_vec,(grid_sz[1], grid_sz[2], 1))
    alt_grid  = np.swapaxes(alt_grid, 0, 2)
    
    #calc reflectivity weighting function
    weight_ref                             = (refl_grid - z_lower_bound)/(z_upper_bound - z_lower_bound)
    weight_ref[refl_grid <= z_lower_bound] = 0
    weight_ref[refl_grid >= z_upper_bound] = 1
    
    #calc hail kenitic energy
    hail_KE = (5 * 10**-6) * 10**(0.084 * refl_grid) * weight_ref
    
    #calc temperature based weighting function
    weight_height = (alt_grid - snd_t_0C) / (snd_t_minus20C - snd_t_0C)
    weight_height[alt_grid <= snd_t_0C]       = 0
    weight_height[alt_grid >= snd_t_minus20C] = 1

    #calc severe hail index
    grid_sz_m = alt_vec[1] - alt_vec[0]
    SHI = 0.1 * np.sum(weight_height * hail_KE, axis=0) * grid_sz_m

    #calc maximum estimated severe hail (mm)
    MESH = 2.54 * SHI**0.5

    #calc warning threshold (J/m/s) NOTE: freezing height must be in meters
    WT   = 57.5 * snd_t_0C - 121

    #calc probability of severe hail (POSH) (%)
    POSH           = 29 * np.log(SHI/WT) + 50
    POSH           = np.real(POSH)
    POSH[POSH<0]   = 0
    POSH[POSH>100] = 100
    
    #add grids to grid object
    hail_KE_field   = {'data': hail_KE, 'units': 'Jm-2s-1', 'long_name': 'Hail Kinetic Energy',
                  'standard_name': 'hail_KE', 'comments': 'Witt et al. 1998'}
    grid.add_field(fnames['hail_ke'], hail_KE_field, replace_existing=True) 
    
    SHI_grid         = np.zeros_like(hail_KE)
    SHI_grid[0,:,:]  = SHI
    SHI_field        = {'data': SHI_grid, 'units': 'J-1s-1', 'long_name': 'Severe Hail Index',
                        'standard_name': 'SHI', 'comments': 'Witt et al. 1998, only valid in the first level'}
    grid.add_field(fnames['shi'], SHI_field, replace_existing=True) 

    MESH_grid        = np.zeros_like(hail_KE)
    MESH_grid[0,:,:] = MESH    
    MESH_field       = {'data': MESH_grid, 'units': 'mm', 'long_name': 'Maximum Expected Size of Hail',
                        'standard_name': 'MESH', 'comments': 'Witt et al. 1998, only valid in the first level'}
    grid.add_field(fnames['mesh'], MESH_field, replace_existing=True) 

    POSH_grid        = np.zeros_like(hail_KE)
    POSH_grid[0,:,:] = POSH    
    POSH_field       = {'data': POSH_grid, 'units': '%', 'long_name': 'Probability of Severe Hail',
                        'standard_name': 'POSH', 'comments': 'Witt et al. 1998, only valid in the first level'}
    grid.add_field(fnames['posh'], POSH_field, replace_existing=True) 
    
    # Saving data to file
    grid.write(out_ffn)
