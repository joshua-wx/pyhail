"""
MESH sub-module of pyhail

Contains the single pol MESH retrieval for gridded radar data.
Required reflectivity and temperature data.

Joshua Soderholm - 15 June 2018
"""
import os

import netCDF4
import numpy as np
import pyart

from pyhail import common

def _get_latlon(grid, ref_name):
    """
    Generates latitude and longitude arrays.

    Parameters
    ----------
    grid : Grid
        Py-ART grid object.
    ref_name : str
        Reflectivity field name.

    Returns
    -------
    longitude : ndarray
        Array of coordinates for all points.
    latitude : ndarray
        Array of coordinates for all points.

	From cpol_processing: https://github.com/vlouf/cpol_processing

    """
    # Declare array, filled 0 in order to not have a masked array.
    lontot = np.zeros_like(grid.fields[ref_name]['data'].filled(0))
    lattot = np.zeros_like(grid.fields[ref_name]['data'].filled(0))

    for lvl in range(grid.nz):
        lontot[lvl, :, :], lattot[lvl, :, :] = grid.get_point_longitude_latitude(lvl)

    longitude = pyart.config.get_metadata('longitude')
    latitude = pyart.config.get_metadata('latitude')

    longitude['data'] = lontot
    latitude['data'] = lattot

    return longitude, latitude

def main(grid, ref_name, snd_input=None, sonde_temp='temp',
         sonde_height='height',
         temph_data=None, posh_field=None, mesh_field=None,
         hail_ke_field=None, shi_field=None, mesh_method='mh2019_95'):

    """
    Hail grids adapted from Witt et al. 1998,  Cintineo et al. 2012. and Murillo and Homeyer 2019
    Expanded to grids (adapted from wdss-ii)

    Gridding set to 1x1x1km on a 20,145x145km domain

    Parameters
    ----------
    grid : Grid
        Py-ART grid object.
    ref_name : str
        Name of reflectivity field in the radar object.
    snd_input : string
    	Sounding full filename (inc path). Default is None. If default
        will see if temph_data is provided.
    sonde_temp, sonde_height : str
        The variable name for the temperature and height data in the
        sounding dataset. Default is 'temp' and 'height'.
    temph_data : list
    	Contains 0C and -20C altitude (m) in first and second element position,
        only used if snd_input is empty. Default is None. If default,
        a sounding file path needs to be provided with the snd_input
        parameter.
    posh_field, mesh_field, hail_ke_field, shi_field : str
        String to name new hail field that will be added to the grid object.
        Default is 'mesh', 'posh', 'hail_ke', 'shi'.
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below

    Returns
    -------
    None, writes to file.

    """
    #MESH constants
    z_lower_bound = 40
    z_upper_bound = 50

    if mesh_field is None:
        mesh_field = 'mesh_' + mesh_method
    if posh_field is None:
        posh_field = 'posh'
    if hail_ke_field is None:
        hail_ke_field = 'hail_ke'
    if shi_field is None:
        shi_field = 'shi'

    if (snd_input is None) and (temph_data is None):
        raise ValueError(
            "The parameters snd_input and temph_data are both None, please "
            "provide a file path to a sounding file or an array with "
            "temperature data.")

    if snd_input is not None:
        #build sounding data
        snd_data = netCDF4.Dataset(snd_input)
        try:
            snd_temp = snd_data.variables[sonde_temp][:]
            snd_geop = snd_data.variables[sonde_height][:]
        except KeyError:
            raise KeyError(
                "Data Variables %s and %s not found. Check sounding "
                "temperature and sounding height variables names within "
                "the sounding dataset." % (sonde_temp, sonde_height))

        # Not used.
        #snd_rh = snd_data.variables["rh"][:]

        snd_data.close()
        # run interpolation
        snd_t_0C = common.sounding_interp(snd_temp, snd_geop, 0)  #m
        snd_t_minus20C = common.sounding_interp(snd_temp, snd_geop, -20) #m
    else:
        snd_t_0C = temph_data[0]
        snd_t_minus20C = temph_data[1]

    # Latitude Longitude field for each point.
    longitude, latitude = _get_latlon(grid, ref_name)
    grid.add_field('longitude', longitude, replace_existing=True)
    grid.add_field('latitude', latitude, replace_existing=True)

    # extract grids
    refl_grid = grid.fields[ref_name]['data']
    grid_sz = np.shape(refl_grid)
    alt_vec = grid.z['data']
    alt_grid = np.tile(alt_vec, (grid_sz[1], grid_sz[2], 1))
    alt_grid = np.swapaxes(alt_grid, 0, 2) #m

    # calc reflectivity weighting function
    weight_ref = (refl_grid - z_lower_bound)/(z_upper_bound - z_lower_bound)
    weight_ref[refl_grid <= z_lower_bound] = 0
    weight_ref[refl_grid >= z_upper_bound] = 1

    # calc hail kenetic energy
    hail_KE = (5 * 10**-6) * 10**(0.084 * refl_grid) * weight_ref

    # calc temperature based weighting function
    weight_height = (alt_grid - snd_t_0C) / (snd_t_minus20C - snd_t_0C)
    weight_height[alt_grid <= snd_t_0C] = 0
    weight_height[alt_grid >= snd_t_minus20C] = 1

    # calc severe hail index
    grid_sz_m = alt_vec[1] - alt_vec[0]
    SHI = 0.1 * np.sum(weight_height * hail_KE, axis=0) * grid_sz_m

    # calc maximum estimated severe hail (mm)
    if mesh_method == 'witt1998': #75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        MESH = 2.54 * SHI**0.5
        mesh_comment = '75th percentil fit from Witt et al. 1998 (fitted to 147 reports)'
    elif mesh_method == 'mh2019_75': #75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 16.566 * SHI**0.181
        mesh_comment = '75th percentile fit from Murillo and Homeyer 2019 (fitted to 5897 reports)'
    elif mesh_method == 'mh2019_95': #95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 17.270 * SHI**0.272
        mesh_comment = '95th percentile fit from Murillo and Homeyer 2019 (fitted to 5897 reports)'
    else:
        raise ValueError('unknown MESH method selects, please use witt1998, mh2019_75 or mh2019_95')
        
    # calc warning threshold (J/m/s) NOTE: freezing height must be in km
    WT = 57.5 * (snd_t_0C/1000) - 121

    # calc probability of severe hail (POSH) (%)
    POSH = 29 * np.log(SHI/WT) + 50
    POSH = np.real(POSH)
    POSH[POSH < 0] = 0
    POSH[POSH > 100] = 100

    # add grids to grid object
    hail_ke_dict = {'data': hail_KE, 'units': 'Jm-2s-1',
                    'long_name': 'Hail Kinetic Energy',
                    'standard_name': 'hail_KE',
                    'comments': 'Witt et al. 1998'}

    grid.add_field(hail_ke_field, hail_ke_dict, replace_existing=True)

    SHI_grid = np.zeros_like(hail_KE)
    SHI_grid[0, :, :] = SHI
    SHI_dict = {'data': SHI_grid, 'units': 'J-1s-1',
                'long_name': 'Severe Hail Index',
                'standard_name': 'SHI',
                'comments': 'Witt et al. 1998, only valid in the first level'}
    grid.add_field(shi_field, SHI_dict, replace_existing=True)

    MESH_grid = np.zeros_like(hail_KE)
    MESH_grid[0, :, :] = MESH
    MESH_dict = {'data': MESH_grid, 'units': 'mm',
                 'long_name': 'Maximum Expected Size of Hail using ' + mesh_method,
                 'standard_name': 'MESH ' + mesh_method,
                 'comments': mesh_comment}
    grid.add_field(mesh_field, MESH_dict, replace_existing=True)

    POSH_grid = np.zeros_like(hail_KE)
    POSH_grid[0, :, :] = POSH
    POSH_dict = {'data': POSH_grid, 'units': '%',
                 'long_name': 'Probability of Severe Hail',
                 'standard_name': 'POSH',
                 'comments': 'Witt et al. 1998, only valid in the first level'}
    grid.add_field(posh_field, POSH_dict, replace_existing=True)

    #return grid object
    return grid
