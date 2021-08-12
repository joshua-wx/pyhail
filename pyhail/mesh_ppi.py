"""
MESH implementation for calculating on PPI data.
This algorthim was originally developed by Witt et al. 1998 and modified by Murillo and Homeyer 2019 

Joshua Soderholm - 15 August 2020
"""

import time
import numpy as np

def main(radar, dbz_fname, levels, min_range=10, max_range=150,
         mesh_method='mh2019_95',
         mesh_fname=None, posh_fname=None, ke_fname=None, shi_fname=None):
    
    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019

    Parameters
    ----------
    radar : object
        Py-ART radar object.
    dbz_fname : str
        Name of reflectivity field in the radar object.
    levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)        
    mesh_fname, posh_fname, ke_fname, shi_fname : str
        String to name new hail field that will be added to the grid object.
        Default is 'mesh', 'posh', 'hail_ke', 'shi'.
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below

    Returns
    -------
    radar : object
        Py-ART radar object.

    """
    # Rain/Hail dBZ boundaries
    Zl = 40
    Zu = 50
    
    #default field names
    if mesh_fname is None:
        mesh_fname = 'mesh_' + mesh_method
    if posh_fname is None:
        posh_fname = 'posh'
    if ke_fname is None:
        ke_fname = 'hail_ke'
    if shi_fname is None:
        shi_fname = 'shi'
    #require levels
    if levels is None:
        raise ValueError(
            "Missing levels data for freezing level and -20C level")
    
    # This dummy proofs the user input. The melting level will always 
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer  = np.min(levels)
    neg20layer = np.max(levels)
    
    # Initialize dimensions
    el = radar.fixed_angle['data']
    sort_idx = list(np.argsort(el))
    el = el[sort_idx]
    n_ppi = len(el)
    az = radar.get_azimuth(0)
    n_rays = len(az)
    rg = radar.range['data']
    n_bins = len(rg)
    
    # Initialize arrays
    DBZ = np.zeros((len(el), len(az), len(rg)))
    X = np.zeros_like(DBZ)
    Y = np.zeros_like(DBZ)
    Z = np.zeros_like(DBZ)
    dZ = np.zeros_like(DBZ)
    SHI = np.zeros((len(az), len(rg)))
    
    #build 3D vol grids of reflectivity and Cartesian coords
    for i, el_idx in enumerate(sort_idx):
        DBZ[i,:,:] = radar.get_field(el_idx, dbz_fname)
        x_ppi, y_ppi, z_ppi = radar.get_gate_x_y_z(el_idx)
        X[i,:,:] = x_ppi
        Y[i,:,:] = y_ppi
        Z[i,:,:] = z_ppi
    #calculate ground range by ignoring Z
    ground_range = np.sqrt(X**2 + Y**2)
    
    #calculate dZ (used for SHI)
    for i in range(n_ppi):
        if i == 0:
            dZ[i,:,:] = Z[i+1,:,:] - Z[i,:,:]
        if (i != 0) & (i != n_ppi-1):
            dZ[i,:,:] = (Z[i+1,:,:] - Z[i-1,:,:])/2
        if i == n_ppi-1:
            dZ[i,:,:] = Z[i,:,:] - Z[i-1,:,:]
    
    # calc hail kenetic energy
    DBZ_weights = (DBZ - Zl)/(Zu - Zl)
    DBZ_weights[DBZ <= Zl] = 0
    DBZ_weights[DBZ >= Zu] = 1
    E = (5.0E-6)*10**(0.084*DBZ)*DBZ_weights
    
    # calc temperature based weighting function
    Wt = (Z-meltlayer)/(neg20layer-meltlayer)
    Wt[Z <= meltlayer] = 0
    Wt[Z >= neg20layer] = 1
    
    # calc severe hail index (element wise for integration)
    SHI_elements = Wt*E*dZ
    # calc valid mask
    valid = (Wt>0) & (E>0) & (ground_range>min_range*1000) & (ground_range<max_range*1000)
    
    #loop through each azimuth
    for az_idx in range(n_rays):
        slice_valid = valid[:,az_idx,:]
        if not np.any(slice_valid):
            continue
        slice_SHI_elements = SHI_elements[:,az_idx,:]
        #if there's samples that are valid, loop through each surface PPI range bin
        for rg_idx in range(n_bins):
            SHI_temp = 0
            surface_rg_value = rg[rg_idx]
            for el_idx in range(n_ppi):
                #skip invalid
                if not slice_valid[el_idx,rg_idx]:
                    continue
                #skip empty values
                if slice_SHI_elements[el_idx, rg_idx] == 0:
                    continue
                #if lowest PPI (always index 0 in the 3D grid), just use the SHI value directory
                if el_idx==0:
                    SHI_temp = slice_SHI_elements[el_idx, rg_idx]
                else:
                    #find the nearest element in range to the surface_rg_value in the current ray
                    ppi_ray_rg = ground_range[el_idx,az_idx,:]
                    closest_idx = np.argmin(np.abs(ppi_ray_rg - surface_rg_value))
                    SHI_temp += slice_SHI_elements[el_idx, closest_idx]
            #insert into SHI if there's a valid value
            if SHI_temp>0:
                SHI[az_idx, rg_idx] = 0.1*np.nansum(SHI_temp)
            
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
    WT = 57.5 * (meltlayer/1000) - 121

    # calc probability of severe hail (POSH) (%)
    POSH = 29 * np.log(SHI/WT) + 50
    POSH = np.real(POSH)
    POSH[POSH < 0] = 0
    POSH[POSH > 100] = 100

    # add grids to radar object
    #unpack E into cfradial representation
    E_cfradial = np.zeros_like(radar.fields[dbz_fname]['data'])
    for i,j in enumerate(sort_idx):
        E_cfradial[radar.get_slice(j)] = E[i,:,:]
        
    ke_dict = {'data': E_cfradial, 'units': 'Jm-2s-1',
                    'long_name': 'Hail Kinetic Energy',
                    'standard_name': 'hail_KE',
                    'comments': 'Witt et al. 1998'}
    radar.add_field(ke_fname, ke_dict, replace_existing=True)

    #SHI,MESH and POSH are only valid at the surface, to represent it in pyart radar objects, insert it into the lowest sweep
    SHI_field = np.zeros_like(radar.fields[dbz_fname]['data'])
    SHI_field[radar.get_slice(sort_idx[0])] = SHI
    SHI_dict = {'data': SHI_field, 'units': 'J-1s-1',
                'long_name': 'Severe Hail Index',
                'standard_name': 'SHI',
                'comments': 'Witt et al. 1998, only valid in the lowest sweep'}
    radar.add_field(shi_fname, SHI_dict, replace_existing=True)

    MESH_field = np.zeros_like(radar.fields[dbz_fname]['data'])
    MESH_field[radar.get_slice(sort_idx[0])] = MESH
    MESH_dict = {'data': MESH_field, 'units': 'mm',
                 'long_name': 'Maximum Expected Size of Hail using ' + mesh_method,
                 'standard_name': 'MESH ' + mesh_method,
                 'comments': mesh_comment + ', only valid in the lowest sweep'}
    radar.add_field(mesh_fname, MESH_dict, replace_existing=True)

    POSH_field = np.zeros_like(radar.fields[dbz_fname]['data'])
    POSH_field[radar.get_slice(sort_idx[0])] = POSH
    POSH_dict = {'data': POSH_field, 'units': '%',
                 'long_name': 'Probability of Severe Hail',
                 'standard_name': 'POSH',
                 'comments': 'Witt et al. 1998, only valid in the lowest sweep'}
    radar.add_field(posh_fname, POSH_dict, replace_existing=True)

    #return radar object
    return radar