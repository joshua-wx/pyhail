"""
MESH implementation for calculating on gridded data.
This algorthim was originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and modified by Murillo and Homeyer 2019 doi:10.1175/JAMC-D-18-0247.1

Joshua Soderholm - 15 June 2020
"""
import os

import netCDF4
import numpy as np
import pyart

from pyhail import common


def _get_latlon(grid, dbz_fname):
    """
    Generates latitude and longitude arrays.

    Parameters
    ----------
    grid : Grid
        Py-ART grid object.
    dbz_fname : str
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
    lontot = np.zeros_like(grid.fields[dbz_fname]["data"].filled(0))
    lattot = np.zeros_like(grid.fields[dbz_fname]["data"].filled(0))

    for lvl in range(grid.nz):
        lontot[lvl, :, :], lattot[lvl, :, :] = grid.get_point_longitude_latitude(lvl)

    longitude = pyart.config.get_metadata("longitude")
    latitude = pyart.config.get_metadata("latitude")

    longitude["data"] = lontot
    latitude["data"] = lattot

    return longitude, latitude


def main(
    grid,
    dbz_fname,
    levels,
    mesh_method="mh2019_95",
    mesh_fname=None,
    posh_fname=None,
    ke_fname=None,
    shi_fname=None,
):

    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019
    Expanded to calculate MESH on gridded data (adapted from wdss-ii)

    Parameters
    ----------
    grid : object
        Py-ART grid object.
    dbz_fname : str
        Name of reflectivity field in the radar object.
    levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    mesh_fname, posh_fname, ke_fname, shi_fname : str
        String to name new hail field that will be added to the grid object.
        Default is 'mesh', 'posh', 'hail_ke', 'shi'.
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below

    Returns
    -------
    output_fields : dictionary
        Dictionary of output fields (KE, SHI, MESH, POSH)
        
    """
    # Rain/Hail dBZ boundaries
    Zl = 40
    Zu = 50

    # default field names
    if mesh_fname is None:
        mesh_fname = "mesh_" + mesh_method
    if posh_fname is None:
        posh_fname = "posh"
    if ke_fname is None:
        ke_fname = "hail_ke"
    if shi_fname is None:
        shi_fname = "shi"
    # require levels
    if levels is None:
        raise ValueError("Missing levels data for freezing level and -20C level")

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer = min(levels)
    neg20layer = max(levels)

    # Latitude Longitude field for each point.
    longitude, latitude = _get_latlon(grid, dbz_fname)
    grid.add_field("longitude", longitude, replace_existing=True)
    grid.add_field("latitude", latitude, replace_existing=True)

    # extract grids
    dbz_grid = grid.fields[dbz_fname]["data"]
    grid_sz = np.shape(dbz_grid)
    alt_vec = grid.z["data"] + grid.radar_altitude['data'][0] #units m at ASL required for NWP data
    alt_grid = np.tile(alt_vec, (grid_sz[1], grid_sz[2], 1))
    alt_grid = np.swapaxes(alt_grid, 0, 2)  # m

    # calc reflectivity weighting function
    DBZ_weights = (dbz_grid - Zl) / (Zu - Zl)
    DBZ_weights[dbz_grid <= Zl] = 0
    DBZ_weights[dbz_grid >= Zu] = 1

    # calc hail kenetic energy
    E = (5 * 10 ** -6) * 10 ** (0.084 * dbz_grid) * DBZ_weights

    # calc temperature based weighting function
    Wt = (alt_grid - meltlayer) / (neg20layer - meltlayer)
    Wt[alt_grid <= meltlayer] = 0
    Wt[alt_grid >= neg20layer] = 1

    # calc severe hail index
    dZ = alt_vec[1] - alt_vec[0]
    SHI = 0.1 * np.sum(Wt * E, axis=0) * dZ

    # calc maximum estimated severe hail (mm)
    if (
        mesh_method == "witt1998"
    ):  # 75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        MESH = 2.54 * SHI ** 0.5
        mesh_description = "Maximum Estimated Size of Hail retreival developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2"
        mesh_comment = "75th percentile fit using 147 hail reports; only valid in the first level of the 3D grid"
        
    elif (
        mesh_method == "mh2019_75"
    ):  # 75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 15.096 * SHI ** 0.206
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1"
        mesh_comment = "75th percentile fit using 5897 hail reports; only valid in the first level of the 3D grid"

    elif (
        mesh_method == "mh2019_95"
    ):  # 95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 22.157 * SHI ** 0.212
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1"
        mesh_comment = "95th percentile fit using 5897 hail reports; only valid in the first level of the 3D grid"
    else:
        raise ValueError(
            "unknown MESH method selects, please use witt1998, mh2019_75 or mh2019_95"
        )

    # calc warning threshold (J/m/s) NOTE: freezing height must be in km
    WT = 57.5 * (meltlayer / 1000) - 121

    # calc probability of severe hail (POSH) (%)
    POSH = 29 * np.log(SHI / WT) + 50
    POSH = np.real(POSH)
    POSH[POSH < 0] = 0
    POSH[POSH > 100] = 100

    output_fields = dict()
    
    # add grids to grid object
    ke_dict = {
        "data": E,
        "units": "Jm-2s-1",
        "long_name": "Hail Kinetic Energy",
        "description": "Hail Kinetic Energy developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2",
    }
    output_fields[ke_fname] = ke_dict

    SHI_grid = np.zeros_like(E)
    SHI_grid[0, :, :] = SHI
    SHI_dict = {
        "data": SHI_grid,
        "units": "J-1s-1",
        "long_name": "Severe Hail Index",
        "description": "Severe Hail Index developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2",
        "comments": "only valid in the first level of the 3D grid",
    }
    output_fields[shi_fname] = SHI_dict

    MESH_grid = np.zeros_like(E)
    MESH_grid[0, :, :] = MESH
    MESH_dict = {
        "data": MESH_grid,
        "units": "mm",
        "long_name": "Maximum Expected Size of Hail using " + mesh_method,
        "description":mesh_description,
        "comments": mesh_comment,
    }
    output_fields[mesh_fname] = MESH_dict

    POSH_grid = np.zeros_like(E)
    POSH_grid[0, :, :] = POSH
    POSH_dict = {
        "data": POSH_grid,
        "units": "%",
        "long_name": "Probability of Severe Hail",
        "description": "Probability of Severe Hail developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2",
        "comments": "only valid in the first level of the 3D grid",
    }
    output_fields[posh_fname] = POSH_dict

    # return output_fields dictionary
    return output_fields
