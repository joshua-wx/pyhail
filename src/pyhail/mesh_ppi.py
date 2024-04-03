"""
MESH implementation for calculating on PPI data.
This algorthim was originally developed by Witt et al. 1998 and modified by Murillo and Homeyer 2019 

Joshua Soderholm - 15 August 2020
"""

import time
import numpy as np

def main(
    radar,
    dbz_fname,
    levels,
    radar_band='C',
    min_range=10,
    max_range=150,
    mesh_method="mh2019_95",
    mesh_fname=None,
    posh_fname=None,
    ke_fname=None,
    shi_fname=None,
    correct_cband_refl=True
):

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
    radar_band: str 
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_fname, posh_fname, ke_fname, shi_fname : str
        String to name new hail field that will be added to the grid object.
        Default is 'mesh', 'posh', 'hail_ke', 'shi'.
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    Returns
    -------
    output_fields : dictionary
        Dictionary of output fields (KE, SHI, MESH, POSH)
    """

    # require C or S band
    if radar_band not in ["C","S"]:
        raise ValueError("radar_band must be a string of value C or S")
    # require levels
    if levels is None:
        raise ValueError("Missing levels data for freezing level and -20C level")
    
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

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer = np.min(levels)
    neg20layer = np.max(levels)

    # Initialize dimensions
    el = radar.fixed_angle["data"]
    sort_idx = list(np.argsort(el))
    el = el[sort_idx]
    n_ppi = len(el)
    az = radar.get_azimuth(0)
    n_rays = len(az)
    rg = radar.range["data"]
    n_bins = len(rg)

    # require more than one sweep
    if len(el) <= 1:
        raise Exception("Require more than one sweep to calculate MESH")
    elif len(el) < 10:
        raise Warning("Number of sweep is less than 10 and not recommended for MESH calculations")
    
    # Initialize arrays
    DBZ = np.zeros((len(el), len(az), len(rg)))
    X = np.zeros_like(DBZ)
    Y = np.zeros_like(DBZ)
    Z = np.zeros_like(DBZ)
    dZ = np.zeros_like(DBZ)
    SHI = np.zeros((len(az), len(rg)))

    # build 3D vol grids of reflectivity and Cartesian coords
    for i, el_idx in enumerate(sort_idx):
        tmp_field = radar.get_field(el_idx, dbz_fname)
        if np.shape(tmp_field) == (len(az), len(rg)):
            DBZ[i, :, :] = tmp_field
        else:
            raise Exception("Corrupt volume, sweeps of different shapes detected")
        x_ppi, y_ppi, z_ppi = radar.get_gate_x_y_z(el_idx)
        X[i, :, :] = x_ppi
        Y[i, :, :] = y_ppi
        Z[i, :, :] = z_ppi + radar.altitude['data'][0] #units m at ASL required for NWP data
    # calculate ground range by ignoring Z
    ground_range = np.sqrt(X ** 2 + Y ** 2)

    #apply C band correction
    hail_refl_correction_description = ''
    if radar_band == 'C' and correct_cband_refl:
        DBZ = DBZ*1.113 - 3.929
        hail_refl_correction_description = "C band hail reflectivity correction applied from Brook et al. 2023 https://arxiv.org/abs/2306.12016"

    # calculate dZ (used for SHI)
    for i in range(n_ppi):
        if i == 0:
            dZ[i, :, :] = Z[i + 1, :, :] - Z[i, :, :]
        if (i != 0) & (i != n_ppi - 1):
            dZ[i, :, :] = (Z[i + 1, :, :] - Z[i - 1, :, :]) / 2
        if i == n_ppi - 1:
            dZ[i, :, :] = Z[i, :, :] - Z[i - 1, :, :]

    # calc hail kenetic energy
    DBZ_weights = (DBZ - Zl) / (Zu - Zl)
    DBZ_weights[DBZ <= Zl] = 0
    DBZ_weights[DBZ >= Zu] = 1
    E = (5.0e-6) * 10 ** (0.084 * DBZ) * DBZ_weights

    # calc temperature based weighting function
    Wt = (Z - meltlayer) / (neg20layer - meltlayer)
    Wt[Z <= meltlayer] = 0
    Wt[Z >= neg20layer] = 1

    # calc severe hail index (element wise for integration)
    SHI_elements = Wt * E * dZ
    # calc valid mask
    valid = (
        (Wt > 0)
        & (E > 0)
        & (ground_range > min_range * 1000)
        & (ground_range < max_range * 1000)
    )

    # loop through each azimuth
    for az_idx in range(n_rays):
        slice_valid = valid[:, az_idx, :]
        if not np.any(slice_valid):
            continue
        slice_SHI_elements = SHI_elements[:, az_idx, :]
        # if there's samples that are valid, loop through each surface PPI range bin
        for rg_idx in range(n_bins):
            SHI_temp = 0
            surface_rg_value = rg[rg_idx]
            for el_idx in range(n_ppi):
                # skip invalid
                if not slice_valid[el_idx, rg_idx]:
                    continue
                # skip empty values
                if slice_SHI_elements[el_idx, rg_idx] == 0:
                    continue
                # if lowest PPI (always index 0 in the 3D grid), just use the SHI value directory
                if el_idx == 0:
                    SHI_temp = slice_SHI_elements[el_idx, rg_idx]
                else:
                    # find the nearest element in range to the surface_rg_value in the current ray
                    ppi_ray_rg = ground_range[el_idx, az_idx, :]
                    closest_idx = np.argmin(np.abs(ppi_ray_rg - surface_rg_value))
                    SHI_temp += slice_SHI_elements[el_idx, closest_idx]
            # insert into SHI if there's a valid value
            if SHI_temp > 0:
                SHI[az_idx, rg_idx] = 0.1 * np.nansum(SHI_temp)

    # calc maximum estimated severe hail (mm)
    if (
        mesh_method == "witt1998"
    ):  # 75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        MESH = 2.54 * SHI ** 0.5
        mesh_description = "Maximum Estimated Size of Hail retreival developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        mesh_comment = "75th percentile fit using 147 hail reports; only valid in the first sweep"
        
    elif (
        mesh_method == "mh2019_75"
    ):  # 75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 15.096 * SHI ** 0.206
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = "75th percentile fit using 5897 hail reports; only valid in the first sweep"
    elif (
        mesh_method == "mh2019_95"
    ):  # 95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        MESH = 22.157 * SHI ** 0.212
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = "95th percentile fit using 5897 hail reports; only valid in the first sweep"
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
    
    # add grids to radar object
    # unpack E into cfradial representation
    E_cfradial = np.zeros_like(radar.fields[dbz_fname]["data"])
    for i, j in enumerate(sort_idx):
        E_cfradial[radar.get_slice(j)] = E[i, :, :]

    ke_dict = {
        "data": E_cfradial,
        "units": "Jm-2s-1",
        "long_name": "Hail Kinetic Energy",
        "description": "Hail Kinetic Energy developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 " + 
        hail_refl_correction_description,
    }
    output_fields[ke_fname] = ke_dict

    # SHI,MESH and POSH are only valid at the surface, to represent it in pyart radar objects, insert it into the lowest sweep
    SHI_field = np.zeros_like(radar.fields[dbz_fname]["data"])
    SHI_field[radar.get_slice(0)] = SHI
    SHI_dict = {
        "data": SHI_field,
        "units": "Jm-1s-1",
        "long_name": "Severe Hail Index",
        "description": "Severe Hail Index developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 " + 
        hail_refl_correction_description,
        "comments": "only valid in the first sweep",
    }
    output_fields[shi_fname] = SHI_dict

    MESH_field = np.zeros_like(radar.fields[dbz_fname]["data"])
    MESH_field[radar.get_slice(0)] = MESH
    MESH_dict = {
        "data": MESH_field,
        "units": "mm",
        "long_name": "Maximum Expected Size of Hail using " + mesh_method,
        "description":mesh_description + hail_refl_correction_description,
        "comments": mesh_comment,
    }
    output_fields[mesh_fname] = MESH_dict
    
    POSH_field = np.zeros_like(radar.fields[dbz_fname]["data"])
    POSH_field[radar.get_slice(0)] = POSH
    POSH_dict = {
        "data": POSH_field,
        "units": "%",
        "long_name": "Probability of Severe Hail",
        "description": "Probability of Severe Hail developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 " +
        hail_refl_correction_description,
        "comments": "only valid in the first sweep",
    }
    output_fields[posh_fname] = POSH_dict
    
    # return output_fields dictionary
    return output_fields
