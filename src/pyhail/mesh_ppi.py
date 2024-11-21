"""
MESH implementation for calculating on PPI data.
This algorthim was originally developed by Witt et al. 1998 and modified by Murillo and Homeyer 2019 

Joshua Soderholm - 15 August 2020
"""

import copy
import numpy as np
from pyhail import common


def pyart(
    radar,
    reflectivity_fname,
    temp_levels,
    ke_fname="ke",
    shi_fname="shi",
    mesh_fname="mesh",
    posh_fname="posh",
    radar_band="S",
    min_range=10,
    max_range=150,
    mesh_method="mh2019_75",
    correct_cband_refl=True,
    minimum_sweeps_raise_expection=4,
    minimum_sweeps_raise_warning=8,
    column_shift_maximum=2500,
):
    """
    PyART Wrapper for PPI MESH

    Parameters:
    ===========
    radar: class
        pyart radar object
    reflectivity_fname: string
        name of reflectivity field
    temp_levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    ke_fname: string
        name of ke field
    shi_fname: string
        name of shi field
    mesh_fname: string
        name of mesh field
    posh_fname: string
        name of posh field
    radar_band: str
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    minimum_sweeps_raise_expection: int
        minimum number of sweeps to raise an exception
    minimum_sweeps_raise_warning: int
        minimum number of sweeps to raise a warning
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    Returns:
    ========
    radar: class
        pyart radar object updated with mesh, ke, shi, posh fields
    """

    # init radar fields
    empty_radar_field = {
        "data": np.zeros((radar.nrays, radar.ngates)),
        "units": "",
        "long_name": "",
        "description": "",
        "comments": "",
    }
    radar.add_field(ke_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(shi_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(mesh_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(posh_fname, copy.deepcopy(empty_radar_field))
    # build datasets
    reflectivity_dataset = []
    elevation_dataset = []
    azimuth_dataset = []
    range_dataset = []
    elevation_dataset = []
    radar_altitude = radar.altitude["data"][0]
    for sweep_idx in range(radar.nsweeps):
        reflectivity_dataset.append(
            radar.get_field(sweep_idx, reflectivity_fname, copy=True).filled(np.nan)
        )
        azimuth_dataset.append(radar.get_azimuth(sweep_idx))
        range_dataset.append(radar.range["data"])
        elevation_dataset.append(radar.fixed_angle["data"][sweep_idx])

    # run retrieval
    ke_dict, shi_dict, mesh_dict, posh_dict = main(
        reflectivity_dataset,
        elevation_dataset,
        azimuth_dataset,
        range_dataset,
        radar_altitude,
        temp_levels,
        radar_band=radar_band,
        min_range=min_range,
        max_range=max_range,
        mesh_method=mesh_method,
        correct_cband_refl=correct_cband_refl,
        minimum_sweeps_raise_expection=minimum_sweeps_raise_expection,
        minimum_sweeps_raise_warning=minimum_sweeps_raise_warning,
        column_shift_maximum=column_shift_maximum,
    )

    # index of lowest sweep
    sweep0_idx = np.argmin(elevation_dataset)

    # update data
    for sweep_idx in range(radar.nsweeps):
        radar.fields[ke_fname]["data"][radar.get_slice(sweep_idx)] = ke_dict["data"][
            sweep_idx
        ]
    radar.fields[shi_fname]["data"][radar.get_slice(sweep0_idx)] = shi_dict["data"]
    radar.fields[mesh_fname]["data"][radar.get_slice(sweep0_idx)] = mesh_dict["data"]
    radar.fields[posh_fname]["data"][radar.get_slice(sweep0_idx)] = posh_dict["data"]

    # update metadata
    radar = common.add_pyart_metadata(radar, ke_fname, ke_dict)
    radar = common.add_pyart_metadata(radar, shi_fname, shi_dict)
    radar = common.add_pyart_metadata(radar, mesh_fname, mesh_dict)
    radar = common.add_pyart_metadata(radar, posh_fname, posh_dict)

    return radar


def pyodim(
    datasets,
    reflectivity_fname,
    temp_levels,
    radar_height_fname="height",
    elevation_fname="elevation",
    azimuth_fname="azimuth",
    range_fname="range",
    ke_fname="ke",
    shi_fname="shi",
    mesh_fname="mesh",
    posh_fname="posh",
    radar_band="S",
    min_range=10,
    max_range=150,
    mesh_method="mh2019_75",
    correct_cband_refl=True,
    minimum_sweeps_raise_expection=4,
    minimum_sweeps_raise_warning=8,
    column_shift_maximum=2500,
):
    """
    Pyodim Wrapper for PPI MESH

    Parameters:
    ===========
    datasets: list of dicts
        pyodim dataset
    reflectivity_fname: string
        name of reflectivity field
    temp_levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    radar_height_fname : string
        name of radar height field
    elevation_fname: string
        name of radar elevation field
    range_fname: string
        name of radar bin range field
    ke_fname: string
        name of ke field
    shi_fname: string
        name of shi field
    mesh_fname: string
        name of mesh field
    posh_fname: string
        name of posh field
    radar_band: str
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    minimum_sweeps_raise_expection: int
        minimum number of sweeps to raise an exception
    minimum_sweeps_raise_warning: int
        minimum number of sweeps to raise a warning
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    Returns:
    ========
    datasets: list of dicts
        pyodim dataset updated with mesh, ke, shi, posh fields

    """

    # build datasets
    reflectivity_dataset = []
    elevation_dataset = []
    azimuth_dataset = []
    range_dataset = []
    radar_altitude = datasets[0].attrs[radar_height_fname]
    n_ppi = len(datasets)
    for sweep_idx in range(n_ppi):
        reflectivity_dataset.append(datasets[sweep_idx][reflectivity_fname].values)
        elevation_dataset.append(datasets[sweep_idx][elevation_fname].data[0])
        azimuth_dataset.append(datasets[sweep_idx][azimuth_fname].values)
        range_dataset.append(datasets[sweep_idx][range_fname].values)
    # run retrieval
    ke_dict, shi_dict, mesh_dict, posh_dict = main(
        reflectivity_dataset,
        elevation_dataset,
        azimuth_dataset,
        range_dataset,
        radar_altitude,
        temp_levels,
        radar_band=radar_band,
        min_range=min_range,
        max_range=max_range,
        mesh_method=mesh_method,
        correct_cband_refl=correct_cband_refl,
        minimum_sweeps_raise_expection=minimum_sweeps_raise_expection,
        minimum_sweeps_raise_warning=minimum_sweeps_raise_warning,
        column_shift_maximum=column_shift_maximum,
    )

    # add 2D fields and metadata
    sweep0_idx = np.argmin(elevation_dataset)
    datasets[sweep0_idx] = datasets[0].merge(
        {
            shi_fname: (("azimuth", "range"), shi_dict["data"]),
            mesh_fname: (("azimuth", "range"), mesh_dict["data"]),
            posh_fname: (("azimuth", "range"), posh_dict["data"]),
        }
    )
    datasets[0][shi_fname] = common.add_pyodim_metadata(
        datasets[0][shi_fname], shi_dict
    )
    datasets[0][mesh_fname] = common.add_pyodim_metadata(
        datasets[0][mesh_fname], mesh_dict
    )
    datasets[0][posh_fname] = common.add_pyodim_metadata(
        datasets[0][posh_fname], posh_dict
    )

    # add 3D field and metadata
    for sweep_idx, _ in enumerate(datasets):
        datasets[sweep_idx] = datasets[sweep_idx].merge(
            {ke_fname: (("azimuth", "range"), ke_dict["data"][sweep_idx])}
        )
        # metadata
        datasets[sweep_idx][ke_fname] = common.add_pyodim_metadata(
            datasets[sweep_idx][ke_fname], ke_dict
        )

    return datasets


def _antenna_to_arc(ranges, elevation):
    """
    Return the great circle distance directly below the radar beam and the
    altitude of the radar beam.
    ----------
    ranges : 1d array
        Distances to the center of the radar gates (bins) in meters.
    elevation : float
        Elevation angle of the radar in degrees.
    Returns
    -------
    s: 1d array
        Distance along the great circle for each radar bin (units: meters)
    z: 1d array
        Altitude above radar level for each radar bin (units: meters)
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = elevation * np.pi / 180.0  # elevation angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    r = ranges

    z = (r**2 + R**2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    return s, z


def _calc_dz(column_z):
    """
    Calculate altitude difference between elements in a 1d array
    Takes into account the boundaries

    Parameters
    ----------
    column_z : 1darray
        altitude of column samples

    Returns
    -------
    dz : 1darray
        Difference between altitude elements
    """
    # need at least two values
    dz = []
    n_ppi = len(column_z)
    for i in range(n_ppi):
        # calculate dz for use in shi calc
        if i == 0:
            value = column_z[i + 1] - column_z[i]
        elif (i != 0) & (i != n_ppi - 1):
            value = (column_z[i + 1] - column_z[i - 1]) / 2
        else:
            value = column_z[i] - column_z[i - 1]
        dz.append(value)
    dz = np.array(dz)
    return dz


def main(
    reflectivity,
    elevation,
    azimuth,
    rangebin,
    radar_altitude,
    levels,
    radar_band="S",
    min_range=10,
    max_range=150,
    mesh_method="mh2019_75",
    correct_cband_refl=True,
    minimum_sweeps_raise_expection=4,
    minimum_sweeps_raise_warning=8,
    column_shift_maximum=2500,
):
    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019

    Parameters
    ----------
    reflectivity : list of 2D ndarrays
        list where each element is the sweep reflectivity data in an array with dimensions (azimuth, range)
    elevation: 1d ndarray of floats
        ndarray where each element is the fixed elevation angle of the sweep
    azimuth: list of 1D ndarrays
        list where each element is the sweep azimuth angles
    rangebin: list of 1D ndarrays
        list where each element is the sweep range distances
    radar_altitude: float
        altitude of radar AMSL
    levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    radar_band: str
        radar frequency band (either C or S)
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    minimum_sweeps_raise_expection: int
        minimum number of sweeps to raise an exception
    minimum_sweeps_raise_warning: int
        minimum number of sweeps to raise a warning
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    Returns
    -------
    output_fields : dictionary
        Dictionary of output fields (KE, SHI, MESH, POSH)
    """

    # require C or S band
    if radar_band not in ["C", "S"]:
        raise ValueError("radar_band must be a string of value C or S")
    # require levels
    if levels is None:
        raise ValueError("Missing levels data for freezing level and -20C level")

    # Rain/Hail dBZ boundaries
    z_l = 40
    z_u = 50

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer = np.min(levels)
    neg20layer = np.max(levels)

    # sort by fixed angle
    sort_idx = list(np.argsort(elevation))
    reflectivity_dataset = [reflectivity[i] for i in sort_idx]
    elevation_dataset = [elevation[i] for i in sort_idx]
    azimuth_dataset = [azimuth[i] for i in sort_idx]
    range_dataset = [rangebin[i] for i in sort_idx]

    # require more than one sweep
    if len(elevation_dataset) <= minimum_sweeps_raise_expection:
        raise RuntimeError(
            f"Require more than {minimum_sweeps_raise_expection} sweeps to calculate MESH"
        )
    elif len(elevation_dataset) < minimum_sweeps_raise_warning:
        raise Warning(
            (
                f"Number of sweep is less than {minimum_sweeps_raise_warning} "
                "and not recommended for MESH calculations"
            )
        )
    # sweep must be sorted from lowest to highest elevation
    dx = np.diff(elevation_dataset)
    if np.all(dx <= 0):
        raise RuntimeError(
            "Datasets have not been sorted so sweeps are increasing monotonically"
        )

    # Initialize sweep coords
    sweep0_nrays = len(azimuth_dataset[0])
    sweep0_nbins = len(range_dataset[0])
    n_ppi = len(elevation_dataset)
    z_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, altitude above ground level (m) of each range bin
    s_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, great circle arc distance (m) of each radar bin
    wt_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, temperature weighting value
    hail_ke_dataset = (
        []
    )  # list (dim: elevation) of 2d array (dim: azimuth, range) for each sweep, hail kinetic energy
    hail_refl_correction_description = ""
    for i in range(n_ppi):
        # calculate cartesian coordinates
        s, z = _antenna_to_arc(range_dataset[i], elevation_dataset[i])
        s_dataset.append(s)
        z_dataset.append(z + radar_altitude)
        # calc temperature based weighting function
        wt = (z_dataset[i] - meltlayer) / (neg20layer - meltlayer)
        wt[z_dataset[i] <= meltlayer] = 0
        wt[z_dataset[i] >= neg20layer] = 1
        wt[wt < 0] = 0
        wt[wt > 1] = 1
        wt_dataset.append(wt)
        # apply C band correction
        if radar_band == "C" and correct_cband_refl:
            reflectivity_dataset[i] = reflectivity_dataset[i] * 1.113 - 3.929
            hail_refl_correction_description = (
                "C band hail reflectivity correction applied"
                " from Brook et al. 2023 https://arxiv.org/abs/2306.12016"
            )
        # calc weights for hail kenetic energy
        reflectivity_weights = (reflectivity_dataset[i] - z_l) / (z_u - z_l)
        reflectivity_weights[reflectivity_dataset[i] <= z_l] = 0
        reflectivity_weights[reflectivity_dataset[i] >= z_u] = 1
        reflectivity_weights[reflectivity_weights < 0] = 0
        reflectivity_weights[reflectivity_weights > 1] = 1
        # limit on DBZ
        reflectivity_dataset[i][reflectivity_dataset[i] > 100] = 100
        reflectivity_dataset[i][reflectivity_dataset[i] < -100] = -100
        # calc hail kenetic energy
        hail_ke = (
            (5.0e-6) * 10 ** (0.084 * reflectivity_dataset[i]) * reflectivity_weights
        )
        hail_ke[np.isnan(hail_ke)] = 0
        hail_ke_dataset.append(hail_ke)

    # generate arc range and dz lookup (note these have different dimensions to the dimension variables)
    dz_dataset = (
        []
    )  # list (dim: range) where each element represents a range bin, 1d array (dim: elevation) where each element represents a sweep, altitude dz for shi integration (m)
    s_lookup_dataset = (
        []
    )  # list (dim: range) where each element represents an the range bin index to use from each sweep above sweep0. ASSUMES ORDERS SWEEP ELEVATION
    for rg_idx in range(sweep0_nbins):
        s_lookup = [0]
        column_z = [z_dataset[0][rg_idx]]
        for sweep_idx in range(1, n_ppi, 1):
            dist_array = np.abs(s_dataset[0][rg_idx] - s_dataset[sweep_idx])
            closest_rng_idx = np.argmin(dist_array)
            # skip sweeps where the horizontal shift is greater than column_shift_maximum (removes birdbaths and when base scan max range is greater than all other scans)
            if dist_array[closest_rng_idx] < column_shift_maximum:
                s_lookup.append(closest_rng_idx)
                column_z.append(z_dataset[sweep_idx][closest_rng_idx])
            # else:
            #     print('skipping', 'distance check', dist_array[closest_rng_idx], 'range idx', rg_idx, 'sweep idx', sweep_idx)
        # check if at least two valid values in the column exists
        if len(s_lookup) > 1:
            s_lookup_dataset.append(np.array(s_lookup))
            dz_dataset.append(_calc_dz(column_z))
        else:
            s_lookup_dataset.append(None)
            dz_dataset.append(None)

    # calculate shi on lowest sweep coordinates
    shi = np.zeros((len(azimuth_dataset[0]), len(range_dataset[0])))
    shi_mask = np.zeros((len(azimuth_dataset[0]), len(range_dataset[0])), dtype=bool)
    # loop through each ray in the lowest sweep
    for az_idx in range(sweep0_nrays):
        sweep0_az = azimuth_dataset[0][az_idx]
        # loop through each range bin for the ray
        for rg_idx in range(sweep0_nbins):
            # check if sweep0 (lowest) range is outside of limits
            if (
                s_dataset[0][rg_idx] < min_range * 1000
                or s_dataset[0][rg_idx] > max_range * 1000
            ):
                shi_mask[az_idx, rg_idx] = True
                continue
            # check if a valid column exists at this range
            if s_lookup_dataset[rg_idx] is None:
                shi_mask[az_idx, rg_idx] = True
                continue
            # init shi elements
            column_shi_elements = [
                hail_ke_dataset[0][az_idx, rg_idx] * wt_dataset[0][rg_idx]
            ]  # HKE * WT
            # loop through the sweep entries in the lookup table. ASSUMES ORDERED SWEEPS, AS IT WILL REMOVE BIRDBATH SCANS THAT FALL AT THE END OF THE VOLUME
            for sweep_idx in range(1, len(s_lookup_dataset[rg_idx]), 1):
                #check if sweep azimuth value matches sweep0
                search_azi = True
                try:
                    if sweep0_az == azimuth_dataset[sweep_idx][az_idx]:
                        search_azi = False
                except Exception as e:
                    #will fall if index exceeds array size
                    pass
                if search_azi:
                    # if not, find nearest azi
                    closest_az_idx = np.argmin(np.abs(azimuth_dataset[sweep_idx]-sweep0_az))
                    #if the azimuth differs by more than 1 degree from sweep0, use a value of -999 in the shi calculation.
                    if azimuth_dataset[sweep_idx][closest_az_idx] - sweep0_az > 1:
                        column_shi_elements.append(-999)
                        continue
                else:
                    closest_az_idx = az_idx
                # find closest point using great circle arc to sweep0 (lowest) location
                closest_rng_idx = s_lookup_dataset[rg_idx][sweep_idx]
                column_shi_elements.append(
                    hail_ke_dataset[sweep_idx][closest_az_idx, closest_rng_idx]
                    * wt_dataset[sweep_idx][closest_rng_idx]
                )
            # insert into SHI if there's a valid value
            if np.max(column_shi_elements) > 0:
                valid_sweep_idx = column_shi_elements != -999
                shi[az_idx, rg_idx] = 0.1 * np.sum(
                    column_shi_elements * dz_dataset[rg_idx][valid_sweep_idx]
                )

    # calc maximum estimated severe hail (mm)
    if (
        mesh_method == "witt1998"
    ):  # 75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        mesh = 2.54 * shi**0.5
        mesh_description = "Maximum Estimated Size of Hail retreival developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        mesh_comment = (
            "75th percentile fit using 147 hail reports; only valid in the first sweep"
        )

    elif (
        mesh_method == "mh2019_75"
    ):  # 75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = 15.096 * shi**0.206
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = (
            "75th percentile fit using 5897 hail reports; only valid in the first sweep"
        )
    elif (
        mesh_method == "mh2019_95"
    ):  # 95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = 22.157 * shi**0.212
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = (
            "95th percentile fit using 5897 hail reports; only valid in the first sweep"
        )
    else:
        raise ValueError(
            "unknown MESH method selects, please use witt1998, mh2019_75 or mh2019_95"
        )

    # calc warning threshold (J/m/s) NOTE: freezing height must be in km
    warning_threshold = 57.5 * (meltlayer / 1000) - 121

    # calc probability of severe hail (POSH) (%)
    posh = 29 * common.safe_log(shi / warning_threshold) + 50
    posh = np.real(posh)
    posh[posh < 0] = 0
    posh[posh > 100] = 100

    # mask outside of coverage with nan
    posh[shi_mask] = np.nan
    shi[shi_mask] = np.nan
    mesh[shi_mask] = np.nan

    # add grids to radar object
    # unpack E into cfradial representation
    ke_dict = {
        "data": hail_ke_dataset,
        "units": "Jm-2s-1",
        "long_name": "Hail Kinetic Energy",
        "description": "Hail Kinetic Energy developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
    }

    # SHI,MESH and POSH are only valid at the surface as a single sweep
    shi_dict = {
        "data": shi,
        "units": "Jm-1s-1",
        "long_name": "Severe Hail Index",
        "description": "Severe Hail Index developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
        "comments": "only valid in the first sweep",
    }

    mesh_dict = {
        "data": mesh,
        "units": "mm",
        "long_name": "Maximum Expected Size of Hail using " + mesh_method,
        "description": mesh_description + hail_refl_correction_description,
        "comments": mesh_comment,
    }

    posh_dict = {
        "data": posh,
        "units": "%",
        "long_name": "Probability of Severe Hail",
        "description": "Probability of Severe Hail developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
        "comments": "only valid in the first sweep",
    }

    # return output_fields dictionary
    return ke_dict, shi_dict, mesh_dict, posh_dict
