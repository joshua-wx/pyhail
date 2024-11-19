"""
hAcc (Hail Accumulation) implementation

Algorthim published by:
Wallace, R., Friedrich, K., Kalina, E. A., & Schlatter, P. (2019). 
Using Operational Radar to Identify Deep Hail Accumulations from Thunderstorms, 
Weather and Forecasting, 34(1), 133-150. 
from https://journals.ametsoc.org/view/journals/wefo/34/1/waf-d-18-0053_1.xml
and
Kalina, E. A., Friedrich, K., Motta, B. C., Deierling, W., Stano, G. T., & Rydell, N. N. (2016).
 Colorado Plowable Hailstorms: Synoptic Weather, Radar, and Lightning Characteristics, 
 Weather and Forecasting, 31(2), 663-693.
   from https://journals.ametsoc.org/view/journals/wefo/31/2/waf-d-15-0037_1.xml
Contains the LASH retrieval for gridded radar data.

Joshua Soderholm - 12 August 2021
"""

import numpy as np
from pyhail import common


def pyart(
    radar,
    reflectivity_fname,
    sweep_idx,
    mesh_idx,
    fz_level,
    pressure,
    hsda_fname="hsda",
    mesh_fname="mesh",
    hacc_fname="hacc",
    sp_reflectivity_threshold=55,
):
    """
    Pyart Wrapper for Hail Accumulation defined by Robinson et al. 2018 and Kalina et al. 2016.

    Parameters:
    ===========
    radar: class
        pyart radar object
    reflectivity_fname: string
        name of reflectivity field
    sweep_idx : integer
        index of reflectivity sweep to use for hacc calculation
    mesh_idx : integer
        integer of sweep continaing MESH data in the radar object
    fz_level: int
        wet bulb freezing level (m)
    pressure: float (1,)
        mean pressure between the surface and the height of the 0C wet-bulb temperature
    hsda_fname: string
        name of hsda field
    mesh_fname: string
        name of mesh field
    hacc_fname: string
        name of hacc field
    sp_reflectivity_threshold: float
        value used to threshold reflectivity for single pol analysis
    Returns:
    ========
    radar: class
        pyart radar object updated with the hacc field

    """

    # init radar fields
    empty_radar_field = {
        "data": np.zeros((radar.nrays, radar.ngates)),
        "units": "",
        "long_name": "",
        "description": "",
        "comments": "",
    }
    radar.add_field(hacc_fname, empty_radar_field)
    # run retrieval
    radar_altitude = radar.altitude["data"][0]
    _, _, sweep0_z = radar.get_gate_x_y_z(sweep_idx)
    # check if hsda exists
    if hsda_fname is None:
        hsda_field = None
    elif hsda_fname not in radar.fields.keys():
        hsda_field = None
    else:
        hsda_field = radar.get_field(sweep_idx, hsda_fname, copy=True)
    hacc_dict = main(
        radar.get_field(sweep_idx, reflectivity_fname, copy=True).filled(np.nan),
        hsda_field,
        radar.get_field(mesh_idx, mesh_fname, copy=True),
        sweep0_z + radar_altitude,
        fz_level,
        pressure,
        sp_reflectivity_threshold=sp_reflectivity_threshold,
    )
    # update data and metadata
    radar.fields[hacc_fname]["data"][radar.get_slice(sweep_idx)] = hacc_dict["data"]
    radar = common.add_pyart_metadata(radar, hacc_fname, hacc_dict)

    return radar


def pyodim(
    datasets,
    reflectivity_fname,
    sweep_idx,
    mesh_idx,
    fz_level,
    pressure,
    hsda_fname="hsda",
    mesh_fname="mesh",
    z_fname="z",
    hacc_fname="hacc",
    sp_reflectivity_threshold=55,
):
    """
    Pyodim Wrapper for Hail Accumulation defined by Robinson et al. 2018 and Kalina et al. 2016.

    Parameters:
    ===========
    datasets: list of dicts
        pyodim dataset
    reflectivity_fname: string
        name of reflectivity field
    sweep_idx : integer
        index of reflectivity sweep to use for hacc calculation
    mesh_idx : integer
        integer of sweep continaing MESH data in the radar object
    fz_level: int
        wet bulb freezing level (m)
    pressure: float (1,)
        mean pressure between the surface and the height of the 0C wet-bulb temperature
    reflectivity_fname: string
        name of reflectivity field
    hsda_fname: string
        name of hsda field
    mesh_fname: string
        name of mesh field
    hacc_fname: string
        name of hacc field
    sp_reflectivity_threshold: float
        value used to threshold reflectivity for single pol analysis
    Returns:
    ========
    datasets: list of dicts
        pyodim dataset updated with hacc field

    """
    if hsda_fname is None:
        hsda_field = None
    elif hsda_fname not in datasets[sweep_idx].keys():
        hsda_field = None
    else:
        hsda_field = datasets[sweep_idx][hsda_fname].values
    # run retrieval
    hacc_dict = main(
        datasets[sweep_idx][reflectivity_fname].values,
        hsda_field,
        datasets[mesh_idx][mesh_fname].values,
        datasets[sweep_idx][z_fname].values,
        fz_level,
        pressure,
        sp_reflectivity_threshold=sp_reflectivity_threshold,
    )

    # update data and metadata
    datasets[sweep_idx] = datasets[sweep_idx].merge(
        {hacc_fname: (("azimuth", "range"), hacc_dict["data"])}
    )
    # metadata
    datasets[sweep_idx][hacc_fname] = common.add_pyodim_metadata(
        datasets[sweep_idx][hacc_fname], hacc_dict
    )
    return datasets


def main(
    reflectivity_sweep,
    hsda_sweep,
    mesh_sweep,
    z_sweep,
    fz_level,
    pressure,
    sp_reflectivity_threshold=55,
):
    """
    Hail Accumulation defined by Robinson et al. 2018 and Kalina et al. 2016.

    Parameters:
    ===========
    reflectivity_sweep : 2D ndarray
        sweep reflectivity data in an array with dimensions (azimuth, range)
    hsda_sweep : 2D ndarray
        sweep hsda data in an array with dimensions (azimuth, range).
        Set to None to use reflectivity threshold.
    mesh_sweep : 2D ndarray
        sweep mesh data in an array with dimensions (azimuth, range)
    z_sweep: 2D ndarray
        sweep altitude above sea level in m (azimuth, range)
    fz_level: int
        wet bulb freezing level (m)
    pressure: float (1,)
        mean pressure between the surface and the height of the 0C wet-bulb temperature
    sp_reflectivity_threshold: float
        value used to threshold reflectivity for single pol analysis
    Returns:
        hAcc_meta: dict
        pyart field dictionary containing hAcc dataset

    """
    # create hail mask dataset
    if hsda_sweep is None:
        # use a simple single pol HCA for hail (fixed threshold)
        hail_hca = reflectivity_sweep >= sp_reflectivity_threshold
    else:
        # use hsda to mask hail
        hail_hca = hsda_sweep > 0

    # for the lowest sweep, calculate the IWC
    n = 0.64  # packing density of monodisperse spheres (Kalina et al. 2016)
    ph = 900  # density of ice (kg m-3)
    epsilon = 0.814

    ze = 10.0 ** (reflectivity_sweep / 10.0)  # convert Z to Ze
    iwc_sweep = (
        (4.4 * 10**-5) * ze ** (0.71) / 1000
    )  # Ice Water Content (kg m-3) derived from Ze follow Heysfield and Miller 1998
    # remove IWC values where hail_hca is not hail (less than 1)
    iwc_sweep[hail_hca < 1] = 0
    # remove IWC values where temperature is at or below 0
    iwc_sweep[z_sweep > fz_level] = 0

    # pressure correction from Heysmfield and Write (2014)
    pc = (1000 / pressure) ** 0.545
    # diameter-fall speed relation from Heysmfield and Wright (2014), units of cm/s
    vt = 488 * (mesh_sweep / 10) ** 0.84 * pc

    # calculate LASH (units of cm/s)
    hacc = (1 / epsilon) * (1 / (n * ph)) * iwc_sweep * vt
    hacc = hacc * 60  # convert cm/s to cm/min

    # propagate nan as needed
    hacc[np.isnan(reflectivity_sweep)] = np.nan

    # create metadata
    hacc_meta = {
        "data": hacc,
        "units": "cm/min",
        "long_name": "hail accumulation",
        "description": (
            "Hail Accumulation Retrieval developed by Wallace et al. (2019)"
            " doi:10.1175/WAF-D-18-0053.1"
        ),
        "comments": "only valid in the first sweep",
    }

    return hacc_meta
