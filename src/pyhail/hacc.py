"""
hAcc (Hail Accumulation) implementation

Algorthim published by:
Wallace, R., Friedrich, K., Kalina, E. A., & Schlatter, P. (2019). Using Operational Radar to Identify Deep Hail Accumulations from Thunderstorms, Weather and Forecasting, 34(1), 133-150. from https://journals.ametsoc.org/view/journals/wefo/34/1/waf-d-18-0053_1.xml
and
Kalina, E. A., Friedrich, K., Motta, B. C., Deierling, W., Stano, G. T., & Rydell, N. N. (2016). Colorado Plowable Hailstorms: Synoptic Weather, Radar, and Lightning Characteristics, Weather and Forecasting, 31(2), 663-693. from https://journals.ametsoc.org/view/journals/wefo/31/2/waf-d-15-0037_1.xml
Contains the LASH retrieval for gridded radar data.

Joshua Soderholm - 12 August 2021
"""

import numpy as np
from pyhail import common


def main(radar, fz_level, pressure, z_fname, hsda_fname, mesh_fname, sp_reflectivity_threshold=55, heights_fieldname='gate_z'):

    """
    Hail Accumulation defined by Robinson et al. 2018 and Kalina et al. 2016.
    If the heights field exists, this will be used and save a small amount of computation time.

    Parameters:
    ===========
    radar : object
        Py-ART radar object.
    fz_level: int
        wet bulb freezing level (m)
    pressure: float (1,)
        mean pressure between the surface and the height of the 0C wet-bulb temperature
    z_fname: str
        reflectivity field name
    hsda_fname: str
        field name for HSDR
    mesh_fname: str
        field name for MESH
    sp_reflectivity_threshold: float
        value used to threshold reflectivity for single pol analysis
    Returns:
        hAcc_meta: dict
        pyart field dictionary containing hAcc dataset

    """
    Z = radar.fields[z_fname]["data"]
    if np.ma.is_masked(Z):
        Z = Z.filled(0)
    if hsda_fname is None:
        #use a simple single pol HCA for hail (fixed threshold)
        hail_hca = Z >=  sp_reflectivity_threshold
    else:
        #use hsda to determine hail
        hail_hca = radar.fields[hsda_fname]["data"]
        if np.ma.is_masked(hail_hca):
            hail_hca = hail_hca.filled(0)
    #load mesh
    mesh = radar.get_field(0, mesh_fname)
    if np.ma.is_masked(mesh):
        mesh = mesh.filled(0)

    # calculate height
    try:
        heights = radar.fields[heights_fieldname]['data']
    except:
        rg, azg = np.meshgrid(radar.range["data"], radar.azimuth["data"])
        rg, eleg = np.meshgrid(radar.range["data"], radar.elevation["data"])
        _, _, heights = common.antenna_to_cartesian(rg / 1000, azg, eleg)

    n = 0.64  # packing density of monodisperse spheres (Kalina et al. 2016)
    ph = 900  # density of ice (kg m-3)
    epsilon = 0.814

    Ze = 10.0 ** (Z / 10.0)  # convert Z to Ze
    IWC = (
        (4.4 * 10 ** -5) * Ze ** (0.71) / 1000
    )  # Ice Water Content (kg m-3) derived from Ze follow Heysfield and Miller 1998
    # remove IWC values where hail_hca is not hail (less than 1)
    IWC[hail_hca < 1] = 0
    # remove IWC values where temperature is at or below 0
    IWC[heights > fz_level] = 0

    # get lowest valid IWC
    # insert sweeps into 3D array (el, az, rg)
    el_sort_idx = np.argsort(radar.fixed_angle["data"])
    az = radar.get_azimuth(0)
    rg = radar.range["data"]
    IWC_3d = np.ma.zeros((len(el_sort_idx), len(az), len(rg)))
    for i, el_idx in enumerate(el_sort_idx):
        IWC_3d[i, :, :] = IWC[radar.get_slice(el_idx)] 
    # mask zero values
    IWC_3d_masked = np.ma.masked_array(IWC_3d, IWC_3d == 0)
    data_shape = IWC_3d_masked.shape
    # find the lowest unmasked value by first finding edges
    edges = np.ma.notmasked_edges(IWC_3d_masked, axis=0)
    # use first edge on axis 0 (lowest in height)
    IWC_lowest_valid = np.zeros_like(mesh)
    IWC_lowest_valid[edges[0][1], edges[0][2]] = IWC_3d_masked[edges[0]]

    # pressure correction from Heysmfield and Write (2014)
    PC = (1000 / pressure) ** 0.545
    # diameter-fall speed relation from Heysmfield and Wright (2014), units of cm/s
    Vt = 488 * (mesh / 10) ** 0.84 * PC

    # calculate LASH (units of cm/s)
    hAcc = (1 / epsilon) * (1 / (n * ph)) * IWC_lowest_valid * Vt
    hAcc = hAcc * 60  # convert cm/s to cm/min

    # hAcc is only valid at the surface, to represent it in pyart radar objects, insert it into the first sweep
    hAcc_field = np.zeros_like(radar.fields[z_fname]["data"])
    hAcc_field[radar.get_slice(0)] = hAcc
    hAcc_meta = {
        "data": hAcc_field,
        "units": "cm/min",
        "long_name": "hail accumulation",
        "description": "Hail Accumulation Retrieval developed by Wallace et al. (2019) doi:10.1175/WAF-D-18-0053.1",
        "comments": "only valid in the first sweep",
    }

    return hAcc_meta
