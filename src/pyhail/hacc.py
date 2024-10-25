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
import common


def main(reflectivity_sweep, 
         hsda_sweep,
         mesh_sweep,
         z_sweep,
         fz_level, pressure,
         sp_reflectivity_threshold=55):

    """
    Hail Accumulation defined by Robinson et al. 2018 and Kalina et al. 2016.

    Parameters:
    ===========
    reflectivity_sweep : 2D ndarray
        sweep reflectivity data in an array with dimensions (azimuth, range)
    hsda_sweep : 2D ndarray
        sweep hsda data in an array with dimensions (azimuth, range). Set to None to use reflectivity threshold.
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
    #create hail mask dataset
    if hsda_sweep is None:
        #use a simple single pol HCA for hail (fixed threshold)
        hail_hca = reflectivity_sweep >=  sp_reflectivity_threshold
    else:
        #use hsda to mask hail
        hail_hca = hsda_sweep > 0

    #for the lowest sweep, calculate the IWC
    n = 0.64  # packing density of monodisperse spheres (Kalina et al. 2016)
    ph = 900  # density of ice (kg m-3)
    epsilon = 0.814

    Ze = 10.0 ** (reflectivity_sweep / 10.0)  # convert Z to Ze
    IWC_sweep = (
        (4.4 * 10 ** -5) * Ze ** (0.71) / 1000
    )  # Ice Water Content (kg m-3) derived from Ze follow Heysfield and Miller 1998
    # remove IWC values where hail_hca is not hail (less than 1)
    IWC_sweep[hail_hca < 1] = 0
    # remove IWC values where temperature is at or below 0
    IWC_sweep[z_sweep > fz_level] = 0

    # pressure correction from Heysmfield and Write (2014)
    PC = (1000 / pressure) ** 0.545
    # diameter-fall speed relation from Heysmfield and Wright (2014), units of cm/s
    Vt = 488 * (mesh_sweep / 10) ** 0.84 * PC

    # calculate LASH (units of cm/s)
    hAcc = (1 / epsilon) * (1 / (n * ph)) * IWC_sweep * Vt
    hAcc = hAcc * 60  # convert cm/s to cm/min

    # create metadata
    hAcc_meta = {
        "data": hAcc,
        "units": "cm/min",
        "long_name": "hail accumulation",
        "description": "Hail Accumulation Retrieval developed by Wallace et al. (2019) doi:10.1175/WAF-D-18-0053.1",
        "comments": "only valid in the first sweep",
    }

    return hAcc_meta
