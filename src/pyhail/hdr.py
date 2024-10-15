"""
Hail Differential Refletivity (HDR) implementation

This algorthim was developed by:
Aydin and Zhao 1990, A computational study of polarmetric radar observables in hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422
Requires reflectivity and differential reflectivity data
Conversion to hail size developed by:
Depue, T. K., Kennedy, P. C., & Rutledge, S. A. (2007). Performance of the hail differential reflectivity (HDR) polarimetric radar hail indicator. Journal of Applied Meteorology and Climatology, 46(8), 1290–1301. https://doi.org/10.1175/JAM2529.1

Joshua Soderholm - 15 June 2018
"""


def main(reflectivity_sweep, differential_reflectivity_sweep):
    """
    Hail Differential Reflectity Retrieval
    Required DBZH and ZDR fields

    Parameters:
    ===========
    reflectivity_sweep: 2d ndarray
        reflectivity data in an array with dimensions (azimuth, range)
    reflectivity_sweep: 2d ndarray
        differential reflectivity data in an array with dimensions (azimuth, range)    
    Returns:
    ========
    hdr_meta: dict
        pyart field dictionary containing HDR dataset
    hdr_size_meta: dict
        pyary field dictionary containing HDR size dataset

    """

    # calculate hdr
    # apply primary function
    zdr_fun = 19 * differential_reflectivity_sweep + 27
    # set limits based on zdr
    zdr_fun[differential_reflectivity_sweep <= 0] = 27
    zdr_fun[differential_reflectivity_sweep > 1.74] = 60
    # apply to zhh
    hdr = reflectivity_sweep - zdr_fun

    # use polynomial from Depue et al. 2009 to transform dB into mm
    hdr_size = 0.0284 * (hdr ** 2) - 0.366 * hdr + 11.69
    hdr_size[hdr <= 0] = 0

    # generate meta
    hdr_meta = {
        "data": hdr,
        "units": "dB",
        "long_name": "Hail Differential Reflectivity",
        "description": "Hail Differential Reflectivity developed by Aydin and Zhao (1990) doi:10.1109/TGRS.1990.572906",
        "comments": ""

    }

    hdr_size_meta = {
        "data": hdr_size,
        "units": "mm",
        "long_name": "HDR hail size estimate",
        "description": "Hail Differential Reflectivity Hail Size developed by Depue et al. (2009) doi:10.1175/JAM2529.1",
        "comments": "transform from HDR (dB) to hail size (mm); function scaled from paper figure"
    }

    # return hdr data
    return hdr_meta, hdr_size_meta
