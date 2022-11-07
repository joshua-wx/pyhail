"""
Hail Differential Refletivity (HDR) implementation

This algorthim was developed by:
Aydin and Zhao 1990, A computational study of polarmetric radar observables in hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422
Requires reflectivity and differential reflectivity data
Conversion to hail size developed by:
Depue, T. K., Kennedy, P. C., & Rutledge, S. A. (2007). Performance of the hail differential reflectivity (HDR) polarimetric radar hail indicator. Journal of Applied Meteorology and Climatology, 46(8), 1290–1301. https://doi.org/10.1175/JAM2529.1

Joshua Soderholm - 15 June 2018
"""


def main(radar_dict):
    """
    Hail Differential Reflectity Retrieval
    Required DBZH and ZDR fields

    Parameters:
    ===========
    radar_dict: dictionary
        contains two entries, dbz and zdr, which contain numpy arrays of their respective fields.
    Returns:
    ========
    hdr_meta: dict
        pyart field dictionary containing HDR dataset
    hdr_size_meta: dict
        pyary field dictionary containing HDR size dataset

    """
    # extract fields
    dbz = radar_dict["dbz"]
    zdr = radar_dict["zdr"]

    # calculate hdr
    # apply primary function
    zdr_fun = 19 * zdr + 27
    # set limits based on zdr
    zdr_fun[zdr <= 0] = 27
    zdr_fun[zdr > 1.74] = 60
    # apply to zhh
    hdr = dbz - zdr_fun

    # use polynomial from Depue et al. 2009 to transform dB into mm
    hdr_size = 0.0284 * (hdr ** 2) - 0.366 * hdr + 11.69
    hdr_size[hdr <= 0] = 0

    # generate meta
    hdr_meta = {
        "data": hdr,
        "units": "dB",
        "long_name": "Hail Differential Reflectivity",
        "description": "Hail Differential Reflectivity developed by Aydin and Zhao (1990) doi:10.1109/TGRS.1990.572906"
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
