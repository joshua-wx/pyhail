"""
Hail Differential Refletivity sub-module of pyhail
Aydin and Zhao 1990, A computational study of polarmetric radar observables in hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422
Requires reflectivity and differential reflectivity data

Contains HDF method

Joshua Soderholm - 15 June 2018
"""

import pyart

def main(radar,fieldnames):
    """
    Hail Differential Reflectity Retrieval
    Required DBZH and ZDR fields

    Parameters:
    ===========
    radar: struct
        pyart radar object
    fieldnames: dict
        map pyart fieldnames

    Returns:
    ========
    hdr:
        ndarray containing hail differential reflectivity (mm)
    """
    #extract fields
    dbz = radar.fields[fieldnames['dbzh_corr']]['data']
    zdr = radar.fields[fieldnames['zdr_corr']]['data']

    #calculate hdr
    #apply primary function
    zdr_fun = 19 * zdr + 27
    #set limits based on zdr
    zdr_fun[zdr <= 0]   = 27
    zdr_fun[zdr > 1.74] = 60
    #apply to zhh
    hdr = dbz - zdr_fun

    #return hdr data 
    return hdr