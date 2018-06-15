"""
Hail Differential Reflectivity implementation using pyart radar object
Aydin and Zhao 1990, A computational study of polarmetric radar observables in hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422
"""

import pyart
import numpy as np

def zdr_fun(zdr):
    #apply primary function
    out = 19 * zdr + 27
    #set limits based on zdr
    out[zdr <= 0]    = 27
    out[zdr > 1.74] = 60
    return out

def main(radar,fieldnames):

    #extract fields
    zhh = radar.fields[fieldnames['dbzh']]['data']
    zdr = radar.fields[fieldnames['zdr']]['data']

    #calculate hdr
    hdr = zhh - zdr_fun(zdr)

    #return hdr data 
    return hdr