"""
Hail Differential Refletivity sub-module of pyhail
Aydin and Zhao 1990, A computational study of polarmetric radar observables in hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422
Requires reflectivity and differential reflectivity data

Contains HDF method

Joshua Soderholm - 15 June 2018
"""



def main(radar_dict):
    """
    Hail Differential Reflectity Retrieval
    Required DBZH and ZDR fields

    Parameters:
    ===========
    radar: struct
        pyart radar object
    ref_name: string
        name of reflecitivty field
    zdr_name: string
        name of zdr field

    Returns:
    ========
    hdr:
        ndarray containing hail differential reflectivity (mm)
    """
    #extract fields
    dbz = radar_dict['dbz']
    zdr = radar_dict['zdr']

    #calculate hdr
    #apply primary function
    zdr_fun = 19 * zdr + 27
    #set limits based on zdr
    zdr_fun[zdr <= 0]   = 27
    zdr_fun[zdr > 1.74] = 60
    #apply to zhh
    hdr = dbz - zdr_fun

    #use polynomial from Depue et al. 2009 to transform dB into mm
    hdr_size = 0.0284*(hdr**2)-0.366*hdr+11.69
    hdr_size[hdr<=0] = 0
    
    #generate meta
    the_comments = "Applies Aydin and Zhao 1990"
    hdr_meta     = {'data': hdr, 'units': 'dB', 'long_name': 'Hail Differential Reflectivity',
                  'standard_name': 'HDR', 'comments': the_comments}

    the_comments = "Applies the transform from hdr to mm used by Depue et al. 2009"
    hdr_size_meta = {'data': hdr_size, 'units': 'mm', 'long_name': 'HDR hail size estimate',
                  'standard_name': 'HDR', 'comments': the_comments}
    
    #return hdr data 
    return hdr_meta, hdr_size_meta