"""
Hail Size Discrimination Algrothim membership functions sub-module of pyhail

Joshua Soderholm - 15 June 2018
"""

####################################################################
# Functions for within membership functions
####################################################################

def c(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    return mf_off

def f1(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = -0.5 + ((2.5 * 10 ** -3) * zh) + ((7.5 * 10 ** -4) * (zh ** 2)) + dzdr
    return out + mf_off


def f2(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = 0.1 * (zh - 50.0) + dzdr
    return out + mf_off


def f3(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = 0.1 * (zh - 60.0) + dzdr
    return out + mf_off


def g1(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = -0.9 + ((1.5 * 10 ** -2) * zh) + ((5.0 * 10 ** -4) * (zh ** 2)) + dzdr
    return out + mf_off


def g2(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = 0.075 * (zh - 50.0) + dzdr
    return out + mf_off


def g3(const, mf_off=0):
    """
    where:
    zh (float): horizontal reflectivity
    dzdr (float): zdr calibration offset
    mf_off (float): offset from membership function
    """
    zh, dzdr = const
    out = 0.075 * (zh - 60.0) + dzdr
    return out + mf_off


def build_mf(alt, zh, dzdr):
    """
    generate weights and membership function tables for HSDA retrieval

    Parameters:
    ===========
    alt: int
        index of altitude where
        0: alt >= wbt_minus25C:
        1: alt >= wbt_0C:
        2: alt >= (wbt_0C - 1000):
        3: alt >= (wbt_0C - 2000):
        4: alt >= (wbt_0C - 3000):
        5: else
    zh: float
        horz reflectivity value
    dzdr: float
        zdr calibration value
    Returns:
    ========
    w: list of floats length 3
        weights for [zh, zdr, rhv]
    mf_h1: 3 sets of lists of length 3 containing floats
        small hail (< 25mm)
        each set of lists represents membership function shape for [zh, zdr, rhv]
        each list contains the lower, mid1, mid2 and upper values
    mf_h2: 3 sets of lists of length 3 containing floats
        large hail (25-50mm)
        each set of lists represents membership function shape for [zh, zdr, rhv]
        each list contains the lower, mid1, mid2 and upper values
    mf_h3: 3 sets of lists of length 3 containing floats
        giant hail (> 50mm)
        each set of lists represents membership function shape for [zh, zdr, rhv]
        each list contains the lower, mid1, mid2 and upper values

    """
    
    const = (zh, dzdr)
    if alt == 0:
        w = [1.0, 0.3, 0.6]
        mf_h1 = [[45, 50, 60, 65], [-0.5, -0.3, 0.3, 0.5], [0.92, 0.96, 0.99, 1.00]]
        mf_h2 = [[48, 58, 63, 68], [-0.5, -0.3, 0.3, 0.5], [0.92, 0.96, 0.99, 1.00]]
        mf_h3 = [[50, 60, 100, 101], [-8.75, -7.75, 0.3, 0.5], [-1.00, 0.00, 0.99, 1.00]]
        
    elif alt == 1:
        w = [1.0, 0.3, 0.6]
        mf_h1 = [[45, 50, 60, 65], [-0.5, -0.3, 0.3, 0.5], [0.92, 0.96, 0.99, 1.00]]
        mf_h2 = [[48, 58, 63, 68], [-0.5, -0.3, 0.3, 0.5], [0.86, 0.90, 0.96, 0.98]]
        mf_h3 = [[50, 60, 100, 101], [-8.75, -7.75, 0.2, 0.5], [-1.00, 0.00, 0.93, 0.98]]
        
    elif alt == 2:
        w = [0.8, 0.5, 0.6]
        mf_h1 = [[45, 50, 60, 65], [-0.1, 0.3, 0.7, 1.2], [0.93, 0.96, 0.99, 1.00]]
        mf_h2 = [[48, 58, 63, 68], [-0.3, 0.1, 0.5, 1.0], [0.80, 0.91, 0.97, 0.98]]
        mf_h3 = [[50, 60, 100, 101], [-8.75, -7.75, 0.2, 0.7], [-1.00, 0.00, 0.94, 0.98]]         
        
    elif alt == 3:
        w = [0.7, 0.8, 0.6]
        mf_h1 = [[45, 52, 62, 67], [g2(const, -0.3), g2(const), g1(const), g1(const, 0.3)],  [0.94, 0.96, 0.98, 1.00]]
        mf_h2 = [[50, 60, 65, 70], [g3(const, -0.3), g3(const), g2(const), g2(const, 0.3)], [0.80, 0.91, 0.97, 0.98]]
        mf_h3 = [[52, 62, 100, 101], [c(const, -8.75), c(const, -7.75), g3(const), g3(const,0.3)], [-1.00, 0.00, 0.96, 0.98]]
        
    elif alt == 4:
        w = [0.7, 1.0, 0.6]
        mf_h1 = [[45, 49, 59, 64], [f2(const, -0.3), f2(const), f1(const), f1(const, 0.3)], [0.91, 0.94, 0.96, 0.99]]
        mf_h2 = [[50, 57, 62, 67], [f3(const, -0.3), f3(const), f2(const), f2(const, 0.3)], [0.80, 0.90, 0.96, 0.99]]
        mf_h3 = [[50, 59, 100, 101], [c(const, -8.75), c(const, -7.75), f3(const), f3(const,0.3)], [-1.00, 0.00, 0.93, 0.98]]
        
    elif alt == 5:
        w = [0.7, 1.0, 0.6]
        mf_h1 = [[45, 47, 57, 62], [f2(const, -0.3), f2(const), f1(const), f1(const, 0.3)], [0.91, 0.94, 0.96, 0.99]]
        mf_h2 = [[50, 55, 60, 65], [f3(const, -0.3), f3(const), f2(const), f2(const, 0.3)], [0.80, 0.90, 0.96, 0.99]]
        mf_h3 = [[50, 57, 100, 101], [c(const, -8.75), c(const, -7.75), f3(const), f3(const, 0.3)], [-1.00, 0.00, 0.93, 0.98]]

    # membership functions

    return w, mf_h1, mf_h2, mf_h3
