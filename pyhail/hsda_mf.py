"""
Hail Size Discrimination Algrothim membership functions sub-module of pyhail

Joshua Soderholm - 15 June 2018
"""

####################################################################
# Functions for within membership functions
####################################################################


def c(offset, zh, dzdr):
    return offset


def f1(offset, zh, dzdr):
    out = -0.5 + ((2.5 * 10 ** -3) * zh) + ((7.5 * 10 ** -4) * (zh ** 2)) + dzdr
    out = out + offset
    return out


def f2(offset, zh, dzdr):
    out = 0.1 * (zh - 50.0) + dzdr
    out = out + offset
    return out


def f3(offset, zh, dzdr):
    out = 0.1 * (zh - 60.0) + dzdr
    out = out + offset
    return out


def g1(offset, zh, dzdr):
    out = -0.9 + ((1.5 * 10 ** -2) * zh) + ((5.0 * 10 ** -4) * (zh ** 2)) + dzdr
    out = out + offset
    return out


def g2(offset, zh, dzdr):
    out = 0.075 * (zh - 50.0) + dzdr
    out = out + offset
    return out


def g3(offset, zh, dzdr):
    out = 0.075 * (zh - 60.0) + dzdr
    out = out + offset
    return out


def build_mf():
    """
        build membership functions for HSDA retrieval

        syntax:
                a#: height interval # (1 is the highest, 6 is the lowest)
                h#: hail size interval # (1 is >25, 2 is 25-50, 3 is >50)

        TODO: build confidence vector

    Parameters:
    ===========

    Returns:
    ========
    w: dict
        Field weights at the 6 height intervals and 3 fields
        mf: dict
                Trapezoid membership function parameters for the 6 height intervals, 3 hail sizes and 3 fields
        q: dict
                Confidence vector (currently just 1)
    """
    # weights
    w = {
        "a1.zh": 1.0,
        "a1.zdr": 0.3,
        "a1.rhv": 0.6,
        "a2.zh": 1.0,
        "a2.zdr": 0.3,
        "a2.rhv": 0.6,
        "a3.zh": 0.8,
        "a3.zdr": 0.5,
        "a3.rhv": 0.6,
        "a4.zh": 0.7,
        "a4.zdr": 0.8,
        "a4.rhv": 0.6,
        "a5.zh": 0.7,
        "a5.zdr": 1.0,
        "a5.rhv": 0.6,
        "a6.zh": 0.7,
        "a6.zdr": 1.0,
        "a6.rhv": 0.6,
    }

    # membership functions

    mf = {
        "a1.h1.zh": [45, 50, 60, 65],
        "a1.h1.zdr": [-0.5, -0.3, 0.3, 0.5],
        "a1.h1.rhv": [0.92, 0.96, 0.99, 1.00],
        "a2.h1.zh": [45, 50, 60, 65],
        "a2.h1.zdr": [-0.5, -0.3, 0.3, 0.5],
        "a2.h1.rhv": [0.92, 0.96, 0.99, 1.00],
        "a3.h1.zh": [45, 50, 60, 65],
        "a3.h1.zdr": [-0.1, 0.3, 0.7, 1.2],
        "a3.h1.rhv": [0.93, 0.96, 0.99, 1.00],
        "a4.h1.zh": [45, 52, 62, 67],
        "a4.h1.zdr": [[g2, -0.3], [g2, 0.0], [g1, 0.0], [g1, 0.3]],
        "a4.h1.rhv": [0.94, 0.96, 0.98, 1.00],
        "a5.h1.zh": [45, 49, 59, 64],
        "a5.h1.zdr": [[f2, -0.3], [f2, 0.0], [f1, 0.0], [f1, 0.3]],
        "a5.h1.rhv": [0.91, 0.94, 0.96, 0.99],
        "a6.h1.zh": [45, 47, 57, 62],
        "a6.h1.zdr": [[f2, -0.3], [f2, 0.0], [f1, 0.0], [f1, 0.3]],
        "a6.h1.rhv": [0.91, 0.94, 0.96, 0.99],
        "a1.h2.zh": [48, 58, 63, 68],
        "a1.h2.zdr": [-0.5, -0.3, 0.3, 0.5],
        "a1.h2.rhv": [0.92, 0.96, 0.99, 1.00],
        "a2.h2.zh": [48, 58, 63, 68],
        "a2.h2.zdr": [-0.5, -0.3, 0.3, 0.5],
        "a2.h2.rhv": [0.86, 0.90, 0.96, 0.98],
        "a3.h2.zh": [48, 58, 63, 68],
        "a3.h2.zdr": [-0.3, 0.1, 0.5, 1.0],
        "a3.h2.rhv": [0.80, 0.91, 0.97, 0.98],
        "a4.h2.zh": [50, 60, 65, 70],
        "a4.h2.zdr": [[g3, -0.3], [g3, 0.0], [g2, 0.0], [g2, 0.3]],
        "a4.h2.rhv": [0.80, 0.91, 0.97, 0.98],
        "a5.h2.zh": [50, 57, 62, 67],
        "a5.h2.zdr": [[f3, -0.3], [f3, 0.0], [f2, 0.0], [f2, 0.3]],
        "a5.h2.rhv": [0.80, 0.90, 0.96, 0.99],
        "a6.h2.zh": [50, 55, 60, 65],
        "a6.h2.zdr": [[f3, -0.3], [f3, 0.0], [f2, 0.0], [f2, 0.3]],
        "a6.h2.rhv": [0.80, 0.90, 0.96, 0.99],
        "a1.h3.zh": [50, 60, 100, 101],
        "a1.h3.zdr": [-8.75, -7.75, 0.3, 0.5],
        "a1.h3.rhv": [-1.00, 0.00, 0.99, 1.00],
        "a2.h3.zh": [50, 60, 100, 101],
        "a2.h3.zdr": [-8.75, -7.75, 0.2, 0.5],
        "a2.h3.rhv": [-1.00, 0.00, 0.93, 0.98],
        "a3.h3.zh": [50, 60, 100, 101],
        "a3.h3.zdr": [-8.75, -7.75, 0.2, 0.7],
        "a3.h3.rhv": [-1.00, 0.00, 0.94, 0.98],
        "a4.h3.zh": [52, 62, 100, 101],
        "a4.h3.zdr": [[c, -8.75], [c, -7.75], [g3, 0.0], [g3, 0.3]],
        "a4.h3.rhv": [-1.00, 0.00, 0.96, 0.98],
        "a5.h3.zh": [50, 59, 100, 101],
        "a5.h3.zdr": [[c, -8.75], [c, -7.75], [f3, 0.0], [f3, 0.3]],
        "a5.h3.rhv": [-1.00, 0.00, 0.93, 0.98],
        "a6.h3.zh": [50, 57, 100, 101],
        "a6.h3.zdr": [[c, -8.75], [c, -7.75], [f3, 0.0], [f3, 0.3]],
        "a6.h3.rhv": [-1.00, 0.00, 0.93, 0.98],
    }

    return w, mf
