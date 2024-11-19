"""
Hail Differential Refletivity (HDR) implementation

This algorthim was developed by:
Aydin and Zhao 1990, A computational study of polarmetric radar observables in
hail. IEEE Trans. Geosci. Remote Sens. 28, 412-422

Requires reflectivity and differential reflectivity data
Conversion to hail size developed by:
Depue, T. K., Kennedy, P. C., & Rutledge, S. A. (2007). 
Performance of the hail differential reflectivity (HDR)
polarimetric radar hail indicator. 
Journal of Applied Meteorology and Climatology, 
46(8), 1290–1301. https://doi.org/10.1175/JAM2529.1

Joshua Soderholm - 15 June 2018
"""

import copy
import numpy as np
from pyhail import common


def pyart(
    radar,
    reflectivity_fname,
    differential_reflectivity_fname,
    hdr_fname="hdr",
    hdr_size_fname="hdr_size",
):
    """

    wrapper function using pyart object

    Parameters:
    ===========
    radar: class
        pyart radar object
    reflectivity_fname: string
        name of reflectivity field
    differential_reflectivity_fname: string
        name of differential reflectivity field
    hdr_fname: string
        name of HDR field
    hdr_size_fname: string
        name of HDR size field
    Returns:
    ========
    radar: pyart radar class
        updated with hdr and hdr_size fields

    """
    # init radar fields
    empty_radar_field = {
        "data": np.zeros((radar.nrays, radar.ngates)),
        "units": "",
        "long_name": "",
        "description": "",
        "comments": "",
    }
    radar.add_field(hdr_fname, copy.deepcopy(empty_radar_field))
    radar.add_field(hdr_size_fname, copy.deepcopy(empty_radar_field))

    # process sweeps
    for sweep in range(radar.nsweeps):
        hdr_meta, hdr_size_meta = main(
            radar.get_field(sweep, reflectivity_fname, copy=True).filled(np.nan),
            radar.get_field(sweep, differential_reflectivity_fname, copy=True).filled(
                np.nan
            ),
        )
        radar.fields[hdr_fname]["data"][radar.get_slice(sweep)] = hdr_meta["data"]
        radar.fields[hdr_size_fname]["data"][radar.get_slice(sweep)] = hdr_size_meta[
            "data"
        ]

    # add metadata
    radar = common.add_pyart_metadata(radar, hdr_fname, hdr_meta)
    radar = common.add_pyart_metadata(radar, hdr_size_fname, hdr_size_meta)

    return radar


def pyodim(
    datasets,
    reflectivity_fname,
    differential_reflectivity_fname,
    hdr_fname="hdr",
    hdr_size_fname="hdr_size",
):
    """

    wrapper function using pyodim object

    Parameters:
    ===========
    datasets: list of dicts
        pyodim dataset
    reflectivity_fname: string
        name of reflectivity field
    differential_reflectivity_fname: string
        name of differential reflectivity field
    hdr_fname: string
        name of HDR field
    hdr_size_fname: string
        name of HDR size field
    Returns:
    ========
    datasets: dict
        updated with hdr and hdr_size fields

    """
    # for each sweep
    for sweep_idx, _ in enumerate(datasets):
        hdr_meta, hdr_size_meta = main(
            datasets[sweep_idx][reflectivity_fname].values,
            datasets[sweep_idx][differential_reflectivity_fname].values,
        )
        # add new fields
        datasets[sweep_idx] = datasets[sweep_idx].merge(
            {
                hdr_fname: (("azimuth", "range"), hdr_meta["data"]),
                hdr_size_fname: (("azimuth", "range"), hdr_size_meta["data"]),
            }
        )
        # metadata
        datasets[sweep_idx][hdr_fname] = common.add_pyodim_metadata(
            datasets[sweep_idx][hdr_fname], hdr_meta
        )
        datasets[sweep_idx][hdr_size_fname] = common.add_pyodim_metadata(
            datasets[sweep_idx][hdr_size_fname], hdr_size_meta
        )
    return datasets


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
    hdr_data = reflectivity_sweep - zdr_fun

    # use polynomial from Depue et al. 2009 to transform dB into mm
    hdr_size = 0.0284 * (hdr_data**2) - 0.366 * hdr_data + 11.69
    hdr_size[hdr_data <= 0] = 0

    # generate meta
    hdr_meta = {
        "data": hdr_data,
        "units": "dB",
        "long_name": "Hail Differential Reflectivity",
        "description": (
            "Hail Differential Reflectivity developed by Aydin and Zhao (1990) "
            "doi:10.1109/TGRS.1990.572906"
        ),
        "comments": "",
    }

    hdr_size_meta = {
        "data": hdr_size,
        "units": "mm",
        "long_name": "HDR hail size estimate",
        "description": (
            "Hail Differential Reflectivity Hail Size developed by Depue et al. (2009) "
            "doi:10.1175/JAM2529.1"
        ),
        "comments": "transform from HDR (dB) to hail size (mm); function scaled from paper figure",
    }

    # return hdr data
    return hdr_meta, hdr_size_meta
