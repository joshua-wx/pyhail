"""
common sub-module of pyhail

Contains commonly used functions.

Joshua Soderholm - 15 June 2018
"""

import numpy as np
import h5py


def get_odim_ncar_hca(elevation, odim_ffn, array_shape, skip_birdbath=True):
    """
    Get the NCAR HCA data from the ODIMH5 file odim_ffn for the sweep

    Parameters
    ----------
    elevation : float
        elevation angle of sweep to use
    odim_ffn: string
        filename of odimh5 file
    array_shape: tuple
        tuple of the data arrange shape with 2 values
    skip_birdbath: boolean
        flag to skip birdbath scans (90 deg elevation)
    fillvalue: int
        Fillvalue used for data array

    Returns
    -------
    hca_meta : dict
        dictionary of ncar HCA for current sweep

    """
    # init
    the_comments = (
        "0: nodata; 1: Cloud; 2: Drizzle; 3: Light_Rain; 4: Moderate_Rain; 5: Heavy_Rain; "
        + "6: Hail; 7: Rain_Hail_Mixture; 8: Graupel_Small_Hail; 9: Graupel_Rain; "
        + "10: Dry_Snow; 11: Wet_Snow; 12: Ice_Crystals; 13: Irreg_Ice_Crystals; "
        + "14: Supercooled_Liquid_Droplets; 15: Flying_Insects; 16: Second_Trip; "
        + "17: Ground_Clutter; 18: misc1; 19: misc2"
    )
    hca = np.zeros(array_shape)
    hca[:] = np.nan
    hca_meta = {
        "data": hca,
        "units": "NA",
        "long_name": "NCAR Hydrometeor classification",
        "description:": ("NCAR Hydrometeor classification developed by Vivekanandan et al. (1999) "
        "doi:10.1175/1520-0477(1999)080<0381:CMRUSB>2.0.CO;2"),
        "comments": the_comments,
    }
    with h5py.File(odim_ffn, "r") as f:
        h5keys = list(f.keys())
        # init
        if "how" in h5keys:
            h5keys.remove("how")
        if "what" in h5keys:
            h5keys.remove("what")
        if "where" in h5keys:
            h5keys.remove("where")
        n_keys = len(h5keys)
        for i in range(n_keys):
            # read dataset
            ds_name = "dataset" + str(i + 1)
            # skip until required elevation angle is found
            if f[ds_name]["where"].attrs["elangle"] != elevation:
                continue
            # skip if birdbath
            if f[ds_name]["where"].attrs["elangle"] == 90 and skip_birdbath:
                return hca_meta
            # read pid data into output dictionary
            hca_data = np.array(f[ds_name]["quality1"]["data"]).astype(float)
            hca_data[hca_data == -1] = np.nan
            shape = hca_data.shape
            hca_meta["data"][: shape[0], : shape[1]] = hca_data
            break

    return hca_meta


def add_pyodim_metadata(
        ds, metadata_dict, skip_key="data"
):
    """
    For each key in metadata_dict, a new attribute is created in sweep_ds with the key value

    Parameters
    ----------
    ds : xarray data
        xarray dataset
    metadata_dict: dict
        dictionary containing keys and values to add into sweep_ds
    skip_key: string
        names of key to skip in metadata_dict

    Returns
    -------
    sweep_ds : xarray data
        sweep xarray dataset

    """

    for key_name in metadata_dict.keys():
        if key_name != skip_key:
            ds.attrs[key_name] = metadata_dict[key_name]
    return ds


def add_pyart_metadata(radar, variable_name, metadata_dict, skip_key="data"):
    """
    For each key in metadata_dict, a new attribute is created in sweep_ds with the key value

    Parameters
    ----------
    radar : class
        pyart radar object
    variable_name: string
        name of variable in sweep_ds to update
    metadata_dict: dict
        dictionary containing keys and values to add into sweep_ds
    skip_key: string
        names of key to skip in metadata_dict

    Returns
    -------
    radar : class
        pyart radar object

    """

    for key_name in metadata_dict.keys():
        if key_name != skip_key:
            radar.fields[variable_name][key_name] = metadata_dict[key_name]
    return radar


def safe_log(x, eps=1e-10):
    """
    Safe log function

    Parameters
    ----------
    x: numpy array

    Returns
    -------
    result : numpy array
"""

    result = np.where(x > eps, x, -10)
    np.log(result, out=result, where=result > 0)
    return result
