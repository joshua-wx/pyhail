"""
common sub-module of pyhail

Contains commonly used functions.

Joshua Soderholm - 15 June 2018
"""

import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy import ndimage as ndi

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
    #init
    the_comments = (
        "0: nodata; 1: Cloud; 2: Drizzle; 3: Light_Rain; 4: Moderate_Rain; 5: Heavy_Rain; "
        + "6: Hail; 7: Rain_Hail_Mixture; 8: Graupel_Small_Hail; 9: Graupel_Rain; "
        + "10: Dry_Snow; 11: Wet_Snow; 12: Ice_Crystals; 13: Irreg_Ice_Crystals; "
        + "14: Supercooled_Liquid_Droplets; 15: Flying_Insects; 16: Second_Trip; 17: Ground_Clutter; "
        + "18: misc1; 19: misc2"
    )
    hca = np.zeros(array_shape)
    hca[:] = np.nan
    hca_meta = {
        "data": hca,
        "units": "NA",
        "long_name": "NCAR Hydrometeor classification",
        "description:": "NCAR Hydrometeor classification developed by Vivekanandan et al. (1999) doi:10.1175/1520-0477(1999)080<0381:CMRUSB>2.0.CO;2",
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
            #read dataset
            ds_name = "dataset" + str(i + 1)
            #skip until required elevation angle is found
            if f[ds_name]["where"].attrs["elangle"] != elevation:
                continue
            #skip if birdbath
            if f[ds_name]["where"].attrs["elangle"] == 90 and skip_birdbath:
                return hca_meta
            #read pid data into output dictionary
            hca_data = np.array(f[ds_name]["quality1"]["data"]).astype(float)
            hca_data[hca_data == -1] = np.nan
            shape = hca_data.shape
            hca_meta['data'][: shape[0], : shape[1]] = hca_data
            break


    return hca_meta


def add_pyodim_sweep_metadata(sweep_ds, variable_name, metadata_dict, skip_keys=['data']):
    """
    For each key in metadata_dict, a new attribute is created in sweep_ds with the key value 

    Parameters
    ----------
    sweep_ds : xarray data
        sweep xarray dataset
    variable_name: string
        name of variable in sweep_ds to update
    metadata_dict: dict
        dictionary containing keys and values to add into sweep_ds
    skip_keys: list of strings
        names of keys to skip in metadata_dict

    Returns
    -------
    sweep_ds : xarray data
        sweep xarray dataset

    """

    for key_name in metadata_dict.keys():
        if key_name in skip_keys:
            continue
        else:
            sweep_ds[variable_name].assign_attrs(key_name=metadata_dict[key_name])
    return sweep_ds

def add_pyart_metadata(radar, variable_name, metadata_dict, skip_keys=['data']):
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
    skip_keys: list of strings
        names of keys to skip in metadata_dict

    Returns
    -------
    radar : class
        pyart radar object

    """

    for key_name in metadata_dict.keys():
        if key_name in skip_keys:
            continue
        else:
            radar.fields[variable_name][key_name]=metadata_dict[key_name]
    return radar

def safe_log(x, eps=1e-10):
    result = np.where(x > eps, x, -10)
    np.log(result, out=result, where=result > 0)
    return result

def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError(
            "Only bool or integer image types are supported. " f"Got {ar.dtype}."
        )

def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """Remove objects smaller than the specified size.

    Copyied from https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/morphology/misc.py#L64-L160

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    See Also
    --------
    skimage.morphology.remove_objects_by_distance

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True

    """
    # Raising type error if not int or bool

    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    if len(component_sizes) == 2 and out.dtype != bool:
        warn(
            "Only one label was provided to `remove_small_objects`. "
            "Did you mean to use a boolean array?"
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def antenna_to_cartesian(ranges, azimuths, elevations):
    """
    Return Cartesian coordinates from antenna coordinates.
    Parameters
    ----------
    ranges : array
        Distances to the center of the radar gates (bins) in kilometers.
    azimuths : array
        Azimuth angle of the radar in degrees.
    elevations : array
        Elevation angle of the radar in degrees.
    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """        
    theta_e = elevations * np.pi / 180.0  # elevation angle in radians.
    theta_a = azimuths * np.pi / 180.0  # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0  # effective radius of earth in meters.
    r = ranges * 1000.0  # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z


def sounding_interp(snd_temp, snd_height, target_temp):
    """
    Provides an linear interpolated height for a target temperature using a sounding vertical profile.
    Looks for first instance of temperature below target_temp from surface upward.
    Parameters:
    ===========
    snd_temp: ndarray
        temperature data (degrees C)
    snd_height: ndarray
        relative height data (m)
    target_temp: float
        target temperature to find height at (m)
    Returns:
    ========
    intp_h: float
        interpolated height of target_temp (m)
    """

    intp_h = np.nan

    # check if target_temp is warmer than lowest level in sounding
    if target_temp > snd_temp[0]:
        print("warning, target temp level below sounding, returning ground level (0m)")
        return 0.0

    # find index above and below freezing level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]

    # index below
    below_ind = above_ind - 1
    # apply linear interplation to points above and below target_temp
    set_interp = interp1d(
        snd_temp[below_ind : above_ind + 1],
        snd_height[below_ind : above_ind + 1],
        kind="linear",
    )
    # apply interpolant
    intp_h = set_interp(target_temp)

    return intp_h


def wbt(temp, rh):
    """
    Calculate wet bulb temperature from temperature and relative humidity.

    Parameters
    ----------
    temp : ndarray
        Temperature data (degrees C).
    rh : ndarray
        Relative humidity data (%).

    Returns
    -------
    wb_temp : ndarray
        Wet bulb temperature (degrees C).

    """
    wb_temp = (
        temp * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(temp + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return wb_temp


def smooth_ppi_rays(ppi_data, n):
    """
    Apply a smoothing average filter of size n over ppi_data
    (rays are columns).

    Parameters
    ----------
    ppi_data : ndarray
        PPI data.
    n : float
        Smoothing kernel size (must be odd).

    Returns
    -------
    out : ndarray
        Ray smoothed ppi.

    """
    # calculate offset from edges
    offset = int((n - 1) / 2)
    # init ppi cumulative sum with zero values in first row
    zero_mat = np.zeros((ppi_data.shape[0], 1))
    ppi_cs = np.hstack((zero_mat, ppi_data))
    # calculate cumulative sum
    ppi_cs = np.nancumsum(ppi_cs, axis=1)
    # calculate simple moving average
    ppi_sma = (ppi_cs[:, n:] - ppi_cs[:, :-n]) / float(n)
    # stack data in output with zeros
    out = np.hstack((ppi_data[:, :offset], ppi_sma, ppi_data[:, -offset:]))

    return out