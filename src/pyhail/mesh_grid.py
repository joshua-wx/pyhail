"""
MESH implementation for calculating on gridded data.
This algorthim was originally developed by 
Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 
and modified by Murillo and Homeyer 2019 doi:10.1175/JAMC-D-18-0247.1

Joshua Soderholm - 15 June 2020
"""

import warnings
import numpy as np
from scipy import ndimage as ndi
from pyhail import common


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
    except ValueError as exc:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`." 
        ) from exc

    if len(component_sizes) == 2 and out.dtype != bool:
        warnings.warn(
            "Only one label was provided to `remove_small_objects`. "
            "Did you mean to use a boolean array?"
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def filter_small_objects(field, threshold=0, size=9):
    """run a filter to remove small objects

     Parameters:
    ===========
    field: ndarray (n,m)
        2D array to filter
    threshold: float
        intensity threshold to run small objects filter
    size: int
        area size (number of pixels) threshold to remove small objects
    Returns:
    ========
    field: narray (n,m)

    """
    # apply intensity threshold to produce a mask
    masked_data = field > threshold
    # remove small objects
    filtered_masked_data = remove_small_objects(masked_data, min_size=size)
    # apply filter to field
    field[filtered_masked_data == 0] = threshold

    return field


def _get_latlon(grid, dbz_fname):
    """
    Generates latitude and longitude arrays.

    Parameters
    ----------
    grid : Grid
        Py-ART grid object.
    dbz_fname : str
        Reflectivity field name.

    Returns
    -------
    longitude : ndarray
        Array of coordinates for all points.
    latitude : ndarray
        Array of coordinates for all points.

        From cpol_processing: https://github.com/vlouf/cpol_processing

    """
    # Declare array, filled 0 in order to not have a masked array.
    lontot = np.zeros_like(grid.fields[dbz_fname]["data"].filled(0))
    lattot = np.zeros_like(grid.fields[dbz_fname]["data"].filled(0))

    for lvl in range(grid.nz):
        lontot[lvl, :, :], lattot[lvl, :, :] = grid.get_point_longitude_latitude(lvl)

    longitude = {
        "long_name": "Longitude",
        "standard_name": "Longitude",
        "units": "degrees_east",
        "data": lontot,
    }
    latitude = {
        "long_name": "Latitude",
        "standard_name": "Latitude",
        "units": "degrees_north",
        "data": lattot,
    }

    return longitude, latitude


def main(
    grid,
    dbz_fname,
    levels,
    radar_band="C",
    mesh_method="mh2019_75",
    mesh_fname=None,
    posh_fname=None,
    ke_fname=None,
    shi_fname=None,
    speckle_filter=True,
    correct_cband_refl=True,
):
    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019
    Expanded to calculate MESH on gridded data (adapted from wdss-ii)

    Parameters
    ----------
    grid : object
        Py-ART grid object.
    dbz_fname : str
        Name of reflectivity field in the radar object.
    levels : list of length 2
        height above sea level (m) of the freezing level and -20C level (in any order)
    radar_band: str
        radar frequency band (either C or S)
    mesh_fname, posh_fname, ke_fname, shi_fname : str
        String to name new hail field that will be added to the grid object.
        Default is 'mesh', 'posh', 'hail_ke', 'shi'.
    mesh_method : string
        either witt1998, mh2019_75 or mh2019_95. see more information below
    speckle_filter: logical
        flag for running the speckle filter
    correct_cband_refl: logical
        flag to trigger C band hail reflectivity correction (if radar_band is C)
    Returns
    -------
    output_fields : dictionary
        Dictionary of output fields (KE, SHI, MESH, POSH)
    """

    # require C or S band
    if radar_band not in ["C", "S"]:
        raise ValueError("radar_band must be a string of value C or S")
    # require levels
    if levels is None:
        raise ValueError("Missing levels data for freezing level and -20C level")

    # Rain/Hail dBZ boundaries
    z_l = 40
    z_u = 50

    # default field names
    if mesh_fname is None:
        mesh_fname = "mesh_" + mesh_method
    if posh_fname is None:
        posh_fname = "posh"
    if ke_fname is None:
        ke_fname = "hail_ke"
    if shi_fname is None:
        shi_fname = "shi"

    # This dummy proofs the user input. The melting level will always
    # be lower in elevation than the negative 20 deg C isotherm
    meltlayer = min(levels)
    neg20layer = max(levels)

    # Latitude Longitude field for each point.
    longitude, latitude = _get_latlon(grid, dbz_fname)
    grid.add_field("longitude", longitude, replace_existing=True)
    grid.add_field("latitude", latitude, replace_existing=True)

    # extract grids
    dbz_grid = grid.fields[dbz_fname]["data"]
    grid_sz = np.shape(dbz_grid)
    alt_vec = (
        grid.z["data"] + grid.radar_altitude["data"][0]
    )  # units m at ASL required for NWP data
    alt_grid = np.tile(alt_vec, (grid_sz[1], grid_sz[2], 1))
    alt_grid = np.swapaxes(alt_grid, 0, 2)  # m

    # apply C band correction
    hail_refl_correction_description = ""
    if radar_band == "C" and correct_cband_refl:
        dbz_grid = dbz_grid * 1.113 - 3.929
        hail_refl_correction_description = "C band hail reflectivity correction applied from Brook et al. 2023 https://arxiv.org/abs/2306.12016"

    # calc reflectivity weighting function
    dbz_weights = (dbz_grid - z_l) / (z_u - z_l)
    dbz_weights[dbz_grid <= z_l] = 0
    dbz_weights[dbz_grid >= z_u] = 1
    dbz_weights[dbz_weights < 0] = 0
    dbz_weights[dbz_weights > 1] = 1

    # limit on dbz_grid
    dbz_grid[dbz_grid > 100] = 100
    dbz_grid[dbz_grid < -100] = -100

    # calc hail kenetic energy
    hke = (5 * 10**-6) * 10 ** (0.084 * dbz_grid) * dbz_weights

    # calc temperature based weighting function
    w_t = (alt_grid - meltlayer) / (neg20layer - meltlayer)
    w_t[alt_grid <= meltlayer] = 0
    w_t[alt_grid >= neg20layer] = 1
    w_t[w_t < 0] = 0
    w_t[w_t > 1] = 1

    # calc severe hail index
    d_z = alt_vec[1] - alt_vec[0]
    shi = 0.1 * np.sum(w_t * hke, axis=0) * d_z

    # calc maximum estimated severe hail (mm)
    if (
        mesh_method == "witt1998"
    ):  # 75th percentil fit from witt et al. 1998 (fitted to 147 reports)
        mesh = 2.54 * shi**0.5
        mesh_description = "Maximum Estimated Size of Hail retreival developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        mesh_comment = "75th percentile fit using 147 hail reports; only valid in the first level of the 3D grid."

    elif (
        mesh_method == "mh2019_75"
    ):  # 75th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = 15.096 * shi**0.206
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = "75th percentile fit using 5897 hail reports; only valid in the first level of the 3D grid."

    elif (
        mesh_method == "mh2019_95"
    ):  # 95th percentile fit from Muillo and Homeyer 2019 (fitted to 5897 reports)
        mesh = 22.157 * shi**0.212
        mesh_description = "Maximum Estimated Size of Hail retreival originally developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 and recalibrated by Murillo and Homeyer (2021) doi:10.1175/JAMC-D-20-0271.1 "
        mesh_comment = "95th percentile fit using 5897 hail reports; only valid in the first level of the 3D grid."
    else:
        raise ValueError(
            "unknown MESH method selects, please use witt1998, mh2019_75 or mh2019_95"
        )

    # calc warning threshold (J/m/s) NOTE: freezing height must be in km
    warning_threshold = 57.5 * (meltlayer / 1000) - 121

    # calc probability of severe hail (POSH) (%)
    posh = 29 * common.safe_log(shi / warning_threshold) + 50
    posh = np.real(posh)
    posh[posh < 0] = 0
    posh[posh > 100] = 100

    # apply speckle filter
    if speckle_filter:
        shi = filter_small_objects(shi)

    output_fields = {}

    # add grids to grid object
    ke_dict = {
        "data": hke,
        "units": "Jm-2s-1",
        "long_name": "Hail Kinetic Energy",
        "description": "Hail Kinetic Energy developed by Witt et al. 1998 doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
    }
    output_fields[ke_fname] = ke_dict

    shi_grid = np.zeros_like(hke)
    shi_grid[0, :, :] = shi
    shi_dict = {
        "data": shi_grid,
        "units": "Jm-1s-1",
        "long_name": "Severe Hail Index",
        "description": "Severe Hail Index developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
        "comments": "only valid in the first level of the 3D grid",
    }
    output_fields[shi_fname] = shi_dict

    mesh_grid = np.zeros_like(hke)
    mesh_grid[0, :, :] = mesh
    mesh_dict = {
        "data": mesh_grid,
        "units": "mm",
        "long_name": "Maximum Expected Size of Hail using " + mesh_method,
        "description": mesh_description + hail_refl_correction_description,
        "comments": mesh_comment,
    }
    output_fields[mesh_fname] = mesh_dict

    posh_grid = np.zeros_like(hke)
    posh_grid[0, :, :] = posh
    posh_dict = {
        "data": posh_grid,
        "units": "%",
        "long_name": "Probability of Severe Hail",
        "description": "Probability of Severe Hail developed by Witt et al. (1998) doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2 "
        + hail_refl_correction_description,
        "comments": "only valid in the first level of the 3D grid",
    }
    output_fields[posh_fname] = posh_dict

    # return output_fields dictionary
    return output_fields
