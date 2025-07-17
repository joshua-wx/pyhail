"""
Optimised approach to find the vertical column maximum in spherical coordinates
"""

import numpy as np

import numba
from numba import jit

def pyodim(
    datasets,
    field_fname,
    elevation_fname="elevation",
    azimuth_fname="azimuth",
    range_fname="range",
    min_range=10,
    max_range=150,
    column_altitude_maximum=2500,
    column_shift_maximum=2500
    ):
    """
    Pyodim Wrapper for PPI MESH

    Parameters:
    ===========
    datasets: list of dicts
        pyodim dataset
    field_fname: string
        name of field
    elevation_fname: string
        name of radar elevation field
    range_fname: string
        name of radar bin range field
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    column_altitude_maximum: float
        maximum altitude (m above radar radar) to use for column search
    column_shift_maximum: float
        maximum horizontal distance a column can shift by
    Returns:
    ========
    field: 2D array
        output with dimensions of range and azimuth

    """

    # build datasets
    field_dataset = []
    elevation_dataset = []
    azimuth_dataset = []
    range_dataset = []
    n_ppi = len(datasets)
    for sweep_idx in range(n_ppi):
        field_dataset.append(datasets[sweep_idx][field_fname].values)
        elevation_dataset.append(datasets[sweep_idx][elevation_fname].data[0])
        azimuth_dataset.append(datasets[sweep_idx][azimuth_fname].values)
        range_dataset.append(datasets[sweep_idx][range_fname].values)

    # run retrieval
    output_field = main(
        field_dataset,
        elevation_dataset,
        azimuth_dataset,
        range_dataset,
        min_range=min_range,
        max_range=max_range,
        column_altitude_maximum=column_altitude_maximum,
        column_shift_maximum=column_shift_maximum,
    )

    return output_field

@jit(nopython=True)
def _antenna_to_arc(ranges, elevation):
    """
    Return the great circle distance directly below the radar beam and the
    altitude of the radar beam.
    ----------
    ranges : 1d array
        Distances to the center of the radar gates (bins) in meters.
    elevation : float
        Elevation angle of the radar in degrees.
    Returns
    -------
    s: 1d array
        Distance along the great circle for each radar bin (units: meters)
    z: 1d array
        Altitude above radar level for each radar bin (units: meters)
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = elevation * np.pi / 180.0
    R = 6371.0 * 1000.0 * 4.0 / 3.0
    
    # Vectorized operations compiled to efficient machine code
    z = np.sqrt(ranges**2 + R**2 + 2.0 * ranges * R * np.sin(theta_e)) - R
    s = R * np.arcsin(ranges * np.cos(theta_e) / (R + z))
    return s, z


@jit(nopython=True)
def find_column_max(field_dataset, s_lookup_dataset,
                    min_range_m, max_range_m,
                    azimuth_dataset, s_dataset):
    """
    Optimized function to find column maximums with better memory access patterns and
    eliminated redundant calculations.
    """
    n_rays = len(azimuth_dataset[0])
    n_bins = len(s_dataset[0])
    
    # Pre-allocate output arrays
    column_max_field = np.zeros((n_rays, n_bins))
    column_max_mask = np.zeros((n_rays, n_bins), dtype=numba.boolean)
    # Pre-compute range mask (vectorized)
    s0 = s_dataset[0]
    range_mask = (s0 < min_range_m) | (s0 > max_range_m)
    
    # Pre-identify valid columns to avoid repeated empty checks
    valid_columns = []
    for rg_idx in range(n_bins):
        if (not range_mask[rg_idx]
            and s_lookup_dataset[rg_idx].size > 0):
            valid_columns.append(rg_idx)
    
    # Process only valid columns - reduces iterations significantly
    for az_idx in range(n_rays):
        # Apply range mask for entire row at once
        column_max_mask[az_idx, range_mask] = True
        
        # Process only valid columns
        for rg_idx in valid_columns:
            column_field = np.zeros(len(s_lookup_dataset[rg_idx]))
            lookup_indices = s_lookup_dataset[rg_idx]
            
            # Vectorized column integration where possible
            for lookup_idx in range(len(lookup_indices)):
                sweep_idx = lookup_idx
                if sweep_idx < len(field_dataset):
                    rng_idx = lookup_indices[lookup_idx]
                    if rng_idx < field_dataset[sweep_idx].shape[1]:
                        column_field[lookup_idx] = field_dataset[sweep_idx][az_idx, rng_idx]
            
            column_max_field[az_idx, rg_idx] = np.nanmax(column_field)
    
    return column_max_field, column_max_mask

def main(
    field,
    elevation,
    azimuth,
    rangebin,
    min_range=10,
    max_range=150,
    column_altitude_maximum=2500,
    column_shift_maximum=2500
):
    """
    Adapted from Witt et al. 1998 and Murillo and Homeyer 2019

    Parameters
    ----------
    field : list of 2D ndarrays
        list where each element is the sweep field data in an array with dimensions (azimuth, range)
    elevation: 1d ndarray of floats
        ndarray where each element is the fixed elevation angle of the sweep
    azimuth: list of 1D ndarrays
        list where each element is the sweep azimuth angles
    rangebin: list of 1D ndarrays
        list where each element is the sweep range distances
    min_range: int
        minimum surface range for MESH retrieval (m)
    max_range: int
        maximum surface range for MESH retrieval (m)
    column_altitude_maximum: float
        maximum altitude (m above radar radar) to use for column search
    column_shift_maximum: float
        maximum shift from column center to use a pixel
    Returns
    -------
    output_field : dictionary

    """

    # sort by fixed angle
    sort_idx = list(np.argsort(elevation))
    field_dataset = [field[i] for i in sort_idx]
    elevation_dataset = [elevation[i] for i in sort_idx]
    azimuth_dataset = [azimuth[i] for i in sort_idx]
    range_dataset = [rangebin[i] for i in sort_idx]

    # Initialize sweep coords
    sweep0_nbins = len(range_dataset[0])
    n_ppi = len(elevation_dataset)
    z_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, altitude above ground level (m) of each range bin
    s_dataset = (
        []
    )  # list (dim: elevation) of 1d array (dim: range) for each sweep, great circle arc distance (m) of each radar bin
    for i in range(n_ppi):
        # calculate cartesian coordinates
        s, z = _antenna_to_arc(range_dataset[i], elevation_dataset[i])
        s_dataset.append(s)
        z_dataset.append(z)
    # generate arc range lookup (note these have different dimensions to the dimension variables)
    s_lookup_dataset = (
        []
    )  # list (dim: range) where each element represents an the range bin index to use from each sweep above sweep0. ASSUMES ORDERS SWEEP ELEVATION
    for rg_idx in range(sweep0_nbins):
        s_lookup = [0]
        for sweep_idx in range(1, n_ppi, 1):
            dist_array = np.abs(s_dataset[0][rg_idx] - s_dataset[sweep_idx])
            closest_rng_idx = np.argmin(dist_array)
            #topped out above column max, break loop
            if z_dataset[sweep_idx][closest_rng_idx] > column_altitude_maximum:
                continue
            # skip sweeps where the horizontal shift is greater than column_shift_maximum (removes birdbaths and when base scan max range is greater than all other scans)
            elif dist_array[closest_rng_idx] > column_shift_maximum:
                continue
            else:
                s_lookup.append(closest_rng_idx)
            # else:
            #     print('skipping', 'distance check', dist_array[closest_rng_idx], 'range idx', rg_idx, 'sweep idx', sweep_idx)
        # check if at least one valid values in the column exists
        if len(s_lookup) >= 1:
            s_lookup_dataset.append(np.array(s_lookup))
        else:
            s_lookup_dataset.append(np.empty(0, dtype=np.int64))

    # Optimized calculation
    min_range_m = min_range * 1000
    max_range_m = max_range * 1000
    column_max, column_max_mask = find_column_max(
        field_dataset, s_lookup_dataset,
        min_range_m, max_range_m,
        azimuth_dataset, s_dataset)
    column_max[column_max_mask] = np.nan

    # return output_fields dictionary
    return column_max
