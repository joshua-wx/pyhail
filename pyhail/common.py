"""
common sub-module of pyhail

Contains commonly used functions.

Joshua Soderholm - 15 June 2018
"""

import numpy as np
from scipy.interpolate import interp1d
import wradlib as wrl


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
    zero_mat = np.ma.zeros((ppi_data.shape[0], 1))
    ppi_cs = np.ma.hstack((zero_mat, ppi_data))
    # calculate cumulative sum
    ppi_cs = ppi_cs.cumsum(axis=1)
    # calculate simple moving average
    ppi_sma = (ppi_cs[:, n:] - ppi_cs[:, :-n]) / float(n)
    # stack data in output with zeros
    out = np.ma.hstack((ppi_data[:, :offset], ppi_sma, ppi_data[:, -offset:]))

    return out


def beam_blocking(radar, srtm_ffn):
    """
    Apply the wradlib beam blocking library for the target volume.

    Parameters
    ----------
    radar : Radar
        Py-ART radar object.
    srtm_ffn : string
        Full path to SRTM geotiff file.

    Returns
    -------
    ccb_dict : dict
        Dictionary containing the cumulative beam blocking (CBB) for
        every pixel in the radar object.

    """
    # site parameters
    radar_lat = radar.latitude["data"][0]
    radar_lon = radar.longitude["data"][0]
    radar_alt = radar.altitude["data"][0]
    sitecoords = (radar_lon, radar_lat, radar_alt)
    nsweeps = radar.nsweeps
    nrays = int(radar.nrays / nsweeps)
    nbins = int(radar.ngates)
    el_list = radar.fixed_angle["data"]
    range_res = radar.range["meters_between_gates"]
    try:
        bw = radar.instrument_parameters["radar_beam_width_h"]["data"][0]
    except KeyError:
        print("beamwidth info missing form volume, using default of 1deg")
        bw = 1
    # grid arrays
    r = np.arange(nbins) * range_res
    beamradius = wrl.util.half_power_radius(r, bw)

    # read geotiff
    ds = wrl.io.open_raster(srtm_ffn)
    rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(
        ds, nodata=-32768
    )

    # build coordiantes
    coord = None
    for el in el_list:
        # calculat spherical coordiantes for a sweep
        sweep_coord = wrl.georef.sweep_centroids(nrays, range_res, nbins, el)
        if coord is None:
            coord = sweep_coord
        else:
            # append spherical coordiantes for a sweep
            coord = np.append(coord, sweep_coord, axis=0)
    # calculate geographical coordinates of spherical space
    coords = wrl.georef.spherical_to_proj(
        coord[..., 0], coord[..., 1], coord[..., 2], sitecoords
    )

    lon = coords[..., 0]
    lat = coords[..., 1]
    alt = coords[..., 2]
    # polar coodinates for mapping terrain (no altitude)
    polcoords = coords[..., :2]

    # Clip the region inside our bounding box
    rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
    ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
    rastercoords_clip = rastercoords.copy()[ind[1] : ind[3], ind[0] : ind[2], ...]
    rastervalues_clip = rastervalues.copy()[ind[1] : ind[3], ind[0] : ind[2]]

    # Map rastervalues to polar grid points
    polarvalues = wrl.ipol.cart_to_irregular_interp(
        rastercoords_clip, rastervalues_clip, polcoords, method="nearest"
    )

    # calculate beam blocking for each bin
    PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
    PBB = np.ma.masked_invalid(PBB)

    # calculate beam blocking along each ray
    CBB = wrl.qual.cum_beam_block_frac(PBB)

    # generate meta
    the_comments = "wradlib cumulative beam blocking"
    cbb_dict = {
        "data": CBB,
        "units": "%",
        "long_name": "cumulative beam blocking percentage",
        "standard_name": "CBB",
        "comments": the_comments,
    }

    return cbb_dict
