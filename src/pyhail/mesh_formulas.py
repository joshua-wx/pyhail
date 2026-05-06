import numpy as np


def mesh_witt1998(shi):
    """
    Maximum Expected Size of Hail (MESH, mm) using the Witt et al. (1998)
    75th-percentile power-law calibration.

      MESH = 2.54 * SHI^0.5

    Fitted to 147 hail reports. Grows relatively quickly with SHI and tends
    to overestimate at high SHI values compared to more recent calibrations.

    Reference
    ---------
    Witt et al. (1998)  doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2

    Parameters
    ----------
    shi : array_like
        Severe Hail Index (J m⁻¹ s⁻¹).  Non-positive values return 0.

    Returns
    -------
    mesh : ndarray
        MESH in mm, same shape as `shi`.
    """
    shi = np.asarray(shi, dtype=float)
    return np.where(shi > 0.0, 2.54 * shi ** 0.5, 0.0)


def mesh_mh2019_75(shi):
    """
    Maximum Expected Size of Hail (MESH, mm) using the Murillo & Homeyer
    (2021) 75th-percentile power-law recalibration.

      MESH = 15.096 * SHI^0.206

    Fitted to 5897 hail reports. The shallower exponent (0.206 vs 0.5)
    causes MESH to grow more slowly at high SHI, better bounding observed
    hail sizes in the upper tail of the distribution.

    Reference
    ---------
    Murillo & Homeyer (2021)  doi:10.1175/JAMC-D-20-0271.1

    Parameters
    ----------
    shi : array_like
        Severe Hail Index (J m⁻¹ s⁻¹).  Non-positive values return 0.

    Returns
    -------
    mesh : ndarray
        MESH in mm, same shape as `shi`.
    """
    shi = np.asarray(shi, dtype=float)
    return np.where(shi > 0.0, 15.096 * shi ** 0.206, 0.0)


def mesh_mh2019_95(shi):
    """
    Maximum Expected Size of Hail (MESH, mm) using the Murillo & Homeyer
    (2021) 95th-percentile power-law recalibration.

      MESH = 22.157 * SHI^0.212

    Fitted to 5897 hail reports. Returns larger values than the 75th-percentile
    fit and is more appropriate when a conservative (upper-bound) hail size
    estimate is required.

    Reference
    ---------
    Murillo & Homeyer (2021)  doi:10.1175/JAMC-D-20-0271.1

    Parameters
    ----------
    shi : array_like
        Severe Hail Index (J m⁻¹ s⁻¹).  Non-positive values return 0.

    Returns
    -------
    mesh : ndarray
        MESH in mm, same shape as `shi`.
    """
    shi = np.asarray(shi, dtype=float)
    return np.where(shi > 0.0, 22.157 * shi ** 0.212, 0.0)


def mesh_smooth_blend(shi, transition_width=200.0):
    """
    Compute a smoothly blended Maximum Expected Size of Hail (MESH, mm).

    Two established power-law calibrations are combined using a logistic
    weight function, avoiding the discontinuous derivative that arises from
    a hard piecewise switch:

      MESH = (1 - w) * 2.54 * SHI^0.5  +  w * 15.096 * SHI^0.206

    where the weight w is the logistic function

      w(SHI) = 1 / (1 + exp(-k * (SHI - SHI*)))

    and SHI* ≈ 429.3 J m⁻¹ s⁻¹ is the analytical intercept of the two
    power laws (the SHI value at which both calibrations return the same
    MESH, ≈ 52.6 mm).  At this pivot w = 0.5 and the blended value equals
    both calibrations exactly.

    The steepness k is chosen so that w moves from 0.1 to 0.9 across the
    interval [SHI* - transition_width/2, SHI* + transition_width/2], giving
    intuitive control over how abruptly the blend transitions:

      k = 2 * ln(9) / transition_width

    Behaviour at the extremes:
      - SHI << SHI*  (w → 0): output converges to Witt (1998), which grows
        faster (SHI^0.5) and is appropriate for lower-intensity hail.
      - SHI >> SHI*  (w → 1): output converges to Murillo & Homeyer (2019)
        75th percentile (SHI^0.206), which flattens at large SHI and better
        represents the observational upper bound on hail size.
      - SHI ≤ 0: output is 0 (no hail signal).

    Calibration references
    ----------------------
    Witt et al. (1998)  doi:10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2
    Murillo & Homeyer (2021)  doi:10.1175/JAMC-D-20-0271.1

    Parameters
    ----------
    shi : array_like
        Severe Hail Index (J m⁻¹ s⁻¹).  Non-positive values return 0.
    transition_width : float, optional
        SHI range (J m⁻¹ s⁻¹) over which the logistic weight moves from
        0.1 to 0.9.  Smaller values approach a hard piecewise switch;
        larger values produce a broader, gentler handoff.  Default 200.

    Returns
    -------
    mesh : ndarray
        MESH in mm, same shape as `shi`.
    """
    WITT_A, WITT_B = 2.54, 0.5            # Witt (1998)
    MH19_A, MH19_B = 15.096, 0.206        # Murillo & Homeyer 75th percentile (2019)
    SHI_INTERCEPT = (MH19_A / WITT_A) ** (1.0 / (WITT_B - MH19_B))

    shi = np.asarray(shi, dtype=float)

    # Logistic steepness: w goes from 0.1 to 0.9 across `transition_width`.
    k = 2.0 * np.log(9.0) / float(transition_width)

    # Clip to avoid overflow at extreme SHI on large grids.
    z = np.clip(-k * (shi - SHI_INTERCEPT), -50.0, 50.0) #clip to reduce numerical noise in exp(): 1/(1+exp(-50))~0 and 1/(1+exp(50))~1
    w = 1.0 / (1.0 + np.exp(z))

    safe_shi = np.maximum(shi, 0.0)
    mesh_witt = WITT_A * np.power(safe_shi, WITT_B)
    mesh_mh19 = MH19_A * np.power(safe_shi, MH19_B)

    mesh = (1.0 - w) * mesh_witt + w * mesh_mh19
    return np.where(shi > 0.0, mesh, 0.0)