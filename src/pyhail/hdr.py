"""
Hail Differential Reflectivity (HDR) Implementation

This module implements the Hail Differential Reflectivity algorithm developed by
Aydin and Zhao (1990) with hail size estimation based on Depue et al. (2007).

The HDR algorithm uses the relationship between horizontal reflectivity (ZH) and
differential reflectivity (ZDR) to detect and estimate hail characteristics. It
exploits the fact that hailstones have different polarimetric signatures compared
to raindrops due to their shape, tumbling motion, and dielectric properties.

References
----------
Aydin, K., & Zhao, Y. (1990). A computational study of polarimetric radar 
observables in hail. IEEE Transactions on Geoscience and Remote Sensing, 
28(3), 412-422. doi:10.1109/TGRS.1990.572906

Depue, T. K., Kennedy, P. C., & Rutledge, S. A. (2007). Performance of the 
hail differential reflectivity (HDR) polarimetric radar hail indicator. 
Journal of Applied Meteorology and Climatology, 46(8), 1290-1301. 
doi:10.1175/JAM2529.1

Author: Joshua Soderholm
Created: 15 June 2018
Modified: [Current Date] - Enhanced with improved error handling, documentation, and performance
"""

import copy
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from numba import jit, vectorize
from numpy.typing import NDArray

from pyhail import common

# Configure module logger
logger = logging.getLogger(__name__)

# HDR Algorithm Constants
HDR_ZDR_LOWER_THRESHOLD = 0.0  # dB - Lower ZDR threshold
HDR_ZDR_UPPER_THRESHOLD = 1.74  # dB - Upper ZDR threshold  
HDR_ZDR_FUNCTION_OFFSET = 27.0  # dB - Base offset for ZDR function
HDR_ZDR_FUNCTION_SLOPE = 19.0  # dB/dB - Slope coefficient for ZDR function
HDR_SIZE_POLY_A = 0.0284  # mm/dB² - Quadratic coefficient for size estimation
HDR_SIZE_POLY_B = -0.366  # mm/dB - Linear coefficient for size estimation
HDR_SIZE_POLY_C = 11.69  # mm - Constant term for size estimation


class HDRError(Exception):
    """Base exception for HDR algorithm errors."""
    pass


class HDRValidationError(HDRError):
    """Raised when HDR input validation fails."""
    pass


class HDRProcessingError(HDRError):
    """Raised when HDR processing encounters an error."""
    pass


def validate_radar_inputs(
    reflectivity: NDArray[np.floating],
    differential_reflectivity: NDArray[np.floating]
) -> None:
    """
    Validate radar input arrays for HDR processing.
    
    Parameters
    ----------
    reflectivity : NDArray[np.floating]
        Horizontal reflectivity data in dBZ
    differential_reflectivity : NDArray[np.floating]
        Differential reflectivity data in dB
        
    Raises
    ------
    HDRValidationError
        If input validation fails
    """
    if not isinstance(reflectivity, np.ndarray):
        raise HDRValidationError(
            f"Reflectivity must be numpy array, got {type(reflectivity)}"
        )
    
    if not isinstance(differential_reflectivity, np.ndarray):
        raise HDRValidationError(
            f"Differential reflectivity must be numpy array, got {type(differential_reflectivity)}"
        )
    
    if reflectivity.shape != differential_reflectivity.shape:
        raise HDRValidationError(
            f"Shape mismatch: reflectivity {reflectivity.shape} != "
            f"differential_reflectivity {differential_reflectivity.shape}"
        )
    
    if reflectivity.size == 0:
        raise HDRValidationError("Input arrays cannot be empty")
    
    if not np.issubdtype(reflectivity.dtype, np.floating):
        warnings.warn(
            f"Reflectivity dtype {reflectivity.dtype} is not floating point. "
            "Converting to float64.",
            UserWarning
        )
    
    if not np.issubdtype(differential_reflectivity.dtype, np.floating):
        warnings.warn(
            f"Differential reflectivity dtype {differential_reflectivity.dtype} "
            "is not floating point. Converting to float64.",
            UserWarning
        )
    
    # Check for reasonable value ranges
    # Note: Quality control and outlier detection should be handled
    # at a higher level before calling HDR processing functions


@jit(nopython=True, cache=True)
def _calculate_hdr_core(zh: float, zdr: float) -> float:
    """
    Core HDR calculation for valid input values.
    
    Parameters
    ----------
    zh : float
        Horizontal reflectivity in dBZ (must be finite)
    zdr : float
        Differential reflectivity in dB (must be finite)
        
    Returns
    -------
    float
        HDR value in dB
    """
    # Calculate ZDR function with threshold limits
    if zdr <= HDR_ZDR_LOWER_THRESHOLD:
        zdr_fun = HDR_ZDR_FUNCTION_OFFSET
    elif zdr > HDR_ZDR_UPPER_THRESHOLD:
        zdr_fun = 60.0
    else:
        zdr_fun = HDR_ZDR_FUNCTION_SLOPE * zdr + HDR_ZDR_FUNCTION_OFFSET
    
    return zh - zdr_fun


@jit(nopython=True, cache=True)
def _calculate_hdr_size_core(hdr: float) -> float:
    """
    Core HDR size estimation for valid input values.
    
    Parameters
    ----------
    hdr : float
        HDR value in dB (must be finite)
        
    Returns
    -------
    float
        Estimated hail size in mm
    """
    if hdr <= 0:
        return 0.0
    
    # Apply polynomial relationship from Depue et al. (2007)
    size = HDR_SIZE_POLY_A * (hdr * hdr) + HDR_SIZE_POLY_B * hdr + HDR_SIZE_POLY_C
    
    # Ensure non-negative size
    return max(size, 0.0)


def _calculate_hdr_array(
    zh_data: NDArray[np.floating], 
    zdr_data: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Calculate HDR for arrays while properly handling invalid values.
    
    Parameters
    ----------
    zh_data : NDArray[np.floating]
        Horizontal reflectivity array in dBZ
    zdr_data : NDArray[np.floating]
        Differential reflectivity array in dB
        
    Returns
    -------
    NDArray[np.floating]
        HDR array with NaN for invalid inputs
    """
    # Initialize output array
    hdr_data = np.full_like(zh_data, np.nan, dtype=np.float64)
    
    # Find valid data points (finite values in both arrays)
    valid_mask = np.isfinite(zh_data) & np.isfinite(zdr_data)
    
    if not np.any(valid_mask):
        return hdr_data
    
    # Extract valid values
    zh_valid = zh_data[valid_mask]
    zdr_valid = zdr_data[valid_mask]
    
    # Process valid values using compiled function
    for i in range(len(zh_valid)):
        hdr_data[valid_mask][i] = _calculate_hdr_core(zh_valid[i], zdr_valid[i])
    
    return hdr_data


def _calculate_hdr_size_array(hdr_data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate HDR size estimates for arrays while properly handling invalid values.
    
    Parameters
    ----------
    hdr_data : NDArray[np.floating]
        HDR array in dB
        
    Returns
    -------
    NDArray[np.floating]
        HDR size array with NaN for invalid inputs
    """
    # Initialize output array
    size_data = np.full_like(hdr_data, np.nan, dtype=np.float64)
    
    # Find valid data points
    valid_mask = np.isfinite(hdr_data)
    
    if not np.any(valid_mask):
        return size_data
    
    # Extract valid values
    hdr_valid = hdr_data[valid_mask]
    
    # Process valid values using compiled function
    for i in range(len(hdr_valid)):
        size_data[valid_mask][i] = _calculate_hdr_size_core(hdr_valid[i])
    
    return size_data


def calculate_hdr(
    reflectivity: NDArray[np.floating],
    differential_reflectivity: NDArray[np.floating],
    validate_inputs: bool = True
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate Hail Differential Reflectivity (HDR) and estimated hail size.
    
    This function implements the HDR algorithm which uses the relationship between
    horizontal reflectivity and differential reflectivity to detect hail. The
    algorithm assumes that hailstones have lower differential reflectivity compared
    to raindrops of equivalent reflectivity.
    
    Parameters
    ----------
    reflectivity : NDArray[np.floating]
        Horizontal reflectivity data in dBZ, typically from weather radar
    differential_reflectivity : NDArray[np.floating]
        Differential reflectivity data in dB, typically from dual-polarization radar
    validate_inputs : bool, optional, default=True
        Whether to validate input arrays for common issues
        
    Returns
    -------
    hdr_data : NDArray[np.floating]
        HDR values in dB. Higher values indicate greater likelihood of hail.
        Typical hail values are > 20 dB.
    hdr_size_data : NDArray[np.floating]
        Estimated hail size in mm based on polynomial relationship.
        Values of 0 indicate no hail detected.
        
    Raises
    ------
    HDRValidationError
        If input validation fails
    HDRProcessingError
        If processing encounters an error
        
    Examples
    --------
    Calculate HDR for synthetic radar data:
    
    >>> import numpy as np
    >>> zh = np.array([[45, 50, 55], [40, 35, 30]])  # dBZ
    >>> zdr = np.array([[0.5, 0.2, -0.1], [1.0, 2.0, 1.5]])  # dB
    >>> hdr, hdr_size = calculate_hdr(zh, zdr)
    >>> print(f"Max HDR: {np.nanmax(hdr):.1f} dB")
    >>> print(f"Max hail size: {np.nanmax(hdr_size):.1f} mm")
    
    Process real radar sweep data:
    
    >>> # Assuming radar_data contains sweep arrays
    >>> hdr_values, hail_sizes = calculate_hdr(
    ...     radar_data['ZH'], radar_data['ZDR']
    ... )
    >>> hail_pixels = hdr_values > 20  # Typical hail threshold
    >>> print(f"Hail detected in {np.sum(hail_pixels)} pixels")
    
    Notes
    -----
    The HDR algorithm works best for:
    - Large hail (> 1 cm diameter)
    - Well-calibrated ZDR measurements
    - Situations where hail is not heavily melting
    
    Algorithm limitations:
    - May produce false positives in areas of attenuation
    - Assumes spherical tumbling motion of hailstones
    - Size estimates are approximate and radar-specific calibration may be needed
    """
    try:
        if validate_inputs:
            validate_radar_inputs(reflectivity, differential_reflectivity)
        
        logger.debug(
            f"Processing HDR for array shape {reflectivity.shape}, "
            f"{np.sum(np.isfinite(reflectivity))} valid reflectivity points"
        )
        
        # Ensure arrays are floating point for processing
        zh_data = np.asarray(reflectivity, dtype=np.float64)
        zdr_data = np.asarray(differential_reflectivity, dtype=np.float64)
        
        # Calculate HDR using optimized array processing
        hdr_data = _calculate_hdr_array(zh_data, zdr_data)
        
        # Calculate hail size estimates
        hdr_size_data = _calculate_hdr_size_array(hdr_data)
        
        # Log processing statistics
        valid_hdr = np.isfinite(hdr_data)
        if np.any(valid_hdr):
            logger.debug(
                f"HDR processing complete. Valid pixels: {np.sum(valid_hdr)}, "
                f"HDR range: [{np.nanmin(hdr_data):.1f}, {np.nanmax(hdr_data):.1f}] dB, "
                f"Size range: [{np.nanmin(hdr_size_data):.1f}, {np.nanmax(hdr_size_data):.1f}] mm"
            )
        
        return hdr_data, hdr_size_data
        
    except Exception as e:
        raise HDRProcessingError(f"HDR calculation failed: {e}") from e


def main(
    reflectivity_sweep: NDArray[np.floating],
    differential_reflectivity_sweep: NDArray[np.floating]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate HDR and HDR size with metadata for pyhail integration.
    
    This is the main function for HDR processing that returns results in the
    format expected by other pyhail modules and radar data frameworks.
    
    Parameters
    ----------
    reflectivity_sweep : NDArray[np.floating], shape (n_rays, n_gates)
        Reflectivity data for a radar sweep in dBZ
    differential_reflectivity_sweep : NDArray[np.floating], shape (n_rays, n_gates)  
        Differential reflectivity data for a radar sweep in dB
        
    Returns
    -------
    hdr_meta : Dict[str, Any]
        Dictionary containing HDR data and metadata with keys:
        - 'data': NDArray containing HDR values in dB
        - 'units': str, data units
        - 'long_name': str, descriptive name
        - 'description': str, algorithm description with references
        - 'comments': str, additional comments
    hdr_size_meta : Dict[str, Any]
        Dictionary containing HDR size data and metadata with keys:
        - 'data': NDArray containing hail size estimates in mm
        - 'units': str, data units  
        - 'long_name': str, descriptive name
        - 'description': str, algorithm description with references
        - 'comments': str, size estimation details
        
    Raises
    ------
    HDRValidationError
        If input validation fails
    HDRProcessingError
        If processing encounters an error
        
    Examples
    --------
    >>> zh_sweep = np.random.rand(360, 500) * 70  # Simulated reflectivity
    >>> zdr_sweep = np.random.rand(360, 500) * 3 - 1  # Simulated ZDR
    >>> hdr_meta, size_meta = main(zh_sweep, zdr_sweep)
    >>> print(f"HDR data shape: {hdr_meta['data'].shape}")
    >>> print(f"Size units: {size_meta['units']}")
    """
    try:
        # Calculate HDR and size using core algorithm
        hdr_data, hdr_size_data = calculate_hdr(
            reflectivity_sweep, 
            differential_reflectivity_sweep
        )
        
        # Create metadata dictionaries
        hdr_meta = {
            "data": hdr_data,
            "units": "dB",
            "long_name": "Hail Differential Reflectivity",
            "description": (
                "Hail Differential Reflectivity developed by Aydin and Zhao (1990). "
                "HDR exploits the relationship between horizontal reflectivity and "
                "differential reflectivity to identify hail signatures. Higher values "
                "indicate greater likelihood of hail presence. "
                "doi:10.1109/TGRS.1990.572906"
            ),
            "comments": (
                f"HDR calculated using ZDR thresholds: lower={HDR_ZDR_LOWER_THRESHOLD} dB, "
                f"upper={HDR_ZDR_UPPER_THRESHOLD} dB. "
                f"Typical hail threshold: > 20 dB"
            ),
            "algorithm": "Aydin-Zhao 1990",
            "valid_min": -50.0,
            "valid_max": 100.0
        }

        hdr_size_meta = {
            "data": hdr_size_data,
            "units": "mm",
            "long_name": "HDR Hail Size Estimate",
            "description": (
                "Hail size estimation from Hail Differential Reflectivity using "
                "polynomial relationship developed by Depue et al. (2007). "
                "Size estimates are approximate and may require radar-specific "
                "calibration for quantitative applications. "
                "doi:10.1175/JAM2529.1"
            ),
            "comments": (
                "Polynomial coefficients from Depue et al. (2007): "
                f"Size = {HDR_SIZE_POLY_A} * HDR² + {HDR_SIZE_POLY_B} * HDR + {HDR_SIZE_POLY_C}. "
                "Size estimates set to 0 for HDR ≤ 0 dB. "
                "Function scaled from paper figure and may need local calibration."
            ),
            "algorithm": "Depue et al. 2007",
            "valid_min": 0.0,
            "valid_max": 100.0
        }

        return hdr_meta, hdr_size_meta
        
    except Exception as e:
        raise HDRProcessingError(f"HDR main processing failed: {e}") from e


def pyart(
    radar,
    reflectivity_fname: str,
    differential_reflectivity_fname: str,
    hdr_fname: str = "hdr",
    hdr_size_fname: str = "hdr_size"
):
    """
    PyART wrapper for HDR processing.
    
    This function processes HDR for all sweeps in a PyART radar object and adds
    the results as new fields. It handles the PyART data format and metadata
    conventions automatically.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        PyART radar object containing reflectivity and differential reflectivity data
    reflectivity_fname : str
        Name of the reflectivity field in the radar object (e.g., 'DBZ', 'DBZH')
    differential_reflectivity_fname : str  
        Name of the differential reflectivity field (e.g., 'ZDR', 'RHOHV')
    hdr_fname : str, optional, default='hdr'
        Name for the new HDR field to be added to the radar object
    hdr_size_fname : str, optional, default='hdr_size'
        Name for the new HDR size field to be added to the radar object
        
    Returns
    -------
    radar : pyart.core.Radar
        Updated radar object with HDR and HDR size fields added
        
    Raises
    ------
    HDRValidationError
        If required fields are missing from radar object
    HDRProcessingError
        If processing fails for any sweep
        
    Examples
    --------
    Process HDR for a PyART radar object:
    
    >>> import pyart
    >>> radar = pyart.io.read('radar_file.nc')
    >>> radar_with_hdr = pyart(radar, 'DBZ', 'ZDR')
    >>> print(f"HDR field added: {'hdr' in radar_with_hdr.fields}")
    >>> print(f"Available fields: {list(radar_with_hdr.fields.keys())}")
    
    Use custom field names:
    
    >>> radar_custom = pyart(
    ...     radar, 'reflectivity', 'differential_reflectivity',
    ...     hdr_fname='hail_detection', hdr_size_fname='hail_size_est'
    ... )
    
    Notes
    -----
    - Processing is applied to all sweeps in the radar volume
    - Masked arrays are automatically handled (masked values become NaN)
    - Original radar object is modified in-place and returned
    """
    try:
        # Validate that required fields exist
        if reflectivity_fname not in radar.fields:
            raise HDRValidationError(
                f"Reflectivity field '{reflectivity_fname}' not found in radar object. "
                f"Available fields: {list(radar.fields.keys())}"
            )
        
        if differential_reflectivity_fname not in radar.fields:
            raise HDRValidationError(
                f"Differential reflectivity field '{differential_reflectivity_fname}' "
                f"not found in radar object. Available fields: {list(radar.fields.keys())}"
            )
        
        logger.info(
            f"Processing HDR for PyART radar with {radar.nsweeps} sweeps, "
            f"{radar.nrays} rays, {radar.ngates} gates"
        )
        
        # Initialize radar fields with empty structure
        empty_radar_field = {
            "data": np.zeros((radar.nrays, radar.ngates)),
            "units": "",
            "long_name": "",
            "description": "",
            "comments": "",
        }
        radar.add_field(hdr_fname, copy.deepcopy(empty_radar_field))
        radar.add_field(hdr_size_fname, copy.deepcopy(empty_radar_field))

        # Process each sweep
        for sweep_idx in range(radar.nsweeps):
            logger.debug(f"Processing sweep {sweep_idx + 1}/{radar.nsweeps}")
            
            # Extract data for current sweep, handling masked arrays
            zh_sweep = radar.get_field(sweep_idx, reflectivity_fname, copy=True)
            zdr_sweep = radar.get_field(sweep_idx, differential_reflectivity_fname, copy=True)
            
            # Convert masked arrays to regular arrays with NaN
            if hasattr(zh_sweep, 'filled'):
                zh_sweep = zh_sweep.filled(np.nan)
            if hasattr(zdr_sweep, 'filled'):
                zdr_sweep = zdr_sweep.filled(np.nan)
            
            # Calculate HDR for current sweep
            hdr_meta, hdr_size_meta = main(zh_sweep, zdr_sweep)
            
            # Update radar fields with results
            radar.fields[hdr_fname]["data"][radar.get_slice(sweep_idx)] = hdr_meta["data"]
            radar.fields[hdr_size_fname]["data"][radar.get_slice(sweep_idx)] = hdr_size_meta["data"]

        # Add metadata to radar fields
        radar = common.add_pyart_metadata(radar, hdr_fname, hdr_meta)
        radar = common.add_pyart_metadata(radar, hdr_size_fname, hdr_size_meta)
        
        logger.info("HDR processing completed successfully for all sweeps")
        return radar
        
    except Exception as e:
        raise HDRProcessingError(f"PyART HDR processing failed: {e}") from e


def pyodim(
    datasets: List[Dict],
    reflectivity_fname: str,
    differential_reflectivity_fname: str,
    hdr_fname: str = "hdr",
    hdr_size_fname: str = "hdr_size"
) -> List[Dict]:
    """
    PyOdim wrapper for HDR processing.
    
    This function processes HDR for PyOdim dataset format and adds the results
    as new fields to each dataset. It handles the xarray-based data structure
    used by PyOdim.
    
    Parameters
    ----------
    datasets : List[Dict]
        List of PyOdim datasets (xarray-based), one per sweep
    reflectivity_fname : str
        Name of the reflectivity field in the datasets
    differential_reflectivity_fname : str
        Name of the differential reflectivity field in the datasets  
    hdr_fname : str, optional, default='hdr'
        Name for the new HDR field to be added to each dataset
    hdr_size_fname : str, optional, default='hdr_size'
        Name for the new HDR size field to be added to each dataset
        
    Returns
    -------
    datasets : List[Dict]
        Updated list of datasets with HDR and HDR size fields added
        
    Raises
    ------
    HDRValidationError
        If required fields are missing from datasets
    HDRProcessingError
        If processing fails for any dataset
        
    Examples
    --------
    Process HDR for PyOdim datasets:
    
    >>> import pyodim
    >>> datasets = pyodim.read_vol('radar_volume.h5')
    >>> datasets_with_hdr = pyodim(datasets, 'DBZH', 'ZDR')
    >>> print(f"Processed {len(datasets_with_hdr)} sweeps")
    
    Check results for first sweep:
    
    >>> first_sweep = datasets_with_hdr[0]
    >>> print(f"HDR data shape: {first_sweep['hdr'].shape}")
    >>> print(f"Max hail size: {first_sweep['hdr_size'].max().values:.1f} mm")
    
    Notes
    -----
    - Each dataset in the list is processed independently
    - Results are added as new data variables with appropriate metadata
    - Original datasets are modified in-place
    """
    try:
        logger.info(f"Processing HDR for {len(datasets)} PyOdim datasets")
        
        # Process each sweep dataset
        for sweep_idx, dataset in enumerate(datasets):
            logger.debug(f"Processing dataset {sweep_idx + 1}/{len(datasets)}")
            
            # Validate required fields exist
            if reflectivity_fname not in dataset:
                raise HDRValidationError(
                    f"Reflectivity field '{reflectivity_fname}' not found in "
                    f"dataset {sweep_idx}. Available fields: {list(dataset.keys())}"
                )
            
            if differential_reflectivity_fname not in dataset:
                raise HDRValidationError(
                    f"Differential reflectivity field '{differential_reflectivity_fname}' "
                    f"not found in dataset {sweep_idx}. Available fields: {list(dataset.keys())}"
                )
            
            # Extract data values
            zh_data = dataset[reflectivity_fname].values
            zdr_data = dataset[differential_reflectivity_fname].values
            
            # Calculate HDR
            hdr_meta, hdr_size_meta = main(zh_data, zdr_data)
            
            # Add new fields to dataset
            dataset = dataset.merge({
                hdr_fname: (("azimuth", "range"), hdr_meta["data"]),
                hdr_size_fname: (("azimuth", "range"), hdr_size_meta["data"]),
            })
            
            # Add metadata using PyOdim conventions
            dataset[hdr_fname] = common.add_pyodim_metadata(dataset[hdr_fname], hdr_meta)
            dataset[hdr_size_fname] = common.add_pyodim_metadata(dataset[hdr_size_fname], hdr_size_meta)
            
            # Update the dataset in the list
            datasets[sweep_idx] = dataset
        
        logger.info("HDR processing completed successfully for all datasets")
        return datasets
        
    except Exception as e:
        raise HDRProcessingError(f"PyOdim HDR processing failed: {e}") from e


# Convenience functions for direct access
def calculate_hdr_simple(
    zh: Union[float, NDArray[np.floating]], 
    zdr: Union[float, NDArray[np.floating]]
) -> Union[float, NDArray[np.floating]]:
    """
    Simple HDR calculation for quick analysis.
    
    Parameters
    ----------
    zh : float or NDArray[np.floating]
        Horizontal reflectivity in dBZ
    zdr : float or NDArray[np.floating]
        Differential reflectivity in dB
        
    Returns
    -------
    float or NDArray[np.floating]
        HDR value(s) in dB
        
    Examples
    --------
    >>> hdr_val = calculate_hdr_simple(50.0, 0.5)
    >>> print(f"HDR: {hdr_val:.1f} dB")
    """
    # Handle scalar inputs
    if np.isscalar(zh) and np.isscalar(zdr):
        if np.isfinite(zh) and np.isfinite(zdr):
            return _calculate_hdr_core(float(zh), float(zdr))
        else:
            return np.nan
    
    # Handle array inputs
    zh_array = np.asarray(zh, dtype=np.float64)
    zdr_array = np.asarray(zdr, dtype=np.float64)
    
    if zh_array.shape != zdr_array.shape:
        raise ValueError(f"Shape mismatch: zh {zh_array.shape} != zdr {zdr_array.shape}")
    
    return _calculate_hdr_array(zh_array, zdr_array)


def calculate_hail_size_simple(
    hdr: Union[float, NDArray[np.floating]]
) -> Union[float, NDArray[np.floating]]:
    """
    Simple hail size estimation for quick analysis.
    
    Parameters
    ----------
    hdr : float or NDArray[np.floating]
        HDR value(s) in dB
        
    Returns
    -------
    float or NDArray[np.floating]
        Estimated hail size(s) in mm
        
    Examples
    --------
    >>> size = calculate_hail_size_simple(25.0)
    >>> print(f"Estimated hail size: {size:.1f} mm")
    """
    # Handle scalar inputs
    if np.isscalar(hdr):
        if np.isfinite(hdr):
            return _calculate_hdr_size_core(float(hdr))
        else:
            return np.nan
    
    # Handle array inputs
    hdr_array = np.asarray(hdr, dtype=np.float64)
    return _calculate_hdr_size_array(hdr_array)