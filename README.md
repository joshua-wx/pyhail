# Python Hail Retrieval Toolkit (PyHail)

This toolkit provides a collection of hail retrieval techniques for
weather radar data using the [Py-ART](https://github.com/ARM-DOE/pyart/) toolkit.

### Dependencies
- [Py-ART](https://github.com/ARM-DOE/pyart/)
- numpy
- netCDF4
- [wradlib](https://github.com/wradlib/wradlib)

### Hail Retrivals
- Hail Size Discrimination Algorithm - HSDA ([Ortega et al. 2016](https://journals.ametsoc.org/doi/10.1175/JAMC-D-15-0203.1))
- Hail Differential Reflectivity - HDR ([Depue et al. 2007](https://doi.org/10.1175/JAM2529.1))
- Maximum Expected Size of Hail - MESH ([Witt et al. 1998](https://journals.ametsoc.org/doi/10.1175/1520-0434%281998%29013%3C0286%3AAEHDAF%3E2.0.CO%3B2))

### Install
To install PyHail, you can either download and unpack the zip file of the source code or use git to checkout the repository:

`git clone git@github.com:joshua-wx/PyHail.git`

To install in your home directory, use:

`python setup.py install --user`

### Libraries used for Dual Pol Corrections
- [CPOL Processing](https://github.com/vlouf/cpol_processing)
- [CSU Radartools](https://github.com/CSU-Radarmet/CSU_RadarTools)

### Dual Pol Corrections
- SNR calculation from radiosonde using Py-ART calculate_snr_from_reflectivity
- Unfolding of PHIDP using Py-ART get_phidp_unf
- Recalculation of KDP using csu_radartools Bringi technique
- Correction of ZDR attenuation using csu_radartools Bringi relationship (KDP)
- Correction of DBZH attenuation using Py-ART calculate_attenuation
- Correction of RHOHV from noise using cpol_processing (Schuur et al. 2003 NOAA report (p7 eq 5))
- Correction of ZDR from noise using cpol_processing (Schuur et al. 2003 NOAA report (p7 eq 6))

### Use
- [Inspection Plot Notebook](https://github.com/joshua-wx/PyHail/blob/master/notebooks/inspection_plot.ipynb): applies dual pol processing (filtering, attenuation corrections)
and hail retrievals to various radar formats (cfradial, odimh5, mdv). Note that radiosonde data must be supplied in netcdf format.
- [Pipeline Notebook](https://github.com/joshua-wx/PyHail/blob/master/notebooks/pipeline.ipynb): plots dual pol fields and all hail retrievals


