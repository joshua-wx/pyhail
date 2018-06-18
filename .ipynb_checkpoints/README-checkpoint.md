# Python Hail Retrieval Toolkit (PyHail)

This toolkit provides a collection of hail retrieval techniques for
weather radar data using the [Py-ART](https://github.com/ARM-DOE/pyart/) toolkit.

![7th nov mt cootha x1 pano-small](https://user-images.githubusercontent.com/16043083/41452799-c07be16c-70b6-11e8-9047-4fd92e66a4fb.jpg)

### Dependencies
- [Py-ART](https://github.com/ARM-DOE/pyart/)
- numpy
- netCDF4

### Hail Retrivals
- Hail Size Discrimination Algorithm - HSDA ([Ortega et al. 2016](https://journals.ametsoc.org/doi/10.1175/JAMC-D-15-0203.1))
- Hail Differential Reflectivity - HDR ([Depue et al. 2007](https://doi.org/10.1175/JAM2529.1))
- Maximum Expected Size of Hail - MESH ([Witt et al. 1998](https://journals.ametsoc.org/doi/10.1175/1520-0434%281998%29013%3C0286%3AAEHDAF%3E2.0.CO%3B2))

### Libraries used for Dual Pol Corrections
- [cpol_processing](https://github.com/vlouf/cpol_processing)
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
- [pipeline notebook](https://github.com/joshuass/pyHail/blob/master/inspection_plot.ipynb): applies dual pol processing (filtering, attenuation corrections)
and hail retrievals to various radar formats (cfradial, odimh5, mdv). Note that radiosonde data must be supplied in netcdf format.
- [inspection_plot notebook](https://github.com/joshuass/pyHail/blob/master/pipeline.ipynb): plots dual pol fields and all hail retrievals


