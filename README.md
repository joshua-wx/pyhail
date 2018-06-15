# Python Hail Retrieval Toolkit (PyHail)

This toolkit provides a collection of hail retrieval techniques for
weather radar data built upon [Py-ART](https://github.com/ARM-DOE/pyart/)
and [col_processing](https://github.com/vlouf/cpol_processing). 

![7th nov mt cootha x1 pano-small](https://user-images.githubusercontent.com/16043083/41452799-c07be16c-70b6-11e8-9047-4fd92e66a4fb.jpg)

### Dependencies
- [CSU Radartools](https://github.com/CSU-Radarmet/CSU_RadarTools)
- [Py-ART](https://github.com/ARM-DOE/pyart/)

### Hail Retrivals
- Hail Size Discrimination Algorithm - HSDA ([Ortega et al. 2016](https://journals.ametsoc.org/doi/10.1175/JAMC-D-15-0203.1))
- Hail Differential Reflectivity - HDR ([Depue et al. 2007](https://doi.org/10.1175/JAM2529.1))
- Maximum Expected Size of Hail - MESH ([Witt et al. 1998](https://journals.ametsoc.org/doi/10.1175/1520-0434%281998%29013%3C0286%3AAEHDAF%3E2.0.CO%3B2))

### Dual Pol Corrections
- SNR calculation from radiosonde using Py-ART calculate_snr_from_reflectivity
- Unfolding of PHIDP using Py-ART get_phidp_unf
- Recalculation of KDP using CSU radartools Bringi technique
- Correction of ZDR attenuation using Bringi relationship (KDP)
- Correction of DBZH attenuation using Py-ART calculate_attenuation
- Correction of RHOHV from noise (Schuur et al. 2003 NOAA report (p7 eq 5))
- Correction of ZDR from noise (Schuur et al. 2003 NOAA report (p7 eq 6))

### Use
- [pipeline notebook](https://github.com/joshuass/pyHail/blob/master/inspection_plot.ipynb): applies dual pol processing (filtering, attenuation corrections)
and hail retrievals to various radar formats (cfradial, odimh5, mdv). Note that radiosonde data must be supplied in netcdf format.
- [inspection_plot notebook](https://github.com/joshuass/pyHail/blob/master/pipeline.ipynb): plots dual pol fields and all hail retrievals


