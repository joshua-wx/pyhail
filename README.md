# Python Hail Retrieval Toolkit (pyhail) ⛈️📡🧊

This toolkit provides a collection of hail retrieval techniques for
weather radar data using the [Py-ART](https://github.com/ARM-DOE/pyart/) toolkit.

### Library Dependencies
- [Py-ART](https://github.com/ARM-DOE/pyart/)
- numpy
- netCDF4
- scipy
### Notebook Dependencies
- matplotlib
- cartopy

### Hail Retrivals
- Hail Size Discrimination Algorithm - HSDA ([Ortega et al. 2016](https://journals.ametsoc.org/doi/10.1175/JAMC-D-15-0203.1))
- Hail Differential Reflectivity - HDR ([Depue et al. 2007](https://doi.org/10.1175/JAM2529.1))
- Maximum Expected Size of Hail - MESH witt1998 ([Witt et al. 1998](https://journals.ametsoc.org/doi/10.1175/1520-0434%281998%29013%3C0286%3AAEHDAF%3E2.0.CO%3B2))
- Maximum Expected Size of Hail - MESH mh2019_75/mh2019_95 ([Murillo and Homeyer 2019](https://journals.ametsoc.org/view/journals/apme/58/5/jamc-d-18-0247.1.xml))
- Accumulated Hail - hAcc ([Wallace et al. 2019](https://journals.ametsoc.org/view/journals/wefo/34/1/waf-d-18-0053_1.xml))

MESH is implemented for both pyart radar (PPI) and grid (Cartesian) data!

### Install
To install pyhail, you can either download and unpack the zip file of the source code or use git to checkout the repository:

`git clone git@github.com:joshua-wx/pyhail.git`

To install in your home directory, use:

`python setup.py install --user`

### Use
- [Example Notebook](https://github.com/joshua-wx/pyhail/blob/master/notebooks/example.ipynb)

