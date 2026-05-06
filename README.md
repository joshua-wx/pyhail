[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/joshua-wx/pyhail)
[![tests](https://github.com/joshua-wx/pyhail/actions/workflows/tests.yml/badge.svg)](https://github.com/joshua-wx/pyhail/actions/workflows/tests.yml)

# Python Hail Retrieval Toolkit (pyhail)

pyhail is a Python library providing hail retrieval algorithms for weather radar data.
It supports both the [Py-ART](https://github.com/ARM-DOE/pyart) and [pyodim](https://github.com/vlouf/pyodim)
radar interfaces and operates directly on native polar (PPI) data, with optional Cartesian gridding for MESH.

---

## Retrievals

| Algorithm | Output field | Units | Description |
|-----------|-------------|-------|-------------|
| **MESH** | `mesh` | mm | Maximum Expected Size of Hail — column-integrated kinetic energy converted to hail diameter |
| **HSDA** | `hsda` | class | Hail Size Discrimination Algorithm — polarimetric classification of hail size |
| **HDR** | `hdr_size` | mm | Hail Differential Reflectivity — hail size estimate from ZH and ZDR |
| **HACC** | `hacc` | cm/min | Hail Accumulation — surface hail accumulation rate |

### MESH formulations

Four calibrations are available via the `mesh_method` parameter:

| `mesh_method` | Formula | Calibrated to | Reference |
|--------------|---------|--------------|-----------|
| `witt1998` | MESH = 2.54 × SHI^0.5 | 147 reports | Witt et al. (1998) |
| `mh2019_75` | MESH = 15.096 × SHI^0.206 | 5897 reports, 75th pct | Murillo & Homeyer (2021) |
| `mh2019_95` | MESH = 22.157 × SHI^0.212 | 5897 reports, 95th pct | Murillo & Homeyer (2021) |
| `blend` *(default)* | Logistic blend of `witt1998` → `mh2019_75` | — | — |

The `blend` formulation transitions smoothly from Witt (1998) at low SHI to Murillo & Homeyer (2021)
at high SHI using a logistic weight, eliminating the discontinuous derivative of a hard piecewise switch.
The crossover point is the analytical intercept of the two power laws (SHI ≈ 429 J m⁻¹ s⁻¹,
MESH ≈ 52.6 mm).

### C-band correction

When processing C-band radar data, MESH can apply a reflectivity correction for Mie scattering
and attenuation effects using `radar_band='C'` and `correct_cband_refl=True`
(Brook et al. 2023, [arXiv:2306.12016](https://arxiv.org/abs/2306.12016)).

> **Note**: The HSDA Q confidence vector (Park et al. 2009) is not implemented;
> all pixels are assigned q = 1.

---

## Installation

### From PyPI

```bash
pip install pyhail
```

### Conda environment (recommended for notebooks)

A full environment including Py-ART, pyodim, cartopy, and Jupyter is provided:

```bash
conda env create -f test_environment.yml
conda activate pyhail-test-env
```

### From source

```bash
git clone git@github.com:joshua-wx/pyhail.git
cd pyhail
pip install -e .
```

---

## Dependencies

**Core** (installed automatically via pip):
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [numba](https://numba.pydata.org/) ≥ 0.59.1

**Radar I/O** (install separately, choose one or both):
- [Py-ART](https://github.com/ARM-DOE/pyart) — `pip install arm-pyart`
- [pyodim](https://github.com/vlouf/pyodim) — `pip install pyodim`

**Notebooks only**: matplotlib, cartopy, ipywidgets

---

## Key parameters

### MESH (`mesh_ppi.pyart` / `mesh_ppi.pyodim`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temp_levels` | — | `[freezing_level, minus20_level]` in m ASL |
| `mesh_method` | `'blend'` | One of `witt1998`, `mh2019_75`, `mh2019_95`, `blend` |
| `radar_band` | `'S'` | `'S'` or `'C'` |
| `correct_cband_refl` | `True` | Apply C-band Mie/attenuation correction |
| `min_range` / `max_range` | `10` / `150` | Range limits (km) |
| `transition_width` | `200` | Blend logistic width (J m⁻¹ s⁻¹); smaller = sharper transition |

### HSDA (`hsda.pyart` / `hsda.pyodim`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `levels` | — | `[freezing_level, minus20_level]` in m ASL |
| `hca_hail_idx` | — | HCA class indices to apply HSDA (e.g. `[6, 7, 8]` for hail classes) |
| `dzdr` | `0` | ZDR calibration offset (dB) |

### HACC (`hacc.pyart` / `hacc.pyodim`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sweep_idx` | — | Sweep index for accumulation calculation |
| `mesh_idx` | — | Sweep index containing the MESH field |
| `fz_level` | — | Freezing level height (m) |
| `pressure` | — | Mean surface-to-freezing-level pressure (hPa) |

---

## Testing

The unit tests cover the pure mathematical functions: MESH calibration formulas, HSDA membership
functions, the trapezoidal MF and aggregate scorer, and `safe_log`.

### Requirements

pytest must be installed in the active environment:

```bash
pip install pytest
```

If you are using the conda environment from `test_environment.yml`, activate it first:

```bash
conda activate pyhail-test-env
pip install pytest
```

### Running the tests

From the repository root:

```bash
python -m pytest tests/ -v
```

To run a single test file:

```bash
python -m pytest tests/test_mesh_formulas.py -v
```

| Test file | What it covers |
|-----------|----------------|
| [`tests/test_mesh_formulas.py`](tests/test_mesh_formulas.py) | `mesh_witt1998`, `mesh_mh2019_75`, `mesh_mh2019_95`, `mesh_smooth_blend` |
| [`tests/test_hsda_mf.py`](tests/test_hsda_mf.py) | HSDA scalar membership functions (`c`, `f1`–`f3`, `g1`–`g3`) and `build_mf` for all altitude levels |
| [`tests/test_hsda_core.py`](tests/test_hsda_core.py) | `trapmf`, `calc_ag` (weighting and rule 1), `h_sz` (rules 2 and 3) |
| [`tests/test_common.py`](tests/test_common.py) | `safe_log` — scalar, array, custom eps, boundary values |

---

## Notebooks

Working examples with bundled data are in [`notebooks/`](notebooks/).

| Notebook | Description |
|----------|-------------|
| [`testing_pyart.ipynb`](notebooks/testing_pyart.ipynb) | All retrievals using the Py-ART interface |
| [`testing_pyodim.ipynb`](notebooks/testing_pyodim.ipynb) | All retrievals using the pyodim interface |
| [`compare_mesh_fits.ipynb`](notebooks/compare_mesh_fits.ipynb) | Comparison of the four MESH calibrations |
| [`hybrid_mesh_blend.ipynb`](notebooks/hybrid_mesh_blend.ipynb) | Interactive exploration of the smooth blend formulation |
| [`c_band_mesh_correction.ipynb`](notebooks/c_band_mesh_correction.ipynb) | Effect of C-band correction on MESH (requires external data) |

---

## References

- Witt et al. (1998) doi:[10.1175/1520-0434(1998)013<0286:AEHDAF>2.0.CO;2](https://doi.org/10.1175/1520-0434(1998)013%3C0286:AEHDAF%3E2.0.CO;2)
- Murillo & Homeyer (2021) doi:[10.1175/JAMC-D-20-0271.1](https://doi.org/10.1175/JAMC-D-20-0271.1)
- Ortega et al. (2016) doi:[10.1175/JAMC-D-15-0203.1](https://doi.org/10.1175/JAMC-D-15-0203.1)
- Ryzhkov et al. (2013) doi:[10.1175/JAMC-D-13-074.1](https://doi.org/10.1175/JAMC-D-13-074.1)
- Depue et al. (2007) doi:[10.1175/JAM2529.1](https://doi.org/10.1175/JAM2529.1)
- Wallace et al. (2019) doi:[10.1175/WAF-D-18-0053.1](https://doi.org/10.1175/WAF-D-18-0053.1)
- Kalina et al. (2016) doi:[10.1175/WAF-D-15-0037.1](https://doi.org/10.1175/WAF-D-15-0037.1)
- Brook et al. (2023) arXiv:[2306.12016](https://arxiv.org/abs/2306.12016)

---

## Issues and contact

Please use the [GitHub issue tracker](https://github.com/joshua-wx/pyhail/issues) to report bugs or request features.
Maintained by Joshua Soderholm (joshua.soderholm@bom.gov.au).
