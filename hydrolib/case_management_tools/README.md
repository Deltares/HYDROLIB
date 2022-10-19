# Case Management Tools (CMT)

Open Source Case Management Tools for [D3DFM/D-HYDRO](https://www.deltares.nl/en/software/delft3d-flexible-mesh-suite/)

CMT includes
* Generation of one or multiple cases for D-HYDRO using an uniform directory structure
* Generation of cases as part of a StochasticTool (HydroConsult) workflow
* Generation of rainfal events from STOWA parameters
* Run D-HYDRO scenarios in single or parallel mode
* On-the-fly post-processing of results on desired location.

---

**Documentation**: [https://deltares.github.io/HYDROLIB/case_management_tools](https://deltares.github.io/HYDROLIB/case_management_tools)

**Source Code**: [https://github.com/Deltares/HYDROLIB/tree/main/hydrolib/case_management_tools](https://github.com/Deltares/HYDROLIB/tree/main/hydrolib/case_management_tools)

---

## Installation

We recommend to build your environment using [Anaconda](https://www.anaconda.com/). You can build an environment ánd install CMT by conda in one go using this <a href="https://github.com/Deltares/HYDROLIB/blob/main/hydrolib/case_management_tools/envs/environment.yml" target="_blank">environment.yml</a> from the command-line:
```
conda env create -f environment.yml
```

In that activated environment you can add cmt with pip in the setup.py directory:
```
pip install .
```

Alternatively CMT can be installed with pip in any environment with the following Python-packages properly installed:

* hydrolib-core 0.3.0
* netCDF4
* openpyxl
* pandas
* simplejson

## About

Case Management Tools is developed and maintained by [D2Hydro](https://d2hydro.nl/) and freely available under an Open Source <a href="https://github.com/Deltares/HYDROLIB/blob/main/hydrolib/case_management_tools/LICENSE" target="_blank">MIT license</a>.
