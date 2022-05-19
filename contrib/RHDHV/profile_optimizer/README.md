# Introduction 
The Profile Optimizer is a python tool that automizes model optimisation for D-Hydro models. 
Based on an existing D-Hydro FM model, a part of the system will be changed to find an optimal situation. 
In this version (v1.0) it is possible to optimize the bottom width of a single hydroobject (non-branching stream)
based on the current bottom level and a chosen slope. 
The bottom width is optimised to reach a desired flow velocity at a chosen location. 

The Profile Optimizer can be adjusted to optimise for other system variables. For example, the slope of the profile or the friction coefficient could be changed. 
Other resulting parameters can also be used to test the system, for example the Profile Optimizer could test for an desired water level. 
These adjustments are not yet developed. 

# Getting Started
It is recommended to use the Profile Optimizer as a part of the HYDROLIB package. 
Please follow the most recent instructions for HYDROLIB (https://github.com/Deltares/HYDROLIB).

Alternatively, the following steps can be used to run the Profile Optimizer as a stand-alone package:
1.  Install a conda distribution. 
2.  Use conda to install the environment.yml delivered with this project:
    - `conda env create --file environment.yml`
    - the environment is called "po_env" and contains all required dependencies, including HYDROLIB-CORE.
3.  Use the example notebook to use the Profile Optimizer for your model

# Contact 
The Profile Optimizer is part of HYDROLIB, an open source community effort for python tools for the D-Hydro software package. 
For more information about this initiative, visit: https://github.com/Deltares/HYDROLIB

The Profile Optimizer is developed by Royal HaskoningDHV:
- lisa.weijers@rhdhv.com
- valerie.demetriades@rhdhv.com