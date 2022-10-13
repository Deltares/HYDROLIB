# Introduction 
The Profile Optimizer is a python tool that automizes model optimisation for D-Hydro models. 
Based on an existing D-Hydro FM model, a part of the system will be changed to find an optimal situation. 
In this version (v1.0) it is possible to optimize the bottom width of a single hydroobject (non-branching stream)
based on the current bottom level and a chosen slope. 
The bottom width is optimised to reach a desired flow velocity at a chosen location. 

The Profile Optimizer can be adjusted to optimise for other system variables. For example, the slope of the profile or the friction coefficient could be changed. 
Other resulting parameters can also be used to test the system, for example the Profile Optimizer could test for an desired water level. 
These adjustments are not yet developed. 

# Important notices
The base functionality of creating iterations of a DHydro model is covered in `optimizer.py`'s `ProfileOptimizer` class. 
It is necassary to have a base DHydro model with XY crosssections. 
Cross section definitions should be unique to avoid unpredictable behaviour. 
For example, if a profile with the definition-id "default_profile" is used at multiple locations, and one of these locations is selected in the optimization, the definition will be changed for all locations. 

In the `preprocessing` script, some supporting functions are supplied to estimate a reasonable initial bottom width. 
These functions are simply guidance, and can be ignored in favor of expert judgement. 
The preprocessing functions rely on both the Manning formula and a simple `Q=V*A`. 
The code cannot solve Manning for a chosen Q (discharge), only a chosen V (velocity), for solving B (bottom width).
Therefore, the first solve of B is then checked with `Q=V*A` if the found profile is befitting of the chosen Q. 
The first solve of B is then adjusted until it fits with Q. 
B is adjusted in steps of 5% until it gets close to Q (default: 5%, can be adjusted by user). 
This process can help, but it can also give a wrong indication if the hydraulic system is complex or if the input is not right. 

Finally: Iterations with assymetric slopes (talud) are possible, but cannot be used as input for preprocessing. 

TL;DR Warnings: 
* Watch out when one profile definition is used at multiple locations 
* Preprocessing is only a suggestion, it can be ignored and/or skipped
* Assymetric slopes are possible! 


# Getting Started
The recommended approach to use the Profile Optimizer as a part of the HYDROLIB package. 
Please follow the most recent instructions for HYDROLIB (https://github.com/Deltares/HYDROLIB).

Alternatively, the following steps can be used to run the Profile Optimizer as a stand-alone package:
1.  Install a conda distribution. 
2.  Use conda to install the environment.yml delivered with this project:  
    `conda env create --file environment.yml`  
    - the environment is called "po_env" and contains all required dependencies, including HYDROLIB-CORE.
3.  Use the example notebook to use the Profile Optimizer for your model.
    - To launch the notebook in this environment, use the following code in anaconda prompt:  
    `conda activate profileoptimizer`  
    `jupyter notebook`

# Contact 
The Profile Optimizer is part of HYDROLIB, an open source community effort for python tools for the D-Hydro software package. 
For more information about this initiative, visit: https://github.com/Deltares/HYDROLIB

The Profile Optimizer is developed by Royal HaskoningDHV:
- rineke.hulsman@rhdhv.com
- lisa.weijers@rhdhv.com
- valerie.demetriades@rhdhv.com