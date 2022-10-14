# Introduction 
The Profile Optimizer is a python tool that automates model optimization for D-Hydro models. 
Based on an existing D-HYDRO 1D2D (D-Flow FM) model, a part of the system will be changed to find an optimal situation. 
In this version (v0.1.3) it is possible to optimize the bottom width of a single hydroobject (non-branching stream)
based on the current bottom level and a chosen slope. 
The bottom width is optimised to reach a desired flow velocity at a chosen location. 

The Profile Optimizer can be adjusted to optimize for other system variables. For example, the slope of the profile or the friction coefficient could be changed. 
Other resulting parameters can also be used to test the system, for example the Profile Optimizer could test for an desired water level. 
These adjustments are not yet developed. 

# Usage
An example workflow is developed in Jupyter Notebook, this notebook explains the possible steps for the profile optimizer. 
But to get started with the basics, the principles are also explained here in less detail. 

```
# Initiate profile optimizer class:
from hydrolib.profile_optimizer.optimizer import ProfileOptimizer
from pathlib import Path

base_model_fn = "example_folder"  # should contain MDU of base model
bat_file = "example_folder/run.bat"  # should run DIMR 

optimize = ProfileOptimizer(base_model_fn = example_folder, 
                            bat_file = bat_file, 
                            work_dir = Path("new_temp_folder"),  
                            output_dir: Path("new_output_folder"),
                            iteration_name='Iteration', 
                            iteration_start_count=1)
``` 
After initiating the class, any number of iterations can be created using the following function:
```
optimize.create_iteration(prof_ids = ['prof_3', 'prof_4', 'prof_5'], 
                          trapezium_pars = dict(bottom_width=4, 
                                                slope_l=2, 
                                                slope_r=1, 
                                                depth=1.5})
```
Every call to this function will create a new iteration of the model, written to the work_dir.  
Another function can be used to run the model, using a copy of the bat_file. 
```
optimize.run_latest()
```
The functions to create a search-window for the ideal bottom width are part of `hydrolib.profile_optimizer.preprocessing` 
and are explained in the jupyter notebook. The optimization based on this is based on `hydrolib.profile_optimizer.optimizer.find_optimum`
and is also explained in the jupyter notebook. 

# Important notices
The base functionality of creating iterations of a D-HYDRO model is covered in `optimizer.py`'s `ProfileOptimizer` class. 
It is necessary to have a base D-HYDRO model with XY crosssections. 
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

Finally: Iterations with asymetric slopes (talud) are possible, but cannot be used as input for preprocessing. 

TL;DR Warnings: 
* Watch out when one profile definition is used at multiple locations 
* Preprocessing is only a suggestion, it can be ignored and/or skipped
* Asymetric slopes are possible! 


# Getting Started
The recommended approach to use the Profile Optimizer as a part of the HYDROLIB package. 
Please follow the most recent instructions for HYDROLIB (https://github.com/Deltares/HYDROLIB).

Alternatively, the following steps can be used to run the Profile Optimizer as a stand-alone package:
1.  Install a conda distribution. 
2.  Use conda to install the environment.yml delivered with this project:  
    `conda env create --file environment.yml`  
    - the environment is called "po_env" and contains all required dependencies, including HYDROLIB-core.
3.  Use the example notebook to use the Profile Optimizer for your model.
    - To launch the notebook in this environment, use the following code in anaconda prompt:  
    `conda activate profileoptimizer`  
    `jupyter notebook`

# Contact 
The Profile Optimizer is part of HYDROLIB, an open source community effort for python tools for the hydraulic/hydrological modelling workflows. 
For more information about this initiative, visit: https://github.com/Deltares/HYDROLIB

The Profile Optimizer is developed by Royal HaskoningDHV:
- rineke.hulsman@rhdhv.com
- lisa.weijers@rhdhv.com
- valerie.demetriades@rhdhv.com
