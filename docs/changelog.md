
## 0.4.1 (2025-06-02)
This patch release includes fixes to use zonal statistics for the rainfall-runoff section in the D-HyDAMO tool. Zonal statistics does not longer accept extended geodataframes.

## 0.4.0 (2025-06-02)
This release contains improvements and bugfixes for the D-HyDAMO tool. Note that the following release notes are only applicable for the D-HyDAMO tool. 

### General
Underlying packages have been updated and the recommended Python version is now 3.12.

### 1D model
- weirs with multiple openings and are now treated differently. Whereas formerly the first opening was assigned to the weir, now for every 'extra' opening a fictional weir is created and combined in a compound structure. The same holds for orifices.
- when pumpstations contain multiple pumps, fictional pumpstations are created and combined in a compound structure. Formerly, in D-HYDRO Suite a pumpstation was created for every pump.
- related to the former item: pumps in D-HYDRO Suite now inherit the ID of the pumpstation. This used to be the ID of the pump.
- code to assign order numbers to branches is now contained in one function "mesh.mesh1d_order_numbers_from_attribute", where an attribute in the hydamo hydroobjects-file (for example a name) is used to base order numbers on;
- a CRS-projection is assigned to the net.nc-file and thus the resulting model. For now, only RijksDriehoeksstelsel (EPSG=28992) is supported.
- the CF-compliance attribute in the net.nc-file has changed to prevent warnings in DHYDRO.
- functionality to construct storage nodes is added. Waterlevel-area relationships and locations should be provided. Storage nodes can only be located at the location of connection nodes.

### 2D model
There are still two approaches to 2D mesh-generation and 1d2d links: gridgeom and meshkernel. Gridgeom is still available, but meshkernel has been improved significantly and is now able to provide all relevant functionality.
- in grid refinement all meshkernel parameters can be passed from D-HyDAMO, providing more control over the resulting
- generation of 2d-to-1d links (both embedded and lateral) has been improved; formerly problems ocurred when a rectangular mesh contained triangular cells (for example after mesh refining).

### RTC model
- bug fix in the parsing of complex structures from XML. It was assumed that the rtcDataConfig-file starts with a reference to the timeseries-file, but this is not necessarily the case (e.g. if no time controllers or controllers with time-varying setpoints are present). Now, the file is also parsed correctly if there is no reference to a timeseries file.

### RR model
- change in the dropping of nodes with an area of 0 m2. Formerly, in some circumstances, the wrong node ID could be dropped.
- similar to paved nodes for sewer areas and overflows, specific greenhouses and outlets can be specified, creating extra greenhouse nodes apart from the regular nodes (where one node per catchment was created based on the greenhouse area in the land use map.
- greenhouse properties (storage per hectare) can now be passed from D-HyDAMO; this used to be hardcoded.
- the coupling between lateral nodes and catchments was confusing and not completely according the HyDAMO standard. Now, the coupling is created based on the fields globalid (in lateral nodes) and 'lateraleknoopid' in the catchments. However, these ID's are not suitable for the model. For ID's the 'code' attributes of both layers are used. To provide more control over the ID's, the ID's are assigned in the workflow. This was always the case, but the syntax in the workflow is now different (and shorter) than before. 

### Changes to the workflow:
The example workflow has been expanded to illustrate the new functionality above. Furthermore:
- examples are included to add all types of structures
- examples to add lateral nodes (both constant and timeseries) are added
- gridgeom functionality is still available in D-HyDAMO but removed from the notebook. Gridgeom is only usable with additional software that is not freely available. We recommend using meshkernel from now on.
- a selection can be made in the notebook whether: 1) all RTC controllers are replaced by timeseries of observed crestlevels/gate heights, for example for calibration, or 2) PID/interval-controllers  are maintained.
- the call to 'hydamo.structures.convert.weirs' has been changed: this was not consistent with other calls and now contains named, optional arguments instead of positional arguments. This prevents issues when not all optional input is provided.
- Important: the syntax of the coupling between lateral nodes and catchments has been changed.
- to add projection information to the model, the funtion 'dimr.add_crs()' is called to post-process the .net file.
- the 'sediment' block in the MDU file is removed to prevent errors in DHYDRO.

## 0.3.0 (2024-09-06)

## 0.2.0 (2023-06-09)

### Feat

- D-HyDAMO: Set branch order to allow interpolation of crosssections
- D-HyDAMO: Allow creation of compound structures
- D-HyDAMO: Example notebook with extended demo of new functionalities and custom MDU settings and 2D functionalities

### Fix
- D-HyDAMO: Crosssections that are assigned to a structure could (erroneously) also be assigned to a branch

## 0.1.2 (2023-01-26)

### Refactor

- HYDROLIB now depends on hydrolib-core 0.4.1, with the new import structure.

## 0.1.1 (2023-01-26)

### Feat

- D-HyDAMO initial version under HYDROLIB package (incl. DAMO 2.2 support and hydrolib-core usage)
- profile optimizer 0.1.3 (#87)
