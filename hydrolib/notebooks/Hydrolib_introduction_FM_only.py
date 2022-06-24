
# Basis
import os
import sys
import shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

sys.path.append('../')      
# Importing relevant classes from delft3dfmpy
from dhydamo.io.hydamo import HyDAMO
from dhydamo.io.convert_hydamo import Structures
#from dhydamo.io.dfm import DFlowFMmodel


#from delft3dfmpy import DFlowRRModel, DFlowRRWriter
 #rom delft3dfmpy.datamodels.common import ExtendedGeoDataFrame

sys.path.append(r'D:\3640.20\HYDROLIB-core')
from hydrolib.core.io.structure.models import *
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.dimr.models import DIMR, FMComponent

# path to the package containing the dummy-data
data_path = os.path.abspath('../tests/data')
print(data_path)

# path to write the models
output_path = os.path.abspath('../tests/model')

gpkg_file = os.path.join(data_path,'Example_model.gpkg')

hydamo = HyDAMO(extent_file=os.path.join(data_path,'OLO_stroomgebied_incl.maas.shp'))

# show content
hydamo.branches.show_gpkg(gpkg_file)
hydamo.branches.read_gpkg_layer(gpkg_file, layer_name='HydroObject', index_col='code')

hydamo.profile.read_gpkg_layer(gpkg_file, layer_name='ProfielPunt', groupby_column = 'profiellijnid', order_column='codevolgnummer', id_col='code') 
hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name='RuwheidProfiel') 
hydamo.profile.snap_to_branch(hydamo.branches, snap_method='intersecting')
hydamo.profile.dropna(axis=0, inplace=True, subset=['branch_offset'])
hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name='profiellijn')
hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name='profielgroep')

hydamo.profile.drop('code', axis=1, inplace=True)
hydamo.profile.rename(columns={'profiellijnid': 'code'}, inplace=True)

hydamo.culverts.read_gpkg_layer(gpkg_file, layer_name='DuikerSifonHevel', index_col='code')
hydamo.culverts.snap_to_branch(hydamo.branches, snap_method='ends', maxdist=5)
hydamo.culverts.dropna(axis=0, inplace=True, subset=['branch_offset'])

hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name='Stuw')
hydamo.weirs.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=10)
hydamo.weirs.dropna(axis=0, inplace=True, subset=['branch_offset'])
hydamo.opening.read_gpkg_layer(gpkg_file, layer_name='Kunstwerkopening')
hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name='Regelmiddel')

idx = hydamo.management_device[hydamo.management_device['duikersifonhevelid'].notnull()].index
for i in idx:
    globid = hydamo.culverts[hydamo.culverts.code==hydamo.management_device.duikersifonhevelid.loc[i]].globalid.values[0]
    hydamo.management_device.at[i,'duikersifonhevelid'] = globid

hydamo.pumpstations.read_gpkg_layer(gpkg_file, layer_name='Gemaal',index_col='code')
hydamo.pumpstations.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=10)
hydamo.pumps.read_gpkg_layer(gpkg_file,layer_name='Pomp', index_col='code')
hydamo.management.read_gpkg_layer(gpkg_file, layer_name='Sturing', index_col='code')


hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name='Brug', index_col='code')
hydamo.bridges.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=1100)
hydamo.bridges.dropna(axis=0, inplace=True, subset=['branch_offset'])

hydamo.boundary_conditions.read_gpkg_layer(gpkg_file, layer_name='hydrologischerandvoorwaarde', index_col='code')
hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=10)

fm = FMModel()
fm.filepath = "test.mdu"

#network = Network()
# crosssections = Crosssections()


struct = Structures()
struct.weirs_from_hydamo(hydamo.weirs, profile_groups=hydamo.profile_group, profile_lines=hydamo.profile_line, profiles=hydamo.profile, opening=hydamo.opening, management_device=hydamo.management_device)
struct.culverts_from_hydamo(hydamo.culverts, management_device=hydamo.management_device)
struct.bridges_from_hydamo(hydamo.bridges, profile_groups=hydamo.profile_group, profile_lines=hydamo.profile_line, profiles=hydamo.profile)
struct.pumps_from_hydamo(hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management)

all_structures = struct.rweirs+struct.orifices+struct.uweirs+struct.bridges+struct.culverts+struct.pumps

fm.geometry.structurefile = [StructureModel(structure=all_structures)]

# wegschrijven
dimr = DIMR()
dimr.component.append(
    FMComponent(name="test", workingDir=".", inputfile=fm.filepath, model=fm)
)
dimr.save(recurse=True)

print('Done!')


# # Collect structures
#from_hydamo(hydamo.weirs, profile_groups=hydamo.profile_group, profile_lines=hydamo.profile_line, profiles=hydamo.profile, opening=hydamo.opening, management_device=hydamo.management_device, management=None)
# dfmmodel.structures.io.culverts_from_hydamo(hydamo.culverts, management_device=hydamo.management_device)
# dfmmodel.structures.io.bridges_from_hydamo(hydamo.bridges, profile_groups=hydamo.profile_group, profile_lines=hydamo.profile_line, profiles=hydamo.profile)
# dfmmodel.structures.io.pumps_from_hydamo(pompen=hydamo.pumps, sturing=hydamo.management, gemalen=hydamo.pumpstations)

# # Add a weir manually (equivalent functions exist for all structures):
# dfmmodel.structures.add_weir(
#     id='extra_weir',
#     branchid='riv_RS1_1810',
#     chainage=950.0,
#     crestlevel=18.00,
#     crestwidth=7.5,
#     corrcoeff=1.0    
# )


# # To use, provide a list of ID's of compound structures, and along with, for every compound structure, a nested list of sub-structures. If there are many, these can be read from files (for example).
# # cmpnd_ids  = ['cmpnd_1']
# # cmpnd_list = [['Orifice_Test1','UWR_test']]
# # dfmmodel.structures.io.compound_structures(cmpnd_ids, cmpnd_list)


# # After this add the branches and generate a grid.

# # In[8]:


# # Create a 1D schematisation
# dfmmodel.network.set_branches(hydamo.branches)
# dfmmodel.network.generate_1dnetwork(one_d_mesh_distance=40.0, seperate_structures=True)


# # Add cross sections. Here two hydamo files are used. First the imported cross sections. If after this there are branch objects left without a cross sections, it is derived from the norm parametrised profile (Dutch: legger).

# # In[9]:


# # Add cross sections from hydamo
# dfmmodel.crosssections.io.from_hydamo(
#     dwarsprofielen=hydamo.crosssections,
#     parametrised=hydamo.parametrised_profiles,
#     branches=hydamo.branches
# )

# print(f'{len(dfmmodel.crosssections.get_branches_without_crosssection())} branches are still missing a cross section.')
# print(f'{len(dfmmodel.crosssections.get_structures_without_crosssection())} structures are still missing a cross section.')


# # If there are still missing cross sections left, add a default one. To do so add a cross section definition, and assign it with a vertical offset (shift).

# # In[10]:


# # Set a default cross section
# default = dfmmodel.crosssections.add_rectangle_definition(
#     height=5.0, width=5.0, closed=False, roughnesstype='Strickler', roughnessvalue=30, name='default')
# dfmmodel.crosssections.set_default_definition(definition=default, shift=5.0)


# # #### Add a 2D mesh

# # To add a mesh, currently 2 options exist:

# # 1) the converter can generate a relatively simple, rectangular mesh, with a rotation or refinement. Note that rotation _and_ refinement is currently not possible. In the section below we generate a refined 2D mesh with the following steps:
# # 
# # - Generate grid within a polygon. The polygon is the extent given to the HyDAMO model.
# # - Refine along the main branch
# # - Determine altitude from a DEM.
# # 
# # The 'refine'-method requires the dflowfm.exe executable. If this is not added to the system path, it can be provided in an optional argument to refine (dflowfm_path).

# # In[ ]:


# # Create mesh object
# mesh = Rectangular()
# cellsize = 25

# # Generate mesh within model bounds
# mesh.generate_within_polygon(hydamo.clipgeo, cellsize=cellsize, rotation=0)

# # Refine the model (2 steps) along the main branch. To do so we generate a buffer around the main branch.
# buffered_branch = hydamo.branches.loc[['riv_RS1_1810', 'riv_RS1_264'], 'geometry'].unary_union.buffer(10)
# mesh.refine(polygon=[buffered_branch], level=[2], cellsize=cellsize, dflowfm_path=dflowfm_path)

# # Determine the altitude from a digital elevation model
# # rasterpath = '../gis/AHNdommel_clipped.tif'
# # mesh.altitude_from_raster(rasterpath)

# # The full DEM is not added to this notebook. Instead a constant bed level is used
# mesh.altitude_constant(15.0)

# # Add to schematisation
# dfmmodel.network.add_mesh2d(mesh)


# # 2) a more complex mesh can be created in other software (such as SMS) and then imported in the converter: (uncomment to activate)

# # In[ ]:


# #from dhydamo.core.mesh2d import Mesh2D
# #mesh = Mesh2D()
# # import the geometry
# #mesh.geom_from_netcdf(r'T:\2Hugo\Grid_Roer_net.nc')
# # fill every cell with an elevation value
# #mesh.altitude_from_raster(rasterpath)
# # and add to the model
# #dfmmodel.network.add_mesh2d(mesh)


# # #### Add the 1D-2D links

# # For linking the 1D and 2D model, three options are available:
# # 1. Generating links from each 1d node to the nearest 2d node.
# # 2. Generating links from each 2d node to the nearest 1d node (intersecting==True)
# # 3. Generating links from each 2d node to the nearest 1d node, while not allowing the links to intersect other cells (intersecting==True).
# # 
# # Intersecting indicates whether or not the 2D cells cross the 1D network (lateral versus embedded links).
# # So, option 3 is relevant when there is no 2d mesh on top of the 1d mesh: the lateral links.
# # 
# # Note that for each option a maximum link length can be chosen, to prevent creating long (and perhaps unrealistic) links.

# # In[ ]:


# del dfmmodel.network.links1d2d.faces2d[:]
# del dfmmodel.network.links1d2d.nodes1d[:]
# dfmmodel.network.links1d2d.generate_1d_to_2d(max_distance=50)


# # In[ ]:


# fig, ax = plt.subplots(figsize=(13, 10))
# ax.set_aspect(1.0)

# segments = dfmmodel.network.mesh2d.get_segments()
# ax.add_collection(LineCollection(segments, color='0.3', linewidths=0.5, label='2D-mesh'))

# links = dfmmodel.network.links1d2d.get_1d2dlinks()
# ax.add_collection(LineCollection(links, color='k', linewidths=0.5))
# ax.plot(links[:, :, 0].ravel(), links[:, :, 1].ravel(), color='k', marker='.', ls='', label='1D2D-links')

# for i, p in enumerate([buffered_branch]):
#     ax.plot(*p.exterior.xy, color='C3', lw=1.5, zorder=10, alpha=0.8, label='Refinement buffer' if i==0 else None)

# hydamo.branches.plot(ax=ax, color='C0', lw=2.5, alpha=0.8, label='1D-mesh')

# ax.legend()

# ax.set_xlim(140900, 141300)
# ax.set_ylim(393400, 393750);


# # ### Boundary conditions for FM
# # 
# # Add boundary conditions to external forcings from a SOBEK time series.

# # In[ ]:


# fn_bcs = os.path.join(data_path, 'sobekdata', 'boundaryconditions.csv')
# bcs = pd.read_csv(fn_bcs, sep=';', index_col=0)
# bcs.index = pd.to_datetime(bcs.index)


# # In[ ]:


# dfmmodel.external_forcings.add_boundary_condition(
#     name='BC_flow_in',
#     pt=(140712.056047, 391893.277878),
#     bctype='discharge',
#     series=bcs['Discharge']
# )

# dfmmodel.external_forcings.add_boundary_condition(
#     name='BC_wlev_down',
#     pt=(141133.788766, 395441.748424),
#     bctype='waterlevel',
#     series=bcs['Waterlevel']
# )


# # In[ ]:


# fig, ax = plt.subplots()

# ax.plot(
#     dfmmodel.external_forcings.boundaries['BC_flow_in']['time'],
#     dfmmodel.external_forcings.boundaries['BC_flow_in']['value'],
#     label='Discharge [m3/s]'
# )

# ax.plot(
#     dfmmodel.external_forcings.boundaries['BC_wlev_down']['time'],
#     dfmmodel.external_forcings.boundaries['BC_wlev_down']['value'],
#     label='Water level [m+NAP]'
# )

# ax.set_ylabel('Value (discharge or waterlevel)')
# ax.set_xlabel('Time [minutes]')

# ax.legend();


# # ### Initial conditions

# # There are four ways to set the initial conditions. First, global water level or depth can be set. In the example, we use a global water depth of 0.5 m, but we could also use the equivalent function "set_initial_waterlevel".

# # In[ ]:


# # Initial water depth is set to 0.5 m
# dfmmodel.external_forcings.set_initial_waterdepth(0.5)


# # It is also possible to define a certain area, using a polygon, with alternative initial conditions (level or depth).

# # In[ ]:


# #init_special = gpd.read_file(data_path+'/GIS/init_waterlevel_special.shp')
# #dfmmodel.external_forcings.set_initial_waterlevel(10.0, polygon=init_special.geometry[0], name='test_polygon')


# # ### Lateral flow

# # Lateral flow can be obtained from the coupling with the RR-model, or by providing time series. Here, these are read from a Sobek model. In the coupling below, nodes that are not linked to a RR-boundary node are assumed to have a prescribed time series.
# # 
# # If a DFM-model is run offline, timeseries should be provided for all laterals.

# # In[ ]:


# ###For adding the lateral inflow we import SOBEK results. To do so we use hkvsobekpy. For more info on this module, see: https://github.com/HKV-products-services/hkvsobekpy
# # # Add the lateral inflows also from the SOBEK results. Naote that the column names in the his-file need to match
# # # the id's of the imported lateral locations at the top of this notebook.
# rehis = hkvsobekpy.read_his.ReadMetadata(data_path+'/sobekdata/QLAT.HIS', hia_file='auto')
# param = [p for p in rehis.GetParameters() if 'disch' in p][0]
# lateral_discharge = rehis.DataFrame().loc[:, param]
# lateral_discharge.drop('lat_986', inplace=True, axis=1)


# # In[ ]:


# dfmmodel.external_forcings.io.read_laterals(hydamo.laterals, lateral_discharges=lateral_discharge)


# # ### Observation points

# # Observation points are now written in the new format, where once can discriminate between 1D ('1d') and 2D ('2d') observation points. This can be done using the optional argument 'locationTypes'. If it is omitted, all points are assumed to be 1d. 1D-points are always snapped to a the nearest branch. 2D-observation points are always defined by their X/Y-coordinates.
# # 
# # Note: add_points can be called only once: once dfmodel.observation_points is filled,the add_points-method is not available anymore. Observation point coordinates can be definied eiher as an (x,y)-tuple or as a shapely Point-object.

# # In[ ]:


# from shapely.geometry import Point
# dfmmodel.observation_points.add_points([Point((141150, 393700)),(141155, 393705)],['ObsPt1','ObsPt2'], locationTypes=['1d','1d'])


# # ### Settings and writing
# # 
# # Finally, we adjust some settings and export the coupled FM-RR model. For more info on the settings: https://content.oss.deltares.nl/delft3d/manuals/D-Flow_FM_User_Manual.pdf
# # 
# # The 1D/2D model (FM) is written to the sub-folder 'fm'; RR-files are written to 'rr'. An XML-file (dimr-config.xml) describes the coupling between the two. Note that both the GUI and Interaktor do not (yet) support RR, so the only way to carry out a coupled simulation is using DIMR.
# # 

# # In[ ]:


# # Runtime and output settings
# # for FM model
# dfmmodel.mdu_parameters['refdate'] = 20000101
# dfmmodel.mdu_parameters['tstart'] = 0.0 * 3600
# dfmmodel.mdu_parameters['tstop'] = 24.0 * 1 * 3600
# dfmmodel.mdu_parameters['hisinterval'] = '120. 0. 0.'
# dfmmodel.mdu_parameters['cflmax'] = 0.7

# # Create writer
# dfmmodel.dimr_path = dimr_path
# fm_writer = DFlowFMWriter(dfmmodel, output_dir=output_path, name='olo_damo')

# # Write as model
# fm_writer.objects_to_ldb()
# fm_writer.write_all()


# # Finished!

# # In[ ]:




