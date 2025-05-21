#!/usr/bin/env python
# coding: utf-8

# # Example of generating a 1D2DRR D-HYDRO model - an overview of functionalities
# 
# This notebook gives an overview of the functionalities of the D-HyDAMO module, part of the Hydrolib environment.
# 
# This notebook is based on previous examples of the python package delft3dfmpy, but now connnected to the Hydrolib-core package, which is used for writing a D-HYDRO model. It contains similar functionalities as delft3dfmpy 2.0.3; input data is expected to be according to HyDAMO DAMO2.2 gpkg-format. The example model used here is based on a part of the Oostrumsche beek in Limburg, added with some fictional dummy data to better illustrate functionalities.
# 
# Because of the dummy data and demonstation of certain features, the resulting model is not optimal from a hydrologic point of view.
# 
# 

# ## Load Python libraries and Hydrolib-core functionality

# In[1]:


# Basis
from pathlib import Path
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import contextily as cx
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# and from hydrolib-core

sys.path.append('.')
from hydrolib.core.dimr.models import DIMR, FMComponent
from hydrolib.core.dflowfm.inifield.models import IniFieldModel, DiskOnlyFileModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel
from hydrolib.core.dflowfm.structure.models import *
from hydrolib.core.dflowfm.crosssection.models import *
from hydrolib.core.dflowfm.ext.models import ExtModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.bc.models import ForcingModel
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.obs.models import ObservationPointModel
from hydrolib.core.dflowfm.storagenode.models import StorageNodeModel


# In[3]:

from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.geometry.mesh2d_gridgeom import Mesh2D_GG, Rectangular
from hydrolib.dhydamo.geometry.gridgeom.links1d2d import Links1d2d
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from hydrolib.dhydamo.geometry.viz import plot_network


# Define in- and output paths

# In[4]:


# path to the package containing the dummy-data
data_path = Path("hydrolib/tests/data").resolve()
# data_path = Path(r'D:\4390.10\HYDROLIB_devel\hydrolib\notebooks\Limburg')
assert data_path.exists()

# path to write the models
output_path = Path("hydrolib/tests/model").resolve()
# assert output_path.exists()


# Define components that should be used in the model. 1D is used in all cases.

# In[5]:


TwoD = True
RR = False
RTC = True


# Two concepts are available for 2D mesh generation. Use 'GG' for gridgeom (works only in windows, identical to delft3dfmpy), or 'MK' for Meshkernel. Meshkernel is eventually the preferred option (platform-independent, more triangulation options) but has limited support for complex geometries as of now.

# In[6]:


TwoD_option = 'MK' # or 'MK'


# Mesh refinement in the GG method depends on an executable (dflowfm.exe) that is not freely available and is not included in current D-Hydro installation. If you have access to it, this path contains its location.

# In[ ]:

# When 'GG' is used, grid generation certain operations depend on files in your D-Hydro installation. The link below should be adapted for your setup and ensures these files can be used.

# In[ ]:
dll_path = r'C:\Program Files\Deltares\D-HYDRO Suite 2024.03 1D2D\plugins\DeltaShell.Dimr\kernels\x64\lib'
dflowfm_path = r'D:\4390.10\software\dflowfm-x64-1.2.105.67228M\dflowfm.exe'

# This switch defines the mode in which RTC is used. Normally it can handle PID-, interval- and time controllers, but for calibration purposes (for instance) it may be desired to pass historical data, such as actual crest levels, to the model. If True, PID- and interval- and regular (seasonal) time controllers are replaced by time controllers containing a provided time series.
# 
# This does require time series to be available. They should be provided as a file from which a dataframe can be read (here CSV), with structure codes as columns and time steps as row index.

# In[7]:


rtc_onlytimeseries = False
rtc_timeseriesdata = data_path / 'rtc_timeseries.csv'


# ## Read HyDAMO DAMO2.2 data

# In[8]:


# all data is contained in one geopackage called 'Example model'
gpkg_file = str(data_path / "Example_model.gpkg")

# initialize a hydamo object
hydamo = HyDAMO(extent_file=data_path / "Oostrumschebeek_extent.shp")

# show content
hydamo.branches.show_gpkg(gpkg_file)


# In[9]:


# read branches
hydamo.branches.read_gpkg_layer(gpkg_file, layer_name="HydroObject", index_col="code")
# read profiles
hydamo.profile.read_gpkg_layer(
    gpkg_file,
    layer_name="ProfielPunt",
    groupby_column="profiellijnid",
    order_column="codevolgnummer",
    id_col="code",
    index_col="code"
)
hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
hydamo.profile.drop("code", axis=1, inplace=True)
hydamo.profile["code"] = hydamo.profile["profiellijnid"]
hydamo.snap_to_branch_and_drop(hydamo.profile, hydamo.branches, snap_method="intersecting", drop_related=True)

# read structures
hydamo.culverts.read_gpkg_layer(gpkg_file, layer_name="DuikerSifonHevel", index_col="code")
hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw", index_col="code")

hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
# idx = hydamo.management_device[hydamo.management_device["duikersifonhevelid"].notnull()].index
hydamo.snap_to_branch_and_drop(hydamo.culverts, hydamo.branches, snap_method="ends", maxdist=5, drop_related=True)
hydamo.snap_to_branch_and_drop(hydamo.weirs, hydamo.branches, snap_method="overal", maxdist=10, drop_related=True)

hydamo.pumpstations.read_gpkg_layer(gpkg_file, layer_name="Gemaal", index_col="code")
hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")
hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal", maxdist=15, drop_related=True)

hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
hydamo.snap_to_branch_and_drop(hydamo.bridges, hydamo.branches, snap_method="overal", maxdist=1100, drop_related=True)

# read boundaries
hydamo.boundary_conditions.read_gpkg_layer(
    gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
)
hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)


##
if False:
    # assign multiple kunstwerkopeningen to two weirs to test new functionality
    stuwid_1 = hydamo.weirs.iloc[10].globalid
    stuwid_2 = hydamo.weirs.iloc[20].globalid
    num  = 0
    for idx,op in hydamo.opening.iterrows(): 
        if op.stuwid not in hydamo.weirs.globalid.to_list():         
            num+=1
            if num==20 or num == 30 or num==40:
                hydamo.opening.loc[idx, 'stuwid'] = stuwid_1
                new_mandev = hydamo.management_device.iloc[0].to_frame().T
                new_mandev['kunstwerkopeningid'] = op.globalid
                new_mandev['code'] = f'testrm_{num}'
                new_mandev['overlaatonderlaat'] = 'Overlaat'
                hydamo.management_device = pd.concat([hydamo.management_device, new_mandev])
            elif num==50:
                hydamo.opening.loc[idx, 'stuwid'] = stuwid_2
                new_mandev = hydamo.management_device.iloc[0].to_frame().T
                new_mandev['kunstwerkopeningid'] = op.globalid
                new_mandev['code'] = f'testrm_{num}'
                new_mandev['overlaatonderlaat'] = 'Overlaat'
                hydamo.management_device = pd.concat([hydamo.management_device, new_mandev])
            #assert not mandev.empty
    gpd.GeoDataFrame(hydamo.opening).to_file(gpkg_file, layer='kunstwerkopening', driver='GPKG')
    gpd.GeoDataFrame(hydamo.management_device).to_file(gpkg_file, layer='regelmiddel', driver='GPKG')
        

 #In[10]:


# read catchments
hydamo.catchments.read_gpkg_layer(gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code", check_geotype=False)


# In[11]:


# read laterals
hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
# for ind, cat in hydamo.catchments.iterrows():
#     hydamo.catchments.loc[ind, "lateraleknoopid"] = hydamo.laterals[
#         hydamo.laterals.globalid == cat.lateraleknoopid
#     ].code.values[0]
hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)
# hydamo.create_laterals()

# In[12]:


# plot the model objects
plt.rcParams["axes.edgecolor"] = "w"

fig, ax = plt.subplots(figsize=(20,20 ))
xmin,ymin,xmax,ymax=hydamo.clipgeo.bounds
ax.set_xlim(round(xmin), round(xmax))
ax.set_ylim(round(ymin), round(ymax))

hydamo.branches.geometry.plot(ax=ax)#, label="Channel", linewidth=2, color="blue")
hydamo.profile.geometry.plot(ax=ax, color="black", label="Cross section", linewidth=4)
hydamo.culverts.geometry.centroid.plot(
    ax=ax, color="brown", label="Culvert", markersize=40, zorder=10, marker="^"
)
hydamo.weirs.geometry.plot(ax=ax, color="green", label="Weir", markersize=25, zorder=10, marker="^")

hydamo.bridges.geometry.plot(ax=ax, color="red", label="Bridge", markersize=20, zorder=10, marker="^")
hydamo.pumpstations.geometry.plot(
    ax=ax,
    color="orange",
    label="Pump",
    marker="s",
    markersize=125,
    zorder=10,
    facecolor="none",
    linewidth=2.5,
)
hydamo.boundary_conditions.geometry.plot(
    ax=ax, color="red", label="Boundary", marker="s", markersize=125, zorder=10, facecolor="red", linewidth=0
)
ax.legend()

# cx.add_basemap(ax, crs=28992, source=cx.providers.OpenStreetMap.Mapnik)
fig.tight_layout()


# ## Data conversion
# 

# ### Structures

# HyDAMO contains methods to convert HyDAMO DAMO data to internal dataframes, which correspond to the D-HYDRO format.
# 
# We first import the structures from the HyDAMO-object, since the structures' positions are necessary for defining the 1D-mesh. Structures can also be added without the HyDAMO imports.
# 
# Note that for importing most structures multiple objects are needed from the GPKG. For more info on how to add structures (directly or from HyDAMO), see: https://hkvconfluence.atlassian.net/wiki/spaces/DHYD/overview.
# 
#  - for weirs, a corresponding profile is looked up in the crossections. If one is found, the weir is implemented as a universal weir. If it is not found, a regular (rectangular) weir will be used. The cross-section ('hydamo.profile') should be related through 'hydamo.profile_line' to a 'hydamo.profile_group', which contains a 'stuwid' column which is equal to the GlobalID of the corresponding weir. The weir object can also include orifices, in that case the field 'overlaatonderlaat' in the 'management_device-object ('regelmiddel') is 'onderlaat'. For weirs it should be 'overlaat'. For regular weirs and orifices the crestlevel is inferred from the 'laagstedoorstroomhoogte' field in the hydamo.opening object. For universal weirs, this is also the case; however if no 'laagstedoorstroomhoogte' is present, or it is None, the crest level is derived from the lowest level of the associated profile. 
#  
#  - for culverts, a regelmiddel can be used to model a 'schuif' and/or a 'terugslagklep'. This is specified by the field 'soortregelmiddel'. In case of a 'terugslagklep', the flow direction is set to 'positive' instead of 'both'. In case of a 'schuif', a valve is implemented. Note that in DAMO 2.2, an 'afsluitmiddel' can contain the same information. For now, only a regelmiddel (management_device) is implemented.
#  
#  - bridges need an associated crosssection. This is idential to universal weirs, but here the 'hydamo.profile_group'-object should contain a field 'brugid'. 
#  
#  - pumps are composed from 'hydamo.pumpstations', 'hydamo.pumps' and 'hydamo.managmement'. Only suction-side direction is implemented. Maximal capacity should be in m3/min.
# 
# In most cases, these 'extra' arguments are optional, i.e. they are not required and can be left out. Some are required:
# - pumps really need all 3 objects ('hydamo.pumpstations', 'hydamo.pumps' and 'hydamo.managmement');
# - bridges really need an associated crosssection (see above);
# 
# For more info on the structure definitions one is referred to the D-Flow FM user manual: https://content.oss.deltares.nl/delft3d/manuals/D-Flow_FM_User_Manual.pdf.

# In[13]:


hydamo.structures.convert.weirs(
    weirs=hydamo.weirs,
    profile_groups=hydamo.profile_group,
    profile_lines=hydamo.profile_line,
    profiles=hydamo.profile,
    opening=hydamo.opening,
    management_device=hydamo.management_device,
)
hydamo.structures.convert.culverts(culverts=hydamo.culverts, management_device=hydamo.management_device)

hydamo.structures.convert.bridges(
    bridges=hydamo.bridges,
    profile_groups=hydamo.profile_group,
    profile_lines=hydamo.profile_line,
    profiles=hydamo.profile,
)

hydamo.structures.convert.pumps(pumpstations=hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management)


# Additional methods are available to add structures:

# In[14]:


hydamo.structures.add_rweir(
    id="rwtest",
    name="rwtest",
    branchid="W_1386_0",
    chainage=2.0,
    crestlevel=12.5,
    crestwidth=3.0,
    corrcoeff=1.0,
)
hydamo.structures.add_orifice(
    id="orifice_test",
    branchid="W_242213_0",
    chainage=43.0,
    crestlevel=18.00,
    gateloweredgelevel=18.5,
    crestwidth=7.5,
    corrcoeff=1.0,
)
hydamo.structures.add_uweir(
    id="uweir_test",
    branchid="W_242213_0",
    chainage=2.0,
    crestlevel=18.00,
    crestwidth=7.5,
    dischargecoeff=1.0,
    numlevels=4,
    yvalues="0.0 1.0 2.0 3.0",
    zvalues="19.0 18.0 18.2 19",
)
hydamo.structures.add_culvert(
    id="culvert_test",
    branchid="W_242213_0",
    chainage=42.0,
    rightlevel=17.2,
    leftlevel=17.1,
    length=5.,
    bedfrictiontype='StricklerKs',
    bedfriction=75,
    inletlosscoeff=0.6,
    outletlosscoeff = 1.0,
    crosssection = {'shape':'circle', 'diameter':0.6}    
)



# The resulting dataframes look like this:

# In[15]:


hydamo.structures.culverts_df.head()


# Indicate structures that are at the same location and should be treated as a compound structure. The D-Hydro GUI does this automatically, but for DIMR-calculations this should be done here.

# In[16]:


cmpnd_ids = ["cmpnd_1","cmpnd_2","cmpnd_3"]
cmpnd_list = [["D_24521", "D_14808"],["D_21450", "D_19758"],["D_19757", "D_21451"]]
hydamo.structures.convert.compound_structures(cmpnd_ids, cmpnd_list)


# # Initialize the FM-model

# At this stage also the start and stoptime are defined, they will be used in the other modules if needed

# In[17]:


fm = FMModel()
# Set start and stop time
fm.time.refdate = 20160601
fm.time.tstop = 2 * 3600 * 24


# ## Add observation points

# Observation points are now written in the new format, where one can discriminate between 1D ('1d') and 2D ('2d') observation points. This can be done using the optional argument 'locationTypes'. If it is omitted, all points are assumed to be 1d. 1D-points are always snapped to a the nearest branch. 2D-observation points are always defined by their X/Y-coordinates.
# 
# Note: add_points can be called only once: once dfmodel.observation_points is filled,the add_points-method is not available anymore. Observation point coordinates can be definied eiher as an (x,y)-tuple or as a shapely Point-object.
# 
# Observation points need to be defined prior to constructing the 1d mesh, as they will be separated from structures and each other by calculation points.

# In[18]:


hydamo.observationpoints.add_points(
    [Point(199617,394885), Point(199421,393769), Point(199398,393770)],
    ["Obs_BV152054", "ObsS_96684","ObsO_test"],
    locationTypes=["1d", "1d", "1d"],
    snap_distance=10.0,
)

hydamo.observationpoints.add_points(
    [Point(200198,396489), Point(201129, 396269), Point(200264, 394761), Point(199665, 395323)],
    ["ObsS_96544", "ObsP_113GIS", 'Obs_UWR', 'Obs_ORIF'],
    locationTypes=["1d", "1d","1d", "1d"],
    snap_distance=10.0,
)
hydamo.observationpoints.observation_points.head()


# # ## Add the 1D mesh

# The above structures are collected in one dataframe and in the generation of calculation poins, as structures should be separated by calculation points.

# In[19]:


structures = hydamo.structures.as_dataframe(
    rweirs=True,
    bridges=True,
    uweirs=True,
    culverts=True,
    orifices=True,
    pumps=True,
)


# Include also observation points

# In[20]:


objects = pd.concat([structures, hydamo.observationpoints.observation_points], axis=0)

# objects = structures
# In[21]:


mesh.mesh1d_add_branches_from_gdf(
    fm.geometry.netfile.network,
    branches=hydamo.branches,
    branch_name_col="code",
    node_distance=20,
    max_dist_to_struc=None,
    structures=objects,
)


# Add cross-sections to the branches. To do this, many HyDAMO objects might be needed: if parameterised profiles occur, they are taken from hydamo.param_profile and, param_profile_values; if crosssections are associated with structures, those are specified in profile_group and profile lines. 
# 
# HyDAMO DAMO2.2 data contains two roughness values (high and low); here it can be specified which one to use.
# For branches without a crosssection, a default profile can be defined.

# Partly, missing crosssections can be resolved by interpolating over the main branch. We set all branches with identical names to the same order numbers and assign those to the branches. D-Hydro will then interpolate the cross-sections over the branches.
# 

# In[22]:


# Here roughness variant "High" ("ruwheidhoog" in HyDAMO) is chosen. Variant "Low" ("ruwheidlaag" in HyDAMO) can also be chosen
# Here roughness variant "High" ("ruwheidhoog" in HyDAMO) is chosen. Variant "Low" ("ruwheidlaag" in HyDAMO) can also be chosen
hydamo.crosssections.convert.profiles(
    crosssections=hydamo.profile,
    crosssection_roughness=hydamo.profile_roughness,
    profile_groups=hydamo.profile_group,
    profile_lines=hydamo.profile_line,
    param_profile=hydamo.param_profile,
    param_profile_values=hydamo.param_profile_values,
    branches=hydamo.branches,
    roughness_variant="High",
)

# Check how many branches do not have a profile.

# In[23]:


missing = hydamo.crosssections.get_branches_without_crosssection()
print(f"{len(missing)} branches are still missing a cross section.")


# We plot the missing ones.

# In[24]:


# plt.rcParams['axes.edgecolor'] = 'w'
# fig, ax = plt.subplots(figsize=(16, 16))
# xmin,ymin,xmax,ymax=hydamo.clipgeo.bounds
# ax.set_xlim(round(xmin), round(xmax))
# ax.set_ylim(round(ymin), round(ymax))
# hydamo.profile.geometry.plot(ax=ax, color='black', label='dwarsprofielen', linewidth=5)
# hydamo.branches.loc[missing,:].geometry.plot(ax=ax, color='C4', label='geen dwarsprofiel',linewidth=10)
# hydamo.branches.geometry.plot(ax=ax, label='Watergangen')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.legend()
# cx.add_basemap(ax, crs=28992, source=cx.providers.OpenStreetMap.Mapnik)
# fig.tight_layout()


# One way to assign crosssections to branches is to assigning order numbers to branches, so the crosssections are interpolated over branches with the same order number. The functino below aassigns the same order number to branches with a common attribute of hydamo.branches. In this example, all branches with the same 'naam' are given the same order number. 
# 
# Only branches that are in the 'missing' list (and do not have a crosssection) are taken into account.
# 
# There are exceptions: for instance the branch "Retentiebekken Rosmolen" has a name, and therefore an ordernumber, but it cannot be interpolated since it consists of only one segment. Its branch-id is W_1386_0. We pass a list of exceptions like that, making sure that no order numbers will be assigned to those branches.

# Retentiebekken Rosmolen has a name, and therefore an ordernumber, but cannot be interpolated. Set the order to 0, so it gets a default profile.

# In[25]:


missing_after_interpolation = mesh.mesh1d_order_numbers_from_attribute(hydamo.branches, 
                                                                       missing, 
                                                                       order_attribute='naam', 
                                                                       network=fm.geometry.netfile.network,  
                                                                       exceptions=['W_1386_0'])


# In[26]:


print(f"After interpolation, {len(missing_after_interpolation)} branches are still missing a cross section.")


# In[27]:


plt.rcParams['axes.edgecolor'] = 'w'
fig, ax = plt.subplots(figsize=(16, 16))
xmin,ymin,xmax,ymax=hydamo.clipgeo.bounds
ax.set_xlim(round(xmin), round(xmax))
ax.set_ylim(round(ymin), round(ymax))
hydamo.profile.geometry.plot(ax=ax, color='C3', label='dwarsprofielen', linewidth=5)
hydamo.branches.loc[missing_after_interpolation,:].geometry.plot(ax=ax, color='C4', label='geen dwarsprofiel',linewidth=10)
hydamo.branches.geometry.plot(ax=ax, label='Watergangen')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.legend()
# cx.add_basemap(ax, crs=28992, source=cx.providers.OpenStreetMap.Mapnik)
fig.tight_layout()


# For these ones, we apply a default profile. In this case an yz-profile, but it can also be a rectangular or other type or profile.

# In[28]:


# Set a default cross section
profiel=np.array([[0,21],[2,19],[7,19],[9,21]])
default = hydamo.crosssections.add_yz_definition(yz=profiel, 
                                                   thalweg = 4.5,
                                                   roughnesstype='StricklerKs',
                                                   roughnessvalue=25.0, 
                                                   name='default'
                                                  )

hydamo.crosssections.set_default_definition(definition=default, shift=0.0)
hydamo.crosssections.set_default_locations(missing_after_interpolation)


# ### Boundary conditions

# The HyDAMO database contains constant boundaries. They are added to the model:

# In[29]:

    # def add_storagenode(
    #     self,
    #     id,
    #     nodeid,
    #     usestreetstorage="true",
    #     nodetype="unspecified",
    #     name=np.nan,
    #     usetable="false",
    #     bedlevel=np.nan,
    #     area=np.nan,
    #     streetlevel=np.nan,
    #     streetstoragearea=np.nan,
    #     storagetype="reservoir",
    #     levels=np.nan,
    #     storagearea=np.nan,
    #     interpolate="linear",
    # ):

# storage nodes
f_storage_areas = data_path / 'bergingspolygonen.shp'
hydamo.storage_areas.read_shp(f_storage_areas,index_col='code')
hydamo.storage_areas.snap_to_branch(hydamo.branches, snap_method='centroid', maxdist=1000)
storage_node_data = pd.read_excel(data_path / 'storage_data.xls')
storage_node_data= storage_node_data.rename(columns={'GEBIEDID': 'code', 'OPPERVLAK':'area', 'MAAIVELD': 'level'})
hydamo.storage_areas['name'] = hydamo.storage_areas['code'] 


hydamo.storagenodes.convert.storagenodes_from_input(storagenodes=hydamo.storage_areas, storagedata=storage_node_data, network=fm.geometry.netfile.network)

hydamo.storagenodes.add_storagenode(       
        id='sto_test',
        xy=(141001, 395030),                
        name='sto_test',
        usetable="true",
        levels=' '.join(np.arange(17.1, 19.6, 0.1).astype(str)),
        storagearea=' '.join(np.arange(100, 1000, 900/25.).astype(str)),
        interpolate="linear",
        network=fm.geometry.netfile.network)

hydamo.external_forcings.convert.boundaries(hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network)


# However, we also need an upstream discharge boundary, which is not constant. We add a fictional time series, which can be read from Excel as well:

# In[30]:


series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 1.0)
series.index = [pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)]
series.plot()


# There is also a fuction to convert laterals, but to run this we also need the RR model. Therefore, see below. It also possible to manually add boundaries and laterals as constants or timeseries. We implement the sinoid above as an upstream streamflow boundary and a lateral:

# In[31]:


hydamo.external_forcings.add_boundary_condition(
    "RVW_01", (197464.0, 392130.0), "dischargebnd", series, fm.geometry.netfile.network
)


# In[32]:


hydamo.dict_to_dataframe(hydamo.external_forcings.boundary_nodes)


# In the same way as for boundaries, it is possible to add laterals using a fuction. Here are two examples. The first one adds a constant flow (1.5 m3/s) to a location on branch W_2399.0:

# In[33]:


hydamo.external_forcings.add_lateral(id='LAT01', branchid='W_2399_0', chainage=100., discharge=1.5 )


# The second example adds a timeseries, provided as a pandas series-object, to another location on the same branch.It can be read from CSV, but here we pass a hypothetical sine-function.

# In[34]:


series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 2.0)
series.index = [pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)]
hydamo.external_forcings.add_lateral(id='LAT02', branchid='W_2399_0', chainage=900., discharge=series )


# ### Initial conditions

# Set the initial water depth to 0.5 m. It is also possible to set a global water level using the equivalent function "set_initial_waterlevel".
# 
# <span style='color:Red'> WARNING: The path of the initialwaterdepth.ini file in fieldFile.ini is not an absolute path, change or the model is not portable!  </span>
# <span style='color:Red'> This is a known issue for the HYDROLIB-core package and will be adressed in future  </span>

# In[35]:


hydamo.external_forcings.set_initial_waterdepth(1.5)


# ### 2D mesh

# As explained above, several options exist for mesh generation.
# 
# 1. Meshkernel (MK). This is eventually the preferred option: it is platform-independent, supports triangular meshes and has orthogenalization functionality. However, with certain input geometries issues occur.
# 2. Gridgeom (GG). Older functionality from delft3dfmpy. Only rectangular meshes are supported, but it is more stable and widely used.
# 3. Import a mesh from other software, such as SMS.
# 
# To generate a mesh we illustrate the following steps:
#     - Generate grid within a polygon. The polygon is the extent given to the HyDAMO model.
#     - Refine along the main branch
#     - clip the mesh around the branches
#     - Determine altitude from a DEM. 
# 
# <span style='color:Blue'> Important note: Triangular meshes are created without optimalization of smoothness or orthogonalization. This can be handled in de D-HYDRO GUI. In future we will implement this in D-HyDAMO.  </span>

# In[36]:


if TwoD:    
    extent = gpd.read_file(data_path / "2D_extent.shp").at[0, "geometry"]
    network = fm.geometry.netfile.network
    cellsize=50.
    rasterpath = data_path / 'rasters/AHN_2m_clipped_filled.tif'


# Add a rectangular mesh::

# In[37]:


if TwoD:
    if TwoD_option == 'MK':
        mesh.mesh2d_add_rectilinear(network, extent, dx=cellsize, dy=cellsize)
    elif TwoD_option == 'GG':
        mesh_gg = Rectangular(dll_path=dll_path)        
        mesh_gg.generate_within_polygon(extent, cellsize=cellsize, rotation=0)


# With MK, also the creation of a triangular mesh is possible. Note that there are maybe issues with the mesh orthogenality and additional steps in the D-Hydro GUI to orthogenalize the mesh may be necessary. In future we will implement this functionality in D-HyDAMO.

# In[38]:


# if TwoD and TwoD_option=='MK':
#     mesh.mesh2d_add_triangular(network, extent, edge_length=50.0)


# Refine the 2D mesh within an arbitrary distance of 50 m from all branches (works on a subselection as well). Note that refining only works for a polygon without holes in it, so the exterior geometry is used.

# In[39]:


# if TwoD:
    # buffer = Polygon(hydamo.branches.buffer(75.0).union_all().exterior)
    # if TwoD_option == 'MK':
    #     print("Nodes before refinement:", network._mesh2d.mesh2d_node_x.size)        
    #     mesh.mesh2d_refine(network, 
    #                        buffer, 
    #                        steps=2)
                                             
    #     print("Nodes after refinement:", network._mesh2d.mesh2d_node_x.size)       
    # elif TwoD_option == 'GG':
    #     print("Nodes before refinement:", len(mesh_gg.meshgeom.get_values('nodex')))
    #     mesh_gg.refine(polygon=[buffer], level=[1], cellsize=cellsize, dflowfm_path=dflowfm_path)
    #     print("Nodes after refinement:", len(mesh_gg.meshgeom.get_values('nodex')))       


# # Clip the 2D mesh in a 20m buffer around all branches. The algorithm is quite sensitive to the geometry, for example holes are not allowed.

# # In[40]:


# if TwoD:
#     if TwoD_option == 'MK':
#         pass        
#         print("Nodes before clipping:", network._mesh2d.mesh2d_node_x.size)
#         # uitknippen = gpd.read_file(data_path / "2D_extent.shp").geometry.union_all().difference(hydamo.branches.geometry.union_all().buffer(10.))
#         uitknippen = hydamo.branches.geometry.union_all().buffer(25.)
#         import meshkernel as mk
#         mesh.mesh2d_clip(network, uitknippen, deletemeshoption= mk.DeleteMeshOption.FACES_WITH_INCLUDED_CIRCUMCENTERS)    
#         print("Nodes after clipping:", network._mesh2d.mesh2d_node_x.size)       
#     elif TwoD_option == 'GG':
#         print("Nodes before clipping:", len(mesh_gg.meshgeom.get_values('nodex')))
#         mesh_gg.clip_mesh_by_polygon(hydamo.branches.union_all().buffer(10.))
#         print("Nodes after clipping:",  len(mesh_gg.meshgeom.get_values('nodex')))      


# Alternatively, the mesh can be read from a netcdf file (option 3):

# In[41]:


# if TwoD:
#     if TwoD_option == 'MK': 
#         mesh.mesh2d_from_netcdf(network, data_path / "import.nc")
#     elif TwoD_option == 'GG':
#         mesh_gg.geom_from_netcdf(mesh_gg, data_path / "import.nc", only2d=True)


# Add elevation data to the cells

# In[42]:


if TwoD:        
    if TwoD_option == 'MK': 
        mesh.mesh2d_altitude_from_raster(network, rasterpath, "face", "mean", fill_value=-999)
    elif TwoD_option == 'GG':
        mesh_gg.altitude_from_raster(rasterpath)


# ### Add 1d-2d links

# Three options exist to add 1d2d links to the network:
#  - from 1d to 2d
#  - from 2d to 1d embedded: allowing overlap (i.e., the 2D cells and 1D branches are allowed to intersect)
#  - from 2d to 1d lateral: there is no overlap and from each cell the closest 1d points are used. 
#  
#  See https://hkvconfluence.atlassian.net/wiki/spaces/DHYD/pages/601030709/1D2D-links for details.

# In[43]:


if TwoD:
    if TwoD_option == 'GG':
        # initialize and clear 1d2d-object. 
        links1d2d = Links1d2d( network, mesh_gg.meshgeom, hydamo)
        del links1d2d.faces2d[:]
        del links1d2d.nodes1d[:]
        
        # Generate 1D2D links with a maximum length of 50 m
        links1d2d.generate_1d_to_2d(max_distance=50)
        # links1d2d.generate_2d_to_1d(max_distance=50, intersecting=False)
        
        # # No links will be generated from 1D nodes with a boundary condition
        links1d2d.remove_1d_endpoints()        
        
        # add the gridgeom mesh and 1d2d links to the fm-network

        links1d2d.convert_to_hydrolib()                    

        # mesh.links1d2d_add_links_1d_to_2d(network)
        # mesh.links1d2d_add_links_2d_to_1d_lateral(network, max_length=50.)
        # mesh.links1d2d_add_links_2d_to_1d_embedded(network)
        # mesh.links1d2d_remove_1d_endpoints(network)


    elif TwoD_option=='MK':
        # mesh.links1d2d_add_links_1d_to_2d(network)
        # mesh.links1d2d_add_links_2d_to_1d_lateral(network, max_length=50.)
        mesh.links1d2d_add_links_2d_to_1d_embedded(network)
        mesh.links1d2d_remove_1d_endpoints(network)


# In[44]:


# plot the network
if TwoD:
    network = fm.geometry.netfile.network
    fig, axs = plt.subplots(figsize=(13.5, 6), ncols=2, constrained_layout=True)
    plot_network(network, ax=axs[0])
    plot_network(network, ax=axs[1], links1d2d_kwargs=dict(lw=3, color="k"))
#     for ax in axs:
#         ax.set_aspect(1.0)
#         ax.plot(*buffer.exterior.coords.xy, color="k", lw=0.5)
    axs[0].autoscale_view()
    axs[1].set_xlim(199200, 200000)
    axs[1].set_ylim(394200, 394700)

    sc = axs[1].scatter(
        x=network._mesh2d.mesh2d_face_x,
        y=network._mesh2d.mesh2d_face_y,
        c=network._mesh2d.mesh2d_face_z,
        s=10,
        vmin=22,
        vmax=27,
    )
    cb = plt.colorbar(sc, ax=axs[1])
    cb.set_label("Face level [m+NAP]")

    plt.show()


# For finalizing the FM-model, we also need the coupling to the other modules. Therefore, we will do that first.

# # Add an RTC model

# RTC contains many different options. These are now implemented in D-HyDAMO: 
# - a PID controller (crest level is determined by water level or discharge at an observation point);
# - an interval controller (idem, but with different options);
# - a time controller (a time series of crest level is provided);
# - the possibility for the users to provide their own XML-files for more complex cases. Depending on the complexity, the integration might not yet work for all cases.

# First, initialize a DRTCModel-object. The input is hydamo (for the data), fm (for the time settings), a path where the model will be created (typically an 'rtc' subfolder), a timestep (default 60 seconds) and, optionally, a folder where the user can put 'custom' XML code that will be integrated in the RTC-model. These files will be parsed now and be integrated later.
# 
# These files can, for example, be obtained by sc

# In[45]:


if RTC:
    if rtc_onlytimeseries:
        timeseries = pd.read_csv(rtc_timeseriesdata, sep=",", index_col='Time', parse_dates=True)
        drtcmodel = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            rtc_timestep=60.0,
            rtc_onlytimeseries=True,
            rtc_timeseriesdata=timeseries
        )
    else:
        drtcmodel = DRTCModel(
            hydamo,
            fm,
            output_path=output_path,
            rtc_timestep=60.0,
            complex_controllers_folder=data_path / "complex_controllers"  # location where user defined XLM-code should be located
        )


# If PID- or interval controllers are present, they need settings that are not included in the HyDAMO DAMO2.2 data. We define those in dictionaries. They can be specified for each structure - in that case the key of the dictionary should match the key in the HyDAMO DAMO2.2 'sturing'-object. If no separate settings are provided the 'global' settings are used.
# 
# Note that it is also possible to add RTC objects individually, in that case the settings should provided in the correct function as shown below. When interval controllers are converted from HyDAMO data, the settings above and below the deadband are taken from the min- and max setting.

# In[46]:


if RTC and not rtc_onlytimeseries:
    pid_settings = {}
    interval_settings = {}
    
    pid_settings["global"] = {
        "ki": 0.001,
        "kp": 0.00,
        "kd": 0.0,
        "maxspeed": 0.00033,
    }
    interval_settings["global"] = {
        "deadband": 0.2,
        "maxspeed": 0.00033,
    }    
    pid_settings["kst_pid"] = {
            "ki": 0.001,
            "kp": 0.0,
            "kd": 0.0,
            "maxspeed": 0.00033,
        }


# The function 'from_hydamo' converts the controllers that are specified in the HyDAMO DAMO2.2 data. The extra input consists of the settings for PID controllers (see above) and a dataframe with time series for the time controllers.

# In[47]:


if RTC and not rtc_onlytimeseries:
    if not hydamo.management.typecontroller.empty:
        timeseries = pd.read_csv(data_path / "timecontrollers.csv")
        timeseries.index = timeseries.Time        

        drtcmodel.from_hydamo(pid_settings=pid_settings, interval_settings=interval_settings, timeseries=timeseries)


# Additional controllers, that are not included in D-HyDAMO DAMO2.2 might be specified like this:

# In[48]:


if RTC and not rtc_onlytimeseries:
    drtcmodel.add_time_controller(
        structure_id="S_96548", steering_variable="Crest level (s)", data=timeseries.loc[:,'S_96548']
    )


# In[49]:


if RTC and not rtc_onlytimeseries:
    # construct a times series to use as a PID controller setpoint
    series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 13.0)
    series.index = [pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)]
    drtcmodel.add_pid_controller(structure_id='S_96544', 
                                observation_location='ObsS_96544', 
                                steering_variable='Crest level (s)', 
                                target_variable='Water level (op)', 
                                setpoint=series,
                                ki = 0.001,
                                kp =0.,
                                kd =0,
                                max_speed = 0.00033,
                                upper_bound=13.4,
                                lower_bound=12.8,
                                interpolation_option = 'LINEAR',
                                extrapolation_option = "BLOCK"
                                )
    
    # define an interval controller with a constant setpoint for an orifice gate
    drtcmodel.add_interval_controller(structure_id='orifice_test', 
                                observation_location='ObsO_test', 
                                steering_variable='Gate lower edge level (s)', 
                                target_variable='Discharge (op)', 
                                setpoint=13.2,
                                setting_below=12.8,
                                setting_above=13.4,
                                max_speed=0.00033,
                                deadband=0.1,
                                interpolation_option = 'LINEAR',
                                extrapolation_option = "BLOCK"
                                )
    
    # and add a PID controller to a pump capacity
    series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 13.0)
    series.index = [pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)]    
    drtcmodel.add_pid_controller(structure_id='113GIS', 
                                observation_location='ObsP_113GIS', 
                                steering_variable='Capacity (p)', 
                                target_variable='Water level (op)', 
                                setpoint=series,
                                ki = 0.001,
                                kp =0.,
                                kd =0,
                                max_speed = 0.00033,
                                upper_bound=0.4,
                                lower_bound=0.2
                                )


# ## Add a rainfall runoff model

# RR has not changed yet compared to delft3dfmpy. Initialize a model:

# In[50]:


if RR:
    drrmodel = DRRModel()


# Catchments are provided in the HyDAMO DAMO2.2 format and included in the GPKG. They can also be read from other formats using 'read_gml', or 'read_shp'. Note that in case of shapefiles column mapping is necessary because the column names are truncated. 
# 
# Note that when catchments have a "MultiPolygon' geometry, the multipolygons are 'exploded' into single polygon geometries. A warning of this is isued, and a suffix is added to every polygons ID to prevent duplicates. 
# 
# For every catchment, the land use areas will be calculated and if appopriate a maximum of four RR-nodes will be created per catchment:
#  - unpaved (based on the Ernst concept)
#  - paved 
#  - greenhouse
#  - open water (not the full Sobek2 open water, but only used to transfer (net) precipitation that falls on open water that is schematized in RR to the 1D/2D network.
#  
# At the moment, two options exist for the schematisation of the paved area:
#  1) simple: the paved fraction of each catchment is modelled with a paved node, directly connected to catchments' boundary node
#  <br>
#  2) more complex: sewer area polygons and overflow points are used a input as well. For each sewer area, the overlapping paved area is the distributed over the overflows that are associated with the sewerarea (the column 'lateraleknoopcode') using the area fraction (column 'fractie') for each overflow. In each catchment, paved area that does not intersect with a sewer area gets an unpaved node as in option (1).
# 

# Load data and settings. RR-parameters can be derived from a raster (using zonal statistics per catchment), or provided as a standard number. Rasters can be in any format that is accepted by the package rasterio: https://gdal.org/drivers/raster/index.html. All common GIS-formats (.asc, .tif) are accepted.

# In[51]:


if RR:
    lu_file = data_path / "rasters" / "sobek_landuse2.tif"
    ahn_file = data_path / "rasters" / "AHN_2m_clipped_filled.tif"
    soil_file = data_path / "rasters" / "sobek_soil.tif"
    surface_storage = 10.0 # [mm]
    infiltration_capacity = 100.0 # [mm/hr]
    initial_gwd = 1.2  # water level depth below surface [m]  
    runoff_resistance = 0.5 # [d]
    infil_resistance = 300.0 # [d]
    layer_depths = [0.0, 1.0, 2.0] # [m]
    layer_resistances = [30, 200, 10000] # [d]


# A different meteo-station can be assigned to each catchment, of a different shape can be provided. Here, 'meteo_areas' are assumed equal to the catchments.

# In[52]:


if RR:
      meteo_areas = hydamo.catchments 


# ## Unpaved nodes

# For land use and soil type a coding is prescribed. For landuse, the legend of the map is expected to be as follows: <br>
#  1 potatoes  <br>
#  2 wheat<br>
#  3 sugar beet<br> 
#  4 corn       <br> 
#  5 other crops <br> 
#  6 bulbous plants<br> 
#  7 orchard<br>
#  8 grass  <br>
#  9 deciduous forest  <br>
# 10 coniferous forest<br>
# 11 nature<br>
# 12 barren<br>
# 13 open water<br>
# 14 built-up<br>
# 15 greenhouses<br>
# 
# For classes 1-12, the areas are calculated from the provided raster and remapped to the classification in the Sobek RR-tables.
# 
# 
# The coding for the soil types:<br>
# 1 'Veengrond met veraarde bovengrond'<br>
#  2 'Veengrond met veraarde bovengrond, zand'<br>
#  3 'Veengrond met kleidek'<br>
#  4 'Veengrond met kleidek op zand'<br>
#  5 'Veengrond met zanddek op zand'<br>
#  6 'Veengrond op ongerijpte klei'<br>
#  7 'Stuifzand'<br>
#  8 'Podzol (Leemarm, fijn zand)'<br>
#  9 'Podzol (zwak lemig, fijn zand)'<br>
# 10 'Podzol (zwak lemig, fijn zand op grof zand)'<br>
# 11 'Podzol (lemig keileem)'<br>
# 12 'Enkeerd (zwak lemig, fijn zand)'<br>
# 13 'Beekeerd (lemig fijn zand)'<br>
# 14 'Podzol (grof zand)'<br>
# 15 'Zavel'<br>
# 16 'Lichte klei'<br>
# 17 'Zware klei'<br>
# 18 'Klei op veen'<br>
# 19 'Klei op zand'<br>
# 20 'Klei op grof zand'<br>
# 21 'Leem'<br>
# 
# 
# And surface elevation needs to be in m+NAP.

# In[53]:



# In[59]:
if RR:
    roof_storage = 5.0 # [mm] 
    basin_storage_class = 3
    hydamo.greenhouse_areas.read_gpkg_layer( data_path / 'greenhouses.gpkg', layer_name='greenhouses', index_col='code')
    hydamo.greenhouse_laterals.read_gpkg_layer(data_path / 'greenhouse_laterals.gpkg', layer_name='laterals')
    hydamo.greenhouse_laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
    hydamo.greenhouse_areas['roof_storage_mm'] = roof_storage 
        # kas_2 and kas_4 get class 7
    hydamo.greenhouse_areas.loc[['kas_2', 'kas_4'],'basin_storage_class'] = 7    
    # kas_3 and kas_5 get class 5.
    hydamo.greenhouse_areas.loc[['kas_3','kas_5'],'basin_storage_class'] = 5

    hydamo.catchments['boundary_node'] = [hydamo.laterals[hydamo.laterals.globalid==c['lateraleknoopid']].code.values[0] for _,c in hydamo.catchments.iterrows()]
    
if RR:
    drrmodel.unpaved.io.unpaved_from_input(
        hydamo.catchments,
        lu_file,
        ahn_file,
        soil_file,
        surface_storage,
        infiltration_capacity,
        initial_gwd,
        meteo_areas,
        greenhouse_areas=hydamo.greenhouse_areas
    )
    drrmodel.unpaved.io.ernst_from_input(
        hydamo.catchments,
        depths=layer_depths,
        resistance=layer_resistances,
        infiltration_resistance=infil_resistance,
        runoff_resistance=runoff_resistance,
    )    


# ## Paved nodes

# Default parameter values for paved nodes. There are three options to give a sewage properies: (pump-overcapacity and sewer storage):
# 1. a default value in mm/h that will be used for every paved node and converted to m3/s;
# 2. a raster with values in mm/h. For each paved node zonal statistics will be used to extract to correct value that will be converted to m3/s;
# 3. as attributes to the sewer areas, where each area will have a storage in mm and POC in m3/s. The pump capacity will be divided over the related overflows.
# 

# In[54]:


if RR:    
    street_storage = 5.0 # [mm]
    sewer_storage = 5.0 # [mm]
    poc_mmh  = data_path / 'rasters/pumpcap.tif'
    # poc_mmh = 0.7 # [mm/h]

    hydamo.sewer_areas.read_shp(str(data_path / 'rioleringsgebieden.shp'), index_col='code', column_mapping={'Code':'code', 'Berging_mm':'riool_berging_mm', 'POC_m3s':'riool_poc_m3s' })
    hydamo.overflows.read_shp(str(data_path / 'overstorten.shp'), column_mapping={'codegerela': 'codegerelateerdobject'})
    hydamo.overflows.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)


# In[57]:
    drrmodel.paved.io.paved_from_input(
            catchments=hydamo.catchments,
            landuse=lu_file,
            surface_level=ahn_file,
            sewer_areas=hydamo.sewer_areas,
            overflows=hydamo.overflows,
            street_storage=street_storage,
            sewer_storage=sewer_storage,
            pump_capacity=poc_mmh,
            meteo_areas=meteo_areas,
            zonalstats_alltouched=True,
        )


# For paved areas, two options are allowed.
# 1) simply assign a paved noded to the catchment area that is paved in the landuse map.

if RR:
    
   
   
    # standard greenhouses    
    drrmodel.greenhouse.io.greenhouse_from_input(
            catchments=hydamo.catchments,
            landuse=lu_file,
            surface_level=ahn_file,
            greenhouse_areas=hydamo.greenhouse_areas,
            greenhouse_laterals=hydamo.greenhouse_laterals,
            roof_storage=roof_storage,
            basin_storage_class=basin_storage_class,
            meteo_areas=meteo_areas,
            zonalstats_alltouched=True            
        )

   

# In[55]:


# if RR:
#     drrmodel.paved.io.paved_from_input(
#             catchments=hydamo.catchments,
#             landuse=lu_file,
#             surface_level=ahn_file,
#             street_storage=street_storage,
#             sewer_storage=sewer_storage,
#             pump_capacity=pumpcapacity,
#             meteo_areas=meteo_areas,
#             zonalstats_alltouched=True,
#         )


#  2. Also use sewer-areas and overflows by providing them to the function. In that case, the 'overflows' shapefile should have a field 'codegerelateerdobject' that contains the 'code' of the sewer area it is linked to, and a 'fraction' (float) that contains the fraction of the sewer area that drains through that overflow. 
# 
# For every overflow, a paved node is created, containing the fraction of the sewer area. The paved area of the catchment that intersects the sewer-area is corrected for this; for the remaining paved area a seperate paved node is created.
# 
# If sewer areas contain POC and storage values, the fields are mapped to 'riool_berging_mm' and 'riool_poc_m3s'.

# In[56]:



# ## Greenhouse nodes

# In[58]:
 

# ## Open water

# As opposed to Sobek, in D-Hydro open water is merely an interface for precpitation and evaporation. No management and water levels are included.

# In[60]:


# RR
if RR:
    drrmodel.openwater.io.openwater_from_input(
        hydamo.catchments, lu_file, meteo_areas, zonalstats_alltouched=True
    )


# ## RR boundaries

# They are different for the (paved) case with and without overflows. For the extra paved nodes, also a boundary should be created. Without overflows:

# In[61]:


# if RR:
#     drrmodel.external_forcings.io.boundary_from_input(hydamo.laterals, hydamo.catchments, drrmodel)


# And with overflows:

# In[62]:


if RR:
    drrmodel.external_forcings.io.boundary_from_input(hydamo.laterals, hydamo.catchments, drrmodel, overflows=hydamo.overflows, greenhouse_laterals=hydamo.greenhouse_laterals)


# ### External forcings
# 
# Three types of external forcing need to be provided:<br>
# - Seepage/drainage
# - Precipitation
# - Evaporation
# 
# All are assumed to be spatially variable and thus need to pe provided as rasters per time step. Only the locations of the folders containing the rasters need to be provided; the time step is then derived from the file names.
# 
# Precipitation and evaporation are assumed to be in mm/d. As for evaporation only one meteostation is used, the meteo_areas are dissolved. For seepage, as the use of Metaswap-rasters is allowed, the unit is assumed to m3/grid cell/timestep.
# 
# Rastertypes can be any type that is recognized by rasterio (in any case Geotiff and ArcASCII rasters). If the file extension is 'IDF', as is the case in Modflow output, the raster is read using the 'imod'-package.
# 
# IMPORTANT: time steps are extracted from the file names. Therefore, the names should cohere to some conditions:
# The filename should consist of at least two parts, separated by underscores. The second part needs to contain time information, which should be formatted as YYYYMMDDHHMMSS (SS may be omitted). Or, for daily data YYYYMMDD.
# 
# For example: 'precip_20200605151500.tif'
# 
# Extracting meteo-data from rasters can be time consuming. If precip_file and evap_file are specified, meteo-files are copied from an existing location.

# In[63]:


if RR:
    seepage_folder = data_path / "rasters" / "seepage"
    precip_file = str(data_path / "DEFAULT.BUI")
    precip_folder =data_path / "rasters" / "precipitation" 
    evap_folder = data_path / "rasters" / "evaporation"
    drrmodel.external_forcings.io.seepage_from_input(hydamo.catchments, seepage_folder)
    drrmodel.external_forcings.io.precip_from_input(meteo_areas, precip_folder=None, precip_file=precip_file)
    drrmodel.external_forcings.io.evap_from_input(meteo_areas, evap_folder=evap_folder, evap_file=None)


# Add the main parameters:

# In[64]:


if RR:   
    drrmodel.d3b_parameters["Timestepsize"] = 300
    drrmodel.d3b_parameters["StartTime"] = "'2016/06/01;00:00:00'"  # should be equal to refdate for D-HYDRO
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
    drrmodel.d3b_parameters["RestartIn"] = 0
    drrmodel.d3b_parameters["RestartOut"] = 0
    drrmodel.d3b_parameters["RestartFileNamePrefix"] = "Test"
    drrmodel.d3b_parameters["UnsaturatedZone"] = 1
    drrmodel.d3b_parameters["UnpavedPercolationLikeSobek213"] = -1
    drrmodel.d3b_parameters["VolumeCheckFactorToCF"] = 100000


# Laterals are different for the case with and without RR. There can be three options:
# 1) laterals from the RR model (RR=True). There will be real-time coupling where RR and FM are calculated in parallel. Note that, again, the overflows are needed because there are extra boundaries. If there are no overflows, it does not have to be provided.
# 2) timeseries: lateral_discharges can be a dataframe with the code of the lateral as column headers and timesteps as index
# 3) constant: lateral_discharges can be a pandas Series with the code of the lateral as the index. This is the case in the example when RR=False.

# In[65]:


if RR:
    hydamo.external_forcings.convert.laterals(
        hydamo.laterals,
        overflows=hydamo.overflows,
        greenhouse_laterals=hydamo.greenhouse_laterals,
        lateral_discharges=None,
        rr_boundaries=drrmodel.external_forcings.boundary_nodes
    )
else:
    lateral_discharges = hydamo.laterals["afvoer"]
    lateral_discharges.index = hydamo.laterals.code
    hydamo.external_forcings.convert.laterals(
        hydamo.laterals, lateral_discharges=lateral_discharges, rr_boundaries=None
    )


# ### Plot the RR model

# In[66]:


def node_geometry(dict):
    # Function to put the node geometries in geodataframes
    from shapely.geometry import Point, LineString

    geoms = []
    links = []
    for i in dict.items():
        if "ar" in i[1]:
            if np.sum([float(s) for s in i[1]["ar"].split(" ")]) > 0:
                geoms.append(Point((float(i[1]["px"]), float(i[1]["py"]))))
                links.append(
                    LineString(
                        (
                            Point(float(i[1]["px"]), float(i[1]["py"])),
                            Point(
                                float(drrmodel.external_forcings.boundary_nodes[i[1]["boundary_node"]]["px"]),
                                float(drrmodel.external_forcings.boundary_nodes[i[1]["boundary_node"]]["py"]),
                            ),
                        )
                    )
                )
        else:
            geoms.append(Point((float(i[1]["px"]), float(i[1]["py"]))))
    return ((gpd.GeoDataFrame(geoms, columns=["geometry"])), gpd.GeoDataFrame(links, columns=["geometry"]))


# In[67]:


if RR:
    ## plt.rcParams['axes.edgecolor'] = 'w'
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    xmin,ymin,xmax,ymax=hydamo.clipgeo.bounds
    ax.set_xlim(round(xmin), round(xmax))
    ax.set_ylim(round(ymin), round(ymax))

    hydamo.catchments.geometry.plot(ax=ax, label="Catchments", edgecolor="black", facecolor="pink", alpha=0.5)
    hydamo.branches.geometry.plot(ax=ax, label="Channel")
    node_geometry(drrmodel.unpaved.unp_nodes)[0].plot(
        ax=ax, markersize=30, marker="s", color="green", label="Unpaved"
    )
    node_geometry(drrmodel.unpaved.unp_nodes)[1].plot(ax=ax, color="black", linewidth=0.5)
    node_geometry(drrmodel.paved.pav_nodes)[0].plot(ax=ax, markersize=20, marker="s", color="red", label="Paved")
    node_geometry(drrmodel.paved.pav_nodes)[1].plot(ax=ax, color="black", linewidth=0.5)
    node_geometry(drrmodel.greenhouse.gh_nodes)[0].plot(ax=ax, markersize=15, color="yellow", label="Greenhouse")
    node_geometry(drrmodel.greenhouse.gh_nodes)[1].plot(ax=ax, color="black", linewidth=0.5)
    node_geometry(drrmodel.openwater.ow_nodes)[0].plot(ax=ax, markersize=10, color="blue", label="Openwater")
    node_geometry(drrmodel.openwater.ow_nodes)[1].plot(ax=ax, color="black", linewidth=0.5, label="RR-link")
    node_geometry(drrmodel.external_forcings.boundary_nodes)[0].plot(
        ax=ax, markersize=15, color="purple", label="RR Boundary"
    )

    # manually add handles for polygon plot
    handles, labels = ax.get_legend_handles_labels()
    poly = mpatches.Patch(facecolor="pink", edgecolor="black", alpha=0.5)    
    ax.legend(handles=handles.append(poly), labels=labels.append("Catchments"))
    cx.add_basemap(ax, crs=28992, source=cx.providers.OpenStreetMap.Mapnik)
    fig.tight_layout()


# ## Writing the model

# Now we call Hydrolib-core functionality to write the model. First, we initialize an object that converts all dataframes to Hydrolib-core objects. Then we add these models to the file structure of the FM model.

# Call a function to convert the dataframes to Hydrolib-core classes:

# In[68]:


models = Df2HydrolibModel(hydamo, assign_default_profiles=True)


# And add the classes to the file structure. Each class requires a different approach, and at the moment Hydrolib-core is still in development. The code below is subject to change in future releases.

# In[69]:


fm.geometry.structurefile = [StructureModel(structure=models.structures)]
fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)
fm.geometry.storagenodefile = StorageNodeModel(storagenode=models.storagenodes)

fm.geometry.frictfile = []
for i, fric_def in enumerate(models.friction_defs):
    fric_model = FrictionModel(global_=fric_def)
    fric_model.filepath = f"roughness_{i}.ini"
    fm.geometry.frictfile.append(fric_model)

if hasattr(hydamo.observationpoints, 'observation_points'):
    fm.output.obsfile = [ObservationPointModel(observationpoint=models.obspoints)]

extmodel = ExtModel()
extmodel.boundary = models.boundaries_ext
extmodel.lateral = models.laterals_ext
fm.external_forcing.extforcefilenew = extmodel

fm.geometry.inifieldfile = IniFieldModel(initial=models.inifields)

for ifield, onedfield in enumerate(models.onedfieldmodels):
    # eventually this is the way, but it has not been implemented yet in Hydrolib core
    # fm.geometry.inifieldfile.initial[ifield].datafile = OneDFieldModel(global_=onedfield)

    # this is a workaround to do the same
    onedfield_filepath = output_path / "fm" / "initialwaterdepth.ini"
    onedfieldmodel = OneDFieldModel(global_=onedfield)
    onedfieldmodel.save(filepath=onedfield_filepath)
    fm.geometry.inifieldfile.initial[ifield].datafile = DiskOnlyFileModel(
        filepath=onedfield_filepath
    )


# Add some setttings to the MDU that are recommended by Deltares. See the latest user guide, that comes with each D-Hydro release, for details.

# In[70]:


fm.geometry.uniformwidth1d = 1.0            # default  breedte 
fm.geometry.bedlevtype = 1                  # 1: at cell center (tiles xz,yz,bl,bob=max(bl)), 2: at face (tiles xu,yu,blu,bob=blu), 3: at face (using mean node values), 4: at face 
fm.geometry.changestructuredimensions = 0   # Change the structure dimensions in case these are inconsistent with the channel dimensions.

fm.numerics.cflmax = 0.7                    # Maximum Courant nr.
# the following two settings are not supportedy by hydrolib-core 0.4.1
# fm.numerics.epsmaxlev = 0.0001            # stop criterion for non-linear solver
# fm.numerics.epsmaxlevm = 0.0001           # stop criterion for Nested Newton loop
fm.numerics.advectype = 33                  # Adv type, 0=no, 33=Perot q(uio-u) fast, 3=Perot q(uio-u).

fm.volumetables.increment = 0.2             # parameter setting advised by Deltares for better performance
fm.volumetables.usevolumetables = 1         # parameter setting advised by Deltares for better performance

fm.restart.restartfile     = None           # Restart file, only from netCDF-file, hence: either *_rst.nc or *_map.nc.
fm.restart.restartdatetime = None           # Restart time [YYYYMMDDHHMMSS], only relevant in case of restart from *_map.nc.

fm.output.mapformat=4                       # parameter setting advised by Deltares for better performance
fm.output.ncformat = 4                      # parameter setting advised by Deltares for better performance
fm.output.ncnoforcedflush = 1               # parameter setting advised by Deltares for better performance
fm.output.ncnounlimited = 1                 # parameter setting advised by Deltares for better performance
fm.output.wrimap_wet_waterdepth_threshold = 0.01                     # Waterdepth threshold above which a grid point counts as 'wet'
fm.output.mapinterval = [1200.0, fm.time.tstart, fm.time.tstop]      # Map file output, given as 'interval' 'start period' 'end period' [s].
fm.output.rstinterval  = [86400.0, fm.time.tstart, fm.time.tstop]    # Restart file output, given as 'interval' 'start period' 'end period' [s].
fm.output.hisinterval = [300., fm.time.tstart, fm.time.tstop]        # History output, given as 'interval' 'start period' 'end period' [s].
fm.output.wrimap_flow_analysis = 1                                   # write information for flow analysis


# In D-Hydro the 1D timestep (dt user) should be at least equal to than the smallest timestep of RR and RTC, otherwise water balance problems may occur.
# The following code sets 'dtuser' equal to the smallest time step.

# In[71]:
delattr(fm, 'sediment')

# check the timesteps:
timesteps = []
if RR:
    timesteps.append(drrmodel.d3b_parameters['Timestepsize'])
if RTC:
    timesteps.append(drtcmodel.time_settings['step'])
if len(timesteps)>0:
    if fm.time.dtuser > np.min(timesteps):
        fm.time.dtuser = np.min(timesteps)


# Now we write the file structure:

# In[72]:


fm.filepath = Path(output_path) / "fm" / "test.mdu"
dimr = DIMR()
dimr.component.append(
    FMComponent(name="DFM", workingDir=Path(output_path) / "fm", model=fm, inputfile=fm.filepath)    
)
dimr.save(recurse=True)


# The writers for RR and RTC are not yet available in the HYDROLIB-core library. We use the original delft3dfmpy writer for RR and a custom writer for RTC:

# In[73]:


if RTC:
    drtcmodel.write_xml_v1()


# Note that with the WWTP-argument, the coordinates for a (fictional) WWTP are provided. From each paved node, a sewage link is connected to this WWTP.

# In[74]:


if RR:
    rr_writer = DRRWriter(drrmodel, output_dir=output_path, name="test", wwtp=(199000.0, 396000.0))
    rr_writer.write_all()


# A run.bat that will run DIMR is written by the following command. Adjust this with your local D-Hydro Suite version.

# In[75]:


dimr = DIMRWriter(output_path=output_path, dimr_path=str(r"C:\Program Files\Deltares\D-HYDRO Suite 2024.03 1D2D\plugins\DeltaShell.Dimr\kernels\x64\bin\run_dimr.bat"))


# In[76]:


if not RR:
    drrmodel = None
if not RTC:
    drtcmodel = None


# In[77]:


dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)


# In[78]:


dimr.write_runbat()
dimr.add_crs()

# In[79]:


print("Done!")


# # In[ ]:
# import netCDF4 as nc
# from hydrolib.core.dflowfm.net.reader import UgridReader
# from hydrolib.core.dflowfm.net.writer import UgridWriter
# file_format = "NETCDF3_CLASSIC"  # "NETCDF4"
# ncfilepath1 = (output_path / 'fm' / 'network.nc')
# # initialiseren nieuw netwerk
# fm = FMModel()
# network = fm.geometry.netfile.network
# # inlezen uit netcdf
# reader = UgridReader(ncfilepath1)
# reader.read_mesh2d(network._mesh2d)
# reader.read_mesh1d_network1d(network._mesh1d)
# reader.read_link1d2d(network._link1d2d)

# # en ook de attributen 
# ncin = nc.Dataset(ncfilepath1, 'r')
# title = ncin.title
# conventions = ncin.Conventions
# source = ncin.source
# history = ncin.history
# institution = ncin.institution
# references = ncin.references
# ncin.close()
# # nieuwe conventie
# conventions = 'CF-1.8 UGRID-1.0 Deltares-0.10'

# # nu schrijven we het bestand opnieuw
# os.remove(ncfilepath1)
# writer = UgridWriter()
# ncfile = nc.Dataset(ncfilepath1, "w", format=file_format)

# writer._write_mesh1d_to(network._mesh1d, ncfile)
# writer._write_mesh2d_to(network._mesh2d, ncfile)
# writer._write_1d2dlinks_to(network._link1d2d, ncfile)


# #  int projected_coordinate_system;
# #       :name = "Amersfoort / RD New";
# #       :epsg = 28992; // int
# #       :grid_mapping_name = "Unknown projected";
# #       :longitude_of_prime_meridian = 0.0; // double
# #       :semi_major_axis = 6377397.155; // double
# #       :semi_minor_axis = 6356078.962818189; // double
# #       :inverse_flattening = 299.1528128; // double
# #       :proj4_params = "+prdoj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs";
# #       :EPSG_code = "EPSG:28992";
# #       :projection_name = "unknown";
# #       :wkt = "PROJCS[\"Amersfoort / RD New\",\n    GEOGCS[\"Amersfoort\",\n        DATUM[\"Amersfoort\",\n            SPHEROID[\"Bessel 1841\",6377397.155,299.1528128,\n     



# proj = ncfile.createVariable('projected_coordinate_system','i4')
# proj.epsg = 28992
# # proj.name= "Amersfoort / RD New"
# proj.grid_mapping_name = "Amersfoort / RD New" #"Rijksdriehoeksstelsel"
# # proj.longitude_of_prime_meridian = 5.38763888888889
# # proj.semi_major_axis = 6378137.0
# # proj.semi_minor_axis = 6356752.314245
# # proj.inverse_flattening = 298.257223563
# proj.longitude_of_prime_meridian = 0.0#; // double
# proj.semi_major_axis = 6377397.155#; // double
# proj.semi_minor_axis = 6356078.962818189#; // double
# proj.inverse_flattening = 299.1528128#; // double
# proj.proj4_params = "+prdoj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs";
# proj.EPSG_code = "EPSG:28992"
# proj.value = "value is equal to EPSG code"
# proj.wkt = "PROJCS[\"Amersfoort / RD New\",\n    GEOGCS[\"Amersfoort\",\n        DATUM[\"Amersfoort\",\n            SPHEROID[\"Bessel 1841\",6377397.155,299.1528128,\n"     

# ncfile.Conventions = conventions
# ncfile.title = title
# ncfile.source = source
# ncfile.history = history
# ncfile.institution = institution
# ncfile.references = references
# ncfile.close()



# # #  int projected_coordinate_system;
# # #       :name = "Amersfoort / RD New";
# # #       :epsg = 28992; // int
# # #       :grid_mapping_name = "Unknown projected";
# # #       :longitude_of_prime_meridian = 0.0; // double
# # #       :semi_major_axis = 6377397.155; // double
# # #       :semi_minor_axis = 6356078.962818189; // double
# # #       :inverse_flattening = 299.1528128; // double
# # #       :proj4_params = "+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs";
# # #       :EPSG_code = "EPSG:28992";
# # #       :projection_name = "unknown";
# # #       :wkt = "PROJCS[\"Amersfoort / RD New\",\n    GEOGCS[\"Amersfoort\",\n        DATUM[\"Amersfoort\",\n            SPHEROID[\"Bessel 1841\",6377397.155,299.1528128,\n     


