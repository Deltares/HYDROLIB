# =========================================================================================
#
# License: LGPL
#
# Author: Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================
import os
import sys
from pathlib import Path
import csv

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from read_dhydro import net_nc2gdf, read_locations
from stationpoints import stationpoints

from hydrolib.core.io.mdu.models import FMModel, FrictionModel
from hydrolib.core.io.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.io import polyfile
import xarray as xr
from read_dhydro import (
    branch_gui2df,
    chainage2gdf,
    map_nc2gdf,
    net_nc2gdf,
    pli2gdf,
    read_nc_data,
)
#TO DO
# Werkt nu alleen voor yz cross sections

# aantal braches bij nodes
# verschil tussen ids en names bij nodes en meshnodes
# offset vs chainage
# bij meshnodes alleen branch nr geen id
# bij nodes geen branch
# volgorde in lijn branch rekenpunt4en


def check_feasibility_1D_waterlevels(mdu_path, nc_path, output_path):
    
    """
       Check wether the 1D waterlevels of the 1D flow networ are higher than 
       the left and right embankments in cross section profiles

       Parameters:
            mdu_path : str
               Path to mdu file containing the D-hydro model structure
            output_path
                Path where the ... file is saved as output

    ___________________________________________________________________________________________________________
       Warning:
               ...
    """
    

    ## READ DATA
    # Read branches and meshnodes from net.nc
    network = net_nc2gdf(nc_path)
    
    branches = network['1d_branches']
    branches['branchnr'] = branches.index
    branches = branches.rename(columns={'id': 'branchid'})
    branches_reduced = branches[['branchid','branchnr']].copy()
    
    meshnodes = network['1d_meshnodes']
    meshnodes = meshnodes.rename(columns={'id': 'meshnodeid'})
    meshnodes = meshnodes.rename(columns={'branch': 'branchnr'})
    meshnodes = pd.merge(meshnodes,branches_reduced,on='branchnr') # The dataframe of the meshnodes do include a branch number, but not a branch id
    
    # The dataset of mesh nodes is not complete
    # Some meshnodes form the start/end nodes of a branch (= connection node) and
    # lie at intersections of different branches
    # Now however, only 1 of the branches is decribed in the dataframe 
    # We have to complement the dataframe because, later on, we have to know exactly which 
    # meshnodes lie on a specific branch
    # The code below tries to correct this and finds out which mesh node is in fact
    # connection node and connects multiple branches
    # The difficulty is that the mesh nodes are a little bit shifted and do not lie at 
    # exact same locations (decimals rounding) as the connection nodes (= start/end points of branches) 
    # and the id's can be different (sometimes an underscorre is added) and     
    total_nodes_per_branch = network['1d_total_nodes_per_branch']
    total_nodes_per_branch = total_nodes_per_branch.rename(columns={'branch': 'branchnr'})
    total_nodes_per_branch = total_nodes_per_branch.rename(columns={'id': 'nodeid'})    
    total_nodes_per_branch = pd.merge(total_nodes_per_branch,branches_reduced,on='branchnr')

    for i, row in branches.iterrows(): 
        branchid = branches.loc[i,'branchid']
        nodes_on_branch = total_nodes_per_branch[total_nodes_per_branch['branchid'] == branchid].copy()
        for j, row2 in nodes_on_branch.iterrows(): 
            selection = meshnodes[meshnodes['meshnodeid'] == nodes_on_branch.loc[j,'nodeid']].copy()                          
            if len(selection) >= 1: 
                # If the node/meshnode (can be both) from the dataframe 'total_nodes_per_branch' 
                # has a similar id as a meshnode that was already in the meshnode dataframe...
                # Now we have to check if the meshnode is connected to a branch we didn't know about
                if nodes_on_branch.loc[j,'branchid'] not in selection['branchid'].values:
                    # We find out that the meshnode is connected to a new branch, we should add that to our dataframe
                    new_row = selection.iloc[[0]].copy()
                    new_row['branchid'] = branchid
                    new_row['branchnr'] = nodes_on_branch.loc[j,'branchnr']
                    new_row['offset'] = nodes_on_branch.loc[j,'offset']
                    meshnodes = meshnodes.append(new_row)
                    meshnodes.reset_index(drop = True, inplace = True)
            else:
                # It is also possible that the row in the dataframe 'total_nodes_per_branch' refers to a 
                # connection node that lies at a similar location as meshnode that was already in the dataframe
                # but the id is a little bit different for meshnodes, sometimes a "_1" is added
                # Moreovere location of the meshnodes is sometimes a little bit shifted
                if nodes_on_branch.loc[j,'nodeid'] + '_1' in meshnodes['meshnodeid'].values:
                    selection = meshnodes[meshnodes['meshnodeid'] == nodes_on_branch.loc[j,'nodeid'] + '_1'].copy()      
                    if nodes_on_branch.loc[j,'branchid'] not in selection['branchid'].values:
                        if nodes_on_branch.loc[j,'geometry'].buffer(0.01).intersects(selection.geometry.iloc[0]):
                            new_row = selection.iloc[0].to_frame().T
                            new_row['branchid'] = branchid
                            new_row['branchnr'] = nodes_on_branch.loc[j,'branchnr']
                            new_row['offset'] = nodes_on_branch.loc[j,'offset']
                            meshnodes = meshnodes.append(new_row)   
                            meshnodes.reset_index(drop = True, inplace = True)                         
                            
    
    # Read calculated waterlevels at meshnodes from map.nc
    waterlevel = map_nc2gdf(nc_path, 'mesh1d_s1')
    waterlevel = waterlevel.max(axis=1,numeric_only=True).to_frame()
    waterlevel = waterlevel.rename(columns={0: 'max_waterlevel'})
    waterlevel['meshnodeid'] = waterlevel.index
    meshnodes = pd.merge(meshnodes,waterlevel,on='meshnodeid')

    # Read locations and definitions of cross sections
    gdfs_results = read_locations(mdu_path,["cross_sections_locations","cross_sections_definition","structures"])
    structures = gdfs_results["structures"]
    crslocs = gdfs_results["cross_sections_locations"]
    crsdefs = gdfs_results["cross_sections_definition"]
    loc_ids = crslocs['id'].tolist()
    
    # LOOP OVER THE CROSS SECTIONS AND COMPARE LEFT AND RIGHT EMBANKMENT HEIGHT 
    # WITH THE MAX CALCULATED WATERLEVEL (DERIVED FROM THE SURROUNDING MESH NODES)
    print('Find the height of the left and right embankment')
    crslocs['left_y_embak'] = -9999
    crslocs['right_y_embak'] = -9999
    crslocs['left_z_embak'] = -9999
    crslocs['right_z_embak'] = -9999
    crslocs['max_waterlevel'] = -9999
    crslocs['method'] = -9999

    print('Readling embankment height and waterlevels for each profile')
    for loc_id in loc_ids:
        print('Progress ' + str(round(loc_ids.index(loc_id)/len(loc_ids)*100,1)) + '%')
        
        def_id = crslocs.loc[crslocs['id'] == loc_id,'definitionid'].iloc[0] # ID cross section definition
        index_loc = crslocs.loc[crslocs['id'] == loc_id,:].index.to_list()[0] # index of the cross section location
        index_def = crsdefs.loc[crsdefs['id'] == def_id,:].index.to_list()[0] # index of the cross section definition 
        
        if crsdefs.loc[index_def,'type'] != 'yz': # this tool can currently only adjust yz-profiles
            crslocs.loc[index_loc,'method'] = 'cannot find embankment height because it is not a yz-profile' 
        else:
            y = crsdefs.loc[index_def,'ycoordinates'].copy()
            z = crsdefs.loc[index_def,'zcoordinates'].copy()
            bot_value = min(z)
            bot_position = [i for i,zz in enumerate(z) if float(zz) == bot_value]
            
            # find height of left embankment
            i = bot_position[0]
            left_z_embak = -9999           
            while i >= 0:
                if z[i] > left_z_embak:
                    left_y_embak = y[i]
                    left_z_embak = z[i]
                i = i - 1

            # find height of right embankment
            i = bot_position[-1]
            right_z_embak = -9999           
            while i <= len(y)-1:
                if z[i] > right_z_embak:
                    right_y_embak = y[i]
                    right_z_embak = z[i]
                i = i + 1
            
            # extra check for anomalies
            if len(bot_position) > 1:
                if max(z[bot_position[0]:bot_position[-1]+1]) > min(left_z_embak,right_z_embak):
                    print('Profile' + str(loc_id) +': in between the lowest points lies a point that is higher than the left and right embankment')
            
            crslocs.loc[index_loc,'left_z_embak'] = left_z_embak
            crslocs.loc[index_loc,'right_z_embak'] = right_z_embak
            crslocs.loc[index_loc,'left_y_embak'] = left_y_embak
            crslocs.loc[index_loc,'right_y_embak'] = right_y_embak            
            crs_chainage = crslocs.loc[index_loc,'chainage'].copy()
            
            # determine the max waterlevel at cross section, based on surrounding mesh nodes
            meshnodes_selection = meshnodes[meshnodes['branchid'] == crslocs.loc[index_loc,'branchid']].copy()
            
            if len(meshnodes_selection) == 0:
                crslocs.loc[index_loc,'method'] = 'No mesh node found on the branch'
            else:
                if len(meshnodes_selection) == 1:
                    crslocs.loc[index_loc,'max_waterlevel'] = meshnodes_selection['max_waterlevel'].iloc[0]
                    crslocs.loc[index_loc,'method'] = 'Only one mesh node found on the branch'
                else:
                    if len(meshnodes_selection) > 2: 
                        #select the mesh nodes at the front and at the back of the cross section
                        meshnodes_selection['distance'] = meshnodes_selection['offset'] - crs_chainage
                        
                        meshnode_attheback = meshnodes_selection[meshnodes_selection['distance'] == meshnodes_selection[meshnodes_selection['distance'] <= 0]['distance'].max()]
                        meshnode_atthefront = meshnodes_selection[meshnodes_selection['distance'] == meshnodes_selection[meshnodes_selection['distance'] > 0]['distance'].min()]
                        meshnodes_selection = pd.concat([meshnode_attheback,meshnode_atthefront])
                    meshnodes_selection = meshnodes_selection.sort_values(by=['offset'])
                    meshnodes_selection.reset_index(drop = True, inplace = True)
                    
                    # determine if there lies an structure in between the profile and one of the mesh nodes
                    structures_selection = structures[structures['branchid'] == crslocs.loc[index_loc,'branchid']]
                    if len(structures_selection) > 0: # there is a structure on the same branch, now determine if one lies in between the cross section and selected mesh nodes 
                        structures_selection = structures_selection[(structures_selection['chainage'] >= meshnodes_selection.loc[0,'offset']) & (structures_selection['chainage'] <= meshnodes_selection.loc[1,'offset'])]
                        
                    if len(structures_selection) > 0: # there lies a structure in between the cross section and selected mesh nodes
                        # to determine the waterlevel at the cross section, use the mesh node that has no structure in between
                        
                        if len(structures_selection[structures_selection['chainage'] == crs_chainage]) >= 1:
                            # Cross section lies at exact the same location as a structure
                            crslocs.loc[index_loc,'method'] = 'Cross section lies at exact the same location as a structure, waterlevel interpolated between mesh nodes'
                            slope = (meshnodes_selection.loc[1,'max_waterlevel'] - meshnodes_selection.loc[0,'max_waterlevel']) / (meshnodes_selection.loc[1,'offset'] - meshnodes_selection.loc[0,'offset'])
                            crslocs.loc[index_loc,'max_waterlevel'] = slope * (crs_chainage - meshnodes_selection.loc[0,'offset']) + meshnodes_selection.loc[0,'max_waterlevel']
                        elif len(structures_selection[(structures_selection['chainage'] < crs_chainage) & (structures_selection['chainage'] >= meshnodes_selection.loc[0,'offset'])]) == 0:
                            # there is no structure in between the cross section and the mesh node at the back of it
                            crslocs.loc[index_loc,'method'] = 'There lies a structure at the front of the cross section, waterlevel determined based on mesh node at the back of it'
                            crslocs.loc[index_loc,'max_waterlevel'] = meshnodes_selection.loc[0,'max_waterlevel']
                        elif len(structures_selection[(structures_selection['chainage'] > crs_chainage) & (structures_selection['chainage'] <= meshnodes_selection.loc[1,'offset'])]) == 0:
                            # there is no structure in between the cross section and the mesh node at the front of it
                            crslocs.loc[index_loc,'method'] = 'There lies a structure at the back of the cross section, waterlevel determined based on mesh node at the front of it'
                            crslocs.loc[index_loc,'max_waterlevel'] = meshnodes_selection.loc[1,'max_waterlevel']
 
                    else: # there is no structure in between the cross section and selected mesh nodes, calculate the water level at the cross section by interpolation
                        crslocs.loc[index_loc,'method'] = 'Cross section is surrounded by two mesh nodes and no structures, waterlevel interpolated between mesh nodes'    
                        slope = (meshnodes_selection.loc[1,'max_waterlevel'] - meshnodes_selection.loc[0,'max_waterlevel']) / (meshnodes_selection.loc[1,'offset'] - meshnodes_selection.loc[0,'offset'])
                        crslocs.loc[index_loc,'max_waterlevel'] = slope * (crs_chainage - meshnodes_selection.loc[0,'offset']) + meshnodes_selection.loc[0,'max_waterlevel']
    A=2
   

if __name__ == "__main__":
    # Read shape
    mdu_path = Path(r"C:\Users\devop\Desktop\Zwolle-Minimodel_inputtestmodel\1D2D-DIMR\dflowfm\flowFM.mdu")
    nc_path = Path(r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\output\FlowFM_map.nc")
    output_path = r'C:\Users\devop\Desktop'
    check_feasibility_1D_waterlevels(mdu_path, nc_path, output_path)