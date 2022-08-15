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
#TO DO
# Werkt nu alleen voor yz cross sections



def change_depth_crosssection(mdu_path, netnc_path, shape_path, column_horizontal, column_vertical, vertical_distance_type, output_path):
    
    """
       Adjust the depth of yz cross sections. Do this over a certain width (deepest part of the profile)
       
       One could think of two scenario's in which this tool is usefil
       - Deepen the bottom of the crosssection, e.g. due to maintenance/dredging activities
       - Heighten/increase the bottom level of the crosssection, e.g. due to sedimentation processes

       Parameters:
            mdu_path : str
               Path to mdu file containing the D-hydro model structure
            shape_path : str
               Path to shape file with polygons describing in which areas to adjust the cross section. The 
               user can define multiple polygons within the shapefile and these can have different setting
            column_horizontal : str
                Horizontal width (always in meters) over which the cross section bottom is adjusted. The 
                horizontal width/section is always placed in the deepest part of (current) cross section. 
                The tool identifies the lowest point of the profile and than looks to the left 
                and right to find the lowest section (with the defined width)
            column_vertical : str
                column name in the shapefile that gives the dimensions with which the cross sections should be
                deepened/raised. column should contain float numbers.
                if type_column_vertical = 'distance'/'uniform', the unit is in meters.
                if type_column_vertical = 'referencelevel', the unit is a reference height with a vertical datum.
            vertical_vertical_distance_type : str, only three options
                The cross section can be adjusted vertically using three options. All options can be used to 
                raise or lower the profile.
                - 'distance' --> indicates that the bottom of the profile for the user-defined width is flattened
                and moved to a certain level. This level is computed by adding up the bottom level of the profile plus
                the vertical displacement. Positive number is increasing the bottom level, negative number is 
                lowering the bottom level. E.g. if the column_vertical contains the number -0.5 and the lowest point 
                is currently +5 m NAP, the bottom of the cross section is flattened (width is user defined) and lies 
                at +4.5 m NAP
                - 'referencelevel' --> indicates that the bottom of the profile for the user-defined width is flattened
                and moved to a certain reference height that is defined with a vertical datum. In the Netherlands, 
                we use meters in Amsterdam Ordnance Datum or Normaal Amsterdams Peil (NAP). E.g. if the column_vertical 
                contains the number 4, the bottom of the cross section is flattened (width is user defined) and lies 
                at +4 m NAP. In this way, the bottom of the profile can lowered/raised. This can also vary within
                the width-section that is selected. Maybe a part of the section is raised, and another part is lowered.
                - 'uniform' --> indicates that the bottom is uniformly/evenly lowered/raised with a certain distance
                The existing variations/irregularities persist. 


            output_path
                Path where the ... file is saved as output

    ___________________________________________________________________________________________________________
       Warning:
               ...
    """
    
    ## READ DATA
    # Read locations and definitions of cross sections
    gdfs_results = read_locations(mdu_path,["cross_sections_locations","cross_sections_definition"])
    
    # Check which profiles should be adjusted
    crslocs = gdfs_results["cross_sections_locations"]
    print("Amersfoort aangenomen als projectie")
    crslocs.crs = "EPSG:28992"
    gdf_frict = gpd.read_file(shape_path)
    loc_ids = crslocs.loc[crslocs.intersects(gdf_frict.dissolve().geometry.iloc[0]),'id'].tolist()  # select only the relevant cross section locations
    
    ## REMOVE SHARED DEFINITIONS 
    crsdefs = gdfs_results["cross_sections_definition"]
    crslocs, crsdefs = split_duplicate_crsdef_references(crslocs, crsdefs, loc_ids)  

    ## Adjust selected profile definitions
    print('Starting to adjust ' + str(len(loc_ids)) + 'profile definitions'
    for loc_id in loc_ids: # Loop over the cross sections that are selected as relevant
        
        def_id = crslocs.loc[crslocs['id'] == loc_id,'definitionid'].iloc[0] # ID cross section definition
        index_loc = crslocs.loc[crslocs['id'] == loc_id,:].index.to_list()[0] # index of the cross section location
        index_def = crsdefs.loc[crsdefs['id'] == def_id,:].index.to_list()[0] # index of the cross section definition 
        
        if crsdefs.loc[index_def,'type'] != 'yz': # this tool can currently only adjust yz-profiles
            print('Cannot adjust profile definition ' + str(def_id) + ' at location ' + str(loc_id) + ' because it is not a yz-profile')
        else:
            
            # Determine the right parameters for this profile
            vertical_distance = gdf_frict.loc[gdf_frict.intersects(crslocs.loc[index_loc,'geometry']),column_vertical].to_list()[0]             
            slot_width = gdf_frict.loc[gdf_frict.intersects(crslocs.loc[index_loc,'geometry']),column_horizontal].to_list()[0]
            if (isinstance(vertical_distance,(int,float)) == False or isinstance(slot_width,(int,float)) == False):
                raise Exception('The input shapefile does not contain int/float numbers for the vertical/horizontal adjustment')
            if ((vertical_distance_type != 'distance') and (vertical_distance_type != 'referencelevel') and (vertical_distance_type != 'uniform')):
                raise Exception('The input shapefile does not contain the right strings to desribte the type of vertical adjustment')
                print('It is either "distance", "referencelevel" or "unifor"')
            
            y = crsdefs.loc[index_def,'ycoordinates'].copy()
            z = crsdefs.loc[index_def,'zcoordinates'].copy()
            bot_value = min(z)

            if slot_width < (max(y) - min(y)): #if only a certain part of the cross section should we changed (not the full width)         
                ## SELECT LOWEST POINT IN THE PROFILE
                # This will be the starting point when finding the optimal area to deepen the profile
                bot_position = [i for i,zz in enumerate(z) if float(zz) == bot_value]
                if len(bot_position) > 2: # If there are two lowest points in de cross section, select the 2 most relevant ones
                    points_next_to_each_other = -1
                    for i in range(len(bot_position)-2):  # Check there are lowest points that lie next to each other
                        if abs(bot_position[i+1] - bot_position[i]) == 1:
                            points_next_to_each_other = i
                    if points_next_to_each_other > -1: # If there are two lowest points next to each other, pick these
                        bot_position = bot_position[points_next_to_each_other:points_next_to_each_other+1]
                    else: # Otherwise pick the first two lowest points
                        bot_position = bot_position[0:2]     
                if len(bot_position) == 2: # If there are two lowest points
                    if (y[bot_position[0]] - y[bot_position[1]]) <= slot_width:
                        index_left = bot_position[0]
                        index_right = bot_position[1]
                    else: # If the distance between the lowest points is too larg
                        if abs(bot_position[-1] - bot_position[0]) != 1: 
                            # If the lowest points are not next to each other, pick point in the middle based in index (not distance)
                            # TO DO: improve selection
                            bot_position = round(abs(bot_position[-1] - bot_position[0])/2) #
                            index_left = bot_position
                            index_right = bot_position
                        else:
                            # If the lowest points are next to each other, add extra point in between them
                            ymiddle = [(y[bot_position[0]] + y[bot_position[1]])/2]
                            y = y[0:bot_position[0]+1] + ymiddle + y[bot_position[1]:-1]
                            z = z[0:bot_position[0]+1] + bot_value + z[bot_position[1]:-1]
                            index_left = bot_position[0]+1
                            index_right = bot_position[0]+1                        
                elif len(bot_position) == 1: # If there is only one lowest point
                    #Find index of lowest point
                    index_left = bot_position[0]
                    index_right = bot_position[0]
                
                y_middle = (y[index_left] + y[index_right]) / 2
                
                ## ADD EXTRA INTERMEDIATE POINTS IN THE PROFILE
                # This makes the profile line more detailed and makes it easier to select the lowest section which should be deepened
                line= []
                for i in range(len(y)):
                    line.append([0,0])
                    line[i][0]=y[i]
                    line[i][1]=z[i]
                line = LineString(line)
                line2 = gpd.GeoDataFrame(index=[0],data=[0])
                line2['geometry']=line
                gdf_points = stationpoints(line2,0.1,midpoint=False)
                #Add the coords from the points
                line = line2.iloc[0]['geometry'].coords[:]
                line  += [list(p.coords)[0] for p in gdf_points['geometry']]          
                # Calculate the distance along the line for each point    
                dists = [line2.iloc[0,:]['geometry'].project(Point(p)) for p in line ]
                #Sort the coords based on the distances    
                line  = [p for (d, p) in sorted(zip(dists, line ))]
                #convert tuple to list
                line = [list(i) for i in line]
                
                # determine new indices of lowest points based on the refined line
                index_left = [i for i in range(len(line)) if line[i] == [y[index_left],z[index_left]]][0]
                index_right = [i for i in range(len(line)) if line[i] == [y[index_right],z[index_right]]][0]
                
                ## FIND LOWEST SECTION OF THE PROFILE OF A CERTAIN WIDTH (USER-DEFINED) 
                # first do an estimation, using the steps of the refined profile line
                # starting at lowest point(s), search to the left and right until the lowest section is found
                index_lowest_left = index_left
                index_lowest_right = index_right
                found = 0
                while found == 0:
                    section_width = float(line[index_right][0]) - float(line[index_left][0])
                    if section_width < slot_width:
                        index_left_new = index_left - 1
                        index_right_new = index_right + 1
                        elevation_left  = float(line[index_left_new][1]) if index_left_new > -1         else 999
                        elevation_right = float(line[index_right_new][1]) if index_right_new < len(line) else 999
                       
                        if (elevation_right-elevation_left) > 0.01: #if left point lies at least 1 cm lower
                            index_left = index_left_new
                        elif (elevation_left-elevation_right) > 0.01: #if right point lies at least 1 cm lower
                            index_right = index_right_new
                        else: #if values are equal
                            if abs(line[index_lowest_left][0] - line[index_left][0]) > abs(line[index_right][0] - line[index_lowest_right][0]): #if left point has already moved further away from starting point
                                index_right = index_right_new
                            else: #if right point has already moved further away from starting point
                                index_left = index_left_new
                            
                    section_width = float(line[index_right][0]) - float(line[index_left][0])
                    if section_width >= slot_width:
                        found = 1                            
                
                # now, make a more refined estimation of the y and z coordinates of the lowest section                             
                line_line=LineString(line)
                y_left = round((line[index_right][0] + line[index_left][0])/2 - slot_width/2,4)
                y_left_line = LineString([[y_left,-999],[y_left,999]])
                z_left = round(line_line.intersection(y_left_line).coords[0][1],4)
                y_right = round((line[index_right][0] + line[index_left][0])/2 + slot_width/2,4)
                y_right_line = LineString([[y_right,-999],[y_right,999]])
                z_right = round(line_line.intersection(y_right_line).coords[0][1],4)
     
                ## ADJUST THE PROFILE IN THE SECTION THAT WAS FOUND, BASED ON THE PARAMETERS ENTERED BY THE USER
                pos1 = 0
                pos2 = len(y)-1
                for i in range(len(y)):
                    if float(y[i]) < y_left:
                        pos1 = i
                    if float(y[i]) > y_right and pos2 == len(y)-1:
                        pos2 = i   
    
                if vertical_distance_type == 'distance':
                    slot_value = bot_value + vertical_distance
                    y_slot = [y[pos1], y_left, y_left + 0.001, y_right - 0.001, y_right]
                    z_slot = [z[pos1], z_left, slot_value, slot_value, z_right]
                    y[pos1:pos2] = y_slot
                    z[pos1:pos2] = z_slot
                elif vertical_distance_type == 'reference':
                    slot_value = vertical_distance
                    y_slot = [y[pos1], y_left, y_left + 0.001, y_right - 0.001, y_right]
                    z_slot = [z[pos1], z_left, slot_value, slot_value, z_right]
                    y[pos1:pos2] = y_slot
                    z[pos1:pos2] = z_slot 
                elif vertical_distance_type == 'uniform':
                    z[pos1:pos2] = z[pos1:pos2] + vertical_distance            
                          
            else:
                if vertical_distance_type == 'distance':
                    slot_value = bot_value + vertical_distance
                    z = [i * 0 + slot_value for i in z]
                elif vertical_distance_type == 'reference':
                    slot_value = vertical_distance
                    z = [i * 0 + slot_value for i in z]
                elif vertical_distance_type == 'uniform':                
                    crslocs.loc[index_loc,'shift'] = crslocs.loc[index_loc,'shift'] + vertical_distance
            
            ## PUT THE DATA BACK INTO THE DATAFRAME
            temp = crsdefs.loc[:,'ycoordinates'].to_list()
            temp[index_def] = y
            crsdefs['ycoordinates'] = pd.Series(temp)
            temp = crsdefs.loc[:,'zcoordinates'].to_list()
            temp[index_def] = z
            crsdefs['zcoordinates'] = pd.Series(temp)
            crsdefs.loc[index_def,'yzcount'] = len(z)
            crsdefs.loc[index_def,'thalweg'] = y_middle

    ## PUT THE CRS DEFINITION BACK INTO THE CROSSDEFMODEL AND WRITE OUTPUT
    tempfile = os.path.join(output_path,'crsdef_temp.ini')
    crsdefs = crsdefs.replace({np.nan: None})
    crossdef_new = CrossDefModel(definition=crsdefs.to_dict("records"))
    crossdef_new.save(Path(tempfile))
    
    ## TEMPORARY BUG FIX, OPEN FILE AND DELETE THE ROWS ‘frictionId’, ‘frictionType’, ‘frictionValue’
    # read lines
    def_data = []
    with open(tempfile,'r') as file_temp:
        data= csv.reader(file_temp)
        for row in data:
            def_data.append(row)

    filename = os.path.join(output_path,'crsdef.ini')
    writefric = False
    with open(filename, 'w') as file_def:
        for row in def_data:
            if row != []:
                if len(row[0]) > 8:
                    if row[0][4:6] == 'id':
                        def_id = row[0].split('= ')[1]
                        index_def = crsdefs.loc[crsdefs['id'] == def_id,:].index.to_list()[0] # index of the cross section definition 
                        if isinstance(crsdefs.loc[index_def,'frictionids'],list):
                            writefric = True
                        else:
                            writefric = False
                    if row[0][4:8] == 'fric':
                        if writefric == True:
                            file_def.write(row[0])
                            file_def.write("\n")
                    else:
                        file_def.write(row[0])
                        file_def.write("\n")
            else:
                file_def.write("\n")
    os.remove(tempfile)
    
    ## PUT THE CRS DEFINITION BACK INTO THE CROSSDEFMODEL AND WRITE OUTPUT
    filename = os.path.join(output_path,'crsloc.ini')
    crslocs = crslocs.replace({np.nan: None})
    crossloc_new = CrossLocModel(crosssection=crslocs.to_dict("records"))
    crossloc_new.save(Path(filename))   
    
    print('Finished adjusting ' + str(len(loc_ids)) + 'profile definitions'   

def split_duplicate_crsdef_references(crslocs,crsdefs,loc_ids=[]):
    """
    If a cross section definition is used at more than 1 cross section locatoin, it is called a shared definition.
    Sometimes, for example when use want to adjust 1 profile at 1 location, it is wise to split these shared definitions.
    Otherwise, also the cross sections elsewhere will have an adjusted profile
    
    
    This function splits the shared defintions for those cross section LOCATIONS that are included in the list loc_ids

    Parameters
    ----------
    crsloc : pandas geodataframe with the cross section locations
    crsdefs : pandas geodataframe with the cross section defintions
    loc_ids : list with the cross section locations for which the defintions should be checked and (if needed) splitted.
            if empty, the function will check all cross section locations

    Returns
    -------
    crsloc : pandas geodataframe with the cross section locations
    crsdefs : pandas geodataframe with the cross section defintions

    """

    counter = 0    
    
    if loc_ids == []: # if list is empyt, all cross section locations will be checked
        loc_ids = crslocs['id'].tolist() 
        
    for loc_id in loc_ids:
        def_id = crslocs.loc[crslocs['id'] == loc_id,'definitionid'].iloc[0] # ID cross section definition
        index_loc = crslocs.loc[crslocs['id'] == loc_id,:].index.to_list()[0] # index of the cross section location
        if (crslocs['definitionid'] == def_id).sum() > 1: # if a cross section definition is used more than once
            if (crslocs['definitionid'] == def_id).sum() == 2:              
                # if a definition is used 2 times, the label shared definition should be adjusted
                # the first location that refers to the shared definition, gets its own new cross section definition
                # the location that still refers to the shared definition remains untouched. However, the definition itself does not get the label 'shared_definition' anymore
                crsdefs.loc[crsdefs.id == def_id,'isshared'] = np.nan # remove the label 'shared_definition'
                # if a definition is used more than 2 times, we can keep the shared definition. But we should make a new definition for our location.

            #determine the new ID of the cross section definition
            i = 1
            new_def_id = str(def_id) + "_-_" + str(i)
            while len(crslocs.loc[crslocs['definitionid'] == new_def_id,:])>0:
                i = i + 1
                new_def_id = str(def_id) + "_-_" + str(i)
    
            # enter the new information in the dataframes
            crslocs.loc[index_loc,'definitionid'] = new_def_id
            new_row = crsdefs.loc[crsdefs.id == def_id,:].copy()
            new_row['id'] = new_def_id
            new_row['isshared'] = np.nan
            new_row.name = len(crsdefs)
            crsdefs = crsdefs.append(new_row)
            counter = counter + 1
            crsdefs.reset_index(drop=True,inplace=True)
    
    if counter > 0: 
        print('Splitted the shared definitions for the selected profile loctions')
        print('Therefore, created '+ str(counter) + ' new profile definitions')
    else:
        print('Did not have to split shared definitions for the selected profile loctions') 
                  
    return crslocs,crsdefs


if __name__ == "__main__":
    # Read shape
    mdu_path = Path(r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\flowFM.mdu")
    netnc_path = r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    # mdu_path = Path(r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel\1D2D-DIMR\dflowfm\flowFM.mdu") 
    shape_path = r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\shapes\change_fric.shp"
    column_vertical = 'bodem_m'
    vertical_vertical_distance_type = 'distance'
    column_horizontal = 'breedte'
    output_path = r'C:\Users\devop\Desktop'
    change_depth_crosssection(mdu_path, netnc_path, shape_path, column_horizontal, column_vertical, vertical_vertical_distance_type, output_path)
