# =========================================================================================
#
# License: LGPL
#
# Author: Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
from read_dhydro import read_locations
from hydrolib.core.io.mdu.models import FMModel, FrictionModel
from hydrolib.core.io.crosssection.models import CrossDefModel, CrossLocModel
from change_friction_channels import friction2dict
from change_depth_crosssections import write_cross_section_data
from clean_dhydro import remove_double_friction_definitions, clean_friction_files, clean_crsdefs, split_duplicate_crsdef_references

# to do
# function currently only works for yz and zwRiver (these will be converted to yz) profiles
# it may be possible in the future to also apply zw and xyz profiles

def natural_embankment_crs_friction(
    mdu_path, shape_path, column_waterlevel, width_embak, friction_type, embank_friction, normal_friction, side
):
    """
    Function uses input shape of frictions and model to change frictions within chosen branches.

    Args:
        mdu_path : Path()
            Path to mdu file containing the D-hydro model structure
        shape_path : str
            Path to shape containing polygon areas in which friction of cross sections should be adjusted
        column_waterlevel: str
            Column in the shapefile that describes the waterlevel with which the water width/line should be calculated
        width_embak : double
            Width (meters) of the natural embankment, measured at the waterline
        friction_type: str or int
            Choice between Chezy, Manning, Strickler, deBosBijkerk (standard is Strickler) 
            Chezy - 0
            Manning - 1
            WallLawNikuradse - 2
            WhiteColebrook - 3
            StricklerNikuradse - 7
            Strickler - 8
            deBosBijkerk - 9
        embank_friction : double
            friction value with which you want the schematisize the natural embankment
        normal_friction : double
            friction value with which you want the schematisize the remaining cross section (flow area)
        side : str
            Choose either "left" or "right"

    Returns:
        Updated crsdef.ini, crsloc.ini, mdu-file and friction files in the existing model

    # to do
    # function only works for yz and zwRiver (these will be converted) profiles
    # it may be possible in the future to also apply zw and xyz profiles

    """
    print("Load input data") 
    
    #Read friction data
    output_path = os.path.dirname(mdu_path)
    fm = FMModel(mdu_path)
    dict_global, dict_frictions = friction2dict(fm)


    # Read cross section defintions and locations
    gdfs_results = read_locations(mdu_path,["cross_sections_locations","cross_sections_definition"])
    crslocs = gdfs_results["cross_sections_locations"]
    crslocs.crs = "EPSG:28992"
    crsdefs = gdfs_results["cross_sections_definition"] 
    
    #link data from shapefile with cross section locations
    areas_selection = gpd.read_file(shape_path)
    if "id" in areas_selection.columns:
        areas_selection.rename(columns={"id": "areas_id"}, inplace=True)
    
    # split shared profiles 
    loc_ids = crslocs.loc[crslocs.intersects(areas_selection.dissolve().geometry.iloc[0]), "id"].tolist()  # select only the relevant cross section locations   
    crslocs, crsdefs = split_duplicate_crsdef_references(crslocs, crsdefs, loc_ids)

    # clean the cross section locations and defintions and friction definitions
    crslocs, crsdefs, dict_frictions = remove_double_friction_definitions(crslocs, crsdefs, dict_frictions)
    dict_frictions = clean_friction_files(dict_frictions)
    crsdefs = clean_crsdefs(crsdefs)     

    # Select only the cross section locations that intersect with the shape
    crslocs_selection = crslocs.copy()   
    crslocs_selection = gpd.sjoin(crslocs_selection, areas_selection, predicate="within")
    if column_waterlevel in crslocs_selection.columns:
        crslocs_selection = crslocs_selection.rename(columns={column_waterlevel: "waterlevel"})
    else:
        print("wrong columnname, not found in attribute table of shapefile")    
    crslocs_selection = crslocs_selection.sort_index(axis=0)
    loc_ids = crslocs_selection["id"].tolist()


    # LOOP OVER THE CROSS SECTIONS LOCATIONS, DETERMINE WATERLINE AND AFTER
    # THAT DETERMINE WHERE NATURAL EMBANKMENT LIES
    print("Find the parameters for the friction sections") 
    
    # Make temporary columns
    crsdefs["waterlevel"] = -9999
    crsdefs["left_y_waterlevel"] = -9999
    crsdefs["right_y_waterlevel"] = -9999   
    crsdefs["method"] = ""
    
    print("Writing embankment height and waterlevels for each cross section")
    for loc_id in loc_ids:
        print(
            "Progress "
            + str(round(loc_ids.index(loc_id) / len(loc_ids) * 100, 1))
            + "%"
        )
        index_loc = crslocs_selection.loc[crslocs_selection["id"] == loc_id].index.to_list()[0]
        def_id = crslocs_selection.loc[index_loc,"definitionid"]  
        index_def = crsdefs.loc[crsdefs["id"] == def_id, :].index.to_list()[0]  # index of the cross section definition
        

        if crsdefs.loc[index_def, "type"] == "zwRiver":
            # convert to yz profile
            y = []
            z = []
            if type(crsdefs.loc[index_def, "totalwidths"]) == list:
                if crsdefs.loc[index_def, "totalwidths"] != []:
                    crsdefs.at[index_def, "flowwidths"] = crsdefs.loc[index_def, "totalwidths"]
            for i in range(int(crsdefs.loc[index_def, "numlevels"])):
                y = [crsdefs.loc[index_def, "flowwidths"][i]/2*-1] + y + [crsdefs.loc[index_def, "flowwidths"][i]/2]
                z = [crsdefs.loc[index_def, "levels"][i]] + z + [crsdefs.loc[index_def, "levels"][i]]           
            crsdefs.at[index_def, "zcoordinates"] = z
            crsdefs.at[index_def, "ycoordinates"] = y
            crsdefs.loc[index_def, "yzcount"] = len(z)
            crsdefs.loc[index_def, "frictionids"] = ["Channels"] # will be overwritten
            crsdefs.loc[index_def, "sectioncount"] = 1
            crsdefs.loc[index_def, "type"] = "yz"
            crsdefs.loc[index_def, "thalweg"] = max(y)/2
            crsdefs.loc[index_def, "singleValuedZ"] = 1
            
            crsdefs.loc[index_def, "numlevels"] = None
            crsdefs.loc[index_def, "levels"] = None
            crsdefs.loc[index_def, "flowwidths"] = None
            crsdefs.loc[index_def, "flowwidths"] = None
            crsdefs.loc[index_def, "leveeflowarea"] = None
            crsdefs.loc[index_def, "leveetotalarea"] = None
            crsdefs.loc[index_def, "leveecrestlevel"] = None
            crsdefs.loc[index_def, "leveebaselevel"] = None
            crsdefs.loc[index_def, "mainwidth"] = None
            crsdefs.loc[index_def, "fp1width"] = None
            crsdefs.loc[index_def, "fp2width"] = None
            crsdefs.loc[index_def, "frictiontypes"] = None
            crsdefs.loc[index_def, "frictionvalues"] = None
        
        if crsdefs.loc[index_def, "type"] == "yz":
            crsdefs.loc[index_def,"method"] ="yz cross section"
            y = crsdefs.loc[index_def, "ycoordinates"].copy()
            z = crsdefs.loc[index_def, "zcoordinates"].copy()
            crsdefs.loc[index_def,"waterlevel"] = crslocs_selection.loc[index_loc,"waterlevel"].copy()

            # Find x coordinate of left point where water line crosses cross section
            found=0
            i=0
            while found == 0 and i < len(y)-2:
                y_step = y[i]
                z_step = z[i]
                y_step_further = y[i+1]
                z_step_further = z[i+1]
                if y_step == crsdefs.loc[index_def,"waterlevel"]:
                    crsdefs.loc[index_def,"left_y_waterlevel"] = y_step
                    found = 1                          
                elif z_step > crsdefs.loc[index_def,"waterlevel"] and z_step_further < crsdefs.loc[index_def,"waterlevel"]:
                    if y_step != y_step_further: #if there is a vertical embankment 
                        crsdefs.loc[index_def,"left_y_waterlevel"] = abs((z_step - crsdefs.loc[index_def,"waterlevel"]) / ((z_step_further-z_step)/(y_step_further-y_step))) + y_step
                    else:
                        crsdefs.loc[index_def,"left_y_waterlevel"] = y_step
                    found = 1
                i = i+1
            if i >= len(y)-2:
               crsdefs.loc[index_def,"left_y_waterlevel"] = y[0]
            
            # Find x coordinate of right point where water line crosses cross section
            found=0
            i = len(y)            
            while found == 0 and i > 0:
                y_step = y[i-1]
                z_step = z[i-1]
                y_step_further = y[i-2]
                z_step_further = z[i-2]
                if y_step == crsdefs.loc[index_def,"waterlevel"]:
                    crsdefs.loc[index_def,"right_y_waterlevel"] = y_step
                    found = 1                              
                elif z_step > crsdefs.loc[index_def,"waterlevel"] and z_step_further < crsdefs.loc[index_def,"waterlevel"]:
                    if y_step != y_step_further: #if there is a vertical embankment
                        crsdefs.loc[index_def,"right_y_waterlevel"] = y_step - abs((z_step - crsdefs.loc[index_def,"waterlevel"])/((z_step_further-z_step)/(y_step_further-y_step)))
                    else:
                        crsdefs.loc[index_def,"right_y_waterlevel"] = y_step
                    found = 1
                i = i-1
            if i <= 0:
               crsdefs.loc[index_def,"right_y_waterlevel"] = y[-1]   

            # find location of natural embankment
            # write friction data in cross section definitions
            if (crsdefs.loc[index_def,"left_y_waterlevel"] == y[0]) and (crsdefs.loc[index_def,"right_y_waterlevel"] == y[-1]):   
                print("Warning, cross section def " + def_id + " lies entirely below/above waterlevel")
            if width_embak + crsdefs.loc[index_def,"left_y_waterlevel"] > crsdefs.loc[index_def,"right_y_waterlevel"]:
                print("Natural embankment for cross section def " + str(def_id), "is wider than wet area, therefore embankment is set equal to wet area")
                crsdefs.loc[index_def,"sectioncount"] = 1
                crsdefs.at[index_def,"frictionpositions"] = [y[0],y[-1]]
                crsdefs.loc[index_def,"frictionids"] = "Naturalembankment"               
            else:
                if side == "left":
                    crsdefs.loc[index_def,"sectioncount"] = 2
                    crsdefs.at[index_def,"frictionpositions"] = [y[0], round(width_embak + crsdefs.loc[index_def,"left_y_waterlevel"],3), y[-1]]
                    crsdefs.at[index_def,"frictionids"] = ["Naturalembankment","Flowsection"]
                else:
                    crsdefs.loc[index_def,"sectioncount"] = 2
                    crsdefs.at[index_def,"frictionpositions"] = [y[0], round(crsdefs.loc[index_def,"right_y_waterlevel"] - width_embak,3) , y[-1]]
                    crsdefs.at[index_def,"frictionids"] = ["Flowsection","Naturalembankment"]
            
            # If friction types and values were used before, they can be deleted
            crsdefs.loc[index_def,"frictiontypes"] = None
            crsdefs.loc[index_def,"frictionvalues"] = None
            
            # Earlier on the roughness-###.ini files were already cleaned
            # Now, a last check. If a profile lies on a branch that also
            # has a friction defintion in another roughness-###.ini file, 
            # this should be deleted
            friction_files = list(dict_frictions.keys())
            branch_id = crslocs_selection.loc[index_loc,"branchid"]
            for check_file in friction_files:
                if len(dict_frictions[check_file]) > 0:
                    branches_check_file = dict_frictions[check_file]["branchid"].to_list()
                    if branch_id in branches_check_file:
                        i = dict_frictions[check_file].index[dict_frictions[check_file]["branchid"] == branch_id][0]
                        dict_frictions[check_file] = dict_frictions[check_file].drop(labels=i, axis=0)    
        
    ## WRITE CROSS SECTION DATA
    crsdefs = crsdefs.drop(columns=["waterlevel", "left_y_waterlevel", "right_y_waterlevel", "method"])
    crslocs = crslocs.drop(columns=["geometry"])
    if "areas_id" in crslocs.columns:
        crslocs = crslocs.drop(columns=["areas_id"])
    if "index_right" in crslocs.columns:
        crslocs = crslocs.drop(columns=["index_right"])         
    write_cross_section_data(crsdefs, crslocs ,output_path)
    
    # WRITE UPDATED FRICTION FILES
    print("Write friction files") 
    # determine where rougheness-###.ini files are saved
    if os.path.isfile(os.path.join(output_path,"roughness-Channels.ini")):
        output_path = output_path
    elif  os.path.isfile(os.path.join(output_path, "roughness","roughness-Channels.ini")):
        output_path = os.path.join(output_path, "roughness")
    else:
        raise Exception("Cannot find roughness-Channels.ini file")
    
    # overwrite existing roughness-###.ini files
    for check_file in friction_files:
        tempfile = Path(output_path) / str("roughness-" + check_file + "_temp.ini")
        writefile = FrictionModel(
            global_=dict_global[list(dict_frictions.keys())[0]].to_dict("records"),
            branch = dict_frictions[check_file].to_dict("records"),
        )
        writefile.save(tempfile)           
      
        # open roughness-###.ini file and clean, remove unnecessary lines               
        with open(tempfile, "r") as file:
            def_data = file.readlines()       
        filename = Path(output_path) / str("roughness-" + check_file + ".ini")

        for num, line in enumerate(def_data, 1):
            if "\n" in line:
                line  = line.replace("\n","")
            if line != "":  
                if "=" in line:
                    if line[-1] == "=":
                        # Dont write line, this is a empty line
                        def_data[num - 1] = ""
                    elif line.split("= ")[1].strip()[0] == "#":
                        # Dont write line, this is a empty line
                        def_data[num - 1] = ""
                    if check_file == "Channels":
                        if line.split("= ")[0].strip() == "numLocations":
                            if line.split("= ")[1].strip()[0] == "0":
                                def_data[num - 1] = ""
        with open(filename, "w") as file:
            file.writelines(def_data)           
        os.remove(tempfile)                   
    
    # write new friction files for embankment and flow section
    new_fricts = pd.DataFrame()
    newglobal = dict_global[list(dict_frictions.keys())[0]]
    newglobal["frictionid"] = "Naturalembankment"
    if type(friction_type) == int:
        if friction_type == 0:
            friction_type = "Chezy"
        elif friction_type == 1:
            friction_type = "Manning"  
        elif friction_type == 2:
            friction_type = "WallLawNikuradse"              
        elif friction_type == 3:
            friction_type = "WhiteColebrook"
        elif friction_type == 7:
            friction_type = "StricklerNikuradse"     
        elif friction_type == 8:
            friction_type = "Strickler"                
        elif friction_type == 9:
            friction_type = "deBosBijkerk" 
        else:
            raise Exception("Friction type was entered as an integer, but wrong integer was given")
    if friction_type not in ["Chezy","Manning","WallLawNikuradse","WhiteColebrook","StricklerNikuradse","Strickler","deBosBijkerk"]:
        raise Exception("Wrong friction type was entered by user")
    newglobal["frictiontype"] = friction_type
    newglobal["frictionvalue"] = embank_friction
    writefile = FrictionModel(
        global_=dict_global[list(dict_frictions.keys())[0]].to_dict("records"),
        branch=new_fricts.to_dict("records"),
    )
    writefile.save(Path(output_path) / "roughness-Naturalembankment.ini")
    
    newglobal["frictionid"] = "Flowsection"
    newglobal["frictionvalue"] = normal_friction
    writefile = FrictionModel(
        global_=dict_global[list(dict_frictions.keys())[0]].to_dict("records"),
        branch=new_fricts.to_dict("records"),
    )
    writefile.save(Path(output_path) / "roughness-Flowsection.ini")
    
    # Write new MDU file
    print("Write MDU file") 
    with open(mdu_path, "r") as file:
        mdu = file.readlines()
    for num, line in enumerate(mdu, 1):
        param = line[:18].strip()
        if param == "FrictFile":
            # model for hydrolib
            if "\n" in line:
                line  = line.replace("\n","")
                if os.path.basename(output_path) == "roughness":
                    if ";roughness/roughness-Flowsection.ini" not in line: 
                        line = line + ";roughness/roughness-Flowsection.ini"
                    if ";roughness/roughness-Naturalembankment.ini" not in line:
                        line = line + ";roughness/roughness-Naturalembankment.ini"
                else:
                    if ";roughness-Flowsection.ini" not in line: 
                        line = line + ";roughness-Flowsection.ini"
                    if ";roughness-Naturalembankment.ini" not in line:
                        line = line + ";roughness-Naturalembankment.ini"                
                line = line + "\n"
            mdu[num - 1] = line
    with open(mdu_path, "w") as file:
        file.writelines(mdu)    

if __name__ == "__main__":
    # Read shape
    mdu_path = Path(r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\Flow1D.mdu")
    shape_path = Path(r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\GIS\nvo_test.shp")
    column_waterlevel = "wl"
    width_embak = 4
    friction_type = "Strickler"
    embank_friction = 1
    normal_friction = 30
    side = "left"
    
    
    natural_embankment_crs_friction(mdu_path, shape_path, column_waterlevel, width_embak, friction_type, embank_friction, normal_friction, side)
