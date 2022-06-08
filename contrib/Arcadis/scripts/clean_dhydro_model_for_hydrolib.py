from pathlib import Path
import os
import shutil
from distutils import dir_util

def clean_model(mdufile):
    '''
    This function cleans up the dflowfm folder to make sure it can be loaded into hydrolib.
    The user should take care and look at the results if something is missing.
    
    The fixed weirs are stored elsewhere and removed from the mdu file. The roughness files are stored in a seperate folder.
    The mdu is changed.
    
    Parameters
    ----------
    mdufile : path 
        Path to mdu file
        
    Returns
    -------
    None. Cleans up folders. Makes a "original" map to make sure the original model is not lost.

    '''
    parentdir = Path(os.path.dirname(mdufile)).parent
    if os.path.isdir(os.path.join(parentdir,"dflowfm_original")) == False:
        os.mkdir(os.path.join(parentdir,"dflowfm_original"))
        dir_util.copy_tree(os.path.dirname(mdufile),os.path.join(parentdir,"dflowfm_original"))
        print("Originele dflowfm folder gekopieerd naar 'dflowfm_original'")
    else:
        print("folder already copied.")
    
    clean_pliz(mdufile)
    clean_roughness(mdufile)
    clean_grw(mdufile)
    

#delete fixed weirs from mdu and place the files in a cleanup map
def clean_pliz(mdufile):
    parentdir = os.path.dirname(mdufile)
    print (parentdir)
    print ("Start opschonen model, pliz bestanden verplaatsen naar cleanup_pliz map.")
    try:
        os.mkdir(os.path.join(parentdir,"cleanup_pliz"))
    except:
        print('cleanup folder al aanwezig')
        pass
    
    for file in os.listdir(parentdir):
        if file.endswith(".pliz") or file.endswith(".pli"):
            shutil.move(os.path.join(parentdir, file), os.path.join(parentdir,"cleanup_pliz"))
    
    print ("Start opschonen mdu file, FixedWeirFile hashtaggen.")        
    with open(mdufile,"r") as file:
        mdu = file.readlines()
        
    for num, line in enumerate(mdu,1):
        param = line[:18].strip()
        if param == "FixedWeirFile":          
            line = "".join([line[:20],"#",line[20:]])
            mdu[num-1] = line
    
    with open(mdufile, "w") as file:
        file.writelines(mdu)
    print ("pliz files opgeschoond")

#place roughness in separate folder to load it into hydrolib
def clean_roughness(mdufile):
    print('Start verplaatsen roughness files naar een aparte map.')
    parentdir = os.path.dirname(mdufile)
    try:
        os.mkdir(os.path.join(parentdir,"roughness"))
    except:
        print("roughness folder al aanwezig")
    
    for roughfile in os.listdir(parentdir):
        if roughfile.startswith("roughness-"):
            shutil.move(os.path.join(parentdir, roughfile), os.path.join(parentdir,"roughness"))
    
    print ('mdu aanpassen dat roughness in aparte map staat')
    with open(mdufile,"r") as file:
        mdu = file.readlines()
        
    for num, line in enumerate(mdu,1):
        param = line[:18].strip()
        if param == "FrictFile":          
            line = line.replace("roughness-","roughness/roughness-")
            mdu[num-1] = line

    with open(mdufile, "w") as file:
        file.writelines(mdu)
        
    print ("Start opschonen roughness definities, enters verwijderen bij frictionValues")
    parentdir = os.path.dirname(mdufile)
    roughnessdir = os.path.join(parentdir, "roughness")
    
    for file in os.listdir(roughnessdir):     
        j= 0
        while j<20:
            with open(os.path.join(roughnessdir,file),"r") as roughnessfile:
                roug = roughnessfile.readlines()
            for num, line in enumerate(roug,0):
                param = line[:18].strip()
                if param == "frictionValues":
                    a = roug[num+1].strip()
                    #Check if the next line is not empty
                    if a:
                        roug[num] = "".join(["    ",roug[num].strip()," "])
                        
            with open(os.path.join(roughnessdir,file), "w") as roughnessfile:
                roughnessfile.writelines(roug)
            j += 1
    print ("roughness opgeschoond")    
    
#Clean [grw] out of the mdufile
def clean_grw(mdufile):
    print ('[grw] verwijderen uit mdu file')
    with open(mdufile,"r") as file:
        mdu = file.readlines()
        
    for num, line in enumerate(mdu,1):
        param = line[:18].strip()
        if param == "[grw]": #or param == "[grw]" or param == "[grw]" or param == "[grw]":          
            line = line.replace(param,"")
            mdu[num-1] = line
    
    with open(mdufile, "w") as file:
        file.writelines(mdu)
    print ('[grw] verwijderd')

if __name__ == "__main__":
    mdu_path = r'C:\scripts\arcadis-hydrology-toolbox\samples\dhydro\dr51_conceptmodel\dflowfm\dr_51.mdu'
    clean_model(mdu_path)
    