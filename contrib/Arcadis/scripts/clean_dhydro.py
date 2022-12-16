# =========================================================================================
#
# License: LGPL
#
# Author:
#           Arjon Buijert and Robert de Lange, Arcadis
#           Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================

import os
import shutil
from distutils import dir_util
from pathlib import Path

import numpy as np


def clean_model(mdufile):
    """
    This function cleans up the dflowfm folder to make sure it can be loaded into hydrolib.
    The user should take care and look at the results if something is missing.

    The fixed weirs are stored elsewhere and removed from the mdu file. The roughness files are stored in a seperate folder.
    The mdu is changed.

    ___________________________________________________________________________________________________________

    Parameters:
        mdufile : path
            Path to mdu file
    ___________________________________________________________________________________________________________

    Returns:
        None. Cleans up folders. Makes a "original" map to make sure the original model is not lost.

    """
    parentdir = Path(os.path.dirname(mdufile)).parent
    if os.path.isdir(os.path.join(parentdir, "dflowfm_original")) == False:
        os.mkdir(os.path.join(parentdir, "dflowfm_original"))
        dir_util.copy_tree(
            os.path.dirname(mdufile), os.path.join(parentdir, "dflowfm_original")
        )
        print("Originele dflowfm folder gekopieerd naar 'dflowfm_original'")
    else:
        print("folder already copied.")

    clean_pliz(mdufile)
    clean_roughness(mdufile)
    clean_grw(mdufile)


# delete fixed weirs from mdu and place the files in a cleanup map
def clean_pliz(mdufile):
    parentdir = os.path.dirname(mdufile)
    print(parentdir)
    print("Start opschonen model, pliz bestanden verplaatsen naar cleanup_pliz map.")
    try:
        os.mkdir(os.path.join(parentdir, "cleanup_pliz"))
    except:
        print("cleanup folder al aanwezig")
        pass

    for file in os.listdir(parentdir):
        if file.endswith(".pliz") or file.endswith(".pli"):
            shutil.move(
                os.path.join(parentdir, file), os.path.join(parentdir, "cleanup_pliz")
            )

    print("Start opschonen mdu file, FixedWeirFile hashtaggen.")
    with open(mdufile, "r") as file:
        mdu = file.readlines()

    for num, line in enumerate(mdu, 1):
        param = line[:18].strip()
        if param == "FixedWeirFile":
            line = "".join([line[:20], "#", line[20:]])
            mdu[num - 1] = line

    with open(mdufile, "w") as file:
        file.writelines(mdu)
    print("pliz files opgeschoond")


# place roughness in separate folder to load it into hydrolib
def clean_roughness(mdufile):
    print("Start verplaatsen roughness files naar een aparte map.")
    parentdir = os.path.dirname(mdufile)
    try:
        os.mkdir(os.path.join(parentdir, "roughness"))
    except:
        print("roughness folder al aanwezig")

    for roughfile in os.listdir(parentdir):
        if roughfile.startswith("roughness-"):
            shutil.move(
                os.path.join(parentdir, roughfile), os.path.join(parentdir, "roughness")
            )

    print("mdu aanpassen dat roughness in aparte map staat")
    with open(mdufile, "r") as file:
        mdu = file.readlines()

    for num, line in enumerate(mdu, 1):
        param = line[:18].strip()
        if param == "FrictFile":
            line = line.replace("roughness-", "roughness/roughness-")
            mdu[num - 1] = line

    with open(mdufile, "w") as file:
        file.writelines(mdu)

    print("Start opschonen roughness definities, enters verwijderen bij frictionValues")
    parentdir = os.path.dirname(mdufile)
    roughnessdir = os.path.join(parentdir, "roughness")

    for file in os.listdir(roughnessdir):
        j = 0
        while j < 20:
            with open(os.path.join(roughnessdir, file), "r") as roughnessfile:
                roug = roughnessfile.readlines()
            for num, line in enumerate(roug, 0):
                param = line[:18].strip()
                if param == "frictionValues":
                    a = roug[num + 1].strip()
                    # Check if the next line is not empty
                    if a:
                        roug[num] = "".join(["    ", roug[num].strip(), " "])

            with open(os.path.join(roughnessdir, file), "w") as roughnessfile:
                roughnessfile.writelines(roug)
            j += 1
    print("roughness opgeschoond")


# Clean [grw] out of the mdufile
def clean_grw(mdufile):
    print("[grw] verwijderen uit mdu file")
    with open(mdufile, "r") as file:
        mdu = file.readlines()

    for num, line in enumerate(mdu, 1):
        param = line[:18].strip()
        if (
            param == "[grw]"
        ):  # or param == "[grw]" or param == "[grw]" or param == "[grw]":
            line = line.replace(param, "")
            mdu[num - 1] = line

    with open(mdufile, "w") as file:
        file.writelines(mdu)
    print("[grw] verwijderd")


def remove_double_friction_definitions(crslocs, crsdefs, dict_frictions):
    """
    In D-Hydro, you can define frictions in different files and different ways.
    However, there is a certain hierarchy. Some friction definitions are
    more important than others

    In a d-hydro model (especially when it is a imported Sobek 2 model)
    the friction definitions are sometimes messy and contradictory.
    Unintentionally, there are multiple friction definitions for the
    same cross section and/or channel. This function tries to repair that,
    it makes sure every channel/cross section has only one friction definition

    This function cleans the crslocs.ini, crsdefs.ini and the separate
    roughness-###.ini files.

    The hierarchy of friction definitions (from high to low importance)
    1. Branch constant/chainge friction from roughness-Channels.ini.
    Therefore, in crsdef.ini, the the cross section should refer to
    frictionID "Channels"
    2. Global value "Channels" from roughness-Channels.ini.
    Therefore, in crsdef.ini, the the cross section should refer to
    frictionID "Channels"
    3. On lanes: constant/chainge friction from roughness-###.ini.
    Therefore, in crsdef.ini, the the cross section should refer to
    frictionID "###"
    4. Global value "###" from roughness-###.ini.
    Therefore, in crsdef.ini, the the cross section should refer to
    frictionID "###"
    In the D-Hydro GUI, this is still called 'on lanes'
    5. Global value from roughness-Channels.ini
    When a branch has no specific constant/chainage in roughness-Channels.ini
    and also the requirements for level 2 and 2 are not met, a global value
    is used. This is the global value "Channels"

    ___________________________________________________________________________________________________________

    Parameters:
        crslocs : DataFrame
            Data of cross section locations
            Imported from crslocs.ini with the function 'read_locations'
        crsdefs : DataFrame
            Data of cross section definitions
            Imported from crslocs.ini with the function 'read_locations'
        dict_frictions: Dictionary
            Data of the branch friction values
            Imported from the roughness-###.ini files with the function 'friction2dict'

    ___________________________________________________________________________________________________________

    Returns:
        Cleaned dataframes crslocs and crsdefs en clead dictionary dict_frictions

    """

    friction_files = list(dict_frictions.keys())

    # Cleaning roughness-###.ini file(s)
    # If a branch is included in a roughness-###.ini, but none of the cross sections
    # on the branch refer to that, the information obsolete/redundant
    for file in friction_files:
        if len(dict_frictions[file]) > 0:  # only loop through file if it consists lines
            branches_list = dict_frictions[file]["branchid"].to_list()
            for branch_id in branches_list:
                indices_loc = list(crslocs[crslocs["branchid"] == branch_id].index)
                roughness_files = []
                for index_loc in indices_loc:
                    def_id = crslocs.loc[index_loc, "definitionid"]
                    index_def = list(crsdefs[crsdefs["id"] == def_id].index)[0]
                    if crsdefs.loc[index_def, "frictionids"] != None:
                        roughness_files = roughness_files + list(
                            set(crsdefs.loc[index_def, "frictionids"])
                        )
                roughness_files = list(set(roughness_files))
            if file not in roughness_files:
                i = dict_frictions[file].index[
                    dict_frictions[file]["branchid"] == branch_id
                ][0]
                dict_frictions[file] = dict_frictions[file].drop(labels=i, axis=0)

    # The "on lanes" principle is meant for frictions that vary over a
    # cross section. Sometimes, yz cross sections get a on lanes friction,
    # while this is unnecessary. In those situations, the cross sections
    # have only 1 friction and they do not vary over the width.
    # This can be corrected. Frictions can be renamed as "Channels"
    for file in friction_files:
        if file != "Channels":
            if (
                len(dict_frictions[file]) > 0
            ):  # only loop through file if it consists lines
                branches_list = dict_frictions[file]["branchid"].to_list()
                for branch_id in branches_list:
                    roughness_files = []
                    indices_loc = list(crslocs[crslocs["branchid"] == branch_id].index)
                    def_ids = crslocs.loc[indices_loc, "definitionid"]
                    for def_id in def_ids:
                        index_def = list(crsdefs[crsdefs["id"] == def_id].index)[0]
                        if file in list(set(crsdefs.loc[index_def, "frictionids"])):
                            roughness_files = roughness_files + list(
                                set(crsdefs.loc[index_def, "frictionids"])
                            )
                    roughness_files = list(set(roughness_files))
                    # If the cross sections on branch X that refer to file/roughness
                    # Y, only refer to that roughness (sectioncount = 1), it is not an
                    # actual on lanes principle. We can move this definition to
                    # roughness-channels.ini if the branch is not already included in
                    # that file or if the roughness was the same
                    if roughness_files == [file]:
                        if (
                            branch_id
                            not in dict_frictions["Channels"]["branchid"].to_list()
                        ):
                            i = dict_frictions[file].index[
                                dict_frictions[file]["branchid"] == branch_id
                            ][0]
                            dict_frictions["Channels"] = dict_frictions[
                                "Channels"
                            ].append(dict_frictions[file].loc[i, :])
                            dict_frictions["Channels"].reset_index(
                                drop=True, inplace=True
                            )
                            dict_frictions[file] = dict_frictions[file].drop(
                                labels=i, axis=0
                            )
                            for def_id in def_ids:
                                index_def = list(
                                    crsdefs[crsdefs["id"] == def_id].index
                                )[0]
                                crsdefs.loc[index_def, "frictionids"] = ["Channels"]
    return crslocs, crsdefs, dict_frictions


def clean_friction_files(dict_frictions):
    """
    Remove unnecessary information from the roughness-###.ini files

    ___________________________________________________________________________________________________________

    Parameters:
        dict_frictions: Dictionary
            Data of the branch friction values
            Imported from the roughness-###.ini files with the function 'friction2dict'

    ___________________________________________________________________________________________________________

    Returns:
        Cleaned dictionary dict_frictions

    """

    friction_files = list(dict_frictions.keys())
    # Cleaning roughness-###.ini file(s)
    for file in friction_files:
        if len(dict_frictions[file]) > 0:  # only loop through file if it consists lines
            for index, row in dict_frictions[file].iterrows():
                if (
                    len(list(set(dict_frictions[file].loc[index, "frictionvalues"])))
                    == 1
                ):  # if only 1 unique friction value is given
                    if file == "Channels":
                        dict_frictions[file].loc[index, "numlocations"] = 0
                        dict_frictions[file].loc[index, "frictionvalues"] = [
                            list(set(dict_frictions[file].loc[index, "frictionvalues"]))
                        ]
                        dict_frictions[file].loc[index, "chainage"] = None
                    else:
                        dict_frictions[file].loc[index, "numlocations"] = 1
                        dict_frictions[file].loc[index, "frictionvalues"] = [
                            list(set(dict_frictions[file].loc[index, "frictionvalues"]))
                        ]
                        dict_frictions[file].loc[index, "chainage"] = [[0]]
    return dict_frictions


def clean_crsdefs(crsdefs):
    """
    Remove unnecessary information from the crsdef.ini file

    ___________________________________________________________________________________________________________

    Parameters:
        crsdefs : DataFrame
            Data of cross section definitions
            Imported from crslocs.ini with the function 'read_locations'

    ___________________________________________________________________________________________________________

    Returns:
        Cleaned dataframe crsdefs

    """

    # Cleaning crsdef.ini
    # Remove unnecessary information, simplify the data
    for index_def, row in crsdefs.iterrows():
        # If yz-profile, than conveyace is default segmentened
        if (
            crsdefs.loc[index_def, "type"] == "yz"
            and crsdefs.loc[index_def, "conveyance"] == "segmented"
        ):
            crsdefs.loc[index_def, "conveyance"] = None
            # If yz-profile and all frictionid are the same, they can be merged
            if crsdefs.loc[index_def, "frictionpositions"] != None:
                if (
                    len(crsdefs.loc[index_def, "frictionpositions"]) > 1
                    and len(list(set(crsdefs.loc[index_def, "frictionids"]))) == 1
                ):
                    crsdefs.loc[index_def, "frictionids"] = list(
                        set(crsdefs.loc[index_def, "frictionids"])
                    )[0]
                    crsdefs.loc[index_def, "sectioncount"] = 1
        # If yz-profile, sectioncount is 1 and frictionID is Channel, no need to
        # write down frictionpositions
        if crsdefs.loc[index_def, "sectioncount"] == 1:
            if list(set(crsdefs.loc[index_def, "frictionids"])) == ["Channels"]:
                crsdefs.loc[index_def, "frictionpositions"] = None
    return crsdefs


def split_duplicate_crsdef_references(crslocs, crsdefs, loc_ids=[]):
    """
    If a cross section definition is used at more than 1 cross section location,
    it is called a shared definition. Sometimes, for example when use want to
    adjust 1 profile at 1 location, it is wise to split these shared definitions.
    Otherwise, also the cross sections elsewhere will have an adjusted profile

    This function splits the shared defintions for those cross section LOCATIONS
    that are included in the list loc_ids
    ___________________________________________________________________________________________________________

    Parameters
        crslocs : DataFrame
            Data of cross section locations
            Imported from crslocs.ini with the function 'read_locations'
        crsdefs : DataFrame
            Data of cross section definitions
            Imported from crslocs.ini with the function 'read_locations'
        loc_ids : list
            List with the cross section locations for which the defintions should
            be checked and (if needed) splitted.
            If empty, the function will check all cross section locations
    ___________________________________________________________________________________________________________
    Returns
        Cleaned dataframes crslocs and crsdefs

    """

    counter = 0

    if loc_ids == []:  # if list is empyt, all cross section locations will be checked
        loc_ids = crslocs["id"].tolist()

    for loc_id in loc_ids:
        def_id = crslocs.loc[crslocs["id"] == loc_id, "definitionid"].iloc[
            0
        ]  # ID cross section definition
        index_loc = crslocs.loc[crslocs["id"] == loc_id, :].index.to_list()[
            0
        ]  # index of the cross section location
        if (
            crslocs["definitionid"] == def_id
        ).sum() > 1:  # if a cross section definition is used more than once
            if (crslocs["definitionid"] == def_id).sum() == 2:
                # if a definition is used 2 times, the label shared definition should be adjusted
                # the first location that refers to the shared definition, gets its own new cross section definition
                # the location that still refers to the shared definition remains untouched. However, the definition itself does not get the label 'shared_definition' anymore
                crsdefs.loc[
                    crsdefs.id == def_id, "isshared"
                ] = np.nan  # remove the label 'shared_definition'
                # if a definition is used more than 2 times, we can keep the shared definition. But we should make a new definition for our location.

            # determine the new ID of the cross section definition
            i = 1
            new_def_id = str(def_id) + "_-_" + str(i)
            while len(crslocs.loc[crslocs["definitionid"] == new_def_id, :]) > 0:
                i = i + 1
                new_def_id = str(def_id) + "_-_" + str(i)

            # enter the new information in the dataframes
            crslocs.loc[index_loc, "definitionid"] = new_def_id
            new_row = crsdefs.loc[crsdefs.id == def_id, :].copy()
            new_row["id"] = new_def_id
            new_row["isshared"] = np.nan
            new_row.name = len(crsdefs)
            crsdefs = crsdefs.append(new_row)
            counter = counter + 1
            crsdefs.reset_index(drop=True, inplace=True)

    if counter > 0:
        print("Splitted the shared definitions for the selected profile loctions")
        print("Therefore, created " + str(counter) + " new profile definitions")
    else:
        print(
            "Did not have to split shared definitions for the selected profile loctions"
        )

    return crslocs, crsdefs


if __name__ == "__main__":
    mdu_path = r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel1D_referentie_clean\dflowfm\FlowFM.mdu"
    clean_model(mdu_path)
