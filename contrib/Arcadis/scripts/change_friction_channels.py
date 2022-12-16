# =============================================================================
#
# License: LGPL
#
# Author: Robbert de Lange Arcadis
#
# =============================================================================

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from read_dhydro import branch_gui2df, net_nc2gdf
from hydrolib.core.io.mdu.models import FMModel, FrictionModel

# =============================================================================
# # TODO geef optie om alle oude frictions te wipen of te houden
# # (3 opties: alles wipen en nieuwe toevoegen, oude houden en nieuwe overschrijven,
# # toevoegen waar alleen global is (is al gedefinieerd of niet?))
# # new, replace & append
# # TODO Er zijn verschillende files, die allemaal overschrijven?
# # Overschrijven per branch en niet per cross section warning.
# # TODO Halverwege watergangen verwerken?
# # TODO soort friction
# # Soort friction als input meegeven.
#
# =============================================================================


def change_friction_shape(
    mdu_path, shape_path, output_path, wipe=False, replace=False, append=False
):
    """
    Function uses input shape of frictions and model to change frictions within chosen branches.
    Buffer used = 10
    ___________________________________________________________________________________________________________
        
    Parameters:
        mdu_path : Path()
            Path to mdu file of the model.
        shape_path : str
            Path to shape containing friction file.
        output_path : str
            Path to desired output folder.
    ___________________________________________________________________________________________________________
    
    Returns:
        Friction files containing new frictions.

    """

    buffer = 10

    fm = FMModel(mdu_path)
    dict_global, dict_frictions = friction2dict(fm)
    gdf_frict = gpd.read_file(shape_path)

    # Read model branches
    netfile = os.path.join(mdu_path.parent, fm.geometry.netfile.filepath)
    branches = net_nc2gdf(netfile)["1d_branches"]

    # Make sure the dataframes do not have
    branches.rename(columns={"id": "branchid"}, inplace=True)
    if "branchid" in gdf_frict.columns:
        gdf_frict.rename(columns={"branchid": "branchid_shape"}, inplace=True)

    # TODO: Filteren op branchtype (wordt nu nog pipe meegenomen)
    branchtypes = branch_gui2df(os.path.join(mdu_path.parent, fm.geometry.branchfile))

    # Intersect shape met branch
    # TODO: definieer id met friction value
    intersect = gpd.sjoin_nearest(branches, gdf_frict, max_distance=buffer)
    # TODO: Probleem Arjon (?)

    # =============================================================================
    #     crs_inf = read_locations(mdu_path, [
    #         "cross_sections_locations",
    #         "cross_sections_definition"]
    #         )
    #     crs_loc = crs_inf["cross_sections_locations"]
    #     crs_def = crs_inf["cross_sections_definition"]
    # =============================================================================

    # Todo: nog in de cross section def duiken de 'on lanes' te verbeteren.
    if replace == True:
        # write friction defintions for all branches that intersect with the user defined shape and buffer
        # remove all old friction defitions on those branches and also neglect frictions on branches that do not intersect
        new_fricts = pd.DataFrame()
        new_fricts["branchid"] = intersect["id"]
        new_fricts["frictionvalues"] = [[i] for i in intersect["frict"].values]
        new_fricts["frictionType"] = ["strickler" for i in intersect["frict"].values]
        new_fricts["numlocations"] = 1
        new_fricts["chainage"] = [[0] for i in intersect["frict"].values]

        # Todo: Assumed global friction is first friction
        writefile = FrictionModel(
            global_=dict_global[list(dict_frictions.keys())[0]].to_dict("records"),
            branch=new_fricts.to_dict("records"),
        )

        writefile.save(
            Path(output_path)
            / (str("roughness-" + list(dict_frictions.keys())[0] + ".ini"))
        )

    # if crs_def['frictionids'] not

    # TODO: Hoe halve watergangen verwerken

    # Loop through friction files, check if shape is in model and write new values

    # =============================================================================
    #     branch_values = []
    #     if append == True:
    #         for key in dict_frictions:
    #             if not dict_frictions[key].empty:
    #                  # check if intersect is in current frictionfile
    #                  in_model = intersect.assign(
    #                      result=intersect["id"].isin(dict_frictions[key].branchid)
    #                  )
    #
    #                  # TODO: Check if the id is always id with gpd.sjoin?
    #                  in_model.rename(columns={"id": "branchid"}, inplace=True)
    #
    #
    #                  # store new values inside new frictionfile
    #                  merge = dict_frictions[key].merge(in_model, how="left", on="branchid")
    #                  branch_values.append(merge.loc[merge.result == True, "branchid"].tolist())
    #         branch_values = np.unique(list(np.concatenate(branch_values).flat))
    #         missing = (set(intersect.id.values) - set(branch_values))
    # =============================================================================
    else:
        for key in dict_frictions:
            if not dict_frictions[key].empty:
                # check if intersect is in current frictionfile
                in_model = intersect.assign(
                    result=intersect["branchid"].isin(dict_frictions[key].branchid)
                )

                # TODO: Check if the id is always id with gpd.sjoin?

                # store new values inside new frictionfile
                merge = dict_frictions[key].merge(in_model, how="left", on="branchid")

                merge.loc[merge.result == True, "frictionvalues"] = merge.loc[
                    merge.result == True, "frict"
                ]

                # Change amount of locations to 0, because the friction is defined on the branch
                merge.loc[merge.result == True, "numlocations"] = 1
                merge.loc[merge.result == True, "chainage"] = 0

                # change all values to list to make importable into hydrolib
                merge["frictionvalues"] = merge["frictionvalues"].apply(
                    lambda x: [x] if isinstance(x, float) else x
                )
                merge["chainage"] = merge["chainage"].apply(
                    lambda x: [x] if isinstance(x, int) else x
                )
                if wipe == True:
                    # write friction defintion to branches that intersect with the user defined
                    # shape and buffer AND already had a friction specified previously
                    new_branches = merge.loc[merge.result == True].copy()

                    new_branches.drop(
                        columns=in_model.columns.difference(
                            dict_frictions[key].columns
                        ),
                        inplace=True,
                    )

                    # write the file into FrictionModel and save to hydrolib
                    writefile = FrictionModel(
                        global_=dict_global[key].to_dict("records"),
                        branch=new_branches.to_dict("records"),
                    )

                    writefile.save(
                        Path(output_path) / (str("roughness-" + key + ".ini"))
                    )

                else:
                    # write friction defintion to branches that intersect with the user defined
                    # shape and buffer AND already had a friction specified previously
                    # but also keep friction definitions of all other branches that do not
                    # intersect with the shape and buffer
                    merge.drop(
                        columns=in_model.columns.difference(
                            dict_frictions[key].columns
                        ),
                        inplace=True,
                    )

                    # write the file into FrictionModel and save to hydrolib
                    writefile = FrictionModel(
                        global_=dict_global[key].to_dict("records"),
                        branch=merge.to_dict("records"),
                    )

                    writefile.save(
                        Path(output_path) / (str("roughness-" + key + ".ini"))
                    )

            else:
                writefile = FrictionModel(global_=dict_global[key].to_dict("records"))
                writefile.save(Path(output_path) / (str("roughness-" + key + ".ini")))


def friction2dict(fm):
    dfsglob = {}
    dfschan = {}

    # loop through friction files and store them in separate dictionaries
    for i in range(len(fm.geometry.frictfile)):
        dfglob = pd.DataFrame([f.__dict__ for f in fm.geometry.frictfile[i].global_])
        dfsglob[str(str(fm.geometry.frictfile[i].global_[0].frictionid))] = dfglob

        dfchan = pd.DataFrame([f.__dict__ for f in fm.geometry.frictfile[i].branch])
        dfschan[str(str(fm.geometry.frictfile[i].global_[0].frictionid))] = dfchan

    return dfsglob, dfschan


if __name__ == "__main__":

    # =============================================================================
    #     # Read shape
    #     shape_path = r"C:\scripts\HYDROLIB\HYDROLIB\contrib\Arcadis\scripts\exampledata\shapes\frictfiles.shp"
    #     # netnc_path = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    #     output_path = r"C:\TEMP\hydrolib"
    #     input_mdu = Path(
    #         r"C:\scripts\HYDROLIB\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\FlowFM.mdu"
    #     )
    #     change_friction_shape(input_mdu, shape_path, output_path)
    # =============================================================================

    # Read shape
    shape_path = r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\GIS\Friction_Dellen_1watergang.shp"
    # netnc_path = r"C:\Users\delanger3781\OneDrive - ARCADIS\Documents\DHydro\Zwolle-Minimodel\Zwolle-Minimodel\1D2D-DIMR\dflowfm\FlowFM_net.nc"
    output_path = r"C:\Users\devop\Desktop\temp"
    input_mdu = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\Flow1D.mdu"
    )
    change_friction_shape(input_mdu, shape_path, output_path, replace=False, wipe=False)
