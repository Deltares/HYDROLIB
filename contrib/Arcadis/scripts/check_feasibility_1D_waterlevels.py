# =========================================================================================
#
# License: LGPL
#
# Author: Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================
import math
import os
from pathlib import Path

import pandas as pd
from read_dhydro import map_nc2gdf, net_nc2gdf, read_locations


def check_feasibility_1D_waterlevels(mdu_path, nc_path, output_path, skip_hours=0):

    """
   Check whether the 1D waterlevels of the 1D flow network are higher than
   the left and right embankments in cross section profiles

   Parameters:
        mdu_path : str
           Path to mdu file containing the D-hydro model structure
        nc_path
            Path to the FlowFM_map.nc, containing the results of the simulation 
        output_path
            Path where the shapefile is saved as output
        skip_hours: int
            Number of hours at the front of the simulation results that
            should be skipped (i.e. because model is not stable yet)

    Returns:
        Shapefile with the margins of the left and right embankment
___________________________________________________________________________________________________________
   Warning: function currently only checks yz and zwRiver cross sections
               ...
    """

    ## READ DATA
    # Read branches and meshnodes from net.nc
    network = net_nc2gdf(nc_path)

    # Read branches
    branches = network["1d_branches"]
    branches["branchnr"] = branches.index
    branches = branches.rename(columns={"id": "branchid"})
    branche_ids = branches[["branchid", "branchnr"]].copy()

    # Read meshnodes
    # The dataframe do include a branch number, but not a branch id
    meshnodes = network["1d_meshnodes"]
    meshnodes = meshnodes.rename(columns={"id": "meshnodeid"})
    meshnodes = meshnodes.rename(columns={"branch": "branchnr"})
    meshnodes = pd.merge(meshnodes, branche_ids, on="branchnr")

    # Read locations and definitions of cross sections + structures
    gdfs_results = read_locations(
        mdu_path,
        ["cross_sections_locations", "cross_sections_definition", "structures"],
    )
    crslocs = gdfs_results["cross_sections_locations"]
    crsdefs = gdfs_results["cross_sections_definition"]
    loc_ids = crslocs["id"].tolist()
    structures = gdfs_results["structures"]

    ## CREATE FULL OVERVIEW OF HOW MESHNODES ARE CONNECTED TO BRANCHES
    # The calculated waterlevels are stored at the meshnodes
    # The dataset of mesh nodes is not complete
    # Some meshnodes form the start/end nodes of a branch (= connection node) and
    # lie at intersections of different branches
    # Now however, only 1 of the branches is listed in the dataframe
    # We have to complement the dataframe because, later on, we have to know exactly which
    # meshnodes lie on a specific branch
    # The code below corrects this and finds out which mesh node is in fact
    # connection node and connects multiple branches
    # The difficulty is that the mesh nodes are a little bit shifted and do not lie at
    # exact same locations (decimals rounding) as the connection nodes (= start/end points of branches)
    # and the id's can be different (sometimes an underscorre is added) and

    # In the function net_nc2gdf a dataframe is made with all nodes and corresponding
    # connected branches, there are many duplicates in the dataframe
    # However, we can use the dataframe to find out which meshnodes are in fact
    # connection nodes
    total_nodes_per_branch = network["1d_total_nodes_per_branch"]
    total_nodes_per_branch = total_nodes_per_branch.rename(
        columns={"branch": "branchnr"}
    )
    total_nodes_per_branch = total_nodes_per_branch.rename(columns={"id": "nodeid"})
    total_nodes_per_branch = pd.merge(
        total_nodes_per_branch, branche_ids, on="branchnr"
    )

    for i, row in branches.iterrows():
        branchid = branches.loc[i, "branchid"]

        # dataframe which nodes lie on the branch
        nodes_on_branch = total_nodes_per_branch[
            total_nodes_per_branch["branchid"] == branchid
        ].copy()

        for j, row2 in nodes_on_branch.iterrows():
            # Check if the node with the same ID was already in the dataframe
            # of the meshnodes
            selection = meshnodes[
                meshnodes["meshnodeid"] == nodes_on_branch.loc[j, "nodeid"]
            ].copy()
            if len(selection) >= 1:  #
                # The node from the dataframe 'total_nodes_per_branch'
                # has a similar id as one that was already in the meshnode dataframe
                # Now,check if the meshnode is connected to a branch we didn't know about
                if (
                    nodes_on_branch.loc[j, "branchid"]
                    not in selection["branchid"].values
                ):
                    # New information, we find out that the meshnode is
                    # connected to a new branch we didnot know about
                    new_row = selection.iloc[[0]].copy()
                    new_row["branchid"] = branchid
                    new_row["branchnr"] = nodes_on_branch.loc[j, "branchnr"]
                    new_row["offset"] = nodes_on_branch.loc[j, "offset"]
                    meshnodes = meshnodes.append(new_row)
                    meshnodes.reset_index(drop=True, inplace=True)
            else:
                selection = pd.DataFrame()
                if (
                    nodes_on_branch.loc[j, "nodeid"] + "_1"
                    in meshnodes["meshnodeid"].values
                ):
                    # It is also possible that the row in the dataframe
                    # 'total_nodes_per_branch' refers to a connection node that
                    # lies almost at a similar location as a known meshnode
                    # but the id is a little bit different for meshnodes
                    # , sometimes a "_1" is added
                    # Check whether this is the case
                    selection = meshnodes[
                        meshnodes["meshnodeid"]
                        == nodes_on_branch.loc[j, "nodeid"] + "_1"
                    ].copy()
                elif nodes_on_branch.loc[j, "nodeid"][0:4].isnumeric():
                    # Sometimes, a mesh node is created with the coordinates as the name
                    # and lies on the connection node
                    selection = meshnodes[
                        meshnodes.intersects(
                            nodes_on_branch.loc[j, "geometry"].buffer(0.01)
                        )
                    ]
                if len(selection) >= 1:
                    if (
                        nodes_on_branch.loc[j, "branchid"]
                        not in selection["branchid"].values
                    ):
                        # Found a suitable node that was not yet registered
                        # Add it to the dataframe of the meshnodes
                        if (
                            nodes_on_branch.loc[j, "geometry"]
                            .buffer(0.01)
                            .intersects(selection.geometry.iloc[0])
                        ):
                            new_row = selection.iloc[0].to_frame().T
                            new_row["branchid"] = branchid
                            new_row["branchnr"] = nodes_on_branch.loc[j, "branchnr"]
                            new_row["offset"] = nodes_on_branch.loc[j, "offset"]
                            meshnodes = meshnodes.append(new_row)
                            meshnodes.reset_index(drop=True, inplace=True)

    ## READ WATERLEVELS
    # Read calculated waterlevels at meshnodes from map.nc
    waterlevel = map_nc2gdf(nc_path, "mesh1d_s1")

    skip_hours = int(skip_hours)
    if skip_hours > 0:
        if (waterlevel.columns[-2] - waterlevel.columns[0]).seconds / 3600 > skip_hours:
            skip_steps = math.ceil(
                skip_hours
                / ((waterlevel.columns[1] - waterlevel.columns[0]).seconds / 3600)
            )
            waterlevel.drop(waterlevel.columns[0:skip_steps], inplace=True, axis=1)

    waterlevel = waterlevel.max(axis=1, numeric_only=True).to_frame()
    waterlevel = waterlevel.rename(columns={0: "max_wl"})
    waterlevel["meshnodeid"] = waterlevel.index
    meshnodes = pd.merge(meshnodes, waterlevel, on="meshnodeid")

    # LOOP OVER THE CROSS SECTIONS AND COMPARE LEFT AND RIGHT EMBANKMENT HEIGHT
    # WITH THE MAX CALCULATED WATERLEVEL (DERIVED FROM THE SURROUNDING MESH NODES)
    print("Find the height of the left and right embankment")
    crslocs["lft_ycrst"] = -9999
    crslocs["rght_ycrst"] = -9999
    crslocs["lft_zcrst"] = -9999
    crslocs["rght_zcrst"] = -9999
    crslocs["max_wl"] = -9999
    crslocs["method"] = -9999
    crslocs["margn_lft"] = -9999
    crslocs["margn_rght"] = -9999

    print("Reading embankment height and waterlevels for each profile")
    for loc_id in loc_ids:
        print(
            "Progress "
            + str(round(loc_ids.index(loc_id) / len(loc_ids) * 100, 1))
            + "%"
        )

        def_id = crslocs.loc[crslocs["id"] == loc_id, "definitionid"].iloc[
            0
        ]  # ID cross section definition
        index_loc = crslocs.loc[crslocs["id"] == loc_id, :].index.to_list()[
            0
        ]  # index of the cross section location
        index_def = crsdefs.loc[crsdefs["id"] == def_id, :].index.to_list()[
            0
        ]  # index of the cross section definition

        y = []
        z = []
        # LEFT AND RIGHT EMBANKMENT HEIGHT
        if crsdefs.loc[index_def, "type"] == "yz":
            y = crsdefs.loc[index_def, "ycoordinates"].copy()
            z = crsdefs.loc[index_def, "zcoordinates"].copy()
        elif crsdefs.loc[index_def, "type"] == "zwRiver":
            for i in range(int(crsdefs.loc[index_def, "numlevels"])):
                y = (
                    [crsdefs.loc[index_def, "flowwidths"][i] / 2 * -1]
                    + y
                    + [crsdefs.loc[index_def, "flowwidths"][i] / 2]
                )
                z = (
                    [crsdefs.loc[index_def, "levels"][i]]
                    + z
                    + [crsdefs.loc[index_def, "levels"][i]]
                )
        else:
            crslocs.loc[
                index_loc, "method"
            ] = "cannot find embankment height because it is not a yz or zwRiver-profile"

        if y != []:
            bot_value = min(z)
            bot_position = [i for i, zz in enumerate(z) if float(zz) == bot_value]

            # find height of left embankment
            i = bot_position[0]
            lft_zcrst = -9999
            while i >= 0:
                if z[i] > lft_zcrst:
                    lft_ycrst = y[i]
                    lft_zcrst = z[i]
                i = i - 1

            # find height of right embankment
            i = bot_position[-1]
            rght_zcrst = -9999
            while i <= len(y) - 1:
                if z[i] > rght_zcrst:
                    rght_ycrst = y[i]
                    rght_zcrst = z[i]
                i = i + 1

            # extra check for anomalies
            if len(bot_position) > 1:
                if max(z[bot_position[0] : bot_position[-1] + 1]) > min(
                    lft_zcrst, rght_zcrst
                ):
                    print(
                        "Profile"
                        + str(loc_id)
                        + ": in between the lowest points lies a point that is higher than the left and right embankment"
                    )

            crslocs.loc[index_loc, "lft_zcrst"] = lft_zcrst
            crslocs.loc[index_loc, "rght_zcrst"] = rght_zcrst
            crslocs.loc[index_loc, "lft_ycrst"] = lft_ycrst
            crslocs.loc[index_loc, "rght_ycrst"] = rght_ycrst
            crs_chainage = crslocs.loc[index_loc, "chainage"].copy()

            # DETERMINE MAX WATERLEVEL BASED ON SURROUNDING MESH NODES
            # Pay attention, the cross section locations are based on chainages
            # For meshnodes, this is called offset

            meshnodes_selection = meshnodes[
                meshnodes["branchid"] == crslocs.loc[index_loc, "branchid"]
            ].copy()

            if len(meshnodes_selection) == 0:
                crslocs.loc[index_loc, "method"] = "No mesh node found on the branch"
            else:
                if len(meshnodes_selection) == 1:
                    crslocs.loc[index_loc, "max_wl"] = meshnodes_selection[
                        "max_wl"
                    ].iloc[0]
                    crslocs.loc[
                        index_loc, "method"
                    ] = "Only one mesh node found on the branch"
                else:
                    if len(meshnodes_selection) > 2:
                        # select the mesh nodes at the front and at the back of the cross section
                        meshnodes_selection["distance"] = (
                            meshnodes_selection["offset"] - crs_chainage
                        )

                        meshnode_attheback = meshnodes_selection[
                            meshnodes_selection["distance"]
                            == meshnodes_selection[
                                meshnodes_selection["distance"] <= 0
                            ]["distance"].max()
                        ]
                        meshnode_atthefront = meshnodes_selection[
                            meshnodes_selection["distance"]
                            == meshnodes_selection[meshnodes_selection["distance"] > 0][
                                "distance"
                            ].min()
                        ]
                        meshnodes_selection = pd.concat(
                            [meshnode_attheback, meshnode_atthefront]
                        )
                    meshnodes_selection = meshnodes_selection.sort_values(by=["offset"])
                    meshnodes_selection.reset_index(drop=True, inplace=True)

                    # determine if there lies an structure in between the profile and one of the mesh nodes
                    structures_selection = structures[
                        structures["branchid"] == crslocs.loc[index_loc, "branchid"]
                    ]
                    if len(structures_selection) > 0:
                        # there is a structure on the same branch
                        # select meshnodes that do not lie behind a structure
                        structures_selection = structures_selection[
                            (
                                structures_selection["chainage"]
                                >= meshnodes_selection.loc[0, "offset"]
                            )
                            & (
                                structures_selection["chainage"]
                                <= meshnodes_selection.loc[1, "offset"]
                            )
                        ]

                    if (
                        len(structures_selection) > 0
                    ):  # there lies a structure in between the cross section and selected mesh nodes
                        # to determine the waterlevel at the cross section, use the mesh node that has no structure in between

                        if (
                            len(
                                structures_selection[
                                    structures_selection["chainage"] == crs_chainage
                                ]
                            )
                            >= 1
                        ):
                            # Cross section lies at exact the same location as a structure
                            crslocs.loc[
                                index_loc, "method"
                            ] = "Cross section lies at exact the same location as a structure, waterlevel interpolated between mesh nodes"
                            slope = (
                                meshnodes_selection.loc[1, "max_wl"]
                                - meshnodes_selection.loc[0, "max_wl"]
                            ) / (
                                meshnodes_selection.loc[1, "offset"]
                                - meshnodes_selection.loc[0, "offset"]
                            )
                            crslocs.loc[index_loc, "max_wl"] = (
                                slope
                                * (crs_chainage - meshnodes_selection.loc[0, "offset"])
                                + meshnodes_selection.loc[0, "max_wl"]
                            )
                        elif (
                            len(
                                structures_selection[
                                    (structures_selection["chainage"] < crs_chainage)
                                    & (
                                        structures_selection["chainage"]
                                        >= meshnodes_selection.loc[0, "offset"]
                                    )
                                ]
                            )
                            == 0
                        ):
                            # there is no structure in between the cross section and the mesh node at the back of it
                            crslocs.loc[
                                index_loc, "method"
                            ] = "There lies a structure at the front of the cross section, waterlevel determined based on mesh node at the back of it"
                            crslocs.loc[index_loc, "max_wl"] = meshnodes_selection.loc[
                                0, "max_wl"
                            ]
                        elif (
                            len(
                                structures_selection[
                                    (structures_selection["chainage"] > crs_chainage)
                                    & (
                                        structures_selection["chainage"]
                                        <= meshnodes_selection.loc[1, "offset"]
                                    )
                                ]
                            )
                            == 0
                        ):
                            # there is no structure in between the cross section and the mesh node at the front of it
                            crslocs.loc[
                                index_loc, "method"
                            ] = "There lies a structure at the back of the cross section, waterlevel determined based on mesh node at the front of it"
                            crslocs.loc[index_loc, "max_wl"] = meshnodes_selection.loc[
                                1, "max_wl"
                            ]

                    else:  # there is no structure in between the cross section and selected mesh nodes, calculate the water level at the cross section by interpolation
                        crslocs.loc[
                            index_loc, "method"
                        ] = "Cross section is surrounded by two mesh nodes and no structures, waterlevel interpolated between mesh nodes"
                        slope = (
                            meshnodes_selection.loc[1, "max_wl"]
                            - meshnodes_selection.loc[0, "max_wl"]
                        ) / (
                            meshnodes_selection.loc[1, "offset"]
                            - meshnodes_selection.loc[0, "offset"]
                        )
                        crslocs.loc[index_loc, "max_wl"] = (
                            slope
                            * (crs_chainage - meshnodes_selection.loc[0, "offset"])
                            + meshnodes_selection.loc[0, "max_wl"]
                        )

                crslocs.loc[index_loc, "margn_lft"] = (
                    crslocs.loc[index_loc, "lft_zcrst"]
                    - crslocs.loc[index_loc, "max_wl"]
                )
                crslocs.loc[index_loc, "margn_rght"] = (
                    crslocs.loc[index_loc, "rght_zcrst"]
                    - crslocs.loc[index_loc, "max_wl"]
                )

    print("Finished, now writing output as shapefile")
    crslocs = crslocs.rename(columns={"definitionid": "def_id", "locationtype": "type"})
    crslocs.to_file(os.path.join(output_path, "feasibility_1D.shp"))


if __name__ == "__main__":
    # Read shape
    mdu_path = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\Flow1D.mdu"
    )
    nc_path = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\output\Flow1D_map.nc"
    )
    output_path = r"C:\Users\devop\Desktop"
    check_feasibility_1D_waterlevels(mdu_path, nc_path, output_path)
