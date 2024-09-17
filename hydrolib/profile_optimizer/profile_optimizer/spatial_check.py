import os
import shutil
import warnings
from pathlib import Path

import colorama
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style

from hydrolib.core.dflowfm.crosssection.models import CrossDefModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dimr.models import DIMR, FMComponent, Start


class SpatialCheck:
    def __init__(
        self, base_model_fn: Path, output_dir: Path, selected_cross_loc, branches
    ):
        self.model_name = base_model_fn.name
        self.source_folder = base_model_fn.parent
        self.gdf_select = selected_cross_loc  # geodataframe
        self.branches = branches  # geodataframe
        self.output_dir = Path(output_dir)

        if os.path.exists(self.output_dir):
            warnings.warn("Output directory already exists: cleaning directory...")
            shutil.rmtree(self.output_dir)
            warnings.warn(f"Output directory cleaned! ({self.output_dir})")

        shutil.copytree(
            self.source_folder, self.output_dir
        )  # copy entire original model folder to output folder
        self.base_model = FMModel(self.output_dir / self.model_name)
        self.cross_def_model = pd.DataFrame(
            [cs.__dict__ for cs in self.base_model.geometry.crossdeffile.definition]
        )

    def preprocessing(self):
        """
        Create dataset of crosssection definition that has information about branchid, start_or_end, edge node of each crosssection

        """
        ids = list(self.gdf_select["definition"])

        # select cross sections that are inside the optimize area
        selectie = self.cross_def_model[self.cross_def_model["id"].isin(ids)]

        # add column branchid and strahler for cross sections
        for i, row in selectie.iterrows():
            selectie.loc[i, "branchid"] = self.gdf_select.loc[
                (self.gdf_select.definition == selectie.loc[i, "id"]), "branchid"
            ].values
            selectie.loc[i, "strahler"] = self.branches.loc[
                (self.branches.branchid.str[:8].values == selectie.loc[i, "branchid"]),
                "Strahler",
            ].values

        # determine start or end edge for each cross section
        selectie["chainage"] = selectie["id"].str[9:]
        # if the id only contain '' or 'down', set chainage value to 0 or 1 respectively
        for i, row in selectie.iterrows():
            if selectie.loc[i, "chainage"] == "":
                selectie.loc[i, "chainage"] = 0
            elif selectie.loc[i, "chainage"] == "down":
                selectie.loc[i, "chainage"] = 1
        selectie["chainage"] = selectie["chainage"].astype(float)

        # set crosssection with lower chainage value as 'start'(upstream) of branch, set crosssection with higher chainage value as 'end'(downstream) of branch
        pair = selectie.loc[
            (selectie.branchid.values == selectie.loc[i, "branchid"]), "chainage"
        ]
        if len(pair) > 1:
            for i, row in selectie.iterrows():
                if selectie.loc[i, "chainage"] == min(pair):
                    selectie.loc[i, "start_or_end"] = "start"
                else:
                    selectie.loc[i, "start_or_end"] = "end"
        else:
            selectie.loc[i, "start_or_end"] = "end"

        # determine edge nodes according to branches
        # check if the branchid in branches only contain id and no '_xx'
        for i, row in selectie.iterrows():
            remark = selectie.loc[i, "start_or_end"]
            selectie.loc[i, "edge_node"] = self.branches.loc[
                (self.branches.branchid.values == selectie.loc[i, "branchid"]),
                f"edge_{remark}",
            ].values

        return selectie

    def route(self, selectie):
        """
        Route to create route number and reachid for cross sections.

        Args:
            selectie: datafrome of crosssection created using function "preprocessing".

        Returns:
            selectie: datafrome of crosssection that contains route number and reachid.
        """
        # Route
        # search for row where 'start_or_end'= end & 'edge_node' appear one time
        selectie.loc[
            (selectie.start_or_end.values == "end")
            & (~selectie.edge_node.duplicated(keep=False)),
            "route",
        ] = 0
        # add route number to cross sections
        while selectie["route"].isnull().any():
            for i, row in selectie.iterrows():
                if pd.isnull(selectie.loc[i, "route"]):
                    if selectie.loc[i, "start_or_end"] == "start":
                        selectie.loc[i, "route"] = (
                            selectie.loc[
                                (
                                    selectie.branchid.values
                                    == selectie.loc[i, "branchid"]
                                )
                                & (selectie.start_or_end.values == "end"),
                                "route",
                            ].values
                            + 1
                        )
                    else:
                        selectie.loc[i, "route"] = (
                            selectie.loc[
                                (
                                    selectie.edge_node.values
                                    == selectie.loc[i, "edge_node"]
                                )
                                & (selectie.start_or_end.values == "start"),
                                "route",
                            ].values
                            + 1
                        )
        # Reach
        # set the very downstream as reach 0
        selectie.loc[
            (selectie.start_or_end.values == "end")
            & (~selectie.edge_node.duplicated(keep=False)),
            "reachid",
        ] = 0
        # find joints: the cross sections that have the same ('end' and 'edge_node'), and add reachid from 1
        joints = selectie.loc[
            selectie.duplicated(["start_or_end", "edge_node"], keep=False)
        ]
        selectie.loc[
            selectie.duplicated(["start_or_end", "edge_node"], keep=False), "reachid"
        ] = range(1, 1 + len(joints))
        # add reachid to all the cross sections
        while selectie["reachid"].isnull().any():
            for i, row in selectie.iterrows():
                if pd.isnull(selectie.loc[i, "reachid"]):
                    if selectie.loc[i, "start_or_end"] == "start":
                        selectie.loc[i, "reachid"] = selectie.loc[
                            (selectie.branchid.values == selectie.loc[i, "branchid"])
                            & (selectie.start_or_end.values == "end"),
                            "reachid",
                        ].values
                    else:
                        selectie.loc[i, "reachid"] = selectie.loc[
                            (selectie.edge_node.values == selectie.loc[i, "edge_node"])
                            & (selectie.start_or_end.values == "start"),
                            "reachid",
                        ].values

        return selectie

    def spatial_check(self, selectie):
        """
        spatial check at joints and within each reach to make sure that the bottom levels does not increase from upstream to downstream

        Args:
            selectie: datafrome of crosssection that contains route number and reachid created using function "route"

        Returns:
            selectie_checked: dataframe of checked crosssection which contains information about (old) lowest_point and new_lowest_point
        """
        colorama.init(autoreset=True)
        # compute lowest_point of cross section
        selectie["lowest_point"] = selectie.zcoordinates.apply(lambda x: min(x))
        # First, spatial check at joints
        # find joints
        joints = selectie.index[
            selectie.duplicated(["start_or_end", "edge_node"], keep=False)
        ]
        # check the bottom levels at joints, if upstream lower than downstream, change the upstream lowest_point as downstream
        edge_nodes = np.zeros(len(joints))
        joints_us = np.zeros(len(joints))
        joints_ds = np.zeros(len(joints))
        for i in range(len(joints)):
            edge_nodes[i] = selectie.loc[joints[i], "edge_node"]
            joints_us[i] = selectie.loc[joints[i], "lowest_point"]
            joints_ds[i] = selectie.loc[
                (selectie.edge_node.values == edge_nodes[i])
                & (selectie.start_or_end.values == "start"),
                "lowest_point",
            ]
            if joints_us[i] < joints_ds[i]:
                print(
                    f"{Fore.RED+ Back.YELLOW + Style.BRIGHT} At joints (edge_node={edge_nodes[i]}), upstream lower than downstream, change upstream bottom level"
                )
                selectie.loc[joints[i], "lowest_point"] = joints_ds[i]
            else:
                print(f"{Fore.RED+ Back.YELLOW + Style.BRIGHT}No depression at joints")

        # Second, spatial check within each reach
        # group by reachid, find and mark the depression in each reach from very downstream to upstream
        grouped = selectie.groupby("reachid")
        selectie_checked = pd.DataFrame()
        for key, item in grouped:
            df = grouped.get_group(key)
            # define bottomlevel_ds as the handle to store the highest bottom level in each step
            bottomlevel_ds = df.loc[df.route == df.route.min(), "lowest_point"].values
            df.loc[
                df.route == df.route.min().astype(int), "new_lowest_point"
            ] = bottomlevel_ds
            for i in range(
                df.route.min().astype(int) + 1,
                df.route.min().astype(int) + 1 + len(df.index),
            ):
                if df.loc[df.route == i, "lowest_point"].values < bottomlevel_ds:
                    df.loc[df.route == i, "new_lowest_point"] = bottomlevel_ds
                else:
                    bottomlevel_ds = df.loc[df.route == i, "lowest_point"].values
                    df.loc[df.route == i, "new_lowest_point"] = bottomlevel_ds
            print(
                df[
                    ["reachid", "route", "lowest_point", "new_lowest_point"]
                ].sort_values(by=["route"])
            )
            if df["lowest_point"].equals(df["new_lowest_point"]):
                print(
                    f"{Fore.RED+ Back.YELLOW + Style.BRIGHT}No depression within this reach"
                )
            else:
                print(f"{Fore.RED+ Back.YELLOW + Style.BRIGHT}Fill depression")
            selectie_checked = pd.concat([selectie_checked, df])

        return selectie_checked

    def fill_depression(self, selectie):
        """
        Fill the depression: increase the depression point bottom level from (old) lowest_point to new_lowest_point

        Args:
           selectie: dataframe of checked crosssection created using function "spatial_check"

        Returns:
            cross_def_new: new crosssection definition file after filling depression point

        """
        colorama.init(autoreset=True)
        if selectie["lowest_point"].equals(selectie["new_lowest_point"]):
            print(f"{Fore.RED+ Back.YELLOW + Style.BRIGHT}No depression")
        else:
            complex = selectie[selectie["yzcount"] != 4]
            simple = selectie[selectie["yzcount"] == 4]
            for i, row in simple.iterrows():
                old_z = row["zcoordinates"]
                old_y = row["ycoordinates"]
                sideslope_l = (old_z[0] - old_z[1]) / (old_y[0] - old_y[1])
                sideslope_r = (old_z[2] - old_z[3]) / (old_y[2] - old_y[3])
                increment_step = (
                    simple.loc[i, "new_lowest_point"] - simple.loc[i, "lowest_point"]
                )
                new_z = [
                    old_z[0],
                    old_z[1] + increment_step,
                    old_z[2] + increment_step,
                    old_z[3],
                ]
                new_y = [
                    old_y[0],
                    old_y[1] + (increment_step / sideslope_l),
                    old_y[2] + (increment_step / sideslope_r),
                    old_y[3],
                ]
                self.cross_def_model.at[i, "zcoordinates"] = new_z
                self.cross_def_model.at[i, "ycoordinates"] = new_y

            for i, row in complex.iterrows():
                old_z = row["zcoordinates"]
                old_y = row["ycoordinates"]
                increment_step = (
                    complex.loc[i, "new_lowest_point"] - complex.loc[i, "lowest_point"]
                )
                new_lowest_level = min(old_z) + increment_step
                low_z_positions = [
                    i
                    for i, value in enumerate(row["zcoordinates"])
                    if value < new_lowest_level
                ]
                new_positions = []
                temp_z = old_z.copy()
                new_y = old_y.copy()
                for position in low_z_positions[::-1]:
                    # Check if any position adjecent to the right is in the list:
                    if not (position + 1) in low_z_positions:
                        new_positions.insert(0, position + 1)
                        next = position + 1
                        zshift = (
                            new_lowest_level - old_z[next]
                        )  # m to move down from next point to the right
                        sideslope = (old_z[position] - old_z[next]) / (
                            old_y[position] - old_y[next]
                        )
                        yshift = zshift / sideslope
                        temp_z.insert(position + 1, old_z[next] + zshift)
                        new_y.insert(position + 1, old_y[next] + yshift)

                    # Check if any position adjecent to the left is in the list:
                    if not (position - 1) in low_z_positions:
                        new_positions.insert(0, position)
                        previous = position - 1
                        zshift = (
                            new_lowest_level - old_z[previous]
                        )  # m to move down from point to the left
                        sideslope = (old_z[position] - old_z[previous]) / (
                            old_y[position] - old_y[previous]
                        )
                        yshift = zshift / sideslope
                        temp_z.insert(position, old_z[previous] + zshift)
                        new_y.insert(position, old_y[previous] + yshift)

                new_z = []
                for z in temp_z:
                    if z < new_lowest_level:
                        new_z.append(new_lowest_level)
                    else:
                        new_z.append(z)

                self.cross_def_model.at[i, "zcoordinates"] = new_z
                self.cross_def_model.at[i, "ycoordinates"] = new_y
                self.cross_def_model.at[i, "yzcount"] = row["yzcount"] + len(
                    new_positions
                )
            print(f"{Fore.RED+ Back.YELLOW + Style.BRIGHT}Fill depression finished!")

        cross_def_df = self.cross_def_model.replace({np.nan: None})
        cross_def_new = CrossDefModel(definition=cross_def_df.to_dict("records"))

        return cross_def_new

    def save_model(self, cross_def_new):
        # save new cross section as new crs definition.ini file
        cross_def_new.save(self.output_dir / "crsdef.ini")
        cross_def_new.filepath = self.output_dir / "crsdef.ini"
        # write new cross section file to mdu
        self.base_model.geometry.crossdeffile = cross_def_new
        name = "spatial_check"
        self.base_model.filepath = self.output_dir / f"{name}.mdu"
        self.base_model.save(recurse=False)
        spatial_check_mdu_path = self.base_model.filepath

        dimr = DIMR()
        dimr.component.append(
            FMComponent(
                name=name,
                workingDir=self.output_dir,
                inputfile=self.base_model.filepath.absolute(),
                model=self.base_model,
            )
        )
        dimr.control.append(Start(name=name))
        dimr_fn = self.output_dir / "dimr_config.xml"
        dimr.save(dimr_fn)
        print("Exported spatial checked model to output folder as spatial_check.mdu")
        return spatial_check_mdu_path
