import os
import shutil
import subprocess
import warnings
from glob import glob
from pathlib import Path

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import Point

from hydrolib.core.dflowfm.crosssection.models import CrossDefModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dimr.models import DIMR, FMComponent, Start


class ProfileOptimizer:
    def __init__(
        self,
        base_model_fn: Path,
        bat_file,
        work_dir: Path,
        output_dir: Path,
        shapefile=None,
        selectie_checked=None,
        iteration_name="Iteration",
        iteration_start_count=0,
    ):
        self.model_name = base_model_fn.name
        self.source_folder = base_model_fn.parent
        self.iteration_nr = iteration_start_count
        self.name = iteration_name
        self.bat_file = bat_file
        self._latest_bat = None
        if shapefile is not None:
            self.shapefile = Path(shapefile)
        self.selectie_checked = selectie_checked  # import selectie_checked csv file to select optimize reaches
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.work_dir):
            warnings.warn("Working directory already exists: cleaning directory...")
            shutil.rmtree(self.work_dir)
            warnings.warn(f"Working directory cleaned! ({self.work_dir})")
        shutil.copytree(self.source_folder, self.work_dir)
        self.base_model = FMModel(self.work_dir / self.model_name)

    def optimize_reach(self, reachid, strahler):
        """
        Select crosssection in the optimize area that you want to optimize, based on reachid or strahler order
        Args:
            reachid: list of reachid that you want to optimize (check the reachid in selectie_checked file)
            strahler: list of strahler order that you want to optimize (check the strahler order in selectie_checked file)
        Returns:
            ids: list of crosssection id that will be optimized
        """
        selectie = self.selectie_checked
        if reachid and strahler:
            ids = list(
                selectie[
                    selectie["reachid"].isin(reachid)
                    & selectie["strahler"].isin(strahler)
                ]["id"]
            )
        elif reachid and (not strahler):
            ids = list(selectie[selectie["reachid"].isin(reachid)]["id"])
        elif (not reachid) and strahler:
            ids = list(selectie[selectie["strahler"].isin(strahler)]["id"])
        else:
            ids = list(selectie["id"])
        if not ids:
            print(
                "WARNING: Please re-select, the reaches you select to optimize is empty"
            )
        return ids

    def constraint_at_joints(self, ids):
        """
        Check if the crosssection selected using function "optimize_reach" contains joints.
        If so, compute bottom level difference between upstream and downstream of joints, and add as a constraint for optimization
        Args:
            ids: list of crosssection id that will be optimized
        Returns:
            constraint_at_joint: constraint at joints during optimization
        """
        print("Begin checking constraint at joints...")
        selectie = self.selectie_checked
        if len(ids) == len(selectie):
            print(
                "RESULT: No constraint at joints, because you will increase all the bottom levels at same time"
            )
            constraint_at_joint = np.nan
        else:
            # check if there are joints, then add another constraint
            joints = selectie.index[
                selectie.duplicated(["start_or_end", "edge_node"], keep=False)
            ]
            if len(joints) == 0:
                print(
                    "RESULT: No constraint at joints, because there is no joint in optimize area"
                )
                constraint_at_joint = np.nan
            else:
                # create a temporary dataframe for joints in optimize area
                tempdf = pd.DataFrame()
                tempdf["edge_nodes"] = selectie.loc[joints, "edge_node"].values
                tempdf["joints_us_id"] = selectie.loc[joints, "id"].values
                tempdf["joints_us"] = selectie.loc[joints, "new_lowest_point"].values
                for i, row in tempdf.iterrows():
                    tempdf.loc[i, "joints_ds_id"] = (
                        selectie.loc[
                            (selectie.edge_node.values == tempdf.loc[i, "edge_nodes"])
                            & (selectie.start_or_end.values == "start"),
                            "id",
                        ].values
                    )[0]
                    tempdf.loc[i, "joints_ds"] = (
                        selectie.loc[
                            (selectie.edge_node.values == tempdf.loc[i, "edge_nodes"])
                            & (selectie.start_or_end.values == "start"),
                            "new_lowest_point",
                        ].values
                    )[0]
                # check if any joints in optimize area are selected
                if any(item in ids for item in tempdf["joints_ds_id"].values):
                    tempdf_sub = tempdf[tempdf["joints_ds_id"].isin(ids)]
                    for i, row in tempdf_sub.iterrows():
                        if (
                            selectie.loc[
                                selectie.id.values == tempdf_sub.loc[i, "joints_us_id"],
                                "strahler",
                            ].values
                            != selectie.loc[
                                selectie.id.values == tempdf_sub.loc[i, "joints_ds_id"],
                                "strahler",
                            ].values
                        ).any():
                            tempdf_sub.loc[i, "delta"] = (
                                tempdf_sub.loc[i, "joints_us"]
                                - tempdf_sub.loc[i, "joints_ds"]
                            )
                        else:
                            tempdf_sub.loc[i, "delta"] = np.nan
                    constraint_at_joint = tempdf_sub["delta"].min()
                    if constraint_at_joint == 0:
                        print(
                            "RESULT: cannot increase this reach bottom level, because at joints bottom level at downstream is already the same as upstream. Or you can increase both at same time"
                        )
                    else:
                        print(
                            f"RESULT: there are constraints at joints, the constraint is {constraint_at_joint}"
                        )
                else:
                    print(
                        "RESULT: No constraint at joints, because there is no downstream of joint in the selected reaches you want to optimize"
                    )
                    constraint_at_joint = np.nan

        return constraint_at_joint

    def increase_bottom(self, ids, increment_step, output_folder):
        """
        Increase bottom level and get new cross section profiles.
        Args:
            ids : list of crosssection id that will be optimized
            increment_step : increment step size
            output_folder : either False (do not export) or a path to the desired output folder

        Returns:
            cross_def_new : new crossdeffile

        """
        cross_def_model = pd.DataFrame(
            [cs.__dict__ for cs in self.base_model.geometry.crossdeffile.definition]
        )
        selectie = cross_def_model[cross_def_model["id"].isin(ids)]
        complex = selectie[selectie["yzcount"] != 4]
        simple = selectie[selectie["yzcount"] == 4]

        # simple
        for i, row in simple.iterrows():
            old_z = row["zcoordinates"]
            old_y = row["ycoordinates"]
            sideslope_l = (old_z[0] - old_z[1]) / (old_y[0] - old_y[1])
            sideslope_r = (old_z[2] - old_z[3]) / (old_y[2] - old_y[3])
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

            cross_def_model.at[i, "zcoordinates"] = new_z
            cross_def_model.at[i, "ycoordinates"] = new_y

        # complex
        for i, row in complex.iterrows():
            old_z = row["zcoordinates"]
            old_y = row["ycoordinates"]
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

            cross_def_model.at[i, "zcoordinates"] = new_z
            cross_def_model.at[i, "ycoordinates"] = new_y
            cross_def_model.at[i, "yzcount"] = row["yzcount"] + len(new_positions)

        cross_def = cross_def_model.replace({np.nan: None})
        cross_def_new = CrossDefModel(definition=cross_def.to_dict("records"))

        return cross_def_new

    @staticmethod
    def _trapezium_coordinates(bottom_level, bottom_width, slope_l, slope_r, depth):
        """General function to create y an z coords for a trapezium profile"""
        slope_width_l = depth * slope_l
        slope_width_r = depth * slope_r
        ycoords = [
            0,
            slope_width_r,
            slope_width_r + bottom_width,
            slope_width_r + bottom_width + slope_width_l,
        ]
        zcoords = [
            bottom_level + depth,
            bottom_level,
            bottom_level,
            bottom_level + depth,
        ]
        return ycoords, zcoords

    def create_iteration_width(self, prof_ids: list, trapezium_pars: dict):
        """Creates a new model, changing the profiles and saving it in the temporary folder.

        Creates:
        - Iteration folder (number is incremental and counted via class)
        - New crossdef file in iteration folder
        - New MDU in the upper model folder
        - DIMR config file in the iteration folder
        - Batch file in the iteration folder
        Args:
            prof_ids: list of profiles that should be changed
            # Caution! if a profile definition is used at multiple locations, this will change the definition for all
            of those locations!
            trapezium_pars: dict of the new trapezium profile parameters (bottom_width, slope_l, slope_r, depth)
        Returns:
            filename to the batch file of this iteration
        """

        cross_def = pd.DataFrame(
            [cs.__dict__ for cs in self.base_model.geometry.crossdeffile.definition]
        )

        to_change_def = cross_def[cross_def["id"].isin(prof_ids)]
        bottom_levels = [min(zcoords) for zcoords in to_change_def["zcoordinates"]]

        yz = [self._trapezium_coordinates(bl, **trapezium_pars) for bl in bottom_levels]

        cross_def.loc[to_change_def.index, "ycoordinates"] = pd.Series(
            [y for y, z in yz], index=to_change_def.index
        )
        cross_def.loc[to_change_def.index, "zcoordinates"] = pd.Series(
            [z for y, z in yz], index=to_change_def.index
        )
        cross_def.loc[to_change_def.index, "frictionpositions"] = pd.Series(
            [[0, y[-1]] for y, z in yz], index=to_change_def.index
        )
        cross_def.loc[to_change_def.index, "yzcount"] = pd.Series(
            [len(y) for y, z in yz], index=to_change_def.index
        )
        cross_def = cross_def.replace({np.nan: None})
        crossdef_new = CrossDefModel(definition=cross_def.to_dict("records"))
        self.iteration_nr += 1
        iteration_name = f"{self.name}_{self.iteration_nr}"
        iteration_folder = self.work_dir / iteration_name
        iteration_folder.mkdir(parents=True, exist_ok=True)

        crossdef_name = f"crossdef_{self.iteration_nr}.ini"
        crossdef_new.save(iteration_folder / crossdef_name)
        crossdef_new.filepath = Path(f"{iteration_name}/{crossdef_name}")
        mdu_copy = self.base_model.copy()
        mdu_copy.geometry.crossdeffile = crossdef_new
        mdu_copy.filepath = self.work_dir / f"{self.name}_{self.iteration_nr}.mdu"
        mdu_copy.save(recurse=False)
        dimr = DIMR()
        dimr.component.append(
            FMComponent(
                name=f"{self.name}_{self.iteration_nr}",
                workingDir=iteration_folder.parent.absolute(),
                inputfile=mdu_copy.filepath.absolute(),
                model=mdu_copy,
            )
        )
        dimr_fn = iteration_folder / "dimr_config.xml"
        dimr.save(dimr_fn)
        with open(dimr_fn, "r") as f:
            content = f.readlines()
        end_documentation_line = [l for l in content if "</documentation>" in l][0]
        end_documentation_loc = content.index(end_documentation_line)
        spaces = end_documentation_line.split("<")[0]
        content.insert(end_documentation_loc + 1, f"{spaces}</control>\n")
        content.insert(
            end_documentation_loc + 1,
            f'{spaces}{spaces}<start name="{self.name}_{self.iteration_nr}" />\n',
        )
        content.insert(end_documentation_loc + 1, f"{spaces}<control>\n")
        with open(dimr_fn, "w+") as f:
            content_string = "".join(content)
            f.write(content_string)
        self._latest_bat = dimr_fn.parent / "run.bat"
        shutil.copy(self.bat_file, self._latest_bat)
        return iteration_folder

    def create_iteration(self, cross_def_new):
        # self.iteration_nr += 1
        iteration_name = f"{self.name}_{self.iteration_nr}"
        iteration_folder = self.work_dir / iteration_name
        iteration_folder.mkdir(parents=True, exist_ok=True)
        # write new crossdef to iteration iteration_folder
        crossdef_name = f"crossdef_{self.iteration_nr}.ini"
        cross_def_new.save(iteration_folder / crossdef_name)
        cross_def_new.filepath = Path(f"{iteration_name}/{crossdef_name}")
        # write new mdu
        mdu_copy = self.base_model.copy()
        mdu_copy.geometry.crossdeffile = cross_def_new
        # mdu_copy.output.outputdir = f"output_{self.name}_{self.iteration_nr}"
        mdu_copy.filepath = self.work_dir / f"{self.name}_{self.iteration_nr}.mdu"
        mdu_copy.save(recurse=False)

        # find and delete the "writeBalanceFile" line in mdu file (delete this part after solving the bug issue)
        f = open(mdu_copy.filepath, "r")
        lines = f.readlines()
        f.close()
        f = open(mdu_copy.filepath, "w")
        for line in lines:
            if "writeBalanceFile" not in line:
                f.write(line)
        f.close()

        dimr = DIMR()
        dimr.component.append(
            FMComponent(
                name=f"{self.name}_{self.iteration_nr}",
                workingDir=iteration_folder.parent.absolute(),
                inputfile=mdu_copy.filepath.absolute(),
                model=mdu_copy,
            )
        )
        dimr.control.append(Start(name=f"{self.name}_{self.iteration_nr}"))
        dimr_fn = iteration_folder / "dimr_config.xml"
        dimr.save(dimr_fn)
        self._latest_bat = dimr_fn.parent / "run_bat.bat"
        shutil.copy(self.bat_file, self._latest_bat)
        return iteration_folder

    def run_model(self, bat_path, model_folder):
        print("Begin running model")
        subprocess.call(
            [str(Path(bat_path).absolute())], cwd=str(Path(model_folder).absolute())
        )
        print("Done running model")

    def get_water_level(self):
        model_folder = self.work_dir / "output"
        map_nc = glob(f"{model_folder}/**_map.nc")[0]
        ds = nc.Dataset(str(map_nc))

        edges = {"x": "mesh1d_node_x", "y": "mesh1d_node_y"}
        var_wl = "mesh1d_s1"  # water level

        x = list(ds.variables[edges["x"]][:])
        y = list(ds.variables[edges["y"]][:])
        coords = [Point(xi, yi) for xi, yi in zip(x, y)]

        wl = ds.variables[var_wl][-1, :].data

        gdf_water_level = gpd.GeoDataFrame(
            geometry=coords, data={"WL": wl}, crs="EPSG:28992"
        )

        # select the water levels in optimize area (shapefile)
        optimize_area = gpd.read_file(self.shapefile)
        selected_water_level = gpd.clip(gdf_water_level, optimize_area)

        return selected_water_level

    def get_constraints(self):
        optimize_area = gpd.read_file(self.shapefile)
        constraints = optimize_area["max allowable WL"][0]
        return constraints

    def increment_step(self, selected_water_level, constraints, constraint_at_joint):
        min_overcapacity = constraints - selected_water_level["WL"].max()
        if min_overcapacity <= 0:
            print("Already exceding flood risk level, cannot raise bottom any higher")
        else:
            increment = min(
                min_overcapacity * 0.5, constraint_at_joint * 0.5, 0.1
            )  # 0.1 maximum increment step, user defined
        return increment

    def run_latest(self):
        if self._latest_bat is not None:
            self.run_model(self._latest_bat, self._latest_bat.parent)
        else:
            raise ValueError(
                "No latest run available. "
                "Use create_iteration() first, or run another model using run_model()."
            )

    def export_model(self, specific_iteration="latest", cleanup=True):
        if specific_iteration == "latest":
            iteration = self.iteration_nr
        else:
            if type(specific_iteration) is int:
                iteration = specific_iteration
            else:
                raise TypeError(
                    f"specific_iteration must be an interger or be 'latest'. "
                    f"Input was: {specific_iteration}"
                )

        mdu_fn = self.work_dir / f"{self.name}_{iteration}.mdu"
        mdu = FMModel(mdu_fn)
        new_mdu = (
            self.output_dir / f"{self.model_name.split('.')[0]}_Profile_Optimizer.mdu"
        )
        mdu.save(filepath=new_mdu, recurse=True)
        shutil.copytree(
            self.work_dir / "output",
            self.output_dir / f"output_{iteration}_Profile_Optimizer",
        )

        other_files = os.listdir(self.work_dir)
        for filename in other_files:
            file = self.work_dir / filename
            if not filename.endswith(".mdu"):
                if os.path.isfile(file):
                    destination = self.output_dir / Path(file).name
                    if not destination.exists():
                        shutil.copy(file, destination)
                        print(f"Copied {destination.name} to destination")

        print(f"Exported iteration {iteration} to output folder as: {new_mdu}")

        if cleanup:
            shutil.rmtree(self.work_dir)
            print(f"Deleted working directory: {self.work_dir}")


def find_optimum(window_b, calculated_v_values, target_v, waterlevel):
    """A function for the optimization of the bottom width of a trapezoidal cross section profile
        for the desired/required flow velocity
    Args:
        window_b: An array of the bottom widths that have been calculated so far in the search window.
        calculated_v_values: An array of the calculated flow velocities for the bottom widths in the search window.
        target_v: desired flow velocity to achieve in the cross section profile (int).
        waterlevel: An array of the calculated water levels.
    Returns:
        df: dataframe with the bottom widths, calculated velocity and the difference between the calculated Velocity
            and the target velocity.
        optimized_bottom_width: The optimalized bottom width for the desired flow velocity.
    """
    lowest_v = min(calculated_v_values)
    highest_v = max(calculated_v_values)
    if target_v < lowest_v or target_v > highest_v:
        print(
            "Velocity target is not in the range of the calculated velocities.\n"
            "Please choose new bottom widths for iterations.\n"
            f"Target velocity: {target_v}.\n"
            f"Range of calculated velocities: {lowest_v:.3f} - {highest_v:.3f}.\n"
            f"Range of input bottom widths: {min(window_b):.3f} - {max(window_b):.3f}"
        )
        raise ValueError(
            "Velocity target is not in the range of the calculated velocities."
        )

    # collect all the relevant data into a dataframe
    gewenste_u_array = np.ones(len(window_b)) * target_v
    data = {
        "bodembreedte": window_b,
        "berekende stroomsnelheid": calculated_v_values,
        "gewenste stroomsnelheid": gewenste_u_array,
        "berekende waterstand": waterlevel,
    }
    df = pd.DataFrame(data=data)
    df["difference"] = df["berekende stroomsnelheid"] - df["gewenste stroomsnelheid"]
    # print (df)

    # interpolate between the point just above the desired flow_velocity & the point just beneath the desired flow_velocity
    interpolation_point_u_max = (
        df[(df.difference > 0)]
        .sort_values(ascending=True, by="difference")
        .iloc[0]["berekende stroomsnelheid"]
    )
    interpolation_point_u_min = (
        df[(df.difference < 0)]
        .sort_values(ascending=False, by="difference")
        .iloc[0]["berekende stroomsnelheid"]
    )
    interpolation_point_width_max = (
        df[(df.difference > 0)]
        .sort_values(ascending=True, by="difference")
        .iloc[0]["bodembreedte"]
    )
    interpolation_point_width_min = (
        df[(df.difference < 0)]
        .sort_values(ascending=False, by="difference")
        .iloc[0]["bodembreedte"]
    )

    gewenste_stroomsnelheid = gewenste_u_array[0]
    x = [interpolation_point_width_min, interpolation_point_width_max]
    y = [interpolation_point_u_min, interpolation_point_u_max]
    optimized_bottom_width = np.interp(gewenste_stroomsnelheid, y, x)

    # plotly figure relatie stroomsnelheid en bodembreedte
    fig = px.scatter(
        df, x="bodembreedte", y="berekende stroomsnelheid", text="bodembreedte"
    )
    fig.update_traces(mode="markers", marker_line_width=2, marker_size=10)

    fig.add_trace(
        go.Scatter(
            x=[interpolation_point_width_min, interpolation_point_width_max],
            y=[interpolation_point_u_min, interpolation_point_u_max],
            mode="lines",
            name="geïnterpoleerde relatie",
            marker_color="blue",
        )
    )
    fig.add_hline(
        y=gewenste_stroomsnelheid, line_width=1, line_dash="dash", line_color="black"
    )
    fig.add_hrect(
        y0=interpolation_point_u_min,
        y1=interpolation_point_u_max,
        fillcolor="grey",
        opacity=0.2,
        annotation_text="interpolatie gebied",
    )
    fig.add_vline(
        x=optimized_bottom_width, line_width=1, line_dash="dash", line_color="black"
    )

    fig.add_trace(
        go.Scatter(
            x=[optimized_bottom_width],
            y=[gewenste_stroomsnelheid],
            mode="markers",
            name="geoptimaliseerde bodembreedte",
            marker_color="green",
            marker_line_width=2,
            marker_size=10,
        )
    )
    fig.update_yaxes(title_text="<b>berekende stroomsnelheid (m/s)</b>")
    # Naming x-axis
    fig.update_xaxes(title_text="<b>bodembreedte (m)</b>")
    fig.update_layout(
        title="<br>Relatie tussen bodembreedte en stroomsnelheid bij het te optimaliseren profiel</b>"
    )
    fig.show()

    # plotly figure relatie stroomsnelheid en bodembreedte en waterlevel
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["bodembreedte"], y=df["berekende stroomsnelheid"], name="bodembreedte"
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title="Relatie tussen bodembreedte, stroomsnelheid en waterlevel bij het te optimaliseren profiel"
    )
    fig.add_hline(
        y=gewenste_stroomsnelheid, line_width=1, line_dash="dash", line_color="black"
    )
    fig.add_hrect(
        y0=interpolation_point_u_min,
        y1=interpolation_point_u_max,
        fillcolor="grey",
        opacity=0.2,
        annotation_text="interpolatie gebied",
    )
    fig.add_vline(
        x=optimized_bottom_width, line_width=1, line_dash="dash", line_color="black"
    )

    fig.add_trace(
        go.Scatter(
            x=[optimized_bottom_width],
            y=[gewenste_stroomsnelheid],
            mode="markers",
            name="geoptimaliseerde bodembreedte",
            marker_color="green",
            marker_line_width=2,
            marker_size=10,
        ),
        secondary_y=False,
    )

    # secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df["bodembreedte"], y=df["berekende waterstand"], name="waterstand"
        ),
        secondary_y=True,
    )
    # Naming x-axis
    fig.update_xaxes(title_text="<b>bodembreedte (m)</b>")

    # Naming y-axes
    fig.update_yaxes(
        title_text="<b>berekende stroomsnelheid (m/s)</b>", secondary_y=False
    )
    fig.update_yaxes(title_text="<b>berekende waterstand (m)</b>", secondary_y=True)
    fig.show()

    return df, optimized_bottom_width
