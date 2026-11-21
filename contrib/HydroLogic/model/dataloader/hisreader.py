# -*- coding: utf-8 -*-
"""
Created on Fri May 13th 2022

@author: Koen Reef Hydrologic
"""

# =============================================================================
# Import
# =============================================================================
import datetime as dt
import pathlib
from typing import List, Optional, Type, TypeVar

import netCDF4 as nc
import numpy as np
import pandas as pd
from pydantic import BaseModel

from plotting import default_plot

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
STRUCTURE_TYPES = ["station", "weirgen"]

"""============================================================================
Provides
    1. Creates an object to convert variables from the his netcdf output files
    from the D-Hydro model (...his.nc) into python objects
    Use: HisResults(inputdir, outputdir)
        Parameters
        ----------
        inputdir : absolute windows directory path to the ...his.nc D-Hydro output file
        outputdir  : absolute windows directory path to the location newly
        created csv files

        Returns
        -------
        HisResults object with lists of objects retrieved from his file
==========================================================================="""


class ExtStructure(BaseModel):
    """Extends the Hydrolib structure datamodel by adding fields for simulated and measured DataFrames.
    Also adds basic plots for quick data inspection.

    Attributes:
        name (str): Name, or identifier of the structure
        xcoordinate (float): x-coordinate of structure
        ycoordinate (float): y-coordinate of structure
        simulated (pd.DataFrame): simulated time-series in a dataframe
        measured (pd.DataFrame): measured time-series in a dataframe
    """

    name: str
    xcoordinate: float
    ycoordinate: float
    simulated: PandasDataFrame
    measured: Optional[PandasDataFrame]

    def simulated_plot(self, variable: str) -> None:
        """Function to plot the simulated value of 'variable'"""
        default_plot(dfs=[self.simulated], variable=variable)

    def measured_plot(self, variable: str) -> None:
        """function to plot the measured value of 'variable'"""
        default_plot(dfs=[self.measured], variable=variable)

    def measured_vs_simulated_plot(self, variable: str) -> None:
        """function to plot the simulated and measured value of 'variable'"""
        default_plot(
            dfs=[self.simulated, self.measured],
            variable=variable,
            labels=["Simulated " + variable, "Measured " + variable],
        )


class HisResults(object):
    """
    Class to read results from a D-HYDRO _his.nc file.

    Attributes:
        __inputdir (str): directory where input file is located
        __outputdir (str): directory where output (e.g. csv files) are stored
        structure_types (List): list of structures to load from input file
    """

    def __init__(self, inputdir: str, outputdir: str, structure_types: List = None) -> None:
        # TODO: let user choose what to execute
        self.__inputdir = inputdir
        self.__outputdir = outputdir
        if structure_types is None:
            structure_types = STRUCTURE_TYPES
        self.structure_list = self.parse_structures(structure_types, ExtStructure)

    def __read_netcdf(self) -> None:
        """Reads the NetCDF file and stores it in __ds attribute.
        Variables are stored in self.variables"""
        fnin = list(pathlib.Path(self.__inputdir).glob("*his.nc"))
        self.__ds = nc.Dataset(fnin[0], "r")
        self.variables = self.__ds.variables

    def read_all_netcdf_variables(self) -> None:
        """User function to create handles for all variables of the NetCDF file
        self.variable corresponds to self.__ds.variables[variable]"""
        for key in self.__ds.variables.keys():
            if "_id" in key:
                setattr(self, key, np.array(nc.chartostring(self.__ds.variables[key][:, :])))

            else:
                setattr(self, key, self.__ds.variables[key])

    def __make_timeframe(self) -> None:
        """Gets the starting time and creates a dataframe containing the model timeframe as index

        Args:
            None
        Returns:
            None
        """
        t_start = (
            self.__ds.variables["time"].units.split("since ")[1].replace("+00:00", "").strip()
        )

        start_moment = dt.datetime.fromisoformat(t_start)
        time_array = (
            np.array(self.__ds.variables["time"][:]) * dt.timedelta(seconds=1) + start_moment
        )
        df = pd.DatetimeIndex(time_array).to_frame(name="time").set_index("time")
        self.timeframe = df

    def parse_structures(
        self,
        structure_types: List,
        structure_obj: Type[ExtStructure],
        structure_names_lists: List = None,
    ) -> List:

        """General function to parse structures from His files. returns an object in self for all structures of
        structure_type and adds their names to a list for convenience.

        Args:
            structure_types (List): list containing the structure types that are to be read from the model results
            structure_obj (ExtStructure): object used to organize the results per structure
            structure_names_lists (List): list containing exactly one list per structure type.
                                          These lists contain the structure names to be loaded.
        Returns:
            structure_list (List): list containing a list per structure type with the loaded structures per type.
        """

        # Check if netcdf is loaded, if not do so
        if not hasattr(self, "__ds"):
            self.__read_netcdf()

        # Check if timeframe exists, if not make it
        if not hasattr(self, "timeframe"):
            self.__make_timeframe()

        structure_list = []
        for s_ix, structure_type in enumerate(structure_types):

            # Firstly, load structure names from stored netcdf data, if not provide as input
            if structure_names_lists is None:
                structure_names = np.array(
                    nc.chartostring(self.__ds.variables[structure_type + r"_id"][:, :])
                )
            else:
                structure_names = structure_names_lists[s_ix]
            # Secondly, loop over structure names and create object with simulated values from model results
            # and initiate an empty dataframe for measured values.
            structure_list.append([])
            for struct_ix, struct_name in enumerate(structure_names):
                data = self.timeframe.copy()
                name = struct_name.strip()

                for variable in self.__ds.variables:
                    if not hasattr(self.__ds.variables[variable], "coordinates"):
                        continue

                    if variable == "bedlevel":
                        # does not work
                        continue

                    if structure_type in self.__ds.variables[variable].coordinates:
                        data[variable] = self.__ds.variables[variable][:, struct_ix]
                try:
                    xcoordinate = self.__ds.variables[structure_type + r"_geom_node_coordx"][
                        struct_ix
                    ]
                    ycoordinate = self.__ds.variables[structure_type + r"_geom_node_coordy"][
                        struct_ix
                    ]
                except KeyError:
                    xcoordinate = -999
                    ycoordinate = -999

                structure = structure_obj(
                    name=name,
                    xcoordinate=xcoordinate,
                    ycoordinate=ycoordinate,
                    measured=self.timeframe.copy(),
                    simulated=data,
                )
                setattr(self, name, structure)
                structure_list[s_ix].append(name)

        del self.__ds
        return structure_list

    def write_csv(self, output_path: str = None, struct_list: List = None):
        """Writes csvs in output_path for all structures in struct_list,
        or for all structures that have been parsed.

        Args:
            output_path (str): the output path to which the csvs are written
            struct_list (List): list containing the names of the structures to be written to files
                                if struct_list is not provided, all structures are written to files
        Returns:
            None
        """
        if output_path is None:
            output_path = self.__outputdir

        if struct_list is not None:
            for struct_name in struct_list:
                structure = getattr(self, struct_name)
                structure.simulated.to_csv(output_path + "\\" + struct_name + r"_simulated.csv")
                if len(structure.measured.columns.to_list()) > 0:
                    structure.measured.to_csv(output_path + "\\" + struct_name + r"_measured.csv")
        else:
            print(self.__dict__.keys())
            for key in self.__dict__.keys():
                if hasattr(getattr(self, key), "simulated"):
                    structure = getattr(self, key)
                    structure.simulated.to_csv(output_path + "\\" + key + r"_simulated.csv")
                    if len(structure.measured.columns.to_list()) > 0:
                        structure.measured.to_csv(output_path + "\\" + key + r"_measured.csv")
