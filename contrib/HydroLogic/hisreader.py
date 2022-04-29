# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:53:59 2021

@author: Krijn Prins Hydrologic
@author: Koen Reef Hydrologic

Update with variables, observation station names, weir names, and writing weir_data
"""

# =============================================================================
# Import
# =============================================================================
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime as dt
import ugfile as uf
import pathlib
from datetime import datetime

"""============================================================================
Provides
    1. Creates an object to convert variables from the his netcdf output files
    from the D-Hydro model (...his.nc) into .csv file format
    Use: hisreader(inputdir, outputdir)
        Parameters
        ----------
        inputdir : absolute windows directory path to the ...his.nc D-Hydro output file
        outputdir  : absolute windows directory path to the location newly
        created csv files

        Returns
        -------
        Hisreader object with supported functions to extract variables:
            observation stations          writeobsstations()
            cross sections                writecrosssections()
            dambreaks                     writedambreaks()
            all of variables above        writeall()
        To create the csv file with the extracted variables use:
                                          writehisdata()
==========================================================================="""


class hisreader(object):
    def __init__(self, inputdir, outputdir):
        self.__inputdir = inputdir
        self.__outputdir = outputdir
        self.__read_ncvar()
        self.variables = self.__ds.variables
        self.obs_names = np.array(
            nc.chartostring(self.__ds.variables["station_name"][:, :])
        )
        self.weir_names = np.array(
            nc.chartostring(self.__ds.variables["weirgen_id"][:, :])
        )

    def __read_ncvar(self):
        fnin = list(pathlib.Path(self.__inputdir).glob("*his.nc"))
        self.__ds = uf.DatasetUG(fnin[0], "r")

        # resultaten uitlezen
        self.__ds.variables.keys()
        columns = [
            "dtype",
            "standard_name",
            "long_name",
            "units",
            "shape",
            "coordinates",
        ]
        variables = pd.DataFrame(columns=columns)
        for var in self.__ds.variables.keys():
            vardata = self.__ds.variables[var]
            variables.loc[var] = [
                getattr(vardata, col) if hasattr(vardata, col) else ""
                for col in columns
            ]

        # tijd toevoegen (obv relatieve tijd ten opzichte van startdatum)
        time_var = self.__ds.variables["time"]
        t = dt.datetime.strptime(
            time_var.units.split("since ")[1].replace("+00:00", "").strip(),
            "%Y-%m-%d %H:%M:%S",
        )
        time = self.__ds.variables["time"][:]
        time2 = np.empty(time.size)
        t2 = []
        for i in range(0, len(time)):
            t2.append(
                (t + dt.timedelta(0, time[i])).strftime("%d-%m-%Y %H:%M")
            )
        time2 = pd.DataFrame(t2)
        df = time2
        df.columns = ["time"]
        self.__df = df

        #%% toevoegen data in csv

    def writeobsstations(
        self, variables=["waterlevel", "waterdepth", "discharge_magnitude"]
    ):
        """
        Extracts and restructures the water level at model defined specified
        observation station locations
        extracts from self.__ds.variables in self.hisreader
        extracts a pandas dataframe with water levels at the appropriate time
        and location to self.__df[obs_names] in self.hisreader

        Parameters
        ----------
        variables : list
            A list of strings with variable_names

        Returns
        -------
        None.

        """
        if "station_name" in self.__ds.variables:
            obs_name = nc.chartostring(
                self.__ds.variables["station_name"][:, :]
            )
            obs_names = np.array(obs_name)
            obs_loc = range(0, len(obs_name))
            for variable in variables:
                variable_ds = self.__ds.variables[variable][:, :]
                self.__df[
                    [f"{obs.strip()}_{variable}" for obs in obs_names]
                ] = pd.DataFrame(variable_ds[:, obs_loc])

    def writeweirs(
        self, variables=["weirgen_crest_level", "weirgen_s1up", "weirgen_s1dn"]
    ):
        """
        Extracts and restructures the water level at model defined specified
        observation station locations
        extracts from self.__ds.variables in self.hisreader
        extracts a pandas dataframe with water levels at the appropriate time
        and location to self.__df[obs_names] in self.hisreader

        Parameters
        ----------
        variables : list
            A list of strings with variable_names

        Returns
        -------
        None.

        """
        if "weirgen_id" in self.__ds.variables:
            obs_name = nc.chartostring(self.__ds.variables["weirgen_id"][:, :])
            obs_names = np.array(obs_name)
            obs_loc = range(0, len(obs_name))
            for variable in variables:
                variable_ds = self.__ds.variables[variable][:, :]
                self.__df[
                    [f"{obs.strip()}_{variable}" for obs in obs_names]
                ] = pd.DataFrame(variable_ds[:, obs_loc])

    def writecrosssections(self):
        """
        Extracts and restructures the water level at model defined specified
        cross section locations
        extracts from self.__ds.variables in self.hisreader
        extracts a pandas dataframe with water levels at the appropriate time
        and location to self.__df[cross_names] in self.hisreader

        Returns
        -------
        None.

        """
        if "cross_section_name" in self.__ds.variables:
            cross_name = nc.chartostring(
                self.__ds.variables["cross_section_name"][:, :]
            )
            cross_names = np.array(cross_name)

            q_cross = self.__ds.variables["cross_section_discharge"][:, :]
            q_cross_section = pd.DataFrame(q_cross)
            self.__df[cross_names] = q_cross_section

    def writedambreaks(self):
        """
        Extracts and restructures the water level at model defined specified
        dam break locations
        extracts from self.__ds.variables in self.hisreader
        extracts a pandas dataframe with water levels at the appropriate time
        and location to self.__df[dambreak_id] in self.hisreader

        Returns
        -------
        None.

        """
        if "dambreak_id" in self.__ds.variables:
            bres_name = nc.chartostring(
                self.__ds.variables["dambreak_id"][:, :]
            )
            bres_names = np.array(bres_name)
            bres = self.__ds.variables["dambreak_crest_width"][:, :]
            bres_debiet = self.__ds.variables["dambreak_discharge"][:, :]
            bres_wl_up = self.__ds.variables["dambreak_s1up"][:, :]
            bres_wl_dn = self.__ds.variables["dambreak_s1dn"][:, :]

            for index, breach_location in enumerate(bres_names):
                self.__df[f"bresbreedte_{breach_location}"] = bres[:, index]
                self.__df[f"bresdebiet_{breach_location}"] = bres_debiet[
                    :, index
                ]
                self.__df[f"bres_wl_up_{breach_location}"] = bres_wl_up[
                    :, index
                ]
                self.__df[f"bres_wl_dn_{breach_location}"] = bres_wl_dn[
                    :, index
                ]

    def writeall(self):
        """
        Extracts the water level at all currently available variables,
        being the dambreak, crosssection and observation station locations.
        extracts from self.__ds.variables in self.hisreader
        extracts a pandas dataframe with water levels at the appropriate time
        and location to self.__df[dambreak_id] in self.hisreader

        Returns
        -------
        None.

        """
        if "station_name" in self.__ds.variables:
            self.writeobsstations()
        if "cross_section_name" in self.__ds.variables:
            self.writecrosssections()
        if "dambreak_id" in self.__ds.variables:
            self.writedambreaks()

    def writehisdata(self):
        """
        function that creates a csv file and writes the extracted
        parameters in this file

        Returns
        -------
        None.

        """
        self.__df.to_csv(
            os.path.join(self.__outputdir, "hisdata.csv"), index=False
        )
