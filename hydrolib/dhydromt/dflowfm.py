"""Implement plugin model class"""

import glob
from os.path import join, basename, isfile
import logging

import pandas as pd
import numpy as np
from rasterio.warp import transform_bounds
import pyproj
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
from typing import Union

import hydromt
from hydromt.models.model_api import Model
from hydromt import gis_utils, io
from hydromt import raster

from .workflows import process_branches
from .workflows import update_data_columns_attributes
from .workflows import update_data_columns_attribute_from_query
from .workflows import validate_branches
from .workflows import generate_roughness
from .workflows import set_branch_crosssections
from .workflows import set_xyz_crosssections

from .workflows import helper
from . import DATADIR

from pathlib import Path

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(Model):
    """General and basic API for models in HydroMT"""

    # FIXME
    _NAME = "dflowfm"
    _CONF = "FMmdu.txt"
    _DATADIR = DATADIR
    # TODO change below mapping table (hydrolib-core convention:shape file convention) to be read from data folder, maybe similar to _intbl for wflow
    # TODO: we also need one reverse table to read from static geom back. maybe a dictionary of data frame is better?
    # TODO: write static geom as geojson dataset, so that we dont get limitation for the 10 characters
    _GEOMS = {}  # FIXME Mapping from hydromt names to model specific names
    _MAPS = {}  # FIXME Mapping from hydromt names to model specific names
    _FOLDERS = ["dflowfm", "staticgeoms"]

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,  # hydromt config contain glob section, anything needed can be added here as args
        data_libs=None,  # yml # TODO: how to choose global mapping files (.csv) and project specific mapping files (.csv)
        logger=logger,
        deltares_data=False,  # data from pdrive,
    ):

        if not isinstance(root, (str, Path)):
            raise ValueError("The 'root' parameter should be a of str or Path.")

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            deltares_data=deltares_data,
            logger=logger,
        )

        # model specific
        # default ini files
        self._region_name = "DFlowFM"
        self._ini_settings = (
            None,
        )  # TODO: add all ini files in one object? e.g. self._intbl in wflow?
        self._datamodel = None
        self._dfmmodel = None  # TODO: replace with hydrolib-core object
        self._branches = gpd.GeoDataFrame()

        # TODO: assign hydrolib-core components

    def setup_basemaps(
        self,
        region: dict,
        region_name: str = None,
        crs: int = None,
    ):
        """Define the model region.

        Adds model layer:

        * **region** geom: model region

        Parameters
        ----------
        region: dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
        region_name: str, optional
            Name of the model region.
        crs : int, optional
            Coordinate system (EPSG number) of the model. If not provided, equal to the region crs
            if "grid" or "geom" option are used, and to 4326 if "bbox" is used.
        """

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind == "bbox":
            if crs:
                geom = gpd.GeoDataFrame(geometry=[box(*region["bbox"])], crs=crs)
            else:
                self.logger.info("No crs given with region bbox, assuming 4326.")
                geom = gpd.GeoDataFrame(geometry=[box(*region["bbox"])], crs=4326)
        elif kind == "grid":
            geom = region["grid"].raster.box
        elif kind == "geom":
            geom = region["geom"]
        else:
            raise ValueError(
                f"Unknown region kind {kind} for DFlowFM, expected one of ['bbox', 'grid', 'geom']."
            )

        if crs:
            geom = geom.to_crs(crs)
        elif geom.crs is None:
            raise AttributeError("region crs can not be None. ")
        else:
            self.logger.info(f"Model region is set to crs: {geom.crs.to_epsg()}")

        # Set the model region geometry (to be accessed through the shortcut self.region).
        self.set_staticgeoms(geom, "region")

        if region_name is not None:
            self._region_name = region_name

        # FIXME: how to deprecate WARNING:root:No staticmaps defined

    def _setup_branches(
        self,
        gdf_br: gpd.GeoDataFrame,
        defaults: pd.DataFrame,
        br_type: str,
        spacing: pd.DataFrame = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
    ):
        """This function is a wrapper for all common steps to add branches type of objects (ie channels, rivers, pipes...).

        Parameters
        ----------
        gdf_br : gpd.GeoDataFrame
            GeoDataFrame with the new branches to add.
        spacing : pd.DataFrame
            DatFrame containing spacing values per 'branchType', 'shape', 'width' or 'diameter'.

        """
        if gdf_br.crs.is_geographic:  # needed for length and splitting
            gdf_br = gdf_br.to_crs(3857)

        self.logger.info("Adding/Filling branches attributes values")
        gdf_br = update_data_columns_attributes(gdf_br, defaults, brtype=br_type)

        # If specific spacing info from spacing_fn, update spacing attribute
        if spacing is not None:
            self.logger.info(f"Updating spacing attributes")
            gdf_br = update_data_columns_attribute_from_query(
                gdf_br, spacing, attribute_name="spacing"
            )

        self.logger.info(f"Processing branches")
        branches, branches_nodes = process_branches(
            gdf_br,
            branch_nodes=None,
            id_col="branchId",
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
            logger=self.logger,
        )

        self.logger.info(f"Validating branches")
        validate_branches(branches)

        # convert to model crs
        branches = branches.to_crs(self.crs)
        branches_nodes = branches_nodes.to_crs(self.crs)

        return branches, branches_nodes

    def setup_channels(
        self,
        channels_fn: str,
        channels_defaults_fn: str = None,
        spacing_fn: str = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
    ):
        """This component prepares the 1D channels and adds to branches 1D network

        Adds model layers:

        * **channels** geom: 1D channels vector
        * **branches** geom: 1D branches vector

        Parameters
        ----------
        channels_fn : str
            Name of data source for branches parameters, see data/data_sources.yml.

            * Required variables: [branchId, branchType] # TODO: now still requires some cross section stuff

            * Optional variables: [spacing, material, shape, diameter, width, t_width, t_width_up, width_up,
              width_dn, t_width_dn, height, height_up, height_dn, inlev_up, inlev_dn, bedlev_up, bedlev_dn,
              closed, manhole_up, manhole_dn]
        channels_defaults_fn : str Path
            Path to a csv file containing all defaults values per 'branchType'.
        spacing : str Path
            Path to a csv file containing spacing values per 'branchType', 'shape', 'width' or 'diameter'.

        """
        self.logger.info(f"Preparing 1D channels.")

        # Read the channels data
        id_col = "branchId"
        gdf_ch = self.data_catalog.get_geodataframe(
            channels_fn, geom=self.region, buffer=10, predicate="contains"
        )
        gdf_ch.index = gdf_ch[id_col]
        gdf_ch.index.name = id_col

        if gdf_ch.index.size == 0:
            self.logger.warning(
                f"No {channels_fn} 1D channel locations found within domain"
            )
            return None

        else:
            # Fill in with default attributes values
            if channels_defaults_fn is None or not channels_defaults_fn.is_file():
                self.logger.warning(
                    f"channels_defaults_fn ({channels_defaults_fn}) does not exist. Fall back choice to defaults. "
                )
                channels_defaults_fn = Path(self._DATADIR).joinpath(
                    "channels", "channels_defaults.csv"
                )
            defaults = pd.read_csv(channels_defaults_fn)
            self.logger.info(
                f"channel default settings read from {channels_defaults_fn}."
            )

            # If specific spacing info from spacing_fn, update spacing attribute
            spacing = None
            if isinstance(spacing_fn, str):
                if not isfile(spacing_fn):
                    self.logger.error(f"Spacing file not found: {spacing_fn}, skipping")
                else:
                    spacing = pd.read_csv(spacing_fn)

            # Build the channels branches and nodes and fill with attributes and spacing
            channels, channel_nodes = self._setup_branches(
                gdf_br=gdf_ch,
                defaults=defaults,
                br_type="channel",
                spacing=spacing,
                snap_offset=snap_offset,
                allow_intersection_snapping=allow_intersection_snapping,
            )

            # setup staticgeoms #TODO do we still need channels?
            self.logger.debug(
                f"Adding branches and branch_nodes vector to staticgeoms."
            )
            self.set_staticgeoms(channels, "channels")
            self.set_staticgeoms(channel_nodes, "channel_nodes")

            # add to branches
            self.add_branches(channels, branchtype="channel")

    def setup_rivers(
        self,
        rivers_fn: str,
        rivers_defaults_fn: str = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
        friction_type: str = "Manning", # what about constructing friction_defaults_fn?
        friction_value: float = 0.023,
        crosssections_defaults_fn: str = None,
        crosssections_fn: str = None,
        crosssections_type:str = None
    ):
        """This component prepares the 1D rivers and adds to branches 1D network.

        If defaults attributes [spacing, material, shape, diameter, width, t_width, t_width_up, width_up,
        width_dn, t_width_dn, height, height_up, height_dn, inlev_up, inlev_dn, bedlev_up, bedlev_dn,
        closed, manhole_up, manhole_dn] are not present in ``rivers_fn``, they are added from defaults values
        in ``rivers_defaults_fn``.

        Friction attributes [friction_type, friction_value] are either taken from ``rivers_fn`` or filled in
        using ``friction_type`` and ``friction_value`` arguments.

        The rivers are also splits into several segments based on either a default 'spacing' value or on
        specific queries defined in ``spacing_fn``.

        Adds/Updates model layers:

        * **rivers** geom: 1D rivers vector
        * **branches** geom: 1D branches vector

        Parameters
        ----------
        rivers_fn : str
            Name of data source for branches parameters, see data/data_sources.yml.
            * Required variables: [branchId, branchType]
            * Optional variables: [material, shape, diameter, width, t_width, height, bedlev, closed, friction_type, friction_value]
        rivers_defaults_fn : str Path
            Path to a csv file containing all defaults values per 'branchType'.
        snap_offset: float, optional
            Snapping tolenrance to automatically connecting branches.
            By default 0.0, no snapping is applied.
        allow_intersection_snapping: bool, optional
            Switch to choose whether snapping of multiple branch ends are allowed when ``snap_offset`` is used.
            By default True.
        friction_type : str, optional
            Type of friction tu use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
            By default "Manning".
        friction_value : float, optional
            Friction value. By default 0.023.
        crosssections_fn : str Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyz"
            * Required variables: crsId, order, z
            * Optional variables:
            By default None, crosssections will be set from branches
        crosssections_defaults_fn : str Path, optional
            Path to a csv file containing all defaults values per 'branchType'.
             By default None, crosssections defaults will be read from "crosssections", "crosssections_defaults.csv"
        crosssections_type : str, optional
            Type of crosssections read from crosssections_fn. One of ["xyz"].
            By default None.
        """
        self.logger.info(f"Preparing 1D rivers.")

        # Read the rivers data
        id_col = "branchId"
        gdf_riv = self.data_catalog.get_geodataframe(
            rivers_fn, geom=self.region, buffer=10, predicate="contains"
        )
        gdf_riv.index = gdf_riv[id_col]
        gdf_riv.index.name = id_col

        if gdf_riv.index.size == 0:
            self.logger.warning(
                f"No {rivers_fn} 1D river locations found within domain"
            )
            return None

        else:
            # Fill in with default attributes values
            if rivers_defaults_fn is None or not rivers_defaults_fn.is_file():
                self.logger.warning(
                    f"rivers_defaults_fn ({rivers_defaults_fn}) does not exist. Fall back choice to defaults. "
                )
                rivers_defaults_fn = Path(self._DATADIR).joinpath(
                    "rivers", "rivers_defaults.csv"
                )
            defaults = pd.read_csv(rivers_defaults_fn)
            self.logger.info(f"river default settings read from {rivers_defaults_fn}.")

            # check for allowed columns and select only the ones required
            _allowed_columns = ["geometry", "branchId", "branchType", "material", "shape", "diameter", "width", "t_width", "height", "bedlev", "closed", "friction_type", "friction_value"]
            allowed_columns = set(_allowed_columns).intersection(gdf_riv.columns)
            gdf_riv = gpd.GeoDataFrame(gdf_riv[allowed_columns])

            # Add friction to defaults
            defaults["frictionType"] = friction_type
            defaults["frictionValue"] = friction_value

            # Build the rivers branches and nodes and fill with attributes and spacing
            rivers, river_nodes = self._setup_branches(
                gdf_br=gdf_riv,
                defaults=defaults,
                br_type="river",
                spacing=None, # does not allow spacing for rivers
                snap_offset=snap_offset,
                allow_intersection_snapping=allow_intersection_snapping,
            )

            # Add friction_id column based on {friction_type}_{friction_value}
            rivers["friction_id"] = [
                f"{ftype}_{fvalue}"
                for ftype, fvalue in zip(
                    rivers["frictionType"], rivers["frictionValue"]
                )
            ]

            # setup crosssection
            self.logger.info(f"Preparing 1D rivers crosssections.")
            crosssections = gpd.GeoDataFrame()

            # TODO: only read files here, pass branches, crosssections and crosssection types to a workflow
            if crosssections_fn is None:

                # read crosssection and fill in with default attributes values
                _gdf_crs = rivers.copy()
                if crosssections_defaults_fn is None or not crosssections_defaults_fn.is_file():
                    self.logger.warning(
                        f"crosssections_defaults_fn ({crosssections_defaults_fn}) does not exist. Fall back choice to defaults. "
                    )
                    crosssections_defaults_fn = Path(self._DATADIR).joinpath(
                        "crosssections", "crosssections_defaults.csv"
                    )
                defaults = pd.read_csv(crosssections_defaults_fn)
                self.logger.info(f"crosssection default settings read from {rivers_defaults_fn}.")
                _gdf_crs = update_data_columns_attributes(_gdf_crs, defaults, brtype="river") 

                # set crosssections
                _gdf_crs = set_branch_crosssections(rivers, _gdf_crs)

                # TODO add to existing crosssections
                crosssections = gpd.GeoDataFrame(pd.concat([crosssections, _gdf_crs]))


            else:
                # setup cross sections from file
                if crosssections_type == None:
                    self.logger.error("Must specify crosssections_type. Use the following options: xyz")

                elif crosssections_type == 'xyz':

                    # read xyz crosssections with a small buffer
                    id_col = "crsId"
                    _gdf_crs = self.data_catalog.get_geodataframe(
                        crosssections_fn, geom=self.region, buffer=10, predicate="contains"
                    )
                    _gdf_crs.index = _gdf_crs[id_col]
                    _gdf_crs.index.name = id_col
                    _gdf_crs_ids = _gdf_crs.index.unique()

                    if _gdf_crs.index.size == 0:
                        self.logger.warning(
                            f"{crosssections_fn} No cross sections found within domain"
                        )

                    else:
                        # read xyz crosssections with a larger buffer and filter the id (in case smaller buffer was too small)
                        _gdf_crs = self.data_catalog.get_geodataframe(
                            crosssections_fn, geom=self.region, buffer=1000, predicate="contains"
                        )
                        _gdf_crs.index = _gdf_crs[id_col]
                        _gdf_crs.index.name = id_col
                        _gdf_crs = _gdf_crs[_gdf_crs.index.isin(_gdf_crs_ids)]

                        self.logger.info(
                            f"{crosssections_fn} No cross sections found within domain"
                        )

                        # set crsloc and crsdef attributes to crosssections
                        _gdf_crs = set_xyz_crosssections(rivers, _gdf_crs)

                        # TODO add to existing crosssections
                        crosssections = gpd.GeoDataFrame(pd.concat([crosssections, _gdf_crs]))

                else:
                    raise NotImplementedError("Method {crosssections_type} is not implemented.")


            # setup staticgeoms #TODO do we still need channels?
            self.logger.debug(f"Adding rivers and river_nodes vector to staticgeoms.")
            self.set_staticgeoms(rivers, "rivers")
            self.set_staticgeoms(river_nodes, "rivers_nodes")

            # add to branches
            self.add_branches(rivers, branchtype="river")

            # setup staticgeoms #TODO do we still need channels?
            self.logger.debug(f"Adding crosssections vector to staticgeoms.")
            self.set_staticgeoms(crosssections, "crosssections")


    # def setup_branches(
    #     self,
    #     branches_fn: str,
    #     branches_ini_fn: str = None,
    #     snap_offset: float = 0.0,
    #     id_col: str = "branchId",
    #     branch_query: str = None,
    #     pipe_query: str = 'branchType == "Channel"',  # TODO update to just TRUE or FALSE keywords instead of full query
    #     channel_query: str = 'branchType == "Pipe"',
    #     **kwargs,
    # ):
    #     """This component prepares the 1D branches

    #     Adds model layers:

    #     * **branches** geom: 1D branches vector

    #     Parameters
    #     ----------
    #     branches_fn : str
    #         Name of data source for branches parameters, see data/data_sources.yml.

    #         * Required variables: branchId, branchType, # TODO: now still requires some cross section stuff

    #         * Optional variables: []

    #     """
    #     self.logger.info(f"Preparing 1D branches.")

    #     # initialise data model
    #     branches_ini = helper.parse_ini(
    #         Path(self._DATADIR).joinpath("dflowfm", f"branch_settings.ini")
    #     )  # TODO: make this default file complete, need 2 more argument, spacing? yes/no, has_in_branch crosssection? yes/no, or maybe move branches, cross sections and roughness into basemap
    #     branches = None
    #     branch_nodes = None
    #     # TODO: initilise hydrolib-core object --> build
    #     # TODO: call hydrolib-core object --> update

    #     # read branch_ini
    #     if branches_ini_fn is None or not branches_ini_fn.is_file():
    #         self.logger.warning(
    #             f"branches_ini_fn ({branches_ini_fn}) does not exist. Fall back choice to defaults. "
    #         )
    #         branches_ini_fn = Path(self._DATADIR).joinpath(
    #             "dflowfm", f"branch_settings.ini"
    #         )

    #     branches_ini.update(helper.parse_ini(branches_ini_fn))
    #     self.logger.info(f"branch default settings read from {branches_ini_fn}.")

    #     # read branches
    #     if branches_fn is None:
    #         raise ValueError("branches_fn must be specified.")

    #     branches = self._get_geodataframe(branches_fn, id_col=id_col, **kwargs)
    #     self.logger.info(f"branches read from {branches_fn} in Data Catalogue.")

    #     branches = helper.append_data_columns_based_on_ini_query(branches, branches_ini)

    #     # select branches to use
    #     if helper.check_geodataframe(branches) and branch_query is not None:
    #         branches = branches.query(branch_query)
    #         self.logger.info(f"Query branches for {branch_query}")

    #     # process branches and generate branch_nodes
    #     if helper.check_geodataframe(branches):
    #         self.logger.info(f"Processing branches")
    #         branches, branch_nodes = process_branches(
    #             branches,
    #             branch_nodes,
    #             branches_ini=branches_ini,  # TODO:  make the branch_setting.ini [global] functions visible in the setup functions. Use kwargs to allow user interaction. Make decisions on what is neccessary and what not
    #             id_col=id_col,
    #             snap_offset=snap_offset,
    #             logger=self.logger,
    #         )

    #     # validate branches
    #     # TODO: integrate below into validate module
    #     if helper.check_geodataframe(branches):
    #         self.logger.info(f"Validating branches")
    #         validate_branches(branches)

    #     # finalise branches
    #     # setup channels
    #     branches.loc[branches.query(channel_query).index, "branchType"] = "Channel"
    #     # setup pipes
    #     branches.loc[branches.query(pipe_query).index, "branchType"] = "Pipe"
    #     # assign crs
    #     branches.crs = self.crs

    #     # setup staticgeoms
    #     self.logger.debug(f"Adding branches and branch_nodes vector to staticgeoms.")
    #     self.set_staticgeoms(branches, "branches")
    #     self.set_staticgeoms(branch_nodes, "branch_nodes")

    #     # TODO: assign hydrolib-core object

    #     return branches, branch_nodes

    def setup_roughness(
        self,
        generate_roughness_from_branches: bool = True,
        roughness_ini_fn: str = None,
        branch_query: str = None,
        **kwargs,
    ):
        """"""

        self.logger.info(f"Preparing 1D roughness.")

        # initialise ini settings and data
        roughness_ini = helper.parse_ini(
            Path(self._DATADIR).joinpath("dflowfm", f"roughness_settings.ini")
        )
        # TODO: how to make sure if defaults are given, we can also know it based on friction name or cross section definiation id (can we use an additional column for that? )
        roughness = None

        # TODO: initilise hydrolib-core object --> build
        # TODO: call hydrolib-core object --> update

        # update ini settings
        if roughness_ini_fn is None or not roughness_ini_fn.is_file():
            self.logger.warning(
                f"roughness_ini_fn ({roughness_ini_fn}) does not exist. Fall back choice to defaults. "
            )
            roughness_ini_fn = Path(self._DATADIR).joinpath(
                "dflowfm", f"roughness_settings.ini"
            )

        roughness_ini.update(helper.parse_ini(roughness_ini_fn))
        self.logger.info(f"roughness default settings read from {roughness_ini}.")

        if generate_roughness_from_branches == True:

            self.logger.info(f"Generating roughness from branches. ")

            # update data by reading user input
            _branches = self.staticgeoms["branches"]

            # update data by combining ini settings
            self.logger.debug(
                f'1D roughness initialised with the following attributes: {list(roughness_ini.get("default", {}))}.'
            )
            branches = helper.append_data_columns_based_on_ini_query(
                _branches, roughness_ini
            )

            # select branches to use e.g. to facilitate setup a selection each time setup_* is called
            if branch_query is not None:
                branches = branches.query(branch_query)
                self.logger.info(f"Query branches for {branch_query}")

            # process data
            if helper.check_geodataframe(branches):
                roughness, branches = generate_roughness(branches, roughness_ini)

            # add staticgeoms
            if helper.check_geodataframe(roughness):
                self.logger.debug(f"Updating branches vector to staticgeoms.")
                self.set_staticgeoms(branches, "branches")

                self.logger.debug(f"Updating roughness vector to staticgeoms.")
                self.set_staticgeoms(roughness, "roughness")

            # TODO: add hydrolib-core object

        else:

            # TODO: setup roughness from other data types

            pass

    def setup_crosssections(
        self,
        generate_crosssections_from_branches: bool = True,
        crosssections_ini_fn: str = None,
        branch_query: str = None,
        **kwargs,
    ):
        """"""

        self.logger.info(f"Preparing 1D crosssections.")

        # initialise ini settings and data
        crosssections_ini = helper.parse_ini(
            Path(self._DATADIR).joinpath("dflowfm", f"crosssection_settings.ini")
        )
        crsdefs = None
        crslocs = None
        # TODO: initilise hydrolib-core object --> build
        # TODO: call hydrolib-core object --> update

        # update ini settings
        if crosssections_ini_fn is None or not crosssections_ini_fn.is_file():
            self.logger.warning(
                f"crosssection_ini_fn ({crosssections_ini_fn}) does not exist. Fall back choice to defaults. "
            )
            crosssections_ini_fn = Path(self._DATADIR).joinpath(
                "dflowfm", f"crosssection_settings.ini"
            )

        crosssections_ini.update(helper.parse_ini(crosssections_ini_fn))
        self.logger.info(
            f"crosssections default settings read from {crosssections_ini}."
        )

        if generate_crosssections_from_branches == True:

            # set crosssections from branches (1D)
            self.logger.info(f"Generating 1D crosssections from 1D branches.")

            # update data by reading user input
            _branches = self.staticgeoms["branches"]

            # update data by combining ini settings
            self.logger.debug(
                f'1D crosssections initialised with the following attributes: {list(crosssections_ini.get("default", {}))}.'
            )
            branches = helper.append_data_columns_based_on_ini_query(
                _branches, crosssections_ini
            )

            # select branches to use e.g. to facilitate setup a selection each time setup_* is called
            if branch_query is not None:
                branches = branches.query(branch_query)
                self.logger.info(f"Query branches for {branch_query}")

            if helper.check_geodataframe(branches):
                crsdefs, crslocs, branches = generate_crosssections(
                    branches, crosssections_ini
                )

            # update new branches with crsdef info to staticgeoms
            self.logger.debug(f"Updating branches vector to staticgeoms.")
            self.set_staticgeoms(branches, "branches")

            # add new crsdefs to staticgeoms
            self.logger.debug(f"Adding crsdefs vector to staticgeoms.")
            self.set_staticgeoms(
                gpd.GeoDataFrame(
                    crsdefs,
                    geometry=gpd.points_from_xy([0] * len(crsdefs), [0] * len(crsdefs)),
                ),
                "crsdefs",
            )  # FIXME: make crsdefs a vector to be add to static geoms. using dummy locations --> might cause issue for structures

            # add new crslocs to staticgeoms
            self.logger.debug(f"Adding crslocs vector to staticgeoms.")
            self.set_staticgeoms(crslocs, "crslocs")

        else:

            # TODO: setup roughness from other data types, e.g. points, xyz

            pass
            # raise NotImplementedError()

    def setup_manholes(
        self,
        manholes_ini_fn: str = None,
        manholes_fn: str = None,
        id_col: str = None,
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing manholes.")
        _branches = self.staticgeoms["branches"]

        # Setup of branches and manholes
        manholes, branches = delft3dfmpy_setupfuncs.setup_manholes(
            _branches,
            manholes_fn=delft3dfmpy_setupfuncs.parse_arg(
                manholes_fn
            ),  # FIXME: hydromt config parser could not parse '' to None
            manholes_ini_fn=delft3dfmpy_setupfuncs.parse_arg(
                manholes_ini_fn
            ),  # FIXME: hydromt config parser could not parse '' to None
            snap_offset=snap_offset,
            id_col=id_col,
            rename_map=delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            required_columns=delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            required_dtypes=delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger=logger,
        )

        self.logger.debug(f"Adding manholes vector to staticgeoms.")
        self.set_staticgeoms(manholes, "manholes")

        self.logger.debug(f"Updating branches vector to staticgeoms.")
        self.set_staticgeoms(branches, "branches")

    def setup_bridges(
        self,
        roughness_ini_fn: str = None,
        bridges_ini_fn: str = None,
        bridges_fn: str = None,
        id_col: str = None,
        branch_query: str = None,
        snap_method: str = "overall",
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing bridges.")
        _branches = self.staticgeoms["branches"]
        _crsdefs = self.staticgeoms["crsdefs"]
        _crslocs = self.staticgeoms["crslocs"]

        bridges, crsdefs = delft3dfmpy_setupfuncs.setup_bridges(
            _branches,
            _crsdefs,
            _crslocs,
            delft3dfmpy_setupfuncs.parse_arg(roughness_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(bridges_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(bridges_fn),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                branch_query
            ),  # TODO: replace with data adaptor
            snap_method,
            snap_offset,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding bridges vector to staticgeoms.")
        self.set_staticgeoms(bridges, "bridges")

        self.logger.debug(f"Updating crsdefs vector to staticgeoms.")
        self.set_staticgeoms(crsdefs, "crsdefs")

    def setup_gates(
        self,
        roughness_ini_fn: str = None,
        gates_ini_fn: str = None,
        gates_fn: str = None,
        id_col: str = None,
        branch_query: str = None,
        snap_method: str = "overall",
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing gates.")
        _branches = self.staticgeoms["branches"]
        _crsdefs = self.staticgeoms["crsdefs"]
        _crslocs = self.staticgeoms["crslocs"]

        gates = delft3dfmpy_setupfuncs.setup_gates(
            _branches,
            _crsdefs,
            _crslocs,
            delft3dfmpy_setupfuncs.parse_arg(roughness_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(gates_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(gates_fn),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                branch_query
            ),  # TODO: replace with data adaptor
            snap_method,
            snap_offset,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding gates vector to staticgeoms.")
        self.set_staticgeoms(gates, "gates")

    def setup_pumps(
        self,
        roughness_ini_fn: str = None,
        pumps_ini_fn: str = None,
        pumps_fn: str = None,
        id_col: str = None,
        branch_query: str = None,
        snap_method: str = "overall",
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing gates.")
        _branches = self.staticgeoms["branches"]
        _crsdefs = self.staticgeoms["crsdefs"]
        _crslocs = self.staticgeoms["crslocs"]

        pumps = delft3dfmpy_setupfuncs.setup_pumps(
            _branches,
            _crsdefs,
            _crslocs,
            delft3dfmpy_setupfuncs.parse_arg(roughness_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(pumps_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(pumps_fn),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                branch_query
            ),  # TODO: replace with data adaptor
            snap_method,
            snap_offset,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding pumps vector to staticgeoms.")
        self.set_staticgeoms(pumps, "pumps")

    def setup_culverts(
        self,
        roughness_ini_fn: str = None,
        culverts_ini_fn: str = None,
        culverts_fn: str = None,
        id_col: str = None,
        branch_query: str = None,
        snap_method: str = "overall",
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing culverts.")
        _branches = self.staticgeoms["branches"]
        _crsdefs = self.staticgeoms["crsdefs"]
        _crslocs = self.staticgeoms["crslocs"]

        culverts, crsdefs = delft3dfmpy_setupfuncs.setup_culverts(
            _branches,
            _crsdefs,
            _crslocs,
            delft3dfmpy_setupfuncs.parse_arg(roughness_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(culverts_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(culverts_fn),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                branch_query
            ),  # TODO: replace with data adaptor
            snap_method,
            snap_offset,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding culverts vector to staticgeoms.")
        self.set_staticgeoms(culverts, "culverts")

        self.logger.debug(f"Updating crsdefs vector to staticgeoms.")
        self.set_staticgeoms(crsdefs, "crsdefs")

    def setup_compounds(
        self,
        roughness_ini_fn: str = None,
        compounds_ini_fn: str = None,
        compounds_fn: str = None,
        id_col: str = None,
        branch_query: str = None,
        snap_method: str = "overall",
        snap_offset: float = 1,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing compounds.")
        _structures = [
            self.staticgeoms[s]
            for s in ["bridges", "gates", "pumps", "culverts"]
            if s in self.staticgeoms.keys()
        ]

        compounds = delft3dfmpy_setupfuncs.setup_compounds(
            _structures,
            delft3dfmpy_setupfuncs.parse_arg(roughness_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(compounds_ini_fn),
            delft3dfmpy_setupfuncs.parse_arg(compounds_fn),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                branch_query
            ),  # TODO: replace with data adaptor
            snap_method,
            snap_offset,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding compounds vector to staticgeoms.")
        self.set_staticgeoms(compounds, "compounds")

    def setup_boundaries(
        self,
        boundaries_fn: str = None,
        boundaries_fn_ini: str = None,
        id_col: str = None,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        logger=logging,
    ):
        """"""

        self.logger.info(f"Preparing boundaries.")
        _structures = [
            self.staticgeoms[s]
            for s in ["bridges", "gates", "pumps", "culverts"]
            if s in self.staticgeoms.keys()
        ]

        boundaries = delft3dfmpy_setupfuncs.setup_boundaries(
            delft3dfmpy_setupfuncs.parse_arg(boundaries_fn),
            delft3dfmpy_setupfuncs.parse_arg(boundaries_fn_ini),
            id_col,
            delft3dfmpy_setupfuncs.parse_arg(
                rename_map
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_columns
            ),  # TODO: replace with data adaptor
            delft3dfmpy_setupfuncs.parse_arg(
                required_dtypes
            ),  # TODO: replace with data adaptor
            logger,
        )

        self.logger.debug(f"Adding boundaries vector to staticgeoms.")
        self.set_staticgeoms(boundaries, "boundaries")

    def _setup_datamodel(self):

        """setup data model using dfm and drr naming conventions"""
        if self._datamodel == None:
            self._datamodel = delft3dfmpy_setupfuncs.setup_dm(
                self.staticgeoms, logger=self.logger
            )

    def setup_dflowfm(
        self,
        model_type: str = "1d",
        one_d_mesh_distance: float = 40,
    ):
        """ """
        self.logger.info(f"Preparing DFlowFM 1D model.")

        self._setup_datamodel()

        self._dfmmodel = delft3dfmpy_setupfuncs.setup_dflowfm(
            self._datamodel,
            model_type=model_type,
            one_d_mesh_distance=one_d_mesh_distance,
            logger=self.logger,
        )

    ## I/O

    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_dfmmodel()

    def write(self):  # complete model
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        if self.config:  # try to read default if not yet set
            self.write_config()  # FIXME: config now isread from default, modified and saved temporaryly in the models folder --> being read by dfm and modify?
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._dfmmodel:
            self.write_dfmmodel()
        if self._forcing:
            self.write_forcing()

    def read_staticmaps(self):
        """Read staticmaps at <root/?/> and parse to xarray Dataset"""
        # to read gdal raster files use: hydromt.open_mfraster()
        # to read netcdf use: xarray.open_dataset()
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        self.set_staticmaps(hydromt.open_mfraster(join(self.root, "*.tif")))

    def write_staticmaps(self):
        """Write staticmaps at <root/?/> in model ready format"""
        # to write to gdal raster files use: self.staticmaps.raster.to_mapstack()
        # to write to netcdf use: self.staticmaps.to_netcdf()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        self.staticmaps.raster.to_mapstack(join(self.root, "dflowfm"))

    def read_staticgeoms(self):
        """Read staticgeoms at <root/?/> and parse to dict of geopandas"""
        if not self._write:
            # start fresh in read-only mode
            self._staticgeoms = dict()
        for fn in glob.glob(join(self.root, "*.xy")):
            name = basename(fn).replace(".xy", "")
            geom = hydromt.open_vector(fn, driver="xy", crs=self.crs)
            self.set_staticgeoms(geom, name)

    def write_staticgeoms(self):  # write_all()
        """Write staticmaps at <root/?/> in model ready format"""
        # TODO: write_data_catalogue with updates of the rename based on mapping table?
        if not self._write:
            raise IOError("Model opened in read-only mode")
        for name, gdf in self.staticgeoms.items():
            fn_out = join(self.root, "staticgeoms", f"{name}.geojson")
            self.staticgeoms[name].reset_index(drop=True).to_file(fn_out, driver='GeoJSON') # FIXME: does not work if does not reset index

    def read_forcing(self):
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        return self._forcing
        # raise NotImplementedError()

    def write_forcing(self):
        """write forcing at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_dfmmodel(self):
        """Read dfmmodel at <root/?/> and parse to model class (deflt3dfmpy)"""
        pass
        # raise NotImplementedError()

    def write_dfmmodel(self):
        """Write dfmmodel at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        delft3dfmpy_setupfuncs.write_dfmmodel(
            self.dfmmodel, output_dir=self.root, name="DFLOWFM", logger=self.logger
        )

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        return self._states
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        return self._results
        # raise NotImplementedError()

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    @property
    def crs(self):
        # return pyproj.CRS.from_epsg(self.get_config("global.epsg", fallback=4326))
        return self.region.crs

    @property
    def region_name(self):
        return self._region_name

    @property
    def dfmmodel(self):
        if self._dfmmodel == None:
            self.init_dfmmodel()
        return self._dfmmodel

    def init_dfmmodel(self):
        self._dfmmodel = delft3dfmpy_setupfuncs.DFlowFMModel()

    @property
    def branches(self):
        """
        Returns the branches (gpd.GeoDataFrame object) representing the 1D network.
        Contains several "branchType" for : channel, river, pipe, tunnel.
        """
        if self._branches.empty:
            # self.read_branches() #not implemented yet
            self._branches = gpd.GeoDataFrame()
        return self._branches

    def set_branches(self, branches: gpd.GeoDataFrame):
        """Updates the branches object as well as the linked staticgeoms."""
        # Check if "branchType" col in new branches
        if "branchType" in branches.columns:
            self._branches = branches
        else:
            self.logger.error(
                "'branchType' column absent from the new branches, could not update."
            )
        # Update channels/pipes in staticgeoms
        _ = self.set_branches_component(name="channel")
        _ = self.set_branches_component(name="pipe")

    def add_branches(self, new_branches: gpd.GeoDataFrame, branchtype: str):
        """Add new branches of branchtype to the branches object"""
        branches = self._branches.copy()
        # Check if "branchType" in new_branches column, else add
        if "branchType" not in new_branches.columns:
            new_branches["branchType"] = np.repeat(branchtype, len(new_branches.index))
        branches = branches.append(new_branches, ignore_index=True)
        # Check if we need to do more check/process to make sure everything is well connected
        validate_branches(branches)
        self.set_branches(branches)

    def set_branches_component(self, name):
        gdf_comp = self.branches[self.branches["branchType"] == name]
        if gdf_comp.index.size > 0:
            self.set_staticgeoms(gdf_comp, name=f"{name}s")
        return gdf_comp

    @property
    def channels(self):
        if "channels" in self.staticgeoms:
            gdf = self.staticgeoms["channels"]
        else:
            gdf = self.set_branches_component("channel")
        return gdf

    @property
    def pipes(self):
        if "pipes" in self.staticgeoms:
            gdf = self.staticgeoms["pipes"]
        else:
            gdf = self.set_branches_component("pipe")
        return gdf
