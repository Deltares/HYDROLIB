"""Implement Delft3D-FM hydromt plugin model class"""

import logging
import os
from datetime import datetime, timedelta
from os.path import basename, dirname, isdir, isfile, join
from pathlib import Path
from turtle import st
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
import meshkernel as mk
from pyproj import CRS
from hydromt.models import MeshModel
from shapely.geometry import box, LineString
from datetime import datetime, timedelta
from meshkernel import MeshKernel

from hydrolib.core.dflowfm import FMModel, IniFieldModel, Mesh1d, Network
from hydrolib.core.dimr import DIMR, FMComponent, Start
from hydrolib.dhydamo.geometry import mesh
from meshkernel import GeometryList


from . import DATADIR, utils, workflows

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(MeshModel):
    """API for Delft3D-FM models in HydroMT"""

    _NAME = "dflowfm"
    _CONF = "DFlowFM.mdu"
    _DATADIR = DATADIR
    _GEOMS = {}
    _API = {
        "crs": CRS,
        "config": Dict[str, Any],
        "region": gpd.GeoDataFrame,
        "geoms": Dict[str, gpd.GeoDataFrame],
        "maps": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "mesh": Union[xr.DataArray, xr.Dataset],
        "forcing": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "results": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "states": Dict[str, Union[xr.DataArray, xr.Dataset]],
    }
    _MAPS = {
        "elevtn": {
            "name": "bedlevel",
            "initype": "initial",
            "interpolation": "triangulation",
            "locationtype": "2d",
        },
        "waterlevel": {
            "name": "waterlevel",
            "initype": "initial",
            "interpolation": "mean",
            "locationtype": "2d",
            "averagingrelsize": 1.01,  # default
        },
        "waterdepth": {
            "name": "waterdepth",
            "initype": "initial",
            "interpolation": "mean",
            "locationtype": "2d",
            "averagingrelsize": 1.01,
        },
        "pet": {
            "name": "PotentialEvaporation",
            "initype": "initial",
            "interpolation": "triangulation",
            "locationtype": "2d",
        },
        "infiltcap": {
            "name": "InfiltrationCapacity",
            "initype": "initial",
            "interpolation": "triangulation",
            "locationtype": "2d",
        },
        "roughness_chezy": {
            "name": "frictioncoefficient",
            "initype": "parameter",
            "interpolation": "triangulation",
            "locationtype": "2d",
            "frictype": 0,
        },
        "roughness_manning": {
            "name": "frictioncoefficient",
            "initype": "parameter",
            "interpolation": "triangulation",
            "locationtype": "2d",
            "frictype": 1,
        },
        "roughness_walllawnikuradse": {
            "name": "frictioncoefficient",
            "initype": "parameter",
            "interpolation": "triangulation",
            "locationtype": "2d",
            "frictype": 2,
        },
        "roughness_whitecolebrook": {
            "name": "frictioncoefficient",
            "initype": "parameter",
            "interpolation": "triangulation",
            "locationtype": "2d",
            "frictype": 3,
        },
    }
    _FOLDERS = ["dflowfm", "geoms", "mesh", "maps"]
    _CLI_ARGS = {"region": "setup_region", "res": "setup_mesh2d"}
    _CATALOGS = join(_DATADIR, "parameters_data.yml")

    def __init__(
        self,
        root: Union[str, Path] = None,
        mode: str = "w",
        config_fn: str = None,  # hydromt config contain glob section, anything needed can be added here as args
        data_libs: List[str] = [],  # yml
        dimr_fn: str = None,
        network_snap_offset=25,
        snap_newbranches_to_branches_at_snapnodes=True,
        openwater_computation_node_distance=40,
        logger=logger,
    ):
        """Initialize the DFlowFMModel.

        Parameters
        ----------
        root : str or Path
            The model root location.
        mode : {'w','r','r+'}
            Write/read/append mode.
            Default is "w".
        config_fn : str, optional
            The D-Flow FM model configuration file (.mdu). If None, default configuration file is used.
            Default is None.
        data_libs : list of str, optional
            List of data catalog yaml files.
            Default is None.
        logger
            The logger used to log messages.
        """

        if not isinstance(root, (str, Path)):
            raise ValueError("The 'root' parameter should be a of str or Path.")

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
        # model specific
        self._branches = None
        self._dimr = None
        self._dimr_fn = "dimr_config.xml" if dimr_fn is None else dimr_fn
        self._dfmmodel = None
        self._config_fn = (
            join("dflowfm", self._CONF) if config_fn is None else config_fn
        )
        self.data_catalog.from_yml(self._CATALOGS)

        self.config

        # Global options for generation of the mesh1d network
        self._network_snap_offset = network_snap_offset
        self._snap_newbranches_to_branches_at_snapnodes = snap_newbranches_to_branches_at_snapnodes
        self._openwater_computation_node_distance = openwater_computation_node_distance
        self._res = None

    def setup_region(
        self,
        region: dict,
        crs: int = None,
    ):
        """Define the 1D model region.

        Adds model layer:

        * **region** geom: model region

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
        crs : int, optional
            Coordinate system (EPSG number) of the model. If not provided, equal to the region crs
            if "grid" or "geom" option are used, and to 4326 if "bbox" is used, i.e. specified crs will be ignored.

        Raises
        ------
        ValueError
            If the region kind in `region` is not supported for D-Flow FM.
            Supported regions are: "bbox", "grid" and "geom".
        """

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind == "bbox":
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
        self.set_geoms(geom, "region")

    def _setup_branches(
        self,
        br_fn: Union[str, Path, gpd.GeoDataFrame],
        defaults_fn: Union[str, Path, pd.DataFrame],
        br_type: str,
        friction_type: str,
        friction_value: float,
        crosssections_shape: str = None,
        crosssections_value: Union[List[float], float] = None,
        spacing: pd.DataFrame = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
        allowed_columns: List[str] = [],
        filter: str = None,
    ):
        """This function is to set all common steps to add branches type of objects (ie channels, rivers, pipes...).
        Default frictions and crossections will also be added.

         Parameters
         ----------
         br_fn : str, gpd.GeoDataFrame
             Either data source in data catalogue or Path for branches file or branches gpd.GeoDataFrame directly.
         defaults_fn : str Path
             Either data source in data catalogue or path to a csv file containing all defaults values per "branchType" or defaults pd.DataFRame directly.
         br_type : str
             branches type. Either "river", "channel", "pipe".
         friction_type : str
             Type of friction to use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
         friction_value : float
             Value corresponding to ''friction_type''. Units are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
         crosssections_shape : str, optional
             Shape of branch crosssections to overwrite defaults. Either "circle" or "rectangle".
         crosssections_value : float or list of float, optional
             Crosssections parameter value to overwrite defaults.
             If ``crosssections_shape`` = "circle", expects a diameter [m], used for br_type == "pipe"
             If ``crosssections_shape`` = "rectangle", expects a list with [width, height] (e.g. [1.0, 1.0]) [m]. used for br_type == "river" or "channel".
         spacing: float, optional
             Spacing value in meters to split the long pipelines lines into shorter pipes. By default inf - no splitting is applied.
        snap_offset: float, optional
             Snapping tolerance to automatically connecting branches. Tolerance must be smaller than the shortest pipe length.
             By default 0.0, no snapping is applied.
         allow_intersection_snapping: bool, optional
             Switch to choose whether snapping of multiple branch ends are allowed when ``snap_offset`` is used.
             By default True.
         allowed_columns: list, optional
             List of columns to filter in branches GeoDataFrame
         filter: str, optional
             Keyword in branchType column of br_fn used to filter lines. If None all lines in br_fn are used (default).


         See also
         --------
         dflowfm.setup_rivers
         dflowfm.setup_pipes
        """
        # 1. Read data and filter within region
        # If needed read the branches GeoDataFrame
        if isinstance(br_fn, str) or isinstance(br_fn, Path):
            gdf_br = self.data_catalog.get_geodataframe(
                br_fn, geom=self.region, buffer=0, predicate="intersects"
            )
        else:
            gdf_br = br_fn
        # Filter features based on filter
        if filter is not None and "branchType" in gdf_br.columns:
            gdf_br = gdf_br[gdf_br["branchType"].str.lower() == filter.lower()]
            self.logger.info(f"Set {filter} locations filtered from {br_fn} as {br_type} .")
        # Check if features in region
        if len(gdf_br) == 0:
            self.logger.warning(f"No 1D {type} locations found within domain")
            return None

        # Read defaults table
        if isinstance(defaults_fn, pd.DataFrame):
            defaults = defaults_fn
        else:
            if defaults_fn is None:
                self.logger.warning(
                    f"defaults_fn ({defaults_fn}) does not exist. Fall back choice to defaults. "
                )
                defaults_fn = Path(self._DATADIR).joinpath(
                    f"{br_type}s", f"{br_type}s_defaults.csv"
                )
            defaults = self.data_catalog.get_dataframe(defaults_fn, time_tuple=())
            self.logger.info(f"{br_type} default settings read from {defaults_fn}.")

        # 2. Add defaults
        # Add branchType and branchId attributes if does not exist
        # overwrite branchType if exist
        gdf_br["branchType"] = pd.Series(
                data=np.repeat(br_type, len(gdf_br)), index=gdf_br.index, dtype=str
            )
        if "branchId" not in gdf_br.columns:
            data = [
                f"{br_type}_{i}"
                for i in np.arange(
                    len(self.branches) + 1, len(self.branches) + len(gdf_br) + 1
                )
            ]  # avoid duplicated ids being generated
            gdf_br["branchId"] = pd.Series(data, index=gdf_br.index, dtype=str)

        # assign id
        id_col = "branchId"
        gdf_br.index = gdf_br[id_col]
        gdf_br.index.name = id_col

        # filter for allowed columns
        allowed_columns = set(allowed_columns).intersection(gdf_br.columns)
        gdf_br = gpd.GeoDataFrame(gdf_br[allowed_columns], crs=gdf_br.crs)

        # Add spacing to defaults
        if spacing is not None:
            defaults["spacing"] = spacing
        # Add friction to defaults
        defaults["frictionType"] = friction_type
        defaults["frictionValue"] = friction_value
        # Add crosssections to defaults
        if crosssections_shape == "circle":
            if isinstance(crosssections_value, float):
                defaults["shape"] = crosssections_shape
                defaults["diameter"] = crosssections_value
            else:
                self.logger.warning(
                    "If crosssections_shape is circle, crosssections_value should be a single float for diameter. Keeping defaults"
                )
        elif crosssections_shape == "rectangle":
            if isinstance(crosssections_value, list) and len(crosssections_value) == 2:
                defaults["shape"] = crosssections_shape
                defaults["width"], defaults["height"] = crosssections_value
                defaults["closed"] = "no"
            else:
                self.logger.warning(
                    "If crosssections_shape is rectangle, crosssections_value should be a list with [width, height] values. Keeping defaults"
                )

        self.logger.info("Adding/Filling branches attributes values")
        gdf_br = workflows.update_data_columns_attributes(
            gdf_br, defaults, brtype=br_type
        )

        # 4. Split and prepare branches
        if gdf_br.crs.is_geographic:  # needed for length and splitting
            gdf_br = gdf_br.to_crs(3857)
        # Line smoothing for pipes
        smooth_branches = br_type == "pipe"

        self.logger.info(f"Processing branches")
        branches, branches_nodes = workflows.process_branches(
            gdf_br,
            id_col="branchId",
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
            smooth_branches=smooth_branches,
            logger=self.logger,
        )

        self.logger.info(f"Validating branches")
        workflows.validate_branches(branches)

        # convert to model crs
        branches = branches.to_crs(self.crs)
        branches_nodes = branches_nodes.to_crs(self.crs)

        # 5. Add friction id
        branches["frictionId"] = [
            f"{ftype}_{fvalue}"
            for ftype, fvalue in zip(
                branches["frictionType"], branches["frictionValue"]
            )
        ]

        return branches, branches_nodes

    def setup_channels(
        self,
        channels_fn: str,
        channels_defaults_fn: str = None,
        channel_filter: str = None,
        friction_type: str = "Manning",
        friction_value: float = 0.023,
        crosssections_fn: str = None,
        crosssections_type: str = None,
        spacing: int = None,
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
            Name of data source for channelsparameters, see data/data_sources.yml.
            Note only the lines that are intersects with the region polygon will be used.
            * Optional variables: [branchId, branchType, branchOrder, material, friction_type, friction_value]
        channels_defaults_fn : str, optional
            Path to a csv file containing all defaults values per 'branchType'.
            Default is None.
        channel_filter: str, optional
            Keyword in branchType column of channels_fn used to filter river lines. If None all lines in channels_fn are used (default).
        friction_type : str, optional
            Type of friction to use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
            By default "Manning".
        friction_value : float, optional.
            Units corresponding to [friction_type] are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
            Friction value. By default 0.023.
        crosssections_fn : str or Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyzpoints"
            * Required variables: [crsId, order, z]
            If ``crosssections_type`` = "points"
            * Required variables: [crsId, order, z]
            By default None, crosssections will be set from branches
        crosssections_type : str, optional
            Type of crosssections read from crosssections_fn. One of ["xyzpoints"].
            By default None.
        snap_offset: float, optional
            Snapping tolerance to automatically connecting branches.
            By default 0.0, no snapping is applied.
        allow_intersection_snapping: bool, optional
            Switch to choose whether snapping of multiple branch ends are allowed when ``snap_offset`` is used.
            By default True.

        See Also
        ----------
        dflowfm._setup_branches
        """
        self.logger.info(f"Preparing 1D channels.")

        # filter for allowed columns
        _allowed_columns = [
            "geometry",
            "branchId",
            "branchType",
            "branchOrder",
            "material",
            "shape",
            "diameter",
            "width",
            "t_width",
            "height",
            "bedlev",
            "closed",
            "friction_type",
            "friction_value",
        ]

        # Build the channels branches and nodes and fill with attributes and spacing
        channels, channel_nodes = self._setup_branches(
            br_fn=channels_fn,
            defaults_fn=channels_defaults_fn,
            br_type="channel",
            friction_type=friction_type,
            friction_value=friction_value,
            spacing=spacing,
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
            allowed_columns=_allowed_columns,
            filter=channel_filter,
        )

        # setup crosssections
        if crosssections_type is None:
            crosssections_type = "branch"  # TODO: maybe assign a specific one for river, like branch_river
        assert {crosssections_type}.issubset({"xyzpoints", "branch"})
        crosssections = self._setup_crosssections(
            branches=channels,
            crosssections_fn=crosssections_fn,
            crosssections_type=crosssections_type,
        )

        # add crosssections to exisiting ones and update geoms
        self.logger.debug(f"Adding crosssections vector to geoms.")
        self.add_crosssections(crosssections)

        # setup geoms
        self.logger.debug(f"Adding branches and branch_nodes vector to geoms.")
        self.set_geoms(channels, "channels")
        self.set_geoms(channel_nodes, "channel_nodes")

        # add to branches
        self.add_branches(
            channels,
            branchtype="channel",
            node_distance=self._openwater_computation_node_distance,
        )

    def setup_rivers_from_dem(
        self,
        hydrography_fn: str,
        river_geom_fn: str = None,
        rivers_defaults_fn: str = None,
        rivdph_method="gvf",
        rivwth_method="geom",
        river_upa=25.0,
        river_len=1000,
        min_rivwth=50.0,
        min_rivdph=1.0,
        rivbank=True,
        rivbankq=25,
        segment_length=3e3,
        smooth_length=10e3,
        friction_type: str = "Manning",
        friction_value: float = 0.023,
        constrain_rivbed=True,
        constrain_estuary=True,
        **kwargs,  # for workflows.get_river_bathymetry method
    ) -> None:
        """
        This component sets the all river parameters from hydrograph and dem maps.

        River cells are based on the `river_mask_fn` raster file if `rivwth_method='mask'`,
        or if `rivwth_method='geom'` the rasterized segments buffered with half a river width
        ("rivwth" [m]) if that attribute is found in `river_geom_fn`.

        If a river segment geometry file `river_geom_fn` with bedlevel column ("zb" [m+REF]) or
        a river depth ("rivdph" [m]) in combination with `rivdph_method='geom'` is provided,
        this attribute is used directly.

        Otherwise, a river depth is estimated based on bankfull discharge ("qbankfull" [m3/s])
        attribute taken from the nearest river segment in `river_geom_fn` or `qbankfull_fn`
        upstream river boundary points if provided.

        The river depth is relative to the bankfull elevation profile if `rivbank=True` (default),
        which is estimated as the `rivbankq` elevation percentile [0-100] of cells neighboring river cells.
        This option requires the flow direction ("flwdir") and upstream area ("uparea") maps to be set
        using the hydromt.flw.flwdir_from_da method. If `rivbank=False` the depth is simply subtracted
        from the elevation of river cells.

        Missing river width and river depth values are filled by propagating valid values downstream and
        using the constant minimum values `min_rivwth` and `min_rivdph` for the remaining missing values.

        Updates model layer:

        * **dep** map: combined elevation/bathymetry [m+ref]

        Adds model layers

        * **rivmsk** map: map of river cells (not used by SFINCS)
        * **rivers** geom: geometry of rivers (not used by SFINCS)

        Parameters
        ----------
        hydrography_fn : str
            Hydrography data to derive river shape and characteristics from.
            * Required variables: ['elevtn']
            * Optional variables: ['flwdir', 'uparea']
        river_geom_fn : str, optional
            Line geometry with river attribute data.
            * Required variable for direct bed level burning: ['zb']
            * Required variable for direct river depth burning: ['rivdph'] (only in combination with rivdph_method='geom')
            * Variables used for river depth estimates: ['qbankfull', 'rivwth']
        rivers_defaults_fn : str Path
            Path to a csv file containing all defaults values per 'branchType'.
        rivdph_method : {'gvf', 'manning', 'powlaw'}
            River depth estimate method, by default 'gvf'
        rivwth_method : {'geom', 'mask'}
            Derive the river with from either the `river_geom_fn` (geom) or
            `river_mask_fn` (mask; default) data.
        river_upa : float, optional
            Minimum upstream area threshold for rivers [km2], by default 25.0
        river_len: float, optional
            Mimimum river length within the model domain threshhold [m], by default 1000 m.
        min_rivwth, min_rivdph: float, optional
            Minimum river width [m] (by default 50.0) and depth [m] (by default 1.0)
        rivbank: bool, optional
            If True (default), approximate the reference elevation for the river depth based
            on the river bankfull elevation at cells neighboring river cells. Otherwise
            use the elevation of the local river cell as reference level.
        rivbankq : float, optional
            quantile [1-100] for river bank estimation, by default 25
        segment_length : float, optional
            Approximate river segment length [m], by default 5e3
        smooth_length : float, optional
            Approximate smoothing length [m], by default 10e3
        friction_type : str, optional
            Type of friction tu use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
            By default "Manning".
        friction_value : float, optional.
            Units corresponding to [friction_type] are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
            Friction value. By default 0.023.
        constrain_estuary : bool, optional
            If True (default) fix the river depth in estuaries based on the upstream river depth.
        constrain_rivbed : bool, optional
            If True (default) correct the river bed level to be hydrologically correct,
            i.e. sloping downward in downstream direction.

        See Also
        ----------
        workflows.get_river_bathymetry

        """
        self.logger.info(f"Preparing river shape from hydrography data.")
        # read data
        ds_hydro = self.data_catalog.get_rasterdataset(
            hydrography_fn, geom=self.region, buffer=10
        )
        if isinstance(ds_hydro, xr.DataArray):
            ds_hydro = ds_hydro.to_dataset()

        # read river line geometry data
        gdf_riv = None
        if river_geom_fn is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                river_geom_fn, geom=self.region
            ).to_crs(ds_hydro.raster.crs)

        # check if flwdir and uparea in ds_hydro
        if "flwdir" not in ds_hydro.data_vars:
            da_flw = hydromt.flw.d8_from_dem(ds_hydro["elevtn"])
        else:
            da_flw = ds_hydro["flwdir"]
        flwdir = hydromt.flw.flwdir_from_da(da_flw, ftype="d8")
        if "uparea" not in ds_hydro.data_vars:
            da_upa = xr.DataArray(
                dims=ds_hydro["elevtn"].raster.dims,
                coords=ds_hydro["elevtn"].raster.coords,
                data=flwdir.upstream_area(unit="km2"),
                name="uparea",
            )
            da_upa.raster.set_nodata(-9999)
            ds_hydro["uparea"] = da_upa

        # get river shape and bathymetry
        if friction_type == "Manning":
            kwargs.update(manning=friction_value)
        elif rivdph_method == "gvf":
            raise ValueError(
                "rivdph_method 'gvf' requires friction_type='Manning'. Use 'geom' or 'powlaw' instead."
            )
        gdf_riv, _ = workflows.get_river_bathymetry(
            ds_hydro,
            flwdir=flwdir,
            gdf_riv=gdf_riv,
            gdf_qbf=None,
            rivdph_method=rivdph_method,
            rivwth_method=rivwth_method,
            river_upa=river_upa,
            river_len=river_len,
            min_rivdph=min_rivdph,
            min_rivwth=min_rivwth,
            rivbank=rivbank,
            rivbankq=rivbankq,
            segment_length=segment_length,
            smooth_length=smooth_length,
            constrain_estuary=constrain_estuary,
            constrain_rivbed=constrain_rivbed,
            logger=self.logger,
            **kwargs,
        )
        # Rename river properties column and reproject
        rm_dict = {"rivwth": "width", "rivdph": "height", "zb": "bedlev"}
        gdf_riv = gdf_riv.rename(columns=rm_dict).to_crs(self.crs)

        # assign default attributes
        if rivers_defaults_fn is None or not rivers_defaults_fn.is_file():
            self.logger.warning(
                f"rivers_defaults_fn ({rivers_defaults_fn}) does not exist. Fall back choice to defaults. "
            )
            rivers_defaults_fn = Path(self._DATADIR).joinpath(
                "rivers", "rivers_defaults.csv"
            )
        defaults = pd.read_csv(rivers_defaults_fn)
        # Make sure default shape is rectangle
        defaults["shape"] = np.array(["rectangle"])
        self.logger.info(f"river default settings read from {rivers_defaults_fn}.")

        # filter for allowed columns
        _allowed_columns = [
            "geometry",
            "branchId",
            "branchType",
            "branchOrder",
            "material",
            "shape",
            "diameter",
            "width",
            "t_width",
            "height",
            "bedlev",
            "closed",
            "friction_type",
            "friction_value",
        ]

        # Build the rivers branches and nodes and fill with attributes and spacing
        rivers, river_nodes = self._setup_branches(
            br_fn=gdf_riv,
            defaults_fn=defaults,
            br_type="river",
            friction_type=friction_type,
            friction_value=friction_value,
            spacing=None,  # does not allow spacing for rivers
            snap_offset=0.0,
            allow_intersection_snapping=True,
            allowed_columns=_allowed_columns,
        )

        # setup crosssections
        crosssections = self._setup_crosssections(
            branches=rivers,
            crosssections_fn=None,
            crosssections_type="branch",
        )

        # add crosssections to exisiting ones and update geoms
        self.logger.debug(f"Adding crosssections vector to geoms.")
        self.add_crosssections(crosssections)

        # setup geoms #TODO do we still need channels?
        self.logger.debug(f"Adding rivers and river_nodes vector to geoms.")
        self.set_geoms(rivers, "rivers")
        self.set_geoms(river_nodes, "rivers_nodes")

        # add to branches
        self.add_branches(
            rivers,
            branchtype="river",
            node_distance=self._openwater_computation_node_distance,
        )

    def setup_rivers(
        self,
        rivers_fn: str,
        rivers_defaults_fn: str = None,
        river_filter: str = None,
        friction_type: str = "Manning",
        friction_value: float = 0.023,
        crosssections_fn: Union[int, list] = None,
        crosssections_type: Union[int, list] = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
    ):
        """Prepares the 1D rivers and adds to 1D branches.

        1D rivers must contain valid geometry, friction and crosssections.

        The river geometry is read from ``rivers_fn``. If defaults attributes
        [branchOrder, spacing, material, shape, width, t_width, height, bedlev, closed] are not present in ``rivers_fn``,
        they are added from defaults values in ``rivers_defaults_fn``. For branchId and branchType, they are created on the fly
        if not available in rivers_fn ("river" for Type and "river_{i}" for Id).

        Friction attributes are either taken from ``rivers_fn`` or filled in using ``friction_type`` and
        ``friction_value`` arguments.
        Note for now only branch friction or global friction is supported.

        Crosssections are read from ``crosssections_fn`` based on the ``crosssections_type``. If there is no
        ``crosssections_fn`` values are derived at the centroid of each river line based on defaults. If there are multiple
        types of crossections, specify them as lists.

        Adds/Updates model layers:

        * **rivers** geom: 1D rivers vector
        * **branches** geom: 1D branches vector
        * **crosssections** geom: 1D crosssection vector

        Parameters
        ----------
        rivers_fn : str
            Name of data source for rivers parameters, see data/data_sources.yml.
            Note only the lines that are intersects with the region polygon will be used.
            * Optional variables: [branchId, branchType, branchOrder, material, friction_type, friction_value]
        rivers_defaults_fn : str Path
            Path to a csv file containing all defaults values per 'branchType'.
            By default None.
        river_filter: str, optional
            Keyword in branchType column of rivers_fn used to filter river lines. If None all lines in rivers_fn are used (default).
        friction_type : str, optional
            Type of friction to use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
            By default "Manning".
        friction_value : float, optional.
            Units corresponding to [friction_type] are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
            Friction value. By default 0.023.
        crosssections_fn : str, Path, or a list of str or Path, optional
            Name of data source for crosssections, see data/data_sources.yml. One or a list corresponding to ``crosssections_type`` .
            If ``crosssections_type`` = "branch"
            crosssections_fn should be None
            If ``crosssections_type`` = "xyz"
            * Required variables: [crsId, order, z]
            If ``crosssections_type`` = "point"
            * Required variables: [crsId, shape, shift]
            By default None, crosssections will be set from branches
        crosssections_type : str, or a list of str, optional
            Type of crosssections read from crosssections_fn. One or a list of ["branch", "xyz", "point"].
            By default None.
        snap_offset: float, optional
            Snapping tolerance to automatically connecting branches.
            By default 0.0, no snapping is applied.
        allow_intersection_snapping: bool, optional
            Switch to choose whether snapping of multiple branch ends are allowed when ``snap_offset`` is used.
            By default True.

        See Also
        ----------
        dflowfm._setup_branches
        dflowfm._setup_crosssections
        """
        self.logger.info(f"Preparing 1D rivers.")
        # filter for allowed columns
        _allowed_columns = [
            "geometry",
            "branchId",
            "branchType",
            "branchOrder",
            "material",
            "shape",
            "width",
            "t_width",
            "height",
            "bedlev",
            "closed",
            "friction_type",
            "friction_value",
        ]

        # Build the rivers branches and nodes and fill with attributes and spacing
        rivers, river_nodes = self._setup_branches(
            br_fn=rivers_fn,
            defaults_fn=rivers_defaults_fn,
            br_type="river",
            friction_type=friction_type,
            friction_value=friction_value,
            spacing=None,  # does not allow spacing for rivers
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
            allowed_columns=_allowed_columns,
            filter=river_filter,
        )

        # setup crosssections
        if crosssections_type is None:
            crosssections_type = []
            crosssections_fn = [] 
        elif isinstance(crosssections_type, str):
            crosssections_type = [crosssections_type]
            crosssections_fn = [crosssections_fn]
        elif isinstance(crosssections_type, list):
            assert len(crosssections_type) == len(crosssections_fn)
        # inser branch as default
        if "branch" not in crosssections_type:
            crosssections_type.insert(0, "branch")
            crosssections_fn.insert(0, None)
        
        for crs_fn, crs_type in zip(crosssections_fn, crosssections_type):
            assert {crs_type}.issubset({"xyz", "point", "branch"})
            crosssections = self._setup_crosssections(
                branches=rivers,
                crosssections_fn=crs_fn,
                crosssections_type=crs_type,
            )
            self.add_crosssections(crosssections)

        # setup branch orders
        # for crossection type yz or xyz, always use branchOrder = -1, because no interpolation can be applied.
        # TODO: change to lower case is needed
        _overwrite_branchorder = self.crosssections[self.crosssections["crsdef_type"].str.contains("yz")]["crsdef_branchId"].tolist()
        if len(_overwrite_branchorder) > 0:
            rivers.loc[rivers["branchId"].isin(_overwrite_branchorder), "branchOrder"] = -1

            # for crossection type yz or xyz, always use branchOrder = -1, because no interpolation can be applied.
            # TODO: change to lower case is needed
            _overwrite_branchorder = self.crosssections[self.crosssections["crsdef_type"].str.contains("yz")]["crsdef_branchId"].tolist()
            if len(_overwrite_branchorder) > 0:
                rivers.loc[rivers["branchId"].isin(_overwrite_branchorder), "branchOrder"] = -1

        # setup geoms
        self.logger.debug(f"Adding rivers and river_nodes vector to geoms.")
        self.set_geoms(rivers, "rivers")
        self.set_geoms(river_nodes, "rivers_nodes")

        # add to branches
        self.add_branches(
            rivers,
            branchtype="river",
            node_distance=self._openwater_computation_node_distance,
        )

    def setup_pipes(
        self,
        pipes_fn: str,
        pipes_defaults_fn: Union[str, None] = None,
        pipe_filter: Union[str, None] = None,
        spacing: float = np.inf,
        friction_type: str = "WhiteColebrook",
        friction_value: float = 0.003,
        crosssections_shape: str = "circle",
        crosssections_value: Union[int, list] = 0.5,
        dem_fn: Union[str, None] = None,
        pipes_depth: float = 2.0,
        pipes_invlev: float = -2.5,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
    ):
        """Prepares the 1D pipes and adds to 1D branches.

        1D pipes must contain valid geometry, friction and crosssections.

        The pipe geometry is read from ``pipes_fn``.
        if branchType is present in ``pipes_fn``, it is possible to filter pipe geometry using an additional filter specificed in``pipe_filter``.
        If defaults attributes ["branchOrder"] are not present in ``pipes_fn``, they are added from defaults values in ``pipes_defaults_fn``.
        For branchId and branchType, if they are not present in ``pipes_fn``, they are created on the fly ("pipe" for branchType and "pipe_{i}" for branchId).
        The pipe geometry can be processed using splitting based on ``spacing``.

        Friction attributes ["frictionType", "frictionValue"] are either taken from ``pipes_fn`` or filled in using ``friction_type`` and
        ``friction_value`` arguments.
        Note for now only branch friction or global friction is supported.

        Crosssections definition attributes ["shape", "diameter", "width", "height", "closed"] are either taken from ``pipes_fn`` or filled in using ``crosssections_shape`` and ``crosssections_value``.
        Crosssections location attributes ["invlev_up", "invlev_dn"] are either taken from ``pipes_fn``, or derived from ``dem_fn`` minus a fixed depth ``pipe_depth`` [m], or from a constant ``pipe_invlev`` [m asl] (not recommended! should be edited before a model run).

        Adds/Updates model layers:

        * **pipes** geom: 1D pipes vector
        * **branches** geom: 1D branches vector
        * **crosssections** geom: 1D crosssection vector

        Parameters
        ----------
        pipes_fn : str
            Name of data source for pipes parameters, see data/data_sources.yml.
            Note only the lines that are within the region polygon will be used.
            * Optional variables: [branchId, branchType, branchOrder, spacing, frictionType, frictionValue, shape, diameter, width, height, closed, invlev_up, invlev_dn]
            #TODO: material table is used for friction which is not implemented
        pipes_defaults_fn : str Path
            Path to a csv file containing all defaults values per "branchType"'".
        pipe_filter: str, optional
            Keyword in branchType column of pipes_fn used to filter pipe lines. If None all lines in pipes_fn are used (default).
        spacing: float, optional
            Spacing value in meters to split the long pipelines lines into shorter pipes. By default inf - no splitting is applied.
        friction_type : str, optional
            Type of friction to use. One of ["Manning", "Chezy", "wallLawNikuradse", "WhiteColebrook", "StricklerNikuradse", "Strickler", "deBosBijkerk"].
            By default "WhiteColeBrook".
        friction_value : float, optional.
            Units corresponding to ''friction_type'' are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
            Friction value. By default 0.003.
        crosssections_shape : str, optional
            Shape of pipe crosssections. Either "circle" (default) or "rectangle".
        crosssections_value : int or list of int, optional
            Crosssections parameter value.
            If ``crosssections_shape`` = "circle", expects a diameter (default with 0.5 m) [m]
            If ``crosssections_shape`` = "rectangle", expects a list with [width, height] (e.g. [1.0, 1.0]) [m]. closed rectangle by default.
        dem_fn: str, optional
            Name of data source for dem data. Used to derive default invert levels values (DEM - pipes_depth - pipes diameter/height).
            * Required variables: [elevtn]
        pipes_depth: float, optional
            Depth of the pipes underground [m] (default 2.0 m). Used to derive defaults invert levels values (DEM - pipes_depth - pipes diameter/height).
        pipes_invlev: float, optional
            Constant default invert levels of the pipes [m asl] (default -2.5 m asl). This method is recommended to be used together with the dem method to fill remaining nan values. It slone is not a recommended method.
        snap_offset: float, optional
            Snapping tolenrance to automatically connecting branches. Tolenrance must be smaller than the shortest pipe length.
            By default 0.0, no snapping is applied.
        allow_intersection_snapping: bool, optional
            Switch to choose whether snapping of multiple branch ends are allowed when ``snap_offset`` is used.
            By default True.

        See Also
        ----------
        dflowfm._setup_branches
        dflowfm._setup_crosssections
        """
        self.logger.info(f"Preparing 1D pipes.")

        # filter for allowed columns
        _allowed_columns = [
            "geometry",
            "branchId",
            "branchType",
            "branchOrder",
            "material",
            "spacing",
            "frictionType",
            "frictionValue",
            "shape",  # circle or rectangle
            "diameter",  # circle
            "width",  # rectangle
            "height",  # rectangle
            "invlev_up",
            "inlev_dn",
        ]

        # Build the rivers branches and nodes and fill with attributes and spacing
        pipes, pipe_nodes = self._setup_branches(
            br_fn=pipes_fn,
            defaults_fn=pipes_defaults_fn,
            br_type="pipe",
            friction_type=friction_type,
            friction_value=friction_value,
            crosssections_shape=crosssections_shape,
            crosssections_value=crosssections_value,
            spacing=spacing,  # for now only single default value implemented, use "spacing" column
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
            allowed_columns=_allowed_columns,
            filter=pipe_filter,
        )

        # setup crosssections
        # setup invert levels
        # 1. check if invlev up and dn are fully filled in (nothing needs to be done)
        if "invlev_up" and "invlev_dn" in pipes.columns:
            inv = pipes[["invlev_up", "invlev_dn"]]
            if inv.isnull().sum().sum() > 0:  # nodata values in pipes for invert levels
                fill_invlev = True
                self.logger.info(
                    f"{pipes_fn} data has {inv.isnull().sum().sum()} no data values for invert levels. Will be filled using dem_fn or default value {pipes_invlev}"
                )
            else:
                fill_invlev = False
        else:
            fill_invlev = True
            self.logger.info(
                f"{pipes_fn} does not have columns [invlev_up, invlev_dn]. Invert levels will be generated from dem_fn or default value {pipes_invlev}"
            )
        # 2. filling use dem_fn + pipe_depth
        if fill_invlev and dem_fn is not None:
            dem = self.data_catalog.get_rasterdataset(
                dem_fn, geom=self.region, variables=["elevtn"]
            )
            pipes = workflows.invert_levels_from_dem(
                gdf=pipes, dem=dem, depth=pipes_depth
            )
            if pipes[["invlev_up", "invlev_dn"]].isnull().sum().sum() > 0:
                fill_invlev = True
            else:
                fill_invlev = False
        # 3. filling use pipes_invlev
        if fill_invlev and pipes_invlev is not None:
            self.logger.warning(
                "!Using a constant up and down invert levels for all pipes. May cause issues when running the delft3dfm model.!"
            )
            df_inv = pd.DataFrame(
                data={
                    "branchType": ["pipe"],
                    "invlev_up": [pipes_invlev],
                    "invlev_dn": [pipes_invlev],
                }
            )
            pipes = workflows.update_data_columns_attributes(
                pipes, df_inv, brtype="pipe"
            )

        # TODO: check that geometry lines are properly oriented from up to dn when deriving invert levels from dem

        # Update crosssections object
        crosssections = self._setup_crosssections(
            pipes,
            crosssections_type="branch",
            midpoint=False,
        )
        # add crosssections to exisiting ones and update geoms
        self.logger.debug(f"Adding crosssections vector to geoms.")
        self.add_crosssections(crosssections)

        # setup geoms
        self.logger.debug(f"Adding pipes and pipe_nodes vector to geoms.")
        self.set_geoms(pipes, "pipes")
        self.set_geoms(pipe_nodes, "pipe_nodes")  # TODO: for manholes

        # add to branches
        self.add_branches(
            pipes, branchtype="pipe", node_distance=np.inf
        )  # use node_distance to generate computational nodes at pipe ends only

    def _setup_crosssections(
        self,
        branches,
        crosssections_fn: str = None,
        crosssections_type: str = "branch",
        midpoint=True,
    ) -> gpd.GeoDataFrame:
        """Prepares 1D crosssections.
        crosssections can be set from branches, points and xyz, # TODO to be extended also from dem data for rivers/channels?
        Crosssection must only be used after friction has been setup.

        Crosssections are read from ``crosssections_fn``.
        Crosssection types of this file is read from ``crosssections_type``

        If ``crosssections_fn`` is not defined, default method is ``crosssections_type`` = 'branch',
        meaning that branch attributes will be used to derive regular crosssections.
        Crosssections are derived at branches mid points if ``midpoints`` is True,
        else at both upstream and downstream extremities of branches if False.

        Adds/Updates model layers:
        * **crosssections** geom: 1D crosssection vector

        Parameters
        ----------
        branches : gpd.GeoDataFrame
            geodataframe of the branches to apply crosssections.
            * Required variables: [branchId, branchType, branchOrder]
            If ``crosssections_type`` = "branch"
                if shape = 'circle': 'diameter'
                if shape = 'rectangle': 'width', 'height', 'closed'
                if shape = 'trapezoid': 'width', 't_width', 'height', 'closed'
            * Optional variables: [material, friction_type, friction_value]
        crosssections_fn : str Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            Note that for point crossections, only ones within the snap_network_offset will be used.
            If ``crosssections_type`` = "xyz"
            Note that only points within the region + 1000m buffer will be read.
            * Required variables: crsId, order, z
            * Optional variables:
            If ``crosssections_type`` = "point"
            * Required variables: crsId, shape, shift
            * Optional variables:
                if shape = 'rectangle': 'width', 'height', 'closed'
                if shape = 'trapezoid': 'width', 't_width', 'height', 'closed'
                if shape = 'yz': 'yzcount','ycoordinates','zcoordinates','closed'
                if shape = 'zw': 'numlevels', 'levels', 'flowwidths','totalwidths', 'closed'.
                if shape = 'zwRiver': Not Supported
                Note that list input must be strings seperated by a whitespace ''.
            By default None, crosssections will be set from branches
        crosssections_type : {'branch', 'xyz', 'point'}
            Type of crosssections read from crosssections_fn. One of ['branch', 'xyz', 'point'].
            By default `branch`.

        Returns
        -------
        gdf_cs : gpd.GeoDataFrame
            geodataframe of the new cross-sections

        Raise:
        ------
        NotImplementedError: if ``crosssection_type`` is not recongnised.
        """

        # setup crosssections
        if crosssections_fn is None and crosssections_type == "branch":
            # TODO: set a seperate type for rivers because other branch types might require upstream/downstream
            # TODO: check for required columns
            # read crosssection from branches
            self.logger.info(f"Preparing crossections from branch.")
            gdf_cs = workflows.set_branch_crosssections(branches, midpoint=midpoint)

        elif crosssections_type == "xyz":

            # Read the crosssection data
            gdf_cs = self.data_catalog.get_geodataframe(
                crosssections_fn,
                geom=self.region,
                buffer=1000,
                predicate="contains",
            )

            # check if feature valid
            if len(gdf_cs) == 0:
                self.logger.warning(
                    f"No {crosssections_fn} 1D xyz crosssections found within domain"
                )
                return None
            valid_attributes = workflows.helper.check_gpd_attributes(
                gdf_cs, required_columns=["crsId", "order", "z"]
            )
            if not valid_attributes:
                self.logger.error(
                    f"Required attributes [crsId, order, z] in xyz crosssections do not exist"
                )
                return None

            # assign id
            id_col = "crsId"
            gdf_cs.index = gdf_cs[id_col]
            gdf_cs.index.name = id_col

            # reproject to model crs
            gdf_cs.to_crs(self.crs)

            # set crsloc and crsdef attributes to crosssections
            self.logger.info(f"Preparing 1D xyz crossections from {crosssections_fn}")
            gdf_cs = workflows.set_xyz_crosssections(branches, gdf_cs)

        elif crosssections_type == "point":

            # Read the crosssection data
            gdf_cs = self.data_catalog.get_geodataframe(
                crosssections_fn,
                geom=self.region,
                buffer=100,
                predicate="contains",
            )

            # check if feature valid
            if len(gdf_cs) == 0:
                self.logger.warning(
                    f"No {crosssections_fn} 1D point crosssections found within domain"
                )
                return None
            valid_attributes = workflows.helper.check_gpd_attributes(
                gdf_cs, required_columns=["crsId", "shape", "shift"]
            )
            if not valid_attributes:
                self.logger.error(
                    f"Required attributes [crsId, shape, shift] in point crosssections do not exist"
                )
                return None

            # assign id
            id_col = "crsId"
            gdf_cs.index = gdf_cs[id_col]
            gdf_cs.index.name = id_col

            # reproject to model crs
            gdf_cs.to_crs(self.crs)

            # set crsloc and crsdef attributes to crosssections
            self.logger.info(f"Preparing 1D point crossections from {crosssections_fn}")
            gdf_cs = workflows.set_point_crosssections(
                branches, gdf_cs, maxdist=self._network_snap_offset
            )

        else:
            raise NotImplementedError(
                f"Method {crosssections_type} is not implemented."
            )
            
        return gdf_cs


    def setup_manholes(
        self,
        manholes_fn: str = None,
        manhole_defaults_fn: str = None,
        bedlevel_shift: float = -0.5,
        dem_fn: str = None,
        snap_offset: float = 1e-3,
    ):
        """
        Prepares the 1D manholes to pipes or tunnels. Can only be used after all branches are setup

        The manholes are generated based on a set of standards specified in ``manhole_defaults_fn`` (default)  and can be overwritten with manholes read from ``manholes_fn``.

        Use ``manholes_fn`` to set the manholes from a dataset of point locations.
        Only locations within the model region are selected. They are snapped to the model
        network nodes locations within a max distance defined in ``snap_offset``.

        Manhole attributes ["area", "streetStorageArea", "storageType", "streetLevel"] are either taken from ``manholes_fn`` or filled in using defaults in ``manhole_defaults_fn``.
        Manhole attribute ["bedLevel"] is always generated from invert levels of the pipe/tunnel network plus a shift defined in ``bedlevel_shift``. This is needed for numerical stability.
        Manhole attribute ["streetLevel"]  can also be overwriten with values dervied from "dem_fn".
        #TODO probably needs another parameter to apply different sampling method for the manholes, e.g. min within 2 m radius.

        Adds/Updates model layers:
            * **manholes** geom: 1D manholes vector

        Parameters
        ----------
        manholes_fn: str Path, optional
            Path or data source name for manholes see data/data_sources.yml.
            Note only the points that are within the region polygon will be used.

            * Optional variables: ["area", "streetStorageArea", "storageType", "streetLevel"]
        manholes_defaults_fn : str Path, optional
            Path to a csv file containing all defaults values per "branchType".
            Use multiple rows to apply defaults per ["shape", "diameter"/"width"] pairs.
            By default `hydrolib.hydromt_delft3dfm.data.manholes.manholes_defaults.csv` is used.

            * Allowed variables: ["area", "streetLevel", "streeStorageArea", "storageType"]
        dem_fn: str, optional
            Name of data source for dem data. Used to derive default invert levels values (DEM - pipes_depth - pipes diameter/height).
            * Required variables: [elevtn]
        bedlevel_shift: float, optional
            Shift applied to lowest pipe invert levels to derive manhole bedlevels [m] (default -0.5 m, meaning bedlevel = pipe invert - 0.5m).
        snap_offset: float, optional
            Snapping tolenrance to automatically connecting manholes to network nodes.
            By default 0.001. Use a higher value if large number of user manholes are missing.
        """
        # geom columns for manholes
        _allowed_columns = [
            "geometry",
            "id",  # storage node id, considered identical to manhole id when using single compartment manholes
            "name",
            "manholeId",
            "nodeId",
            "area",
            "bedLevel",
            "streetLevel",
            "streetStorageArea",
            "storageType",
            "useTable",
        ]

        # generate manhole locations and bedlevels
        self.logger.info(f"generating manholes locations and bedlevels. ")
        manholes, branches = workflows.generate_manholes_on_branches(
            self.branches,
            bedlevel_shift=bedlevel_shift,
            use_branch_variables=["diameter", "width"],
            id_prefix="manhole_",
            id_suffix="_generated",
            logger=self.logger,
        )
        # FIXME Xiaohan: why do we need set_branches here? Because of branches.gui --> add a high level write_gui files same level as write_mesh
        self.set_branches(branches)

        # add manhole attributes from defaults
        if manhole_defaults_fn is None or not manhole_defaults_fn.is_file():
            self.logger.warning(
                f"manhole_defaults_fn ({manhole_defaults_fn}) does not exist. Fall back choice to defaults. "
            )
            manhole_defaults_fn = Path(self._DATADIR).joinpath(
                "manholes", "manholes_defaults.csv"
            )
        defaults = pd.read_csv(manhole_defaults_fn)
        self.logger.info(f"manhole default settings read from {manhole_defaults_fn}.")
        # add defaults
        manholes = workflows.update_data_columns_attributes(manholes, defaults)

        # read user manhole
        if manholes_fn:
            self.logger.info(f"reading manholes street level from file {manholes_fn}. ")
            # read
            gdf_manhole = self.data_catalog.get_geodataframe(
                manholes_fn, geom=self.region, buffer=0, predicate="contains"
            )
            # reproject
            if gdf_manhole.crs != self.crs:
                gdf_manhole = gdf_manhole.to_crs(self.crs)
            # filter for allowed columns
            allowed_columns = set(_allowed_columns).intersection(gdf_manhole.columns)
            self.logger.debug(
                f'filtering for allowed columns:{",".join(allowed_columns)}'
            )
            gdf_manhole = gpd.GeoDataFrame(
                gdf_manhole[allowed_columns], crs=gdf_manhole.crs
            )
            # replace generated manhole using user manholes
            self.logger.debug(f"overwriting generated manholes using user manholes.")
            manholes = hydromt.gis_utils.nearest_merge(
                manholes, gdf_manhole, max_dist=snap_offset, overwrite=True
            )

        # generate manhole streetlevels from dem
        if dem_fn is not None:
            self.logger.info(f"overwriting manholes street level from dem. ")
            dem = self.data_catalog.get_rasterdataset(
                dem_fn, geom=self.region, variables=["elevtn"]
            )
            # reproject of dem is done in sample method
            manholes["_streetLevel_dem"] = dem.raster.sample(manholes).values
            manholes["_streetLevel_dem"].fillna(manholes["streetLevel"], inplace=True)
            manholes["streetLevel"] = manholes["_streetLevel_dem"]
            self.logger.debug(
                f'street level mean is {np.mean(manholes["streetLevel"])}'
            )

        # internal administration
        # drop duplicated manholeId
        self.logger.debug(f"dropping duplicated manholeId")
        manholes.drop_duplicates(subset="manholeId")
        # add nodeId to manholes
        manholes = hydromt.gis_utils.nearest_merge(
            manholes, self.network1d_nodes, max_dist=0.1, overwrite=False
        )
        # add additional required columns
        manholes["id"] = manholes[
            "nodeId"
        ]  # id of the storage nodes id, identical to manholeId when single compartment manholes are used
        manholes["name"] = manholes["manholeId"]
        manholes["useTable"] = False

        # validate
        if manholes[_allowed_columns].isna().any().any():
            self.logger.error(
                "manholes contain no data. use manholes_defaults_fn to apply no data filling."
            )

        # setup geoms
        self.logger.debug(f"Adding manholes vector to geoms.")
        self.set_geoms(manholes, "manholes")

    def setup_1dboundary(
        self,
        boundaries_geodataset_fn: str = None,
        boundaries_timeseries_fn: str = None,
        boundary_value: float = -2.5,
        branch_type: str = "river",
        boundary_type: str = "waterlevel",
        boundary_unit: str = "m",
        boundary_locs: str = "downstream",
        snap_offset: float = 1.0,
    ):
        """
        Prepares the 1D ``boundary_type`` boundaries to branches using timeseries or a constant for a
        specific ``branch_type`` at the ``boundary_locs`` locations.
        E.g. 'waterlevel' boundaries for 'downstream''river' branches.

        The values can either be a constant using ``boundary_value`` (default) or timeseries read from ``boundaries_geodataset_fn``.

        Use ``boundaries_geodataset_fn`` to set the boundary values from a dataset of point location
        timeseries. Only locations within the possible model boundary locations + snap_offset are used. They are snapped to the model
        boundary locations within a max distance defined in ``snap_offset``. If ``boundaries_geodataset_fn``
        has missing values, the constant ``boundary_value`` will be used.

        The dataset/timeseries are clipped to the model time based on the model config
        tstart and tstop entries.

        Adds/Updates model layers:
            * **boundary1d_{boundary_type}bnd_{branch_type}** forcing: 1D boundaries DataArray

        Parameters
        ----------
        boundaries_geodataset_fn : str, Path
            Path or data source name for geospatial point timeseries file.
            This can either be a netcdf file with geospatial coordinates
            or a combined point location file with a timeseries data csv file
            which can be setup through the data_catalog yml file.

            * Required variables if netcdf: ['discharge', 'waterlevel'] depending on ``boundary_type``
            * Required coordinates if netcdf: ['time', 'index', 'y', 'x']

            * Required variables if a combined point location file: ['index'] with type int
            * Required index types if a time series data csv file: int
            NOTE: Require equidistant time series
        boundaries_timeseries_fn: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            NOTE: tabulated timeseries files can only in combination with point location
            coordinates be set as a geodataset in the data_catalog yml file.
            NOTE: Require equidistant time series
        boundary_value : float, optional
            Constant value to use for all boundaries if ``boundaries_geodataset_fn`` is None and to
            fill in missing data. By default -2.5 m.
        branch_type: str
            Type of branch to apply boundaries on. One of ["river", "pipe"].
        boundary_type : str, optional
            Type of boundary tu use. One of ["waterlevel", "discharge"].
            By default "waterlevel".
        boundary_unit : str, optional.
            Unit corresponding to [boundary_type].
            If ``boundary_type`` = "waterlevel"
                Allowed unit is [m]
            if ''boundary_type`` = "discharge":
               Allowed unit is [m3/s]
            By default m.
        boundary_locs:
            Boundary locations to consider. One of ["upstream", "downstream", "both"].
            Only used for river waterlevel which can be upstream, downstream or both. By default "downstream".
            For the others, it is automatically derived from branch_type and boundary_type.
        snap_offset : float, optional
                Snapping tolerance to automatically applying boundaries at the correct network nodes.
            By default 0.1, a small snapping is applied to avoid precision errors.
        """
        self.logger.info(f"Preparing 1D {boundary_type} boundaries for {branch_type}.")
        boundaries = self.boundaries.copy()
        refdate, tstart, tstop = self.get_model_time()  # time slice

        # 1. get potential boundary locations based on branch_type and boundary_type
        boundaries_branch_type = workflows.select_boundary_type(
            boundaries, branch_type, boundary_type, boundary_locs
        )

        # 2. read boundary from user data
        if boundaries_geodataset_fn is not None:
            da_bnd = self.data_catalog.get_geodataset(
                boundaries_geodataset_fn,
                geom=boundaries.buffer(snap_offset), # only select data close to the boundary of the model region
                variables=[boundary_type],
                time_tuple=(tstart, tstop),
                crs=self.crs.to_epsg(),  # assume model crs if none defined
            ).rename(boundary_type)
            # error if time mismatch
            if np.logical_and(
                pd.to_datetime(da_bnd.time.values[0]) == pd.to_datetime(tstart),
                pd.to_datetime(da_bnd.time.values[-1]) == pd.to_datetime(tstop),
            ):
                pass
            else:
                self.logger.error(
                    f"forcing has different start and end time. Please check the forcing file. support yyyy-mm-dd HH:MM:SS. "
                )
            # reproject if needed and convert to location
            if da_bnd.vector.crs != self.crs:
                da_bnd.vector.to_crs(self.crs)
        elif boundaries_timeseries_fn is not None:
            raise NotImplementedError()
        else:
            da_bnd = None

        # 3. Derive DataArray with boundary values at boundary locations in boundaries_branch_type
        da_out = workflows.compute_boundary_values(
            boundaries=boundaries_branch_type,
            da_bnd=da_bnd,
            boundary_value=boundary_value,
            boundary_type=boundary_type,
            boundary_unit=boundary_unit,
            snap_offset=snap_offset,
            logger=self.logger,
        )

        # 4. set boundaries
        self.set_forcing(
            da_out, name=f"boundary1d_{da_out.name}_{branch_type}"
        )  # FIXME: this format cannot be read back due to lack of branch type info from model files

    def _setup_1dstructures(self,
                          st_type: str,
                          st_fn: Union[str, Path, gpd.GeoDataFrame] = None,
                          defaults_fn: Union[str, Path, pd.DataFrame] = None,
                          filter: Optional[str] = None,
                          snap_offset: Optional[float] = None,
                          ):
        """Setup 1D structures.
        Include the universal weir (Not Implemented) , culvert and bridge (``str_type`` = 'bridge'), which can only be used in a single 1D channel.

        The structures are first read from ``st_fn`` (only locations within the region will be read), and if any missing, filled with information provided in ``defaults_fn``.
        Read locations are then filtered for value specified in ``filter`` on the column ``st_type``.
        They are then snapped to the existing network within a max distance defined in ``snap_offset`` and will be dropped if not snapped.
        Finally, crossections are read and set up for the remaining structures.


        Parameters
        ----------
        st_type: str, required
            Name of structure type

        st_fn: str Path, optional
            Path or data source name for structures, see data/data_sources.yml.
            Note only the points that are within the region polygon will be used.

            * Optional variables: see the setup function for each ``st_type``
        
        defaults_fn : str Path, optional
            Path to a csv file containing all defaults values per "structure_type".
            By default `hydrolib.hydromt_delft3dfm.data.bridges.bridges_defaults.csv` is used. This file describes a minimum rectangle bridge profile.

            * Allowed variables: see the setup function for each ``st_type``
            
        filter: str, optional
            Keyword in "structure_type" column of ``st_fn`` used to filter features. If None all features are used (default).

        snap_offset: float, optional
            Snapping tolenrance to automatically snap bridges to network and add ['branchid', 'chainage'] attributes.
            By default None. In this case, global variable "network_snap_offset" will be used.
        
        Returns
        -------
        gdf_st: geopandas.GeoDataFrame
            geodataframe of the structures and crossections        
        
        
        See Also
        ----------
        dflowfm.setup_bridges
        dflowfm.setup_culverts
        """

        if snap_offset is None:
            snap_offset = self._network_snap_offset
            self.logger.info(f"Snap_offset is set to global network snap offset: {snap_offset}")
        
        type_col = "structure_type"
        id_col = "structure_id"
        
        branches = self.branches.set_index("branchId")

        # 1. Read data and filter within region
        # If needed read the structures GeoDataFrame
        if isinstance(st_fn, str) or isinstance(st_fn, Path):
            gdf_st = self.data_catalog.get_geodataframe(
                st_fn, geom=self.region, buffer=0, predicate="contains"
            )
        else:
            gdf_st = st_fn
        # Filter features based on filter on type_col
        if filter is not None and type_col in gdf_st.columns:
            gdf_st = gdf_st[gdf_st[type_col].str.lower() == filter.lower()]
            self.logger.info(f"Set {filter} locations filtered from {st_fn} as {st_type} .")
        # Check if features in region
        if len(gdf_st) == 0:
            self.logger.warning(f"No 1D {type} locations found within domain")
            return None
        # reproject to model crs
        gdf_st.to_crs(self.crs)

        # 2.  Read defaults table
        if isinstance(defaults_fn, pd.DataFrame):
            defaults = defaults_fn
        else:
            if defaults_fn is None:
                self.logger.warning(
                    f"defaults_fn ({defaults_fn}) does not exist. Fall back choice to defaults. "
                )
                defaults_fn = Path(self._DATADIR).joinpath(
                    f"{st_type}s", f"{st_type}s_defaults.csv"
                )
            defaults = self.data_catalog.get_dataframe(defaults_fn)
            self.logger.info(f"{st_type} default settings read from {defaults_fn}.")

        # 3. Add defaults
        # overwrite type and add id attributes if does not exist
        gdf_st[type_col] = pd.Series(
                data=np.repeat(st_type, len(gdf_st)), index=gdf_st.index, dtype=str
            )
        if id_col not in gdf_st.columns:
            data = [
                f"{st_type}_{i}"
                for i in np.arange(
                    len(self.structures) + 1, len(self.structures) + len(gdf_st) + 1
                )
            ]  # avoid duplicated ids being generated
            gdf_st[id_col] = pd.Series(data, index=gdf_st.index, dtype=str)
        # assign id
        gdf_st.index = gdf_st[id_col]
        gdf_st.index.name = id_col
        # filter for allowed columns
        allowed_columns = set(defaults.columns).intersection(gdf_st.columns)
        allowed_columns.update({"geometry"})
        gdf_st = gpd.GeoDataFrame(gdf_st[allowed_columns], crs=gdf_st.crs)
        self.logger.info("Adding/Filling default attributes values")
        gdf_st = workflows.update_data_columns_attributes_based_on_filter(
            gdf_st, defaults, type_col, st_type
        )

        # 4. snap structures to branches
        # setup branch_id - snap structures to branch (inplace of structures, will add branch_id and branch_offset columns)
        workflows.find_nearest_branch(
            branches=branches, geometries=gdf_st,  maxdist = snap_offset
        )
        # setup failed - drop based on branch_offset that are not snapped to branch
        _old_ids = gdf_st.index.to_list()
        gdf_st.dropna(axis=0, inplace=True, subset=["branch_offset"])
        _new_ids = gdf_st.index.to_list()
        if len(_old_ids) != len(_new_ids):
            logger.warning(
                f"structure with id: {list(set(_old_ids) - set(_new_ids))} are dropped: unable to find closest branch. "
            )
        if len(_new_ids) == 0:
            self.logger.warning(f"No 1D {type} locations found within the proximity of the network")
            return None
        else:
            # setup success, add branchid and chainage
            gdf_st["structure_branchid"] = gdf_st["branch_id"]
            gdf_st["structure_chainage"] = gdf_st["branch_offset"]

        # 5. add structure crossections
        # add a dummy "shift" for crossection locations if missing (e.g. culverts), because structures only needs crossection definitions.
        if "shift" not in gdf_st.columns:
            gdf_st["shift"] = np.nan
        # derive crosssections 
        gdf_st_crossections = workflows.set_point_crosssections(
                branches, gdf_st, maxdist=snap_offset
            )
        # remove crossection locations and any friction from the setup
        gdf_st_crsdefs = gdf_st_crossections.drop(columns = [c for c in gdf_st_crossections.columns if c.startswith("crsloc") or "friction" in c])
        # add to structures
        gdf_st = gdf_st.merge(gdf_st_crsdefs.drop(columns = "geometry"), left_index = True, right_index=True)
       
        # 6. replace np.nan as None
        gdf_st = gdf_st.replace(np.nan, None)
        
        # 7. remove index
        gdf_st = gdf_st.reset_index()

        return gdf_st


    def setup_bridges(self,
                      bridges_fn: Optional[str] = None,
                      bridges_defaults_fn: Optional[str] = None,
                      bridge_filter: Optional[str] = None,
                      snap_offset: Optional[float] = None,
                      ):
        """Prepares bridges, including bridge locations and bridge crossections.

        The bridges are read from ``bridges_fn`` and if any missing, filled with information provided in ``bridges_defaults_fn``.

        When reading ``bridges_fn``, only locations within the region will be read. 
        Read locations are then filtered for value specified in ``bridge_filter`` on the column "structure_type" .
        Remaining locations are snapped to the existing network within a max distance defined in ``snap_offset`` and will be dropped if not snapped.

        A default rectangle bridge profile can be found in ``bridges_defaults_fn`` as an example.

        Structure attributes ['structure_id', 'structure_type'] are either taken from data or generated in the script.
        Structure attributes ['shape', 'diameter', 'width', 't_width', 'height', 'closed', 'shift', 'length', 'pillarwidth', 'formfactor', 'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
            are either taken from data,  or in case of missing read from defaults.

        Adds/Updates model layers:
            * **bridges** geom: 1D bridges vector

        Parameters
        ----------
        bridges_fn: str Path, optional
            Path or data source name for bridges, see data/data_sources.yml.
            Note only the points that are within the region polygon will be used.

            * Optional variables: ['structure_id', 'structure_type', 'shape', 'diameter', 'width', 't_width', 'height', 'closed', 'shift', 'length', 'pillarwidth', 'formfactor', 'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
        
        bridges_defaults_fn : str Path, optional
            Path to a csv file containing all defaults values per "structure_type".
            By default `hydrolib.hydromt_delft3dfm.data.bridges.bridges_defaults.csv` is used. This file describes a minimum rectangle bridge profile.

            * Allowed variables: ['structure_type', 'shape', 'diameter', 'width', 't_width', 'height', 'closed', 'shift', 'length', 'pillarwidth', 'formfactor', 'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
        
        river_filter: str, optional
            Keyword in "structure_type" column of ``bridges_fn`` used to filter bridge features. If None all features are used (default).

        snap_offset: float, optional
            Snapping tolenrance to automatically snap bridges to network and add ['branchid', 'chainage'] attributes.
            By default None. In this case, global variable "network_snap_offset" will be used..
        
        See Also
        ----------
        dflowfm._setup_1dstructures
        """
        
        # keywords in hydrolib-core
        _st_type = "bridge"
        _allowed_columns =['id', 'type', 'branchid', 'chainage', 'allowedflowdir', 'pillarwidth', 'formfactor', 'csdefid', 'shift', 'inletlosscoeff', 'outletlosscoeff', 'frictiontype', 'friction', 'length']
        
        # setup general 1d structures
        bridges = self._setup_1dstructures(_st_type, bridges_fn, bridges_defaults_fn, bridge_filter, snap_offset)
        
        # setup crossection definitions
        self.add_crosssections(bridges[[c for c in bridges.columns if c.startswith("crsdef")]])
        
        # setup bridges
        bridges.columns = bridges.columns.str.lower()
        bridges.rename(columns = {"structure_id": "id",
                                  "structure_type": "type",
                                  "structure_branchid": "branchid",
                                  "structure_chainage": "chainage",
                                  "crsdef_id": "csdefid",
                                  "friction_type": "frictiontype",
                                  "friction_value": "friction"}, inplace = True)
        # filter for allowed columns
        allowed_columns = set(_allowed_columns).intersection(bridges.columns)
        allowed_columns.update({"geometry"})
        bridges = gpd.GeoDataFrame(bridges[allowed_columns], crs=bridges.crs)
        self.set_geoms(bridges, "bridges")


    def setup_culverts(self,
                      culverts_fn: Optional[str] = None,
                      culverts_defaults_fn: Optional[str] = None,
                      culvert_filter: Optional[str] = None,
                      snap_offset: Optional[float] = None,
                      ):
        
        """Prepares culverts, including locations and crossections. Note that only subtype culvert is supported, i.e. inverted siphon is not supported

        The culverts are read from ``culverts_fn`` and if any missing, filled with information provided in ``culverts_defaults_fn``.

        When reading ``culverts_fn``, only locations within the region will be read. 
        Read locations are then filtered for value specified in ``culvert_filter`` on the column "structure_type" .
        Remaining locations are snapped to the existing network within a max distance defined in ``snap_offset`` and will be dropped if not snapped.

        A default ``culverts_defaults_fn`` that defines a circle culvert profile can be found in dflowfm.data.culverts as an example.

        Structure attributes ['structure_id', 'structure_type'] are either taken from data or generated in the script.
        Structure attributes ['shape', 'diameter', 'width', 't_width', 'height', 'closed', 'leftlevel', 'rightlevel', 'length',
            'valveonoff', 'valveopeningheight', 'numlosscoeff', 'relopening', 'losscoeff', 
            'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
            are either taken from data,  or in case of missing read from defaults.

        Adds/Updates model layers:
            * **culverts** geom: 1D culverts vector

        Parameters
        ----------
        culverts_fn: str Path, optional
            Path or data source name for culverts, see data/data_sources.yml.
            Note only the points that are within the region polygon will be used.

            * Optional variables: ['structure_id', 'structure_type', 'shape', 'diameter', 'width', 't_width', 'height', 'closed', 
                                    'leftlevel', 'rightlevel', 'length', 'valveonoff', 'valveopeningheight', 
                                    'numlosscoeff', 'relopening', 'losscoeff', 
                                    'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
        
        culverts_defaults_fn : str Path, optional
            Path to a csv file containing all defaults values per "structure_type".
            By default `hydrolib.hydromt_delft3dfm.data.culverts.culverts_defaults.csv` is used. 
            This file describes a default circle culvert profile.

            * Allowed variables: ['structure_type', 'shape', 'diameter', 'width', 't_width', 'height', 'closed', 
                                    'leftlevel', 'rightlevel', 'length', 'valveonoff', 'valveopeningheight', 
                                    'numlosscoeff', 'relopening', 'losscoeff', 
                                    'friction_type', 'friction_value', 'allowedflowdir',  'inletlosscoeff', 'outletlosscoeff']
        
        culvert_filter: str, optional
            Keyword in "structure_type" column of ``culverts_fn`` used to filter culvert features. If None all features are used (default).

        snap_offset: float, optional
            Snapping tolenrance to automatically snap culverts to network and add ['branchid', 'chainage'] attributes.
            By default None. In this case, global variable "network_snap_offset" will be used..
        
        See Also
        ----------
        dflowfm._setup_1dstructures
        """
        
        # keywords in hydrolib-core
        _st_type = "culvert"
        _allowed_columns =['id', 'type', 'branchid', 'chainage', 'allowedflowdir', 'leftlevel', 'rightlevel', 'csdefid', 'length', 'inletlosscoeff', 'outletlosscoeff', 
                           'valveonoff', 'valveopeningheight', 'numlosscoeff', 'relopening', 'losscoeff', 'bedfrictiontype', 'bedfriction']
        
        # setup general 1d structures
        culverts = self._setup_1dstructures(_st_type, culverts_fn, culverts_defaults_fn, culvert_filter, snap_offset)
        
        # setup crossection definitions
        self.add_crosssections(culverts[[c for c in culverts.columns if c.startswith("crsdef")]])
        
        # setup culverts
        culverts.columns = culverts.columns.str.lower()
        culverts.rename(columns = {"structure_id": "id",
                                  "structure_type": "type",
                                  "structure_branchid": "branchid",
                                  "structure_chainage": "chainage",
                                  "crsdef_id": "csdefid",
                                  "friction_type": "bedfrictiontype", 
                                  "friction_value": "bedfriction"}, inplace = True)
        # filter for allowed columns
        allowed_columns = set(_allowed_columns).intersection(culverts.columns)
        allowed_columns.update({"geometry"})
        culverts = gpd.GeoDataFrame(culverts[allowed_columns], crs=culverts.crs)
        self.set_geoms(culverts, "culverts")


    def setup_mesh2d(
        self,
        mesh2d_fn: Optional[str] = None,
        geom_fn: Optional[str] = None,
        bbox: Optional[list] = None,
        res: float = 100.0,
    ):
        """Creates an 2D unstructured mesh or prepares an existing 2D mesh according UGRID conventions.

        An 2D unstructured mesh is created as 2D rectangular grid from a geometry (geom_fn) or bbox. If an existing
        2D mesh is given, then no new mesh is generated

        2D mesh contains mesh2d nfaces, x-coordinate of mesh2d nodes, y-coordinate of mesh2d nodes,
        mesh2d face nodes, x-coordinates of face, y-coordinate of face and global atrributes (conventions and
        coordinates).

        Note that:
        (1) Refinement of the mesh is a seperate setup function, however an existing grid with refinement (mesh_fn)
        can already be read.
        (2) If no geometry, bbox or existing grid is specified for this setup function, then the self.region is used as
         mesh extent to generate the unstructured mesh.
        (3) At mesh border, cells that intersect with geometry border will be kept.
        (4) Validation checks have been added to check if the mesh extent is within model region.
        (5) Only existing meshed with only 2D grid can be read.
        (6) 1D2D network files are not supported as mesh2d_fn.

        Adds/Updates model layers:

        * **1D2D links** geom: By any changes in 2D grid

        Parameters
        ----------
        mesh2D_fn : str Path, optional
            Name of data source for an existing unstructured 2D mesh
        geom_fn : str Path, optional
            Path to a polygon used to generate unstructured 2D mesh
        bbox: list, optional
            Describing the mesh extent of interest [xmin, ymin, xmax, ymax]. Please specify it in the model coordinate
            system.
        res: float, optional
            Resolution used to generate 2D mesh. By default a value of 100 m is applied.

        Raises
        ------
        IndexError
            If the grid of the spatial domain contains 0 x-coordinates or 0 y-coordinates.

        See Also
        ----------

        """

        # Function moved to MeshModel in hydromt core
        # Recreate region dict for core function
        if mesh2d_fn is not None:
            region = {"mesh": mesh2d_fn}
        elif geom_fn is not None:
            region = {"geom": str(geom_fn)}
        elif bbox is not None:
            bbox = [float(v) for v in bbox]  # needs to be str in config file
            region = {"bbox": bbox}
        else:  # use model region
            self.logger.warning("Mesh2d is generated for model region.")
            region = {"geom": self.region}

        # reserve existing mesh
        # _mesh = self._mesh

        # Get and set the 2dmesh
        mesh2d = super().setup_mesh(region=region, crs=self.crs, res=res)
        # TODO: if we do not want to clip mesh 2d comment the following line
        # Check if intersects with region
        xmin, ymin, xmax, ymax = self.bounds
        subset = mesh2d.ugrid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        err = f"RasterDataset: No data within model region."
        subset = subset.ugrid.assign_node_coords()
        if subset.ugrid.grid.node_x.size == 0 or subset.ugrid.grid.node_y.size == 0:
            raise IndexError(err)
        self._mesh = subset  # reinitialise mesh2d grid (set_mesh is used in super)

        # add to hydrolib-core net object
        self.set_mesh2d()

        # update the 2dmesh to self._mesh
        # if _mesh is not None:
        #    _mesh = _mesh.drop_vars(set(_mesh.variables.keys()).intersection(subset.variables.keys()))
        #    self._mesh = _mesh.merge(subset)
        # else already set in setup_mesh

        # update res
        self._res = res

    def setup_link1d2d(
            self,
            link_direction: Optional[str] = "1d_to_2d",
            link_type: Optional[str] = "embedded",
            polygon_fn: Optional[str] = None,
            branch_type: Optional[str] = None,
            max_length: Union[float, None] = np.inf,
            dist_factor: Union[float, None] = 2.0,
            **kwargs,
    ):
        """Generate 1d2d links that link mesh1d and mesh2d according UGRID conventions.

        1d2d links are added to allow water exchange between 1d and 2d for an 1d2d model.
        They can only be added if both mesh1d and mesh2d are present. By default, 1d_to_2d links are generated for the entire mesh1d except boundary locations.
        When ''polygon_fn'' is specified, only links within the polygon will be added.
        When ''branch_type'' is specified, only 1d branches matching the specified type will be used for generating 1d2d link.
        # TODO: This option should also allows more customised setup for pipes and tunnels: 1d2d links will also be generated at boundary locations.

        Parameters
        ----------
        link_direction : str, optional
            Direction of the links: ["1d_to_2d", "2d_to_1d"].
            Default to 1d_to_2d.
        link_type : str, optional
            Type of the links to be generated: ["embedded", "lateral"]. Only used when ''link_direction'' = '2d_to_1d'.
            Default to None.
        polygon_fn: str Path, optional
             Source name of polygon data in data_catalog.
             Area within 1D2D links are generated.
             Default to None.
        branch_type: str, Optional
             Type of branch to be used for 1d: ["river","pipe","channel", "tunnel"].
             When ''branch_type'' = "pipe" or "tunnel" are specified, 1d2d links will also be generated at end nodes.
             This is not yet implemented.
             Default to None. Add 1d2d links for the all branches at non-end node locations.
        max_length : Union[float, None], optional
             Max allowed edge length for generated links.
             Only used when ''link_direction'' = '2d_to_1d'  and ''link_type'' = 'lateral'.
             Defaults to infinity.
        dist_factor : Union[float, None], optional:
             Factor to determine which links are kept.
             Only used when ''link_direction'' = '2d_to_1d'  and ''link_type'' = 'lateral'.
             Defaults to 2.0. Links with an intersection distance larger than 2 (dist_factor) times the center to edge distance of the cell, are removed.

         See Also
         ----------
         mesh.links1d2d_add_links_1d_to_2d
         mesh.links1d2d_add_links_2d_to_1d_embedded
         mesh.links1d2d_add_links_2d_to_1d_lateral
        """

        # check existing network
        if self.mesh1d.is_empty() or self.mesh2d.is_empty():
            self.logger.error(
                "cannot setup link1d2d: either mesh1d or mesh2d or both do not exist"
            )

        if not self.link1d2d.is_empty():
            self.logger.warning("adding to existing link1d2d: link1d2d already exists")
            # FIXME: question - how to seperate if the user wants to update the entire 1d2d links object or simply wants to add another set of links?
            # TODO: would be nice in hydrolib to allow clear of subset of 1d2d links for specific branches

        # check input
        if polygon_fn is not None:
            within = self.data_catalog.get_geodataframe(polygon_fn).geometry
            self.logger.info(f"adding 1d2d links only within polygon {polygon_fn}")
        else:
            within = None

        if branch_type is not None:
            branchids = self.branches[
                self.branches.branchType == branch_type
                ].branchId.to_list()  # use selective branches
            self.logger.info(f"adding 1d2d links for {branch_type} branches.")
        else:
            branchids = None  # use all branches
            self.logger.warning(f"adding 1d2d links for all branches at non boundary locations.")

        # setup 1d2d links
        if link_direction == "1d_to_2d":
            self.logger.info("setting up 1d_to_2d links.")
            # recompute max_length based on the diagonal distance of the max mesh area
            max_length = np.sqrt(self._mesh.ugrid.to_geodataframe().area.max()) * np.sqrt(2)
            mesh.links1d2d_add_links_1d_to_2d(
                self.network, branchids=branchids, within=within, max_length=max_length)

        elif link_direction == "2d_to_1d":
            if link_type == "embedded":
                self.logger.info("setting up 2d_to_1d embedded links.")

                mesh.links1d2d_add_links_2d_to_1d_embedded(
                    self.network, branchids=branchids, within=within
                )
            elif link_type == "lateral":

                self.logger.info("setting up 2d_to_1d lateral links.")
                mesh.links1d2d_add_links_2d_to_1d_lateral(
                    self.network,
                    branchids=branchids,
                    within=within,
                    max_length=max_length,
                    dist_factor=dist_factor,
                )
            else:
                self.logger.error(f"link_type {link_type} is not recognised.")

        else:
            self.logger.error(f"link_direction {link_direction} is not recognised.")

    # TODO: Create link1d2d mesh in xu Ugrid

    # FIXME: use the below function as a workflow in setup_mesh2d would be better - to be discussed
    def setup_mesh2d_refine(self,
                            polygon_fn: Optional[str] = None,
                            sample_fn: Optional[str] = None,
                            steps: Optional[int] = None,
                            ):
        """Refine the 2d mesh within the geometry based on polygon `polygon_fn` or raster samples `sample_fn`.
        The number of refinement is defined by `steps` if `polygon_fn` is used.

        Note that this function can only be applied on an existing regular rectangle 2d mesh.
        # FIXME: how to identify if the mesh is uniform rectangle 2d mesh? regular irregular?

        Adds/Updates model layers:

        * **2D mesh** geom: By any changes in 2D grid
        * **1D2D links** geom: By any changes in 2D grid #TODO: check if this is needed, possible to issue an error if 1d2d links are present

        Parameters
        ----------
        polygon_fn : str Path, optional
            Path to a polygon or MultiPolygon used to refine the 2D mesh
        sample_fn  : str Path, optional
            Path to a raster sample file used to refine the 2D mesh.
            The value of each sample point is the number of steps to refine the mesh. Allow only single values.
            The resolution of the raster should be the same as the desired end result resolution.

            * Required variable: ['steps']
        steps : int, optional
            Number of steps in the refinement when `polygon_fn' is used.
            By default 1, i.e. no refinement is applied.

        """

        if self.mesh2d is None:
            logger.error("2d mesh is not available.")
            return

        if polygon_fn is not None and steps is not None:
            self.logger.info(f"reading geometry from file {polygon_fn}. ")
            # read
            gdf = self.data_catalog.get_geodataframe(
                polygon_fn, geom=self.region, buffer=0, predicate="contains"
            )
            # reproject
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            # refine
            _old_size = len(self.mesh2d.mesh2d_face_x)
            
            # Create a list of coordinate lists
            polygon = gdf.geometry.unary_union
            cls = GeometryList()
            # Add exterior
            x_ext, y_ext = np.array(polygon.exterior.coords[:]).T
            x_crds = [x_ext]
            y_crds = [y_ext]
            # Add interiors, seperated by inner_outer_separator
            for interior in polygon.interiors:
                x_int, y_int = np.array(interior.coords[:]).T
                x_crds.append([cls.inner_outer_separator])
                x_crds.append(x_int)
                y_crds.append([cls.inner_outer_separator])
                y_crds.append(y_int)
            gl = GeometryList(
                x_coordinates=np.concatenate(x_crds), y_coordinates=np.concatenate(y_crds)
            )
            self.network.mesh2d_refine(gl, steps)
            _new_size = len(self.mesh2d.mesh2d_face_x)
            # log
            if _new_size != _old_size:
                logger.info(f"2d mesh refined within polygon. Number of faces before: {_old_size} and after: {_new_size}")
            else:
                logger.warning("2d mesh unrefined.")
            # update res # FIXME: might not be the best solution
            self._res = self._res/2**steps

        elif sample_fn is not None:
            self.logger.info(f"reading samples from file {sample_fn}. ")
            # read
            da = self.data_catalog.get_rasterdataset(
                sample_fn, geom=self.region, buffer=0, predicate="contains", variables=["steps"], single_var_as_array=True,
            ).astype(np.float64) # float64 is needed by mesh kernel to convert into c double
            # reproject
            if da.raster.crs != self.crs:
                self.logger.warning(f"Sample grid has a different resolution than model. Reprojecting with nearest but some information might be lost.")
                da = da.raster.reproject(self.crs, method="nearest")
            # get sample point
            xv, yv = np.meshgrid(da.x.values, da.y.values)
            zv = da.values
            mask = zv > 0
            samples = GeometryList(xv[mask].flatten(), yv[mask].flatten(), zv[mask].flatten())
            # get steps
            steps = int(zv.max())
            # refine using mesh kernel
            _old_size = len(self.mesh2d.mesh2d_face_x)
            # refine parameters
            parameters = mk.MeshRefinementParameters(
                refinement_type= 2,                 # Enumerator describing the different refinement types (WaveCourant = 1, RefinementLevels = 2)
                max_refinement_iterations= steps, # Maximum number of refinement iterations, set to 1 if only one refinement is wanted (default 10)
                min_face_size= self.res / 2 ** steps,  # calculate - Minimum cell size
                account_for_samples_outside_face=False,  # default, not useful - Take samples outside face into account , 1 yes 0 no
                connect_hanging_nodes=True,  # default, not useful - Connect hanging nodes at the end of the iteration, 1 yes or 0 no
                refine_intersected=False,  # default, not useful - Whether to compute faces intersected by polygon (yes=1/no=0)
                use_mass_center_when_refining=False,  # default, not useful - Whether to use the mass center when splitting a face in the refinement process (yes=1/no=0)
            )
            # refine assuming sample spacing is the same as end result resolution (hence relative_search_radius=1, minimum_num_samples=1 )
            self.mesh2d.meshkernel.mesh2d_refine_based_on_samples(samples,
                                                       relative_search_radius=1,
                                                       minimum_num_samples=1,
                                                       mesh_refinement_params=parameters)
            self.set_mesh2d(self.mesh2d.meshkernel.mesh2d_get()) # because the resulting mesh2d is only stored in the meshkernel object
            _new_size = len(self.mesh2d.mesh2d_face_x)
            # log
            if _new_size != _old_size:
                logger.info(f"2d mesh refined based on samples. Number of faces before: {_old_size} and after: {_new_size}")
            else:
                logger.warning("2d mesh unrefined.")
            # update res # FIXME: might not be the best solution
            self._res = self._res/2**steps
        else:
            logger.warning("2d mesh unrefined.")

    def setup_link1d2d(
            self,
            link_direction: Optional[str] = "1d_to_2d",
            link_type: Optional[str] = "embedded",
            polygon_fn: Optional[str] = None,
            branch_type: Optional[str] = None,
            max_length: Union[float, None] = np.inf,
            dist_factor: Union[float, None] = 2.0,
            **kwargs,
    ):
        """Generate 1d2d links that link mesh1d and mesh2d according UGRID conventions.

        1d2d links are added to allow water exchange between 1d and 2d for a 1d2d model.
        They can only be added if both mesh1d and mesh2d are present. By default, 1d_to_2d links are generated for the entire mesh1d except boundary locations.
        When ''polygon_fn'' is specified, only links within the polygon will be added.
        When ''branch_type'' is specified, only 1d branches matching the specified type will be used for generating 1d2d link.
        # TODO: This option should also allows more customised setup for pipes and tunnels: 1d2d links will also be generated at boundary locations.

        Parameters
        ----------
        link_direction : str, optional
            Direction of the links: ["1d_to_2d", "2d_to_1d"].
            Default to 1d_to_2d.
        link_type : str, optional
            Type of the links to be generated: ["embedded", "lateral"]. only used when ''link_direction'' = '2d_to_1d'.
            Default to None.
        polygon_fn: str Path, optional
             Source name of raster data in data_catalog.
             Default to None.
        branch_type: str, Optional
             Type of branch to be used for 1d: ["river","pipe","channel", "tunnel"].
             When ''branch_type'' = "pipe" or "tunnel" are specified, 1d2d links will also be generated at boundary locations.
             Default to None. Add 1d2d links for the all branches at non-boundary locations.
        max_length : Union[float, None], optional
             Max allowed edge length for generated links.
             Only used when ''link_direction'' = '2d_to_1d'  and ''link_type'' = 'lateral'.
             Defaults to infinity.
        dist_factor : Union[float, None], optional:
             Factor to determine which links are kept.
             Only used when ''link_direction'' = '2d_to_1d'  and ''link_type'' = 'lateral'.
             Defaults to 2.0. Links with an intersection distance larger than 2 times the center to edge distance of the cell, are removed.

         See Also
         ----------
         mesh.links1d2d_add_links_1d_to_2d
         mesh.links1d2d_add_links_2d_to_1d_embedded
         mesh.links1d2d_add_links_2d_to_1d_lateral
        """

        # check existing network
        if self.mesh1d.is_empty() or self.mesh2d.is_empty():
            self.logger.error(
                "cannot setup link1d2d: either mesh1d or mesh2d or both do not exist"
            )

        if not self.link1d2d.is_empty():
            self.logger.warning("adding to existing link1d2d: link1d2d already exists")
            # FIXME: question - how to seperate if the user wants to update the entire 1d2d links object or simply wants to add another set of links?
            # TODO: would be nice in hydrolib to allow clear of subset of 1d2d links for specific branches

        # check input
        if polygon_fn is not None:
            within = self.data_catalog.get_geodataframe(polygon_fn).gemetry
            self.logger.info(f"adding 1d2d links only within polygon {polygon_fn}")
        else:
            within = None

        if branch_type is not None:
            branchids = self.branches[
                self.branches.branchType == branch_type
                ].branchId.to_list()  # use selective branches
            self.logger.info(f"adding 1d2d links for {branch_type} branches.")
        else:
            branchids = None  # use all branches
            self.logger.warning(f"adding 1d2d links for all branches at non boundary locations.")

        # setup 1d2d links
        if link_direction == "1d_to_2d":
            self.logger.info("setting up 1d_to_2d links.")
            # recompute max_length based on the diagnal distance of the max mesh area
            max_length = np.sqrt(self._mesh.ugrid.to_geodataframe().area.max()) * np.sqrt(2)
            mesh.links1d2d_add_links_1d_to_2d(
                self.network, branchids=branchids, within=within, max_length=max_length)

        elif link_direction == "2d_to_1d":
            if link_type == "embedded":
                self.logger.info("setting up 2d_to_1d embedded links.")

                mesh.links1d2d_add_links_2d_to_1d_embedded(
                    self.network, branchids=branchids, within=within
                )
            elif link_type == "lateral":

                self.logger.info("setting up 2d_to_1d lateral links.")
                mesh.links1d2d_add_links_2d_to_1d_lateral(
                    self.network,
                    branchids=branchids,
                    within=within,
                    max_length=max_length,
                    dist_factor=dist_factor,
                )
            else:
                self.logger.error(f"link_type {link_type} is not recognised.")

        else:
            self.logger.error(f"link_direction {link_direction} is not recognised.")

    # TODO: Create link1d2d mesh in xu Ugrid

    
    
    def setup_maps_from_raster(
        self,
        raster_fn: str,
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[str] = "nearest",
        interpolation_method: Optional[str] = "triangulation",
        locationtype: Optional[str] = "2d",
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ) -> None:
        """
        This component adds data variable(s) from ``raster_fn`` to maps object.

        If raster is a dataset, all variables will be added unless ``variables`` list is specified.

        Adds model layers:

        * **raster.name** maps: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog.
        variables: list, optional
            List of variables to add to maps from raster_fn. By default all.
            Available variables: ['elevtn', 'waterlevel', 'waterdepth', 'pet', 'infiltcap', 'roughness_chezy', 'roughness_manning', 'roughness_walllawnikuradse', 'roughness_whitecolebrook']
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method. Available methods
            are ['linear', 'nearest', 'cubic', 'rio_idw'].
        reproject_method : str, optional
            CRS reprojection method from rasterio.enums.Resampling. By default nearest.
            Available methods: [ 'nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode',
            'gauss', 'max', 'min', 'med', 'q1', 'q3', 'sum', 'rms']
        interpolation_method : str, optional
            Interpolation method for DFlow-FM. By default triangulation. Except for waterlevel and
            waterdepth then the default is mean.
            When methods other than 'triangulation', the relative search cell size will be estimated based on resolution of the raster.
            Available methods: ['triangulation', 'mean', 'nearestNb', 'max', 'min', 'invDist', 'minAbs', 'median']
        locationtype : str, optional
            LocationType in initial fields. Either 2d (default), 1d or all.
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset, either add it as is to maps or split it into several xarray.DataArrays.
            Default to True.
        """
        # check for name when split_dataset is False
        if split_dataset is False and name is None:
            self.logger.error("name must be specified when split_dataset = False")

        # Call super method
        variables = super().setup_maps_from_raster(
            raster_fn=raster_fn,
            variables=variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            name=name,
            split_dataset=split_dataset,
        )

        allowed_methods = [
            "triangulation",
            "mean",
            "nearestNb",
            "max",
            "min",
            "invDist",
            "minAbs",
            "median",
        ]
        if not np.isin(interpolation_method, allowed_methods):
            raise ValueError(
                f"Interpolation method {interpolation_method} not allowed. Select from {allowed_methods}"
            )
        if not np.isin(locationtype, ["2d", "1d", "all"]):
            raise ValueError(
                f"Locationtype {locationtype} not allowed. Select from ['2d', '1d', 'all']"
            )
        for var in variables:
            if var in self._MAPS:
                self._MAPS[var]["locationtype"] = locationtype
                self._MAPS[var]["interpolation"] = interpolation_method
                if interpolation_method != "triangulation":
                    # adjust relative search cell size for averaging methods
                    relsize = np.round(
                        np.abs(self.maps[var].raster.res[0]) / self.res * np.sqrt(2)
                        + 0.05,
                        2,
                    )
                    self._MAPS[var]["averagingrelsize"] = relsize

    def setup_maps_from_raster_reclass(
        self,
        raster_fn: str,
        reclass_table_fn: str,
        reclass_variables: list,
        fill_method: Optional[str] = None,
        reproject_method: Optional[str] = "nearest",
        interpolation_method: Optional[str] = "triangulation",
        locationtype: Optional[str] = "2d",
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """
        This component adds data variable(s) to maps object by combining values in ``raster_mapping_fn`` to
        spatial layer ``raster_fn``. The ``mapping_variables`` rasters are first created by mapping variables values
        from ``raster_mapping_fn`` to value in the ``raster_fn`` grid.

        Adds model layers:
        * **mapping_variables** maps: data from raster_mapping_fn spatially ditributed with raster_fn
        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog. Should be a DataArray. Else use **kwargs to select
            variables/time_tuple in hydromt.data_catalog.get_rasterdataset method
        reclass_table_fn: str
            Source name of mapping table of raster_fn in data_catalog. Make sure the data type is consistant for a ``reclass_variables`` including nodata.
            For example, for roughness, it is common that the data type is float, then use no data value as -999.0.
        reclass_variables: list
            List of mapping_variables from raster_mapping_fn table to add to mesh. Index column should match values
            in raster_fn.
            Available variables: ['elevtn', 'waterlevel', 'waterdepth', 'pet', 'infiltcap', 'roughness_chezy', 'roughness_manning', 'roughness_walllawnikuradse', 'roughness_whitecolebrook']
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method. Available methods
            are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        reproject_method : str, optional
            CRS reprojection method from rasterio.enums.Resampling. By default nearest.
            Available methods: [ 'nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode',
            'gauss', 'max', 'min', 'med', 'q1', 'q3', 'sum', 'rms']
        interpolation_method : str, optional
            Interpolation method for DFlow-FM. By default triangulation. Except for waterlevel and waterdepth then
            the default is mean.
            When methods other than 'triangulation', the relative search cell size will be estimated based on resolution of the raster.
            Available methods: ['triangulation', 'mean', 'nearestNb', 'max', 'min', 'invDist', 'minAbs', 'median']
        locationtype : str, optional
            LocationType in initial fields. Either 2d (default), 1d or all.
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset, either add it as is to maps or split it into several xarray.DataArrays.
            Default to True.
        """
        # check for name when split_dataset is False
        if split_dataset is False and name is None:
            self.logger.error("name must be specified when split_dataset = False")

        # Call super method
        reclass_variables = super().setup_maps_from_raster_reclass(
            raster_fn=raster_fn,
            reclass_table_fn=reclass_table_fn,
            reclass_variables=reclass_variables,
            fill_method=fill_method,
            reproject_method=reproject_method,
            name=name,
            split_dataset=split_dataset,
            **kwargs,
        )

        allowed_methods = [
            "triangulation",
            "mean",
            "nearestNb",
            "max",
            "min",
            "invDist",
            "minAbs",
            "median",
        ]
        if not np.isin(interpolation_method, allowed_methods):
            raise ValueError(
                f"Interpolation method {interpolation_method} not allowed. Select from {allowed_methods}"
            )
        if not np.isin(locationtype, ["2d", "1d", "all"]):
            raise ValueError(
                f"Locationtype {locationtype} not allowed. Select from ['2d', '1d', 'all']"
            )
        for var in reclass_variables:
            if var in self._MAPS:
                self._MAPS[var]["locationtype"] = locationtype
                self._MAPS[var]["interpolation"] = interpolation_method
                if interpolation_method != "triangulation":
                    # increase relative search cell size if raster resolution is coarser than model resolution
                    if self.maps[var].raster.res[0] > self.res:
                        relsize = np.round(
                            np.abs(self.maps[var].raster.res[0]) / self.res * np.sqrt(2)
                            + 0.05,
                            2,
                        )
                    else:
                        relsize = 1.01
                    self._MAPS[var]["averagingrelsize"] = relsize

    def setup_2dboundary(
        self,
        boundaries_fn: str = None,
        boundaries_timeseries_fn: str = None,
        boundary_value: float = 0.0,
        boundary_type: str = "waterlevel",
        tolerance: float = 3.0,
    ):
        """
        Prepares the 2D boundaries from line geometries.

        The values can either be a spatially-uniform constant using ``boundaries_fn`` and ``boundary_value`` (default),
        or spatially-varying timeseries using ``boundaries_fn`` and  ``boundaries_timeseries_fn``
        The ``boundary_type`` can either be "waterlevel" or "discharge".

        If ``boundaries_timeseries_fn`` has missing values, the constant ``boundary_value`` will be used.

        The dataset/timeseries are clipped to the model region (see below note), and model time based on the model config
        tstart and tstop entries.

        Note that:
        (1) Only line geometry that are contained within the distance of ``tolenrance`` to grid cells are allowed.
        (2) Because of the above, this function must be called before the mesh refinement. #FIXME: check this after deciding on mesh refinement being a workflow or function
        (3) when using constant boundary, the output forcing will be written to time series with constant values.

        Adds/Updates model layers:
            * **boundary2d_{boundary_name}** forcing: 2D boundaries DataArray

        Parameters
        ----------
        boundaries_fn: str Path
            Path or data source name for line geometry file.
            * Required variables if a combined time series data csv file: ["boundary_id"] with type int
        boundaries_timeseries_fn: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location index with type int in the first row, matching "boundary_id" in ``boundaries_fn`.
            see :py:meth:`hydromt.get_dataframe`, for details.
            NOTE: Require equidistant time series
        boundary_value : float, optional
            Constant value to use for all boundaries, and to
            fill in missing data. By default 0.0 m.
        boundary_type : str, optional
            Type of boundary tu use. One of ["waterlevel", "discharge"].
            By default "waterlevel".
        tolerance: float, optional
            Search tolerance factor between boundary polyline and grid cells.
            Unit: in cell size units (i.e., not meters)
            By default, 3.0

        Raises
        ------
        AssertionError
            if "boundary_id" in "boundaries_fn" does not match the columns of ``boundaries_timeseries_fn``.

        """
        self.logger.info(f"Preparing 2D boundaries.")

        if boundary_type == "waterlevel":
            boundary_unit = "m"
        if boundary_type == "discharge":
            boundary_unit = "m3/s"

        _mesh_region = self._mesh.ugrid.to_geodataframe().unary_union
        _boundary_region = _mesh_region.buffer(tolerance * self.res).difference(
            _mesh_region
        )  # region where 2d boundary is allowed
        _boundary_region = gpd.GeoDataFrame(
            {"geometry": [_boundary_region]}, crs=self.crs
        )

        refdate, tstart, tstop = self.get_model_time()  # time slice

        # 1. read boundary geometries
        if boundaries_fn is not None:
            gdf_bnd = self.data_catalog.get_geodataframe(
                boundaries_fn,
                geom=_boundary_region,
                crs=self.crs,
                predicate="contains",
            )
            if len(gdf_bnd) == 0:
                self.logger.error(
                    "Boundaries are not found. Check if the boundary are outside of recognisable boundary region (cell size * tolerance to the mesh). "
                )
            # preprocess
            gdf_bnd = gdf_bnd.explode()
            # set index
            if "boundary_id" not in gdf_bnd:
                gdf_bnd["boundary_id"] = [
                    f"2dboundary_{i}" for i in range(len(gdf_bnd))
                ]
            else:
                gdf_bnd["boundary_id"] = gdf_bnd["boundary_id"].astype(str)
        else:
            gdf_bnd = None
        # 2. read timeseries boundaries
        if boundaries_timeseries_fn is not None:
            self.logger.info("reading timeseries boundaries")
            df_bnd = self.data_catalog.get_dataframe(boundaries_timeseries_fn,
                                                     time_tuple=(tstart, tstop)) # could not use open_geodataset due to line geometry
            # error if time mismatch or wrong parsing of dates
            if np.dtype(df_bnd.index).type != np.datetime64:
                raise ValueError(
                    "Dates in boundaries_timeseries_fn were not parsed correctly. "
                    "Update the source kwargs in the DataCatalog based on the driver function arguments (eg pandas.read_csv for csv driver).")
            if (df_bnd.index[-1] -  df_bnd.index[0]) < (tstop - tstart):
                raise ValueError(
                    "Time in boundaries_timeseries_fn were shorter than model simulation time. "
                    "Update the source kwargs in the DataCatalog based on the driver function arguments (eg pandas.read_csv for csv driver).")
            if gdf_bnd is not None:
                # check if all boundary_id are in df_bnd
                assert all(
                    [
                        bnd in df_bnd.columns
                        for bnd in gdf_bnd["boundary_id"].unique()
                    ]
                ), "Not all boundary_id are in df_bnd"
        else:
            # default timeseries
            d_bnd = {bnd_id: np.nan for bnd_id in gdf_bnd["boundary_id"].unique()}
            d_bnd.update({"time": pd.date_range(
                        start=pd.to_datetime(tstart),
                        end=pd.to_datetime(tstop),
                        freq="D",
                    )})
            df_bnd = pd.DataFrame(d_bnd).set_index("time")

        # 4. Derive DataArray with boundary values at boundary locations in boundaries_branch_type
        da_out_dict = workflows.compute_2dboundary_values(
            boundaries=gdf_bnd,
            df_bnd=df_bnd.reset_index(),
            boundary_value=boundary_value,
            boundary_type=boundary_type,
            boundary_unit=boundary_unit,
            logger=self.logger,
        )

        # 5. set boundaries
        for da_out_name,da_out in da_out_dict.items():
            self.set_forcing(da_out, name=f"boundary2d_{da_out_name}")

        # adjust parameters
        self.set_config("geometry.openboundarytolerance", tolerance)
    
    
    def setup_rainfall_from_constant(
        self,
        constant_value: float,
    ):
        """
        Prepares constant daily rainfall_rate timeseries for the 2D grid based on ``constant_value``.

        Adds/Updates model layers:
            * **meteo_{meteo_type}** forcing: DataArray
        
        Parameters
        ----------
        constant_value: float
            Constant value for the rainfall_rate timeseries in mm/day.
        """
        self.logger.info(f"Preparing rainfall meteo forcing from uniform timeseries.")

        refdate, tstart, tstop = self.get_model_time()  # time slice
        meteo_location = (self.region.centroid.x, self.region.centroid.y) # global station location

        df_meteo = pd.DataFrame({
            "time": pd.date_range(start=pd.to_datetime(tstart), end = pd.to_datetime(tstop), freq='D'),
            "precip": constant_value,
        })

        # 3. Derive DataArray with meteo values
        da_out = workflows.compute_meteo_forcings(
            df_meteo = df_meteo,
            fill_value=constant_value,
            is_rate=True,
            meteo_location=meteo_location,
            logger=self.logger,
        )

        # 4. set meteo forcing
        self.set_forcing(da_out, name=f"meteo_{da_out.name}")
        
        # 5. set meteo in mdu
        self.set_config("external_forcing.rainfall", 1)
    
    def setup_rainfall_from_uniform_timeseries(
        self,
        meteo_timeseries_fn: Union[str, Path],
        fill_value: float = 0.,
        is_rate: bool = True,
    ):
        """
        Prepares spatially uniform rainfall forcings to 2d grid using timeseries from ``meteo_timeseries_fn``.
        For now only support global  (spatially uniform) timeseries.

        If ``meteo_timeseries_fn`` has missing values, the constant ``fill_value`` will be used, e.g. 0.

        The dataset/timeseries are clipped to the model time based on the model config
        tstart and tstop entries.

        Adds/Updates model layers:
            * **meteo_{meteo_type}** forcing: DataArray

        Parameters
        ----------
        meteo_timeseries_fn: str, Path
            Path or data source name to tabulated timeseries csv file with time index in first column.
            
            * Required variables : ['precip']
            
            see :py:meth:`hydromt.get_dataframe`, for details.
            NOTE: Require equidistant time series
        fill_value : float, optional
            Constant value to use to fill in missing data. By default 0. 
        is_rate : bool, optional
            Specify if the type of meteo data is direct "rainfall" (False) or "rainfall_rate" (True).
            By default True for "rainfall_rate". Note that Delft3DFM 1D2D Suite 2022.04 supports only "rainfall_rate".
            If rate, unit is expected to be in mm/day and else mm.

        """
        self.logger.info(f"Preparing rainfall meteo forcing from uniform timeseries.")

        refdate, tstart, tstop = self.get_model_time()  # time slice
        meteo_location = (self.region.centroid.x, self.region.centroid.y) # global station location

        # get meteo timeseries
        df_meteo = self.data_catalog.get_dataframe(
            meteo_timeseries_fn,
            variables = ["precip"],
            time_tuple=(tstart, tstop))
        # error if time mismatch or wrong parsing of dates
        if np.dtype(df_meteo.index).type != np.datetime64:
            raise ValueError(
                "Dates in meteo_timeseries_fn were not parsed correctly. "
                "Update the source kwargs in the DataCatalog based on the driver function arguments (eg pandas.read_csv for csv driver).")
        if (df_meteo.index[-1] -  df_meteo.index[0]) < (tstop - tstart):
                raise ValueError(
                    "Time in meteo_timeseries_fn were shorter than model simulation time. "
                    "Update the source kwargs in the DataCatalog based on the driver function arguments (eg pandas.read_csv for csv driver).")
        df_meteo["time"] = df_meteo.index

        # 3. Derive DataArray with meteo values
        da_out = workflows.compute_meteo_forcings(
            df_meteo = df_meteo,
            fill_value=fill_value,
            is_rate=is_rate,
            meteo_location=meteo_location,
            logger=self.logger,
        )

        # 4. set meteo forcing
        self.set_forcing(da_out, name=f"meteo_{da_out.name}")
        
        # 5. set meteo in mdu
        self.set_config("external_forcing.rainfall", 1)

    # ## I/O
    # TODO: remove after hydromt 0.6.1 release
    @property
    def _assert_write_mode(self):
        if not self._write:
            raise IOError("Model opened in read-only mode")

    # TODO: remove after hydromt 0.6.1 release
    @property
    def _assert_read_mode(self):
        if not self._read:
            raise IOError("Model opened in write-only mode")

    def read(self):
        """Method to read the complete model schematization and configuration from file.
        # FIXME: where to read crs? """
        self.logger.info(f"Reading model data from {self.root}")
        self.read_dimr()
        self.read_config()
        self.read_mesh()
        self.read_maps()
        self.read_geoms()  # needs mesh so should be done after
        self.read_forcing()

    def write(self):  # complete model
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return

        if self._maps:
            self.write_maps()
        if self._geoms:
            self.write_geoms()
        if self._mesh or not self.branches.empty:
            self.write_mesh()
        if self._forcing:
            self.write_forcing()
        if self.config:  # dflowfm config, should always be last!
            self.write_config()
        if self.dimr:  # dimr config, should always be last after dflowfm config!
            self.write_dimr()
        self.write_data_catalog()

    def read_config(self) -> None:
        """Use Hydrolib-core reader and return to dictionnary"""
        # Read via init_dfmmodel
        if self._dfmmodel is None:
            self.init_dfmmodel()
        # Convert to full dictionnary without hydrolib-core objects
        cf_dict = dict()
        for k, v in self._dfmmodel.__dict__.items():
            if v is None or k == "filepath":
                cf_dict[k] = v
            else:
                ci_dict = dict()
                for ki, vi in v.__dict__.items():
                    if ki == "frictfile" and isinstance(vi, list):  # list of filepath
                        ci_dict[ki] = ";".join([str(vj.filepath) for vj in vi])
                    elif ki != "comments":
                        if hasattr(vi, "filepath"):
                            # need to change the filepath object to path
                            ci_dict[ki] = vi.filepath
                        else:
                            ci_dict[ki] = vi
                cf_dict[k] = ci_dict
        self._config = cf_dict

    def write_config(self) -> None:
        """From config dict to Hydrolib MDU"""
        # Not sure if this is worth it compared to just calling write_config super method
        # advantage is the validator but the whole model is then read when initialising FMModel
        self._assert_write_mode

        cf_dict = self._config.copy()
        # Need to switch to dflowfm folder for files to be found and properly added
        mdu_fn = cf_dict.pop("filepath", None)
        mdu_fn = Path(join(self.root, self._config_fn))
        cwd = os.getcwd()
        os.chdir(dirname(mdu_fn))
        mdu = FMModel(**cf_dict)
        # add filepath
        mdu.filepath = mdu_fn
        # write
        mdu.save(recurse=False)
        # Go back to working dir
        os.chdir(cwd)

    def read_maps(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Read maps from initialfield and parse to dict of xr.DataArray"""
        self._assert_read_mode
        # Read initial fields
        inifield_model = self.dfmmodel.geometry.inifieldfile
        if inifield_model is not None:
            # Loop over initial / parameter to read the geotif
            inilist = inifield_model.initial.copy()
            inilist.extend(inifield_model.parameter)

            if len(inilist) > 0:
                # DFM map names
                rm_dict = dict()
                for v in self._MAPS:
                    rm_dict[self._MAPS[v]["name"]] = v
                for inidict in inilist:
                    _fn = inidict.datafile.filepath
                    # Bug: when initialising IniFieldModel hydrolib-core does not parse correclty the relative path
                    # For now re-update manually....
                    if not isfile(_fn):
                        _fn = join(self.root, "maps", _fn.name)
                    inimap = hydromt.io.open_raster(_fn)
                    name = inidict.quantity
                    # Need to get frictiontype from config
                    if name == "frictioncoefficient":
                        frictype = self.get_config("physics.UniFrictType", 1)
                        fricname = [
                            n
                            for n in self._MAPS
                            if self._MAPS[n].get("frictype", None) == frictype
                        ]
                        rm_dict[name] = fricname[0]
                    # Check if name in self._MAPS to update properties
                    if name in rm_dict:
                        # update all keywords
                        inidict.__dict__.pop("comments")
                        self._MAPS[rm_dict[name]].update(inidict)
                        # Update default interpolation method
                        if inidict.interpolationmethod == "averaging":
                            self._MAPS[rm_dict[name]][
                                "interpolation"
                            ] = inidict.averagingtype
                        else:
                            self._MAPS[rm_dict[name]][
                                "interpolation"
                            ] = inidict.interpolationmethod
                        # Rename to hydromt name
                        name = rm_dict[name]
                    # Add to maps
                    self.set_maps(inimap, name)

            return self._maps

    def write_maps(self) -> None:
        """Write maps as tif files in maps folder and update initial fields"""
        if len(self._maps) == 0:
            self.logger.debug("No maps data found, skip writing.")
            return
        self._assert_write_mode
        # Global parameters
        mapsroot = join(self.root, "maps")
        inilist = []
        paramlist = []
        self.logger.info(f"Writing maps files to {mapsroot}")

        def _prepare_inifields(da_dict, da):
            # Write tif files
            name = da_dict["name"]
            type = da_dict["initype"]
            interp_method = da_dict["interpolation"]
            locationtype = da_dict["locationtype"]
            _fn = join(mapsroot, f"{name}.tif")
            if da.raster.nodata is None or np.isnan(da.raster.nodata):
                da.raster.set_nodata(-999)
            da.raster.to_raster(_fn)
            # Prepare dict
            if interp_method == "triangulation":
                inidict = {
                    "quantity": name,
                    "dataFile": f"../maps/{name}.tif",
                    "dataFileType": "GeoTIFF",
                    "interpolationMethod": interp_method,
                    "operand": da_dict.get("oprand", 'O'),
                    "locationType": locationtype,
                }
            else: 
                inidict = {
                    "quantity": name,
                    "dataFile": f"../maps/{name}.tif",
                    "dataFileType": "GeoTIFF",
                    "interpolationMethod": "averaging",
                    "operand": da_dict.get("oprand", 'O'),
                    "averagingType": interp_method,
                    "averagingRelSize": da_dict.get("averagingrelsize"),
                    "locationType": locationtype,
                }
            if type == "initial":
                inilist.append(inidict)
            elif type == "parameter":
                paramlist.append(inidict)

        # Only write maps that are listed in self._MAPS, rename tif on the fly
        # TODO raise value error if both waterdepth and waterlevel are given as maps.items
        for name, ds in self._maps.items():
            if isinstance(ds, xr.DataArray):
                if name in self._MAPS:
                    _prepare_inifields(self._MAPS[name], ds)
                    # update config if friction
                    if "frictype" in self._MAPS[name]:
                        self.set_config(
                            "physics.UniFrictType", self._MAPS[name]["frictype"]
                        )
                    # update config if infiltration
                    if name == "infiltcap":
                        self.set_config(
                            "grw.infiltrationmodel", 2
                        )

            elif isinstance(ds, xr.Dataset):
                for v in ds.data_vars:
                    if v in self._MAPS:
                        _prepare_inifields(self._MAPS[v], ds[v])
                        # update config if frcition
                        if "frictype" in self._MAPS[name]:
                            self.set_config(
                                "physics.UniFrictType", self._MAPS[name]["frictype"]
                            )
                        # update config if infiltration
                        if name == "infiltcap":
                            self.set_config(
                                "grw.infiltrationmodel", 2
                            )

        # Assign initial fields to model and write
        inifield_model = IniFieldModel(initial=inilist, parameter=paramlist)
        # Bug: when initialising IniFieldModel hydrolib-core does not parse correclty the relative path
        # For now re-update manually....
        for i in range(len(inifield_model.initial)):
            path = Path(f"../maps/{inifield_model.initial[i].datafile.filepath.name}")
            inifield_model.initial[i].datafile.filepath = path
        for i in range(len(inifield_model.parameter)):
            path = Path(f"../maps/{inifield_model.parameter[i].datafile.filepath.name}")
            inifield_model.parameter[i].datafile.filepath = path
        # Write inifield file
        inifield_model_filename = inifield_model._filename() + ".ini"
        fm_dir = dirname(join(self.root, self._config_fn))
        inifield_model.save(
            join(fm_dir, inifield_model_filename),
            recurse=False,
        )
        # save filepath in the config
        self.set_config("geometry.inifieldfile", inifield_model_filename)

    def read_geoms(self) -> None:  # FIXME: gives an error when only 2D model.
        """
        Read model geometries files at <root>/<geoms> and add to geoms property.

        For branches / boundaries etc... the reading of hydrolib-core objects happens in read_mesh
        There the geoms geojson copies are re-set based on dflowfm files content.
        """
        self._assert_read_mode
        super().read_geoms(fn="geoms/region.geojson")

        if self.dfmmodel.geometry.crosslocfile is not None:
            # Read cross-sections and friction
            # Add crosssections properties, should be done before friction
            # Branches are needed do derive locations, self.branches should start the read if not done yet
            self.logger.info("Reading cross-sections files")
            crosssections = utils.read_crosssections(self.branches, self.dfmmodel)

            # Add friction properties from roughness files
            self.logger.info("Reading friction files")
            crosssections = utils.read_friction(crosssections, self.dfmmodel)
            self.set_geoms(crosssections, "crosssections")

        # Read manholes
        if self.dfmmodel.geometry.storagenodefile is not None:
            self.logger.info("Reading manholes file")
            manholes = utils.read_manholes(self.network1d_nodes, self.dfmmodel)
            self.set_geoms(manholes, "manholes")
        
        # Read structures
        if self.dfmmodel.geometry.structurefile is not None:
            self.logger.info("Reading structures file")
            structures = utils.read_structures(self.branches, self.dfmmodel)
            for st_type in structures["type"].unique():
                self.set_geoms(structures[structures["type"]==st_type], f"{st_type}s")

    def write_geoms(self) -> None:
        """Write model geometries to a GeoJSON file at <root>/<geoms>"""
        self._assert_write_mode

        # Write geojson equivalent of all objects. Note that these files are not directly used when updating the model
        super().write_geoms(fn="geoms/{name}.geojson")

        # Write dfm files
        savedir = dirname(join(self.root, self._config_fn))

        # Write cross-sections (inc. friction)
        if "crosssections" in self._geoms:
            # Crosssections
            gdf_crs = self.geoms["crosssections"]
            self.logger.info("Writting cross-sections files crsdef and crsloc")
            crsdef_fn, crsloc_fn = utils.write_crosssections(gdf_crs, savedir)
            self.set_config("geometry.crossdeffile", crsdef_fn)
            self.set_config("geometry.crosslocfile", crsloc_fn)

            # Friction
            self.logger.info("Writting friction file(s)")
            friction_fns = utils.write_friction(gdf_crs, savedir)
            self.set_config("geometry.frictfile", ";".join(friction_fns))

        # Write structures
        # Manholes
        if "manholes" in self._geoms:
            self.logger.info(f"Writting manholes file.")
            storage_fn = utils.write_manholes(
                self.geoms["manholes"],
                savedir,
            )
            self.set_config("geometry.storagenodefile", storage_fn)
            
        # Write structures
        existing_structures = [st for st in ["bridges", "culverts"] if st in self.geoms]
        if len(existing_structures) > 0: 
            # combine all structures
            structures = []
            for st in existing_structures: 
                structures.append(self.geoms.get(st).to_dict("records"))
            structures = list(itertools.chain.from_iterable(structures))
            structures = pd.DataFrame(structures).replace(np.nan, None)
            # write
            self.logger.info(f"Writting structures file.")
            structures_fn = utils.write_structures(
                structures,
                savedir,
            )
            self.set_config("geometry.structurefile", structures_fn)
            

    def read_forcing(
        self,
    ) -> None:  # FIXME reading of forcing should include boundary, lateral and meteo
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        self._assert_read_mode
        # Read external forcing
        ext_model = self.dfmmodel.external_forcing.extforcefilenew
        if ext_model is not None:
            # boundary
            if len(ext_model.boundary) > 0:
                df_ext = pd.DataFrame([f.__dict__ for f in ext_model.boundary])
                # 1d boundary
                df_ext_1d = df_ext.loc[~df_ext.nodeid.isna(), :]
                if len(df_ext_1d) > 0:
                    # Forcing data arrays to prepare for each quantity
                    forcing_names = np.unique(df_ext_1d.quantity).tolist()
                    # Loop over forcing names to build data arrays
                    for name in forcing_names:
                        # Get the dataframe corresponding to the current variable
                        df = df_ext_1d[df_ext_1d.quantity == name]
                        # Get the corresponding nodes gdf
                        node_geoms = self.network1d_nodes[
                            np.isin(self.network1d_nodes["nodeId"], df.nodeid.values)
                        ]
                        da_out = utils.read_1dboundary(df, quantity=name, nodes=node_geoms)
                        # Add to forcing
                        self.set_forcing(da_out)
                # 2d boundary
                df_ext_2d = df_ext.loc[df_ext.nodeid.isna(), :]
                if len(df_ext_2d) > 0:
                    for _, df in df_ext_2d.iterrows():
                        da_out = utils.read_2dboundary(
                            df, workdir=self.dfmmodel.filepath.parent
                        )
                        # Add to forcing
                        self.set_forcing(da_out)
            # meteo
            if len(ext_model.meteo) > 0:
                df_ext = pd.DataFrame([f.__dict__ for f in ext_model.meteo])
                # Forcing dataarrays to prepare for each quantity
                forcing_names = np.unique(df_ext.quantity).tolist()
                # Loop over forcing names to build data arrays
                for name in forcing_names:
                    # Get the dataframe corresponding to the current variable
                    df = df_ext[df_ext.quantity == name]
                    da_out = utils.read_meteo(df, quantity=name)
                    # Add to forcing
                    self.set_forcing(da_out)
            # TODO lateral

    def write_forcing(self) -> None:
        """write forcing into hydrolib-core ext and forcing models"""
        if len(self._forcing) == 0:
            self.logger.debug("No forcing data found, skip writing.")
        else:
            self._assert_write_mode
            self.logger.info("Writting forcing files.")
            savedir = dirname(join(self.root, self._config_fn))
            # create new external forcing file
            ext_fn = "bnd.ext"
            Path(join(savedir, ext_fn)).unlink(missing_ok=True)
            # populate external forcing file
            utils.write_1dboundary(self.forcing, savedir, ext_fn=ext_fn)
            utils.write_2dboundary(self.forcing, savedir, ext_fn=ext_fn)
            utils.write_meteo(self.forcing, savedir, ext_fn=ext_fn)
            self.set_config("external_forcing.extforcefilenew", ext_fn)

    def read_mesh(self):
        """Read network file with Hydrolib-core and extract 2D mesh/branches info"""
        self._assert_read_mode

        # Read mesh
        # hydrolib-core convention
        net = self.dfmmodel.geometry.netfile.network
        # FIXME: crs info is not available in dfmmodel, so get it from region.geojson
        # Cannot use read_geoms yet because for some some geoms (crosssections, manholes) mesh needs to be read first...
        region_fn = join(self.root, "geoms", "region.geojson")
        if isfile(region_fn):
            crs = gpd.read_file(region_fn).crs
        else:
            crs = None
        # add mesh2d to self.mesh
        if not net._mesh2d.is_empty():
            net._mesh2d._set_mesh2d()
            mesh2d = net._mesh2d.get_mesh2d()
            # Create Ugrid2d object
            grid = xu.Ugrid2d.from_meshkernel(mesh2d)
            # n_max_node = mesh2d.nodes_per_face.max()
            # grid = xu.Ugrid2d(
            #     node_x=mesh2d.node_x,
            #     node_y=mesh2d.node_y,
            #     fill_value=-1,
            #     face_node_connectivity=mesh2d.face_nodes.reshape((-1, n_max_node)),
            #     crs=crs,
            # )
            # grid._mesh = mesh2d
            # Create UgridDataset
            da = xr.DataArray(
                data=np.arange(grid.n_face),
                dims=[grid.face_dimension],
            )
            uda = xu.UgridDataArray(da, grid)
            uds = uda.to_dataset(name="index")
            uds = uds.assign_coords(
                coords={
                    "mesh2d_node_x": ("mesh2d_nNodes", grid.node_x),
                    "mesh2d_node_y": ("mesh2d_nNodes", grid.node_y),
                }
            )
            uds.ugrid.grid.set_crs(grid.crs)
            self.set_mesh(uds)
            if self._res is None:
                np.diff(grid.node_x)

        # add mesh1d to self.mesh # TODO: for now self.mesh is 2D only
        # if not net._mesh1d.is_empty():
        #    self._add_mesh1d(net._mesh1d)

        # Read mesh1d related geometry (branches)
        if not net._mesh1d.is_empty():
            mesh1d = self.mesh1d
            branch_id = mesh1d.network1d_branch_id
            # Create the GeoDataFrame
            branches = gpd.GeoDataFrame(
                geometry=[LineString(mesh1d.branches[i].geometry) for i in branch_id],
                crs=crs,
            )
            branches["branchId"] = branch_id
            branches["branchOrder"] = mesh1d.network1d_branch_order

            # Add branchType, properties from branches.gui file
            self.logger.info("Reading branches GUI file")
            branches = utils.read_branches_gui(branches, self.dfmmodel)

            # Add branches
            self._branches = branches 

    def write_mesh(self, write_gui=True):
        """Write 1D branches and 2D mesh at <root/dflowfm/fm_net.nc> in model ready format"""
        self._assert_write_mode
        savedir = dirname(join(self.root, self._config_fn))

        # write mesh
        # hydromt convention
        # super().write_mesh(fn="mesh/fm_net.nc")
        if self._mesh:
            fn = "mesh/fm_net.nc"
            _fn = join(self.root, fn)
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            ds_out = xr.Dataset()
            for grid in self._mesh.ugrid.grids:
                ds_out = ds_out.merge(grid.to_dataset())
            if grid.crs is not None:
                # save crs to spatial_ref coordinate
                ds_out = ds_out.rio.write_crs(grid.crs)
            ds_out.to_netcdf(_fn, mode="w")

        # hydrolib-core convention (meshkernel)
        mesh_filename = "fm_net.nc"
        self.logger.info(f"Writing mesh to {join(savedir, mesh_filename)}")
        self.dfmmodel.geometry.netfile.save(
            join(savedir, mesh_filename),
            recurse=False,
        )
        # save relative path to mdu
        self.set_config("geometry.netfile", mesh_filename)

        # other mesh1d related geometry
        if not self.mesh1d.is_empty() and write_gui:
            self.logger.info("Writting branches.gui file")
            manholes = None
            if "manholes" in self.geoms:
                manholes = self.geoms["manholes"]
            _ = utils.write_branches_gui(self.branches, savedir)

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
    def bounds(self) -> Tuple:
        """Returns model mesh bounds."""
        return self.region.total_bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns geometry of region of the model area of interest."""
        # First tries in geoms
        if "region" in self.geoms:
            region = self.geoms["region"]
        # Else derives from mesh or branches
        else:
            if self.mesh is not None:
                bounds = self.mesh.ugrid.total_bounds
                crs = self.mesh.ugrid.grid.crs
            elif self.branches is not None:
                bounds = self.branches.total_bounds
                crs = self.branches.crs
            else:
                # Finally raise error assuming model is empty
                self.logger.error(
                    "Could not derive region from geoms, or mesh. Model may be empty."
                )
            region = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs)
            self.set_geoms(region, "region")

        return region

    @property
    def dfmmodel(self):
        if self._dfmmodel is None:
            self.init_dfmmodel()
        return self._dfmmodel

    def init_dfmmodel(self):
        # create a new MDU-Model
        mdu_fn = Path(join(self.root, self._config_fn))
        if isfile(mdu_fn) and self._read:
            self.logger.info(f"Reading mdu file at {mdu_fn}")
            self._dfmmodel = FMModel(filepath=mdu_fn)
        else:  # use hydrolib template
            self._assert_write_mode
            self.logger.info("Initialising empty mdu file")
            self._dfmmodel = FMModel()
            self._dfmmodel.filepath = mdu_fn

    @property
    def dimr(self):
        """DIMR file object"""
        if not self._dimr:
            self.read_dimr()
        return self._dimr

    def read_dimr(self, dimr_fn: Optional[str] = None) -> None:
        """Read DIMR from file and else create from hydrolib-core"""
        if dimr_fn is None:
            dimr_fn = join(self.root, self._dimr_fn)
        # if file exist, read
        if isfile(dimr_fn) and self._read:
            self.logger.info(f"Reading dimr file at {dimr_fn}")
            dimr = DIMR(filepath=Path(dimr_fn))
        # else initialise
        else:
            self._assert_write_mode
            self.logger.info("Initialising empty dimr file")
            dimr = DIMR()
        self._dimr = dimr

    def write_dimr(self, dimr_fn: Optional[str] = None):
        """Writes the dmir file. In write mode, updates first the FMModel component"""
        # force read
        self.dimr
        if dimr_fn is not None:
            self._dimr.filepath = join(self.root, dimr_fn)
        else:
            self._dimr.filepath = join(self.root, self._dimr_fn)

        if not self._read:
            # Updates the dimr file first before writing
            self.logger.info("Adding dflowfm component to dimr config")

            # update component
            components = self._dimr.component
            if len(components) != 0:
                components = []
            fmcomponent = FMComponent(
                name="dflowfm",
                workingdir="dflowfm",
                inputfile=basename(self._config_fn),
                model=self.dfmmodel,
            )
            components.append(fmcomponent)
            self._dimr.component = components
            # update control
            controls = self._dimr.control
            if len(controls) != 0:
                controls = []
            control = Start(name="dflowfm")
            controls.append(control)
            self._dimr.control = control

        # write
        self.logger.info(f"Writing model dimr file to {self._dimr.filepath}")
        self.dimr.save(recurse=False)

    @property
    def branches(self):
        """
        Returns the branches (gpd.GeoDataFrame object) representing the 1D network.
        Contains several "branchType" for : channel, river, pipe, tunnel.
        """
        if self._branches is None and self._read:
            self.read_mesh()
        elif self._branches is None:
            self._branches = gpd.GeoDataFrame()
        return self._branches

    def set_branches(self, branches: gpd.GeoDataFrame):
        """Updates the branches object as well as the linked geoms."""
        # Check if "branchType" col in new branches
        if "branchType" in branches.columns:
            self._branches = branches
        else:
            self.logger.error(
                "'branchType' column absent from the new branches, could not update."
            )

        # Update channels/pipes in geoms
        _ = self.set_branches_component(name="river")
        _ = self.set_branches_component(name="channel")
        _ = self.set_branches_component(name="pipe")

        # update geom
        self.logger.debug(f"Adding branches vector to geoms.")
        self.set_geoms(branches, "branches")

        # set hydrolib-core net object and update self._mesh
        self.set_mesh1d()

        # update boundaries
        self.set_geoms(self.get_boundaries(), "boundaries")

        self.logger.debug(f"Updating branches in network.")

    def add_branches(
        self,
        new_branches: gpd.GeoDataFrame,
        branchtype: str,
        node_distance: float = 40.0,
    ):

        """Add new branches of branchtype to the branches and mesh1d object"""

        snap_newbranches_to_branches_at_snapnodes = self._snap_newbranches_to_branches_at_snapnodes

        snap_offset = self._network_snap_offset

        branches = self.branches.copy()

        if snap_newbranches_to_branches_at_snapnodes is True and len(self.opensystem) > 0:
            self.logger.info(
                f"snapping {branchtype} ends to existing network (opensystem only)"
            )

            # get possible connection points from new branches
            if branchtype in ["pipe", "tunnel"]:
                endnodes = workflows.generate_boundaries_from_branches(
                    new_branches, where="downstream"
                )  # only connect downstream ends to existing network
            else:
                endnodes = workflows.generate_boundaries_from_branches(
                    new_branches, where="both"
                )  # only connect both ends to existing network

            # get possible connection points (allow connection only at mesh nodes) from existing open system
            mesh1d_nodes = self.mesh1d_nodes.copy()
            mesh1d_nodes_open = mesh1d_nodes.loc[
                mesh1d_nodes.branch_name.isin(self.opensystem.branchId.tolist())
            ]

            # snap the new endnodes to existing mesh1d_nodes_open
            snapnodes = hydromt.gis_utils.nearest_merge(
                endnodes, mesh1d_nodes_open, max_dist=snap_offset, overwrite=False
            )
            snapnodes = snapnodes[snapnodes.index_right != -1]  # drop not snapped
            snapnodes["geometry_left"] = snapnodes["geometry"]
            snapnodes["geometry_right"] = [
                mesh1d_nodes_open.at[i, "geometry"] for i in snapnodes["index_right"]
            ]
            logger.debug(f"snapped features: {len(snapnodes)}")
            (
                new_branches_snapped,
                branches_snapped,
            ) = workflows.snap_newbranches_to_branches_at_snapnodes(
                new_branches, branches, snapnodes
            )

            # update the branches
            branches = branches_snapped.append(new_branches_snapped, ignore_index=True)
        else:
            # update the branches
            branches = branches.append(new_branches, ignore_index=True)

        if (
            branches.crs is None
        ):  # Cannot instantiate empty geodataframe with crs anymore
            branches.crs = new_branches.crs

        # Check if we need to do more check/process to make sure everything is well-connected
        workflows.validate_branches(branches)
        # set geom
        self.set_branches(branches)

    def set_branches_component(self, name: str):
        gdf_comp = self.branches[self.branches["branchType"] == name]
        if gdf_comp.index.size > 0:
            self.set_geoms(gdf_comp, name=f"{name}s")
        return gdf_comp

    @property
    def rivers(self):
        if "rivers" in self.geoms:
            gdf = self.geoms["rivers"]
        else:
            gdf = self.set_branches_component("river")
        return gdf

    @property
    def channels(self):
        if "channels" in self.geoms:
            gdf = self.geoms["channels"]
        else:
            gdf = self.set_branches_component("channel")
        return gdf

    @property
    def pipes(self):
        if "pipes" in self.geoms:
            gdf = self.geoms["pipes"]
        else:
            gdf = self.set_branches_component("pipe")
        return gdf

    @property
    def opensystem(self):
        if len(self.branches) > 0:
            gdf = self.branches[self.branches["branchType"].isin(["river", "channel"])]
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    @property
    def closedsystem(self):
        if len(self.branches) > 0:
            gdf = self.branches[self.branches["branchType"].isin(["pipe", "tunnel"])]
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    @property
    def mesh1d(self):
        """
        Returns the mesh1d (hydrolib-core Mesh1d object) representing the 1D mesh.
        """
        return self.dfmmodel.geometry.netfile.network._mesh1d

    @property
    def mesh1d_nodes(self):
        """Returns the nodes of mesh 1D as geodataframe"""
        mesh1d_nodes = gpd.points_from_xy(
            x=self.mesh1d.mesh1d_node_x,
            y=self.mesh1d.mesh1d_node_y,
            crs=self.crs,
        )
        mesh1d_nodes = gpd.GeoDataFrame(
            data={
                "branch_id": self.mesh1d.mesh1d_node_branch_id,
                "branch_name": [
                    list(self.mesh1d.branches.keys())[i]
                    for i in self.mesh1d.mesh1d_node_branch_id
                ],
                "branch_chainage": self.mesh1d.mesh1d_node_branch_offset,
                "geometry": mesh1d_nodes,
            }
        )
        return mesh1d_nodes

    def set_mesh1d(self):
        """update the mesh1d in hydrolib-core net object by overwrite and add to self._mesh"""
        # reinitialise mesh1d (TODO: a clear() function in hydrolib-core could be handy)
        self.dfmmodel.geometry.netfile.network._mesh1d = Mesh1d(
            meshkernel=self.dfmmodel.geometry.netfile.network.meshkernel
        )

        # add open system mesh1d
        opensystem = self.opensystem
        node_distance = self._openwater_computation_node_distance
        mesh.mesh1d_add_branch(
            self._dfmmodel.geometry.netfile.network,
            opensystem.geometry.to_list(),
            node_distance=node_distance,
            branch_names=opensystem.branchId.to_list(),
            branch_orders=opensystem.branchOrder.to_list(),
        )

        # add closed system mesh1d
        closedsystem = self.closedsystem
        node_distance = np.inf
        mesh.mesh1d_add_branch(
            self._dfmmodel.geometry.netfile.network,
            closedsystem.geometry.to_list(),
            node_distance=node_distance,
            branch_names=closedsystem.branchId.to_list(),
            branch_orders=closedsystem.branchOrder.to_list(),
        )

        # set mesh1d
        self._dfmmodel.geometry.netfile.network._mesh1d._set_mesh1d()

        # add to self.mesh # TODO: for now self.mesh is 2D only
        # self._add_mesh1d(self.mesh1d)

    def _add_mesh1d(self, mesh1d: Mesh1d):
        """update the mesh1d in self.mesh object"""

        # Create Ugrid1d object for mesh
        # TODO: after release of xugrid use grid = xu.Ugrid1d.from_meshkernel(mesh1d)
        grid = xu.Ugrid1d(
            node_x=mesh1d.mesh1d_node_x,
            node_y=mesh1d.mesh1d_node_y,
            fill_value=-1,
            edge_node_connectivity=mesh1d.mesh1d_edge_nodes,
            name="mesh1d",
            crs=self.crs,
        )
        # grid._mesh = mesh1d
        # Create UgridDataset
        ds = xr.Dataset(
            data_vars={
                "mesh1d_node_x": ("mesh1d_nNodes", mesh1d.mesh1d_node_x),
                "mesh1d_node_y": ("mesh1d_nNodes", mesh1d.mesh1d_node_y),
                "mesh1d_node_id": ("mesh1d_nNodes", mesh1d.mesh1d_node_id),
                "mesh1d_node_long_name": (
                    "mesh1d_nNodes",
                    mesh1d.mesh1d_node_long_name,
                ),
                "mesh1d_node_branch_id": (
                    "mesh1d_nNodes",
                    mesh1d.mesh1d_node_branch_id,
                ),
                "mesh1d_node_branch_offset": (
                    "mesh1d_nNodes",
                    mesh1d.mesh1d_node_branch_offset,
                ),
                "mesh1d_edge_nodes": (
                    ["mesh1d_nEdges", "two"],
                    mesh1d.mesh1d_edge_nodes,
                ),
                "mesh1d_edge_x": ("mesh1d_nEdges", mesh1d.mesh1d_edge_x),
                "mesh1d_edge_y": ("mesh1d_nEdges", mesh1d.mesh1d_edge_y),
                "mesh1d_edge_branch_id": (
                    "mesh1d_nEdges",
                    mesh1d.mesh1d_edge_branch_id,
                ),
                "mesh1d_edge_branch_offset": (
                    "mesh1d_nEdges",
                    mesh1d.mesh1d_edge_branch_offset,
                ),
            }
        )
        uds = xu.UgridDataset(ds, grid)
        uds.ugrid.grid.set_crs(grid.crs)
        uds_mesh = uds.copy()

        # Create Ugrid1d object for network
        # TODO: after release of xugrid use grid = xu.Ugrid1d.from_meshkernel(mesh1d)
        grid = xu.Ugrid1d(
            node_x=mesh1d.network1d_node_x,
            node_y=mesh1d.network1d_node_y,
            fill_value=-1,
            edge_node_connectivity=mesh1d.network1d_edge_nodes,
            name="network1d",
            crs=self.crs,
        )
        # grid._mesh = mesh1d
        # Create UgridDataset
        ds = xr.Dataset(
            data_vars={
                "network1d_node_x": ("network1d_nNodes", mesh1d.network1d_node_x),
                "network1d_node_y": ("network1d_nNodes", mesh1d.network1d_node_y),
                "network1d_node_id": ("network1d_nNodes", mesh1d.network1d_node_id),
                "network1d_node_long_name": (
                    "network1d_nNodes",
                    mesh1d.network1d_node_long_name,
                ),
                "network1d_edge_nodes": (
                    ["network1d_nEdges", "two"],
                    mesh1d.network1d_edge_nodes,
                ),
                "network1d_geom_x": (
                    "network1d_nGeometryNodes",
                    mesh1d.network1d_geom_x,
                ),
                "network1d_geom_y": (
                    "network1d_nGeometryNodes",
                    mesh1d.network1d_geom_y,
                ),
                "network1d_branch_id": ("network1d_nEdges", mesh1d.network1d_branch_id),
                "network1d_branch_length": (
                    "network1d_nEdges",
                    mesh1d.network1d_branch_length,
                ),
                "network1d_branch_long_name": (
                    "network1d_nEdges",
                    mesh1d.network1d_branch_long_name,
                ),
                "network1d_branch_order": (
                    "network1d_nEdges",
                    mesh1d.network1d_branch_order,
                ),
                "network1d_part_node_count": (
                    "network1d_nEdges",
                    mesh1d.network1d_part_node_count,
                ),
            },
        )
        uds = xu.UgridDataset(ds, grid)
        uds.ugrid.grid.set_crs(grid.crs)
        uds_network = uds.copy()

        # combine
        uds_1d = uds_mesh.merge(uds_network)

        # set
        if self._mesh is not None:
            self._mesh = self._mesh.drop_vars(
                set(self._mesh.variables.keys()).intersection(uds_1d.variables.keys())
            )
        self.set_mesh(uds_1d)

    @property
    def crosssections(self):
        """Quick accessor to crosssections geoms"""
        if "crosssections" in self.geoms:
            gdf = self.geoms["crosssections"]
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    def add_crosssections(self, crosssections: gpd.GeoDataFrame):
        """Updates crosssections in geoms with new ones"""
        # TODO: sort out the crosssections, e.g. remove branch crosssections if point/xyz exist etc
        # TODO: setup river crosssections, set contrains based on branch types
        if len(self.crosssections) > 0:
            # add new crossections
            crosssections = gpd.GeoDataFrame(
                pd.concat([self.crosssections, crosssections]), crs=self.crs
            )
            # drop duplicated locations, keep the first one
            _crosssections_locations = crosssections[~crosssections.crsloc_id.isna()]
            _crosssections_definitions = crosssections[crosssections.crsloc_id.isna()] # retain crs defs for nan locations - structures
            _crosssections_locations["temp_id"] = _crosssections_locations.apply(lambda x:f'{x["crsloc_branchId"]}_{x["crsloc_chainage"]:.2f}', axis = 1)
            if _crosssections_locations["temp_id"].duplicated().any():
                logger.warning(f"Duplicate crosssections locations found, removing duplicates")
                # Remove duplicates based on the branch_id, branch_offset column, keeping the first occurrence (with minimum branch_distance)
                _crosssections_locations = _crosssections_locations.drop_duplicates(subset=["temp_id"], keep='first')
            crosssections = pd.concat([_crosssections_locations, _crosssections_definitions])
            # remove existing generated branch crosssections
            crossections_branch = crosssections[crosssections.crsdef_id.str.endswith("_branch")]
            crossections_others = crosssections[~crosssections.crsdef_id.str.endswith("_branch")]
            mask_to_remove = crossections_branch["crsloc_branchId"].isin(crossections_others["crsloc_branchId"])
            if mask_to_remove.sum() > 0:
                self.logger.warning(f'Overwrite branch crossections where user-defined crossections are used.')
                crosssections = gpd.GeoDataFrame(
                    pd.concat([crossections_branch[~mask_to_remove], crossections_others]), crs=self.crs
                )
        self.set_geoms(crosssections, name="crosssections")

    @property
    def boundaries(self) -> gpd.GeoDataFrame:
        """Quick accessor to boundaries geoms"""
        if "boundaries" in self.geoms:
            gdf = self.geoms["boundaries"]
        else:
            gdf = self.get_boundaries()
        return gdf

    def get_boundaries(self) -> gpd.GeoDataFrame:
        """Get all boundary locations from the network
        branch ends are possible locations for boundaries
        for open system, both upstream and downstream ends are allowed to have boundaries
        for closed system, only downstream ends are allowed to have boundaries
        """

        # generate all possible and allowed boundary locations
        _boundaries = workflows.generate_boundaries_from_branches(
            self.branches, where="both"
        )

        # get networkids to complete the boundaries
        boundaries = hydromt.gis_utils.nearest_merge(
            _boundaries, self.network1d_nodes, max_dist=0.1, overwrite=False
        )
        return boundaries

    def set_boundaries(self, boundaries: gpd.GeoDataFrame):
        """Updates boundaries in geoms with new ones"""
        if len(self.boundaries) > 0:
            task_last = lambda s1, s2: s2
            boundaries = self.boundaries.combine(
                boundaries, func=task_last, overwrite=True
            )
        self.set_geoms(boundaries, name="boundaries")

    def get_model_time(self):
        """Return (refdate, tstart, tstop) tuple with parsed model reference datem start and end time"""
        refdate = datetime.strptime(str(self.get_config("time.refdate")), "%Y%m%d")
        tstart = refdate + timedelta(seconds=float(self.get_config("time.tstart")))
        tstop = refdate + timedelta(seconds=float(self.get_config("time.tstop")))
        return refdate, tstart, tstop

    # @property
    # def network(self):
    #     """
    #     Returns the network (hydrolib-core Network object) based on self.mesh content.
    #     """
    #     return self.dfmmodel.geometry.netfile.network

    @property
    def network1d_nodes(self):
        """get network1d nodes as gdp"""
        # get networkids to complete the boundaries
        _network1d_nodes = gpd.points_from_xy(
            x=self.dfmmodel.geometry.netfile.network._mesh1d.network1d_node_x,
            y=self.dfmmodel.geometry.netfile.network._mesh1d.network1d_node_y,
            crs=self.crs,
        )
        _network1d_nodes = gpd.GeoDataFrame(
            data={
                "nodeId": self.dfmmodel.geometry.netfile.network._mesh1d.network1d_node_id,
                "geometry": _network1d_nodes,
            }
        )
        return _network1d_nodes

    @property
    def res(self):
        "resolution of the mesh2d"
        if self._res is not None:
            return self._res

    @property
    def mesh2d(self):
        """
        Returns the mesh2d (hydrolib-core Mesh2d object) representing the 2D mesh.
        """
        return (
            self.dfmmodel.geometry.netfile.network._mesh2d
        )  # needed to setup 1d2d links

    def set_mesh2d(self, mesh2d: mk.py_structures.Mesh2d = None):
        """update the mesh2d in hydrolib-core net object by overwrite"""
        # process mesh2d to net object
        if mesh2d is None and self._mesh is not None:
            mesh2d = self._mesh.ugrid.grid.mesh
        self.dfmmodel.geometry.netfile.network._mesh2d._process(mesh2d)

    @property
    def link1d2d(self):
        """
        Returns the link1d2d (hydrolib-core Link1d2d object) representing the 1d2d links.
        """
        return self.dfmmodel.geometry.netfile.network._link1d2d

    @property
    def network(self):
        """
        Returns the network (hydrolib-core Network object) representing the entire network file.
        """
        return self.dfmmodel.geometry.netfile.network
    
    @property
    def _model_has_2d(self):
        if not self.mesh2d.is_empty():
            return True
        else:
            return False
        
    @property
    def _model_has_1d(self):
        if not self.mesh1d.is_empty():
            return True
        else:
            return False
        