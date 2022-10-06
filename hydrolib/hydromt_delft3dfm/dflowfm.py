"""Implement Delft3D-FM hydromt plugin model class"""

import glob
import logging
from datetime import datetime, timedelta
from os import times
from os.path import basename, isfile, join
from pathlib import Path
from turtle import st
from typing import List, Tuple, Union

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from hydromt.models import MeshModel
from hydromt.models.model_auxmaps import AuxmapsMixin
from shapely.geometry import Point, box

from hydrolib.core.io.bc.models import *
from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.dimr.models import DIMR, FMComponent, Start
from hydrolib.core.io.ext.models import *
from hydrolib.core.io.friction.models import *
from hydrolib.core.io.gui.models import *
from hydrolib.core.io.inifield.models import IniFieldModel
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.net.models import *
from hydrolib.core.io.storagenode.models import StorageNodeModel
from hydrolib.dhydamo.geometry import common, mesh, viz

from . import DATADIR, workflows

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(AuxmapsMixin, MeshModel):
    """API for Delft3D-FM models in HydroMT"""

    _NAME = "dflowfm"
    _CONF = "DFlowFM.mdu"
    _DATADIR = DATADIR
    _GEOMS = {}
    _AUXMAPS = {
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
        },
        "waterdepth": {
            "name": "waterdepth",
            "initype": "initial",
            "interpolation": "mean",
            "locationtype": "2d",
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
    _FOLDERS = ["dflowfm", "geoms", "mesh", "auxmaps"]
    _CLI_ARGS = {"region": "setup_region", "res": "setup_mesh2d"}
    _CATALOGS = join(_DATADIR, "parameters_data.yml")

    def __init__(
        self,
        root: Union[str, Path] = None,
        mode: str = "w",
        config_fn: str = None,  # hydromt config contain glob section, anything needed can be added here as args
        data_libs: List[
            str
        ] = [],  # yml # TODO: how to choose global mapping files (.csv) and project specific mapping files (.csv)
        dimr_fn: str = None,
        network_snap_offset=25,
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
        self._config_fn = (
            join("dflowfm", self._CONF) if config_fn is None else config_fn
        )
        self.data_catalog.from_yml(self._CATALOGS)
        self.write_config()  #  create the mdu file in order to initialise dfmmodedl properly and at correct output location
        self._dfmmodel = self.init_dfmmodel()
        # Gloabl options for generation of the mesh1d network
        self._network_snap_offset = network_snap_offset
        self._openwater_computation_node_distance = openwater_computation_node_distance

    def setup_region(
        self,
        region: dict,
        crs: int = None,
    ):
        """Define the model region.

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
            DataFrame containing spacing values per 'branchType', 'shape', 'width' or 'diameter'.

        """
        if gdf_br.crs.is_geographic:  # needed for length and splitting
            gdf_br = gdf_br.to_crs(3857)

        self.logger.info("Adding/Filling branches attributes values")
        gdf_br = workflows.update_data_columns_attributes(
            gdf_br, defaults, brtype=br_type
        )

        # If specific spacing info from spacing_fn, update spacing attribute
        if spacing is not None:
            self.logger.info(f"Updating spacing attributes")
            gdf_br = workflows.update_data_columns_attribute_from_query(
                gdf_br, spacing, attribute_name="spacing"
            )
        # Line smoothing for pipes
        smooth_branches = br_type == "pipe"

        self.logger.info(f"Processing branches")
        branches, branches_nodes = workflows.process_branches(
            gdf_br,
            branch_nodes=None,
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
        channels_defaults_fn : str, optional
            Path to a csv file containing all defaults values per 'branchType'.
            Default is None.
        spacing_fn : str, optional
            Path to a csv file containing spacing values per 'branchType', 'shape', 'width' or 'diameter'.
            Default is None.
        snap_offset : float, optional
            Maximum distance between branch end points. If the distance is larger, they are not snapped.
            Default is 0.0.
        allow_intersection_snapping : bool, optional
            Allow snapping at all branch ends, including intersections.
            Default is True.
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

            # setup geoms #TODO do we still need channels?
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

        # Add defaults
        # Add branchType and branchId attributes if does not exist
        if "branchType" not in gdf_riv.columns:
            gdf_riv["branchType"] = pd.Series(
                data=np.repeat("river", len(gdf_riv)), index=gdf_riv.index, dtype=str
            )
        if "branchId" not in gdf_riv.columns:
            data = [f"river_{i}" for i in np.arange(1, len(gdf_riv) + 1)]
            gdf_riv["branchId"] = pd.Series(data, index=gdf_riv.index, dtype=str)

        # assign id
        id_col = "branchId"
        gdf_riv.index = gdf_riv[id_col]
        gdf_riv.index.name = id_col

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
        allowed_columns = set(_allowed_columns).intersection(gdf_riv.columns)
        gdf_riv = gpd.GeoDataFrame(gdf_riv[allowed_columns], crs=gdf_riv.crs)

        # Add friction to defaults
        defaults["frictionType"] = friction_type
        defaults["frictionValue"] = friction_value

        # Build the rivers branches and nodes and fill with attributes and spacing
        rivers, river_nodes = self._setup_branches(
            gdf_br=gdf_riv,
            defaults=defaults,
            br_type="river",
            spacing=None,  # does not allow spacing for rivers
            snap_offset=0.0,
            allow_intersection_snapping=True,
        )

        # Add friction_id column based on {friction_type}_{friction_value}
        rivers["frictionId"] = [
            f"{ftype}_{fvalue}"
            for ftype, fvalue in zip(rivers["frictionType"], rivers["frictionValue"])
        ]

        # setup crosssections
        crosssections = self._setup_crosssections(
            branches=rivers,
            crosssections_fn=None,
            crosssections_type="branch",
        )

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
        friction_type: str = "Manning",  # what about constructing friction_defaults_fn?
        friction_value: float = 0.023,
        crosssections_fn: str = None,
        crosssections_type: str = None,
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
        ``crosssections_fn`` values are derived at the centroid of each river line based on defaults.

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
        crosssections_fn : str or Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyzpoints"
            * Required variables: [crsId, order, z]
            If ``crosssections_type`` = "point"
            * Required variables: [crsId, shape, shift]
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
        self.logger.info(f"Preparing 1D rivers.")

        # Read the rivers data
        gdf_riv = self.data_catalog.get_geodataframe(
            rivers_fn, geom=self.region, buffer=0, predicate="intersects"
        )
        # Filter features based on river_filter
        if river_filter is not None and "branchType" in gdf_riv.columns:
            gdf_riv = gdf_riv[gdf_riv["branchType"] == river_filter]
        # Check if features in region
        if len(gdf_riv) == 0:
            self.logger.warning(
                f"No {rivers_fn} 1D river locations found within domain"
            )
            return None

        # Add branchType and branchId attributes if does not exist
        if "branchType" not in gdf_riv.columns:
            gdf_riv["branchType"] = pd.Series(
                data=np.repeat("river", len(gdf_riv)), index=gdf_riv.index, dtype=str
            )
        if "branchId" not in gdf_riv.columns:
            data = [f"river_{i}" for i in np.arange(1, len(gdf_riv) + 1)]
            gdf_riv["branchId"] = pd.Series(data, index=gdf_riv.index, dtype=str)

        # assign id
        id_col = "branchId"
        gdf_riv.index = gdf_riv[id_col]
        gdf_riv.index.name = id_col

        # assign default attributes
        if rivers_defaults_fn is None or not rivers_defaults_fn.is_file():
            self.logger.warning(
                f"rivers_defaults_fn ({rivers_defaults_fn}) does not exist. Fall back choice to defaults. "
            )
            rivers_defaults_fn = Path(self._DATADIR).joinpath(
                "rivers", "rivers_defaults.csv"
            )
        defaults = pd.read_csv(rivers_defaults_fn)
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
        allowed_columns = set(_allowed_columns).intersection(gdf_riv.columns)
        gdf_riv = gpd.GeoDataFrame(gdf_riv[allowed_columns], crs=gdf_riv.crs)

        # Add friction to defaults
        defaults["frictionType"] = friction_type
        defaults["frictionValue"] = friction_value

        # Build the rivers branches and nodes and fill with attributes and spacing
        rivers, river_nodes = self._setup_branches(
            gdf_br=gdf_riv,
            defaults=defaults,
            br_type="river",
            spacing=None,  # does not allow spacing for rivers
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
        )

        # Add friction_id column based on {friction_type}_{friction_value}
        rivers["frictionId"] = [
            f"{ftype}_{fvalue}"
            for ftype, fvalue in zip(rivers["frictionType"], rivers["frictionValue"])
        ]

        # setup crosssections
        if crosssections_type is None:
            crosssections_type = "branch"  # TODO: maybe assign a specific one for river, like branch_river
        assert {crosssections_type}.issubset({"xyzpoints", "point", "branch"})
        crosssections = self._setup_crosssections(
            branches=rivers,
            crosssections_fn=crosssections_fn,
            crosssections_type=crosssections_type,
        )

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
            # FIXME: dem method can have remaining nan values. For now no interpolation method is used for filling in nan value. Use ``pipes_invlev`` to further fill in nodata.
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

        # Read the pipes data
        gdf_pipe = self.data_catalog.get_geodataframe(
            pipes_fn, geom=self.region, buffer=0, predicate="contains"
        )
        # reproject
        gdf_pipe = gdf_pipe.to_crs(self.crs)

        # Filter features based on pipe_filter
        if pipe_filter is not None and "branchType" in gdf_pipe.columns:
            gdf_pipe = gdf_pipe[gdf_pipe["branchType"] == pipe_filter]
        # Check if features in region
        if len(gdf_pipe) == 0:
            self.logger.warning(f"No {pipes_fn} pipe locations found within domain")
            return None

        # Add branchType and branchId attributes if does not exist
        if "branchType" not in gdf_pipe.columns:
            gdf_pipe["branchType"] = pd.Series(
                data=np.repeat("pipe", len(gdf_pipe)), index=gdf_pipe.index, dtype=str
            )
        if "branchId" not in gdf_pipe.columns:
            data = [f"pipe_{i}" for i in np.arange(1, len(gdf_pipe) + 1)]
            gdf_pipe["branchId"] = pd.Series(data, index=gdf_pipe.index, dtype=str)

        # assign id
        id_col = "branchId"
        gdf_pipe.index = gdf_pipe[id_col]
        gdf_pipe.index.name = id_col

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
        allowed_columns = set(_allowed_columns).intersection(gdf_pipe.columns)
        gdf_pipe = gpd.GeoDataFrame(gdf_pipe[allowed_columns], crs=gdf_pipe.crs)

        # assign default attributes
        if pipes_defaults_fn is None or not pipes_defaults_fn.is_file():
            self.logger.warning(
                f"pipes_defaults_fn ({pipes_defaults_fn}) does not exist. Fall back choice to defaults. "
            )
            pipes_defaults_fn = Path(self._DATADIR).joinpath(
                "pipes", "pipes_defaults.csv"
            )
        defaults = pd.read_csv(pipes_defaults_fn)
        self.logger.info(f"pipe default settings read from {pipes_defaults_fn}.")

        # Add spacing o defaults
        defaults["spacing"] = spacing

        # add friction to defaults
        defaults["frictionType"] = friction_type
        defaults["frictionValue"] = friction_value

        # Add crosssections to defaults
        if crosssections_shape == "circle":
            if isinstance(crosssections_value, float):
                defaults["shape"] = crosssections_shape
                defaults["diameter"] = crosssections_value
            else:
                # TODO: warning or error?
                self.logger.warning(
                    "If crosssections_shape is circle, crosssections_value should be a single float for diameter. Skipping setup_pipes."
                )
                return
        elif crosssections_shape == "rectangle":
            if isinstance(crosssections_value, list) and len(crosssections_value) == 2:
                defaults["shape"] = crosssections_shape
                defaults["width"], defaults["height"] = crosssections_value
                defaults[
                    "closed"
                ] = "yes"  # default rectangle crosssection for pipes are closed
            else:
                # TODO: warning or error?
                self.logger.warning(
                    "If crosssections_shape is rectangle, crosssections_value should be a list with [width, height] values. Skipping setup_pipes."
                )
                return
        else:
            self.logger.warning(
                f"crosssections_shape {crosssections_shape} argument not understood. Should be one of [circle, rectangle]. Skipping setup_pipes"
            )
            return

        # Build the rivers branches and nodes and fill with attributes and spacing
        pipes, pipe_nodes = self._setup_branches(
            gdf_br=gdf_pipe,
            defaults=defaults,
            br_type="pipe",
            spacing=None,  # for now only single default value implemented, use "spacing" column
            snap_offset=snap_offset,
            allow_intersection_snapping=allow_intersection_snapping,
        )

        # setup roughness
        # Add friction_id column based on {friction_type}_{friction_value}
        pipes["frictionId"] = [
            f"{ftype}_{fvalue}"
            for ftype, fvalue in zip(pipes["frictionType"], pipes["frictionValue"])
        ]

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
        self._setup_crosssections(pipes, crosssections_type="branch", midpoint=False)

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
    ):
        """Prepares 1D crosssections.
        crosssections can be set from branches, points and xyzpoints, # TODO to be extended also from dem data for rivers/channels?
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
            * Optional variables: [material, friction_type, friction_value]
        crosssections_fn : str Path, optional # TODO: allow multiple crosssection filenames
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyzpoints"
            Note that only points within the region + 1000m buffer will be read.
            * Required variables: crsId, order, z
            * Optional variables:
            If ``crosssections_type`` = "point"
            * Required variables: crsId, shape, shift  #TODO: do we need frictions from crosssection functions?
            * Optional variables:
                if shape = 'rectangle': 'width', 'height', 'closed'
                if shape = 'trapezoid': 'width', 't_width', 'height', 'closed'
                if shape = 'yz': 'yzcount','ycoordinates','zcoordinates','closed'
                if shape = 'zw': 'numlevels', 'levels', 'flowwidths','totalwidths', 'closed'.
                if shape = 'zwRiver': Not Supported
                Note that list input must be strings seperated by a whitespace ''.
            By default None, crosssections will be set from branches
        crosssections_type : {'branch', 'xyz', 'point'}
            Type of crosssections read from crosssections_fn. One of ["xyzpoints"].
            By default `branch`.
        """

        # setup crosssections
        self.logger.info(f"Preparing 1D crosssections.")

        if crosssections_fn is None and crosssections_type == "branch":
            # TODO: set a seperate type for rivers because other branch types might require upstream/downstream

            # read crosssection from branches
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
            gdf_cs = workflows.set_point_crosssections(branches, gdf_cs)

        else:
            raise NotImplementedError(
                f"Method {crosssections_type} is not implemented."
            )

        # add crosssections to exisiting ones and update geoms
        self.logger.debug(f"Adding crosssections vector to geoms.")
        self.set_crosssections(gdf_cs)
        # TODO: sort out the crosssections, e.g. remove branch crosssections if point/xyz exist etc
        # TODO: setup river crosssections, set contrains based on branch types

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
        #FIXME the above will change once auxmaps are implemented from hydromt.
        #TODO probably needs another parameter to apply different samplinf method for the manholes, e.g. min within 2 m radius.

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
            # FIXME: dem method can have remaining nan values. For now no interpolation method is used for filling in nan value. Use ``pipes_invlev`` to further fill in nodata.
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
        timeseries. Only locations within the model region + 10m are selected. They are snapped to the model
        boundary locations within a max distance defined in ``snap_offset``. If ``boundaries_geodataset_fn``
        has missing values, the constant ``boundary_value`` will be used.

        The dataset/timeseries are clipped to the model time based on the model config
        tstart and tstop entries.

        Adds/Updates model layers:
            * **{boundary_type}bnd_{branch_type}** forcing: 1D boundaries DataArray

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
                geom=self.region,
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
        self.set_forcing(da_out, name=f"{da_out.name}_{branch_type}")

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
        (3) Validation checks have been added to check if the mesh extent is within model region.
        (4) Only existing meshed with only 2D grid can be read.
        (5) 1D2D network files are not supported as mesh2d_fn.

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
            # raise ValueError(
            #    "At least one argument of mesh2d_fn, geom_fn or bbox must be provided."
            # )
            region = {"geom": self.region}
        # Get the 2dmesh TODO when ready with generation, pass other arg like resolution
        mesh2d = super().setup_mesh(region=region, crs=self.crs, res=res)
        # Check if intersects with region
        xmin, ymin, xmax, ymax = self.bounds
        subset = mesh2d.ugrid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        err = f"RasterDataset: No data within spatial domain for mesh."
        if subset.grid.node_x.size == 0 or subset.grid.node_y.size == 0:
            raise IndexError(err)
        # TODO: if we want to keep the clipped mesh 2d uncomment the following line
        # Else mesh2d is used as mesh instead of susbet
        self._mesh = subset  # reinitialise mesh2d grid (set_mesh is used in super)

    def setup_auxmaps_from_raster(
        self,
        raster_fn: str,
        variables: Optional[list] = None,
        fill_method: Optional[str] = None,
        interpolation_method: Optional[str] = "triangulation",
        locationtype: Optional[str] = "2d",
        name: Optional[str] = None,
        split_dataset: Optional[bool] = False,
    ) -> None:
        """
        This component adds data variable(s) from ``raster_fn`` to auxmaps object.

        If raster is a dataset, all variables will be added unless ``variables`` list is specified.

        Adds model layers:

        * **raster.name** auxmaps: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog.
        variables: list, optional
            List of variables to add to auxmaps from raster_fn. By default all.
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method. Available methods
            are ['linear', 'nearest', 'cubic', 'rio_idw'].
        interpolation_method : str, optional
            Interpolation method for DFlow-FM. By default triangulation. Except for waterlevel and
            waterdepth then the default is mean.
            Available methods: ['triangulation', 'mean', 'nearestNb', 'max', 'min', 'invDist', 'minAbs', 'median']
        locationtype : str, optional
            LocationType in initial fields. Either 2d (default), 1d or all.
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset, either add it as is to auxmaps or split it into several xarray.DataArrays.
        """
        # Call super method
        ds = super().setup_auxmaps_from_raster(
            raster_fn=raster_fn,
            variables=variables,
            fill_method=fill_method,
            name=name,
            split_dataset=split_dataset,
        )

        # Updates the interpolation method in self._AUXMAPS
        if variables is None:
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset()
            variables = ds.data_vars
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
            if var in self._AUXMAPS:
                self._AUXMAPS[var]["interpolation"] = interpolation_method
                self._AUXMAPS[var]["locationtype"] = locationtype

    def setup_auxmaps_from_rastermapping(
        self,
        raster_fn: str,
        raster_mapping_fn: str,
        mapping_variables: list,
        fill_method: Optional[str] = None,
        interpolation_method: Optional[str] = "triangulation",
        locationtype: Optional[str] = "2d",
        name: Optional[str] = None,
        split_dataset: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        This component adds data variable(s) to auxmaps object by combining values in ``raster_mapping_fn`` to
        spatial layer ``raster_fn``. The ``mapping_variables`` rasters are first created by mapping variables values
        from ``raster_mapping_fn`` to value in the ``raster_fn`` grid.

        Adds model layers:
        * **mapping_variables** auxmaps: data from raster_mapping_fn spatially ditributed with raster_fn
        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog. Should be a DataArray. Else use **kwargs to select
            variables/time_tuple in hydromt.data_catalog.get_rasterdataset method
        raster_mapping_fn: str
            Source name of mapping table of raster_fn in data_catalog.
        mapping_variables: list
            List of mapping_variables from raster_mapping_fn table to add to mesh. Index column should match values
            in raster_fn.
        fill_method : str, optional
            If specified, fills no data values using fill_nodata method. Available methods
            are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        interpolation_method : str, optional
            Interpolation method for DFlow-FM. By default triangulation. Except for waterlevel and waterdepth then
            the default is mean.
            Available methods: ['triangulation', 'mean', 'nearestNb', 'max', 'min', 'invDist', 'minAbs', 'median']
        locationtype : str, optional
            LocationType in initial fields. Either 2d (default), 1d or all.
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset, either add it as is to auxmaps or split it into several xarray.DataArrays.
        """
        # Call super method
        ds = super().setup_auxmaps_from_rastermapping(
            raster_fn=raster_fn,
            raster_mapping_fn=raster_mapping_fn,
            mapping_variables=mapping_variables,
            fill_method=fill_method,
            name=name,
            split_dataset=split_dataset,
        )

        # Updates the interpolation method in self._AUXMAPS
        if mapping_variables is None:
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset()
            mapping_variables = ds.data_vars
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
        for var in mapping_variables:
            if var in self._AUXMAPS:
                self._AUXMAPS[var]["interpolation"] = interpolation_method
                self._AUXMAPS[var]["locationtype"] = locationtype

    # ## I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        self.read_dimr()
        self.read_config()
        self.read_auxmaps()
        self.read_geoms()
        self.read_mesh(fn="mesh/FlowFM_2D_net.nc")
        self.read_dfmmodel()

    def write(self):  # complete model
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return

        if self.dimr:
            self.write_dimr()
        if self.config:  # try to read default if not yet set
            self.write_config()
        if self._auxmaps:
            self.write_auxmaps()
        if self._geoms:
            self.write_geoms()
        if self._mesh:
            self.write_mesh(fn="mesh/FlowFM_2D_net.nc")
        if self._forcing:
            self.write_forcing()
        if self.dfmmodel:
            self.write_dfmmodel()
        self.write_data_catalog()

    def read_auxmaps(self) -> None:
        """Read auxmaps at <root/?/> and parse to dict of xr.DataArray"""
        return self._auxmaps
        # raise NotImplementedError()

    def write_auxmaps(self) -> None:
        """Write auxmaps as tif files in auxmaps folder and update initial fields"""
        # Global parameters
        auxroot = join(self.root, "auxmaps")
        inilist = []
        paramlist = []
        self.logger.info(f"Writing auxmaps files to {auxroot}")

        def _prepare_inifields(da_dict, da):
            # Write tif files
            name = da_dict["name"]
            type = da_dict["initype"]
            interp_method = da_dict["interpolation"]
            locationtype = da_dict["locationtype"]
            _fn = join(auxroot, f"{name}.tif")
            if np.isnan(da.raster.nodata):
                da.raster.set_nodata(-999)
            da.raster.to_raster(_fn)
            # Prepare dict
            if interp_method == "triangulation":
                inidict = {
                    "quantity": name,
                    "dataFile": f"../auxmaps/{name}.tif",
                    "dataFileType": "GeoTIFF",
                    "interpolationMethod": interp_method,
                    "operand": "O",
                    "locationType": locationtype,
                }
            else:  # averaging
                inidict = {
                    "quantity": name,
                    "dataFile": f"../auxmaps/{name}.tif",
                    "dataFileType": "GeoTIFF",
                    "interpolationMethod": "averaging",
                    "averagingType": interp_method,
                    "operand": "O",
                    "locationType": locationtype,
                }
            if type == "initial":
                inilist.append(inidict)
            elif type == "parameter":
                paramlist.append(inidict)

        # Only write auxmaps that are listed in self._AUXMAPS, rename tif on the fly
        # TODO raise value error if both waterdepth and waterlevel are given as auxmaps.items
        for name, ds in self._auxmaps.items():
            if isinstance(ds, xr.DataArray):
                if name in self._AUXMAPS:
                    _prepare_inifields(self._AUXMAPS[name], ds)
                    # update config if frcition
                    if "frictype" in self._AUXMAPS[name]:
                        self.set_config(
                            "physics.UniFrictType", self._AUXMAPS[name]["frictype"]
                        )
            elif isinstance(ds, xr.Dataset):
                for v in ds.data_vars:
                    if v in self._AUXMAPS:
                        _prepare_inifields(self._AUXMAPS[v], ds[v])
                        # update config if frcition
                        if "frictype" in self._AUXMAPS[name]:
                            self.set_config(
                                "physics.UniFrictType", self._AUXMAPS[name]["frictype"]
                            )
        # rewrite config
        self.write_config()

        # Assign initial fields to model and write
        inifield_model = IniFieldModel(initial=inilist, parameter=paramlist)
        inifield_model_filename = inifield_model._filename() + ".ini"
        self.dfmmodel.geometry.inifieldfile = inifield_model
        self.dfmmodel.geometry.inifieldfile.save(
            self.dfmmodel.filepath.with_name(inifield_model_filename),
            recurse=False,
        )
        # save relative path to mdu
        self.dfmmodel.geometry.inifieldfile.filepath = inifield_model_filename

    def read_geoms(self) -> None:
        """Read geoms at <root/?/> and parse to dict of geopandas"""
        if not self._write:
            # start fresh in read-only mode
            self._geoms = dict()
        for fn in glob.glob(join(self.root, "*.xy")):
            name = basename(fn).replace(".xy", "")
            geom = hydromt.open_vector(fn, driver="xy", crs=self.crs)
            self.set_geoms(geom, name)

    def write_geoms(self) -> None:  # write_all()
        """Write geoms at <root/?/> in model ready format"""
        # TODO: write_data_catalogue with updates of the rename based on mapping table?
        if not self._write:
            raise IOError("Model opened in read-only mode")
        for name, gdf in self.geoms.items():
            fn_out = join(self.root, "geoms", f"{name}.geojson")
            gdf.reset_index(drop=True).to_file(
                fn_out, driver="GeoJSON"
            )  # FIXME: does not work if does not reset index

    def read_forcing(self) -> None:
        """Read forcing at <root/?/> and parse to dict of xr.DataArray"""
        return self._forcing
        # raise NotImplementedError()

    def write_forcing(self) -> None:
        """write forcing into hydrolib-core ext and forcing models"""
        extdict = list()
        bcdict = list()
        # Loop over forcing dict
        for name, da in self.forcing.items():
            for i in da.index.values:
                bc = da.attrs.copy()
                # Boundary
                ext = dict()
                ext["quantity"] = bc["quantity"]
                ext["nodeId"] = i
                extdict.append(ext)
                # Forcing
                bc["name"] = i
                if bc["function"] == "constant":
                    # one quantityunitpair
                    bc["quantityunitpair"] = [tuple((da.name, bc["units"]))]
                    # only one value column (no times)
                    bc["datablock"] = [[da.sel(index=i).values.item()]]
                else:
                    # two quantityunitpair
                    bc["quantityunitpair"] = [
                        tuple(("time", bc["time_unit"])),
                        tuple((da.name, bc["units"])),
                    ]
                    bc.pop("time_unit")
                    # time/value datablock
                    bc["datablock"] = [
                        [t, x] for t, x in zip(da.time.values, da.sel(index=i).values)
                    ]
                bc.pop("quantity")
                bc.pop("units")
                bcdict.append(bc)

        forcing_model = ForcingModel(forcing=bcdict)
        forcing_model_filename = forcing_model._filename() + ".bc"

        ext_model = ExtModel()
        ext_model_filename = ext_model._filename() + ".ext"
        for i in range(len(extdict)):
            ext_model.boundary.append(
                Boundary(**{**extdict[i], "forcingFile": forcing_model})
            )
        # assign to model
        self.dfmmodel.external_forcing.extforcefilenew = ext_model
        self.dfmmodel.external_forcing.extforcefilenew.save(
            self.dfmmodel.filepath.with_name(ext_model_filename), recurse=True
        )
        # save relative path to mdu
        self.dfmmodel.external_forcing.extforcefilenew.filepath = ext_model_filename

    def read_dfmmodel(self):
        """Read dfmmodel at <root/?/> and parse to model class (deflt3dfmpy)"""
        pass
        # raise NotImplementedError()

    def write_dfmmodel(self):
        """Write dfmmodel at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        self.logger.info(f"Writing dfmmodel in {self.root}")
        # write 1D mesh
        # self._write_mesh1d()  # FIXME None handling
        # write 2d mesh
        if self._mesh:
            self._write_mesh2d()
        # TODO: create self._write_mesh2d() using hydrolib-core funcitonalities
        # write branches
        if self.pipes:
            self._write_branches()
        # write friction
        self._write_friction()  # FIXME: ask Rinske, add global section correctly
        # write crosssections
        self._write_crosssections()  # FIXME None handling, if there are no crosssections
        # write manholes
        if "manholes" in self._geoms:
            self._write_manholes()

        # save model
        self.dfmmodel.save(recurse=True)

    def _write_mesh2d(self):
        """
        TODO: write docstring

        :return:
        """
        # Get meshkernel Mesh2d objec
        mesh2d = self._mesh.ugrid.grid.mesh

        # add mesh2d
        # FIXME: improve the way of adding a 2D mesh
        self.dfmmodel.geometry.netfile.network._mesh2d._process(mesh2d)

    def _write_branches(self):
        """write branches.gui
        #TODO combine with others"""
        branches = self._geoms["branches"][
            ["branchId", "branchType", "manhole_up", "manhole_dn"]
        ]
        branches = branches.rename(
            columns={
                "branchId": "name",
                "manhole_up": "sourceCompartmentName",
                "manhole_dn": "targetCompartmentName",
            }
        )
        branches["branchType"] = branches["branchType"].replace(
            {"river": 0, "channel": 0, "pipe": 2, "tunnel": 2, "sewerconnection": 1}
        )
        branchgui_model = BranchModel(branch=branches.to_dict("records"))
        branchgui_model.filepath = self.dfmmodel.filepath.with_name(
            branchgui_model._filename() + branchgui_model._ext()
        )
        branchgui_model.save()

    def _write_friction(self):

        #
        frictions = self._geoms["branches"][
            ["frictionId", "frictionValue", "frictionType"]
        ]
        frictions = frictions.drop_duplicates(subset="frictionId")

        self.dfmmodel.geometry.frictfile = []
        # create a new friction
        for i, row in frictions.iterrows():
            fric_model = FrictionModel(global_=row.to_dict())
            fric_model.filepath = f"roughness_{i}.ini"
            self.dfmmodel.geometry.frictfile.append(fric_model)

    def _write_crosssections(self):
        """write crosssections into hydrolib-core crsloc and crsdef objects"""

        # preprocessing for crosssections from geoms
        gpd_crs = self._geoms["crosssections"]

        # crsdef
        # get crsdef from crosssections gpd # FIXME: change this for update case
        gpd_crsdef = gpd_crs[[c for c in gpd_crs.columns if c.startswith("crsdef")]]
        gpd_crsdef = gpd_crsdef.rename(
            columns={c: c.split("_")[1] for c in gpd_crsdef.columns}
        )
        gpd_crsdef = gpd_crsdef.drop_duplicates(subset="id")
        crsdef = CrossDefModel(definition=gpd_crsdef.to_dict("records"))
        self.dfmmodel.geometry.crossdeffile = crsdef

        # crsloc
        # get crsloc from crosssections gpd # FIXME: change this for update case
        gpd_crsloc = gpd_crs[[c for c in gpd_crs.columns if c.startswith("crsloc")]]
        gpd_crsloc = gpd_crsloc.rename(
            columns={c: c.split("_")[1] for c in gpd_crsloc.columns}
        )

        crsloc = CrossLocModel(crosssection=gpd_crsloc.to_dict("records"))
        self.dfmmodel.geometry.crosslocfile = crsloc

    def _write_manholes(self):
        """write manholes into hydrolib-core storage nodes objects"""

        # preprocessing for manholes from geoms
        gpd_mh = self._geoms["manholes"]

        storagenodes = StorageNodeModel(storagenode=gpd_mh.to_dict("records"))
        self.dfmmodel.geometry.storagenodefile = storagenodes

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
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        return region

    @property
    def dfmmodel(self):
        if self._dfmmodel == None:
            self.init_dfmmodel()
        return self._dfmmodel

    def init_dfmmodel(self):
        # Create output directories
        outputdir = Path(self.root).joinpath("dflowfm")
        outputdir.mkdir(parents=True, exist_ok=True)
        # TODO: check that HydroMT already wrote the updated config
        # create a new MDU-Model
        self._dfmmodel = FMModel(filepath=Path(join(self.root, self._config_fn)))
        self._dfmmodel.geometry.netfile = NetworkModel()
        self._dfmmodel.geometry.netfile.filepath = (
            "fm_net.nc"  # because hydrolib.core writes this argument as absolute path
        )
        self._dfmmodel.geometry.crossdeffile = CrossDefModel()
        self._dfmmodel.geometry.crossdeffile.filepath = outputdir.joinpath("crsdef.ini")
        self._dfmmodel.geometry.crosslocfile = CrossLocModel()
        self._dfmmodel.geometry.crosslocfile.filepath = outputdir.joinpath("crsloc.ini")
        # self._dfmmodel.geometry.frictfile = [FrictionModel()]
        # self._dfmmodel.geometry.frictfile[0].filepath = outputdir.joinpath(
        #    "roughness.ini"
        # )

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
        if isfile(dimr_fn):
            self.logger.info(f"Reading dimr file at {dimr_fn}")
            dimr = DIMR(filepath=Path(dimr_fn))
        # else initialise
        else:
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
            self.logger.info("Adding dflofm component to dimr file")

            # update component
            components = self._dimr.component
            if len(components) != 0:
                components = (
                    []
                )  # FIXME: for now only support control single component of dflowfm
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
                controls = (
                    []
                )  # FIXME: for now only support control single component of dflowfm
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
        if self._branches is None:
            # self.read_branches() #not implemented yet
            self._branches = gpd.GeoDataFrame(crs=self.crs)
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
        self.set_geoms(gpd.GeoDataFrame(branches, crs=self.crs), "branches")

        self.logger.debug(f"Updating branches in network.")

    def add_branches(
        self,
        new_branches: gpd.GeoDataFrame,
        branchtype: str,
        node_distance: float = 40.0,
    ):
        """Add new branches of branchtype to the branches and mesh1d object"""

        snap_offset = self._network_snap_offset

        branches = self.branches.copy()

        # Check if "branchType" in new_branches column, else add
        if "branchType" not in new_branches.columns:
            new_branches["branchType"] = np.repeat(branchtype, len(new_branches.index))

        if len(self.opensystem) > 0:
            self.logger.info(
                f"snapping {branchtype} ends to exisiting network (opensystem only)"
            )

            # get possible connection points from new branches
            if branchtype in ["pipe", "tunnel"]:
                endnodes = workflows.generate_boundaries_from_branches(
                    new_branches, where="downstream"
                )  # FIXME: make generate_boundaries_from_branches function more available
            else:
                endnodes = workflows.generate_boundaries_from_branches(
                    new_branches, where="both"
                )

            # get possible connection points from exisiting open system
            mesh1d_nodes = self.mesh1d_nodes.copy()
            mesh1d_nodes_open = mesh1d_nodes.loc[
                mesh1d_nodes.branch_name.isin(self.opensystem.branchId.tolist())
            ]

            # snap the new to exisiting
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

        # Check if we need to do more check/process to make sure everything is well connected
        workflows.validate_branches(branches)

        # set geom and mesh1d
        self.set_branches(branches)
        self.set_mesh1d()

    def set_branches_component(self, name: str):
        gdf_comp = self.branches[self.branches["branchType"] == name]
        if gdf_comp.index.size > 0:
            self.set_geoms(gdf_comp, name=f"{name}s")
        return gdf_comp

    @property
    def rivers(self):
        if "rivers" in self.sgeoms:
            gdf = self.geoms["rivers"]
        else:
            gdf = self.set_branches_component("rivers")
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
            gdf = gpd.GeoDataFrame(crs=self.crs)
        return gdf

    @property
    def closedsystem(self):
        if len(self.branches) > 0:
            gdf = self.branches[self.branches["branchType"].isin(["pipe", "tunnel"])]
        else:
            gdf = gpd.GeoDataFrame(crs=self.crs)
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
        """update the mesh1d in hydrolib-core net object by overwrite and #TODO the xugrid mesh1d"""

        # init mesh1d
        self.dfmmodel.geometry.netfile.network._mesh1d = Mesh1d()

        # add open system mesh
        opensystem = self.opensystem
        node_distance = self._openwater_computation_node_distance
        mesh.mesh1d_add_branch(
            self.dfmmodel.geometry.netfile.network,
            opensystem.geometry.to_list(),
            node_distance=node_distance,
            branch_names=opensystem.branchId.to_list(),
            branch_orders=opensystem.branchOrder.to_list(),
        )

        # add closed system mesh
        closedsystem = self.closedsystem
        node_distance = np.inf
        mesh.mesh1d_add_branch(
            self.dfmmodel.geometry.netfile.network,
            closedsystem.geometry.to_list(),
            node_distance=node_distance,
            branch_names=closedsystem.branchId.to_list(),
            branch_orders=closedsystem.branchOrder.to_list(),
        )

    @property
    def crosssections(self):
        """Quick accessor to crosssections geoms"""
        if "crosssections" in self.geoms:
            gdf = self.geoms["crosssections"]
        else:
            gdf = gpd.GeoDataFrame(crs=self.crs)
        return gdf

    def set_crosssections(self, crosssections: gpd.GeoDataFrame):
        """Updates crosssections in geoms with new ones"""
        if len(self.crosssections) > 0:
            crosssections = gpd.GeoDataFrame(
                pd.concat([self.crosssections, crosssections]), crs=self.crs
            )
        self.set_geoms(crosssections, name="crosssections")

    @property
    def boundaries(self):
        """Quick accessor to boundaries geoms"""
        if "boundaries" in self.geoms:
            gdf = self.geoms["boundaries"]
        else:
            gdf = self.get_boundaries()
        return gdf

    def get_boundaries(self):
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
        refdate = datetime.strptime(
            str(self.get_config("time.RefDate")), "%Y%m%d"
        )  # FIXME: case senstivie might cause problem when changing template, consider use hydrolib.core reader for mdu files later.
        tstart = refdate + timedelta(seconds=float(self.get_config("time.TStart")))
        tstop = refdate + timedelta(seconds=float(self.get_config("time.TStop")))
        return refdate, tstart, tstop

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
