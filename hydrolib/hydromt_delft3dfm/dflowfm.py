"""Implement Delft3D-FM hydromt plugin model class"""

import glob
import logging
from os import times
from os.path import basename, isfile, join
from pathlib import Path
from typing import Union, Tuple

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import xugrid as xu
from hydromt.models import MeshModel
from hydromt.models.model_auxmaps import AuxmapsMixin
from rasterio.warp import transform_bounds
from shapely.geometry import box, Point
from datetime import datetime, timedelta

from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.friction.models import *
from hydrolib.core.io.ext.models import *
from hydrolib.core.io.bc.models import *
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.net.models import *
from hydrolib.core.io.inifield.models import IniFieldModel
from hydrolib.dhydamo.geometry import common, mesh, viz

from . import DATADIR
from .workflows import (
    generate_roughness,
    helper,
    process_branches,
    set_branch_crosssections,
    set_xyz_crosssections,
    update_data_columns_attribute_from_query,
    update_data_columns_attributes,
    validate_branches,
    generate_boundaries_from_branches,
    select_boundary_type,
    validate_boundaries,
    compute_boundary_values,
)

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(AuxmapsMixin, MeshModel):
    """API for Delft3D-FM models in HydroMT"""

    _NAME = "dflowfm"
    _CONF = "DFlowFM.mdu"
    _DATADIR = DATADIR
    _GEOMS = {}
    _AUXMAPS = {
        "elevtn": {"name": "bedlevel", "type": "initial"},
        "roughness_manning": {"name": "frictioncoefficient", "type": "parameter"},
    }
    _FOLDERS = ["dflowfm", "geoms", "mesh", "auxmaps"]
    _CLI_ARGS = {"region": "setup_region", "res": "setup_mesh2d"}
    _CATALOGS = join(_DATADIR, "parameters_data.yml")

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,
        data_libs=None,
        logger=logger,
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
        # model specific
        self._branches = None
        self._config_fn = (
            join("dflowfm", self._CONF) if config_fn is None else config_fn
        )
        self.data_catalog.from_yml(self._CATALOGS)
        self.write_config()  #  create the mdu file in order to initialise dfmmodedl properly and at correct output location
        self._dfmmodel = self.init_dfmmodel()

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
        region: dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
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

            # setup geoms #TODO do we still need channels?
            self.logger.debug(f"Adding branches and branch_nodes vector to geoms.")
            self.set_geoms(channels, "channels")
            self.set_geoms(channel_nodes, "channel_nodes")

            # add to branches
            self.add_branches(channels, branchtype="channel")

    def _check_gpd_attributes(
        self, gdf: gpd.GeoDataFrame, required_columns: list, raise_error: bool = False
    ):
        """check if the geodataframe contains all required columns

        Parameters
        ----------
        gdf : gpd.GeoDataFrame, required
            GeoDataFrame to be checked
        required_columns: list of strings, optional
            Check if the geodataframe contains all required columns
        raise_error: boolean, optional
            Raise error if the check failed
        """
        if not (set(required_columns).issubset(gdf.columns)):
            if raise_error:
                raise ValueError(
                    f"GeoDataFrame do not contains all required attributes: {required_columns}."
                )
            else:
                logger.warning(
                    f"GeoDataFrame do not contains all required attributes: {required_columns}."
                )
            return False
        return True

    def setup_rivers(
        self,
        rivers_fn: str,
        rivers_defaults_fn: str = None,
        snap_offset: float = 0.0,
        allow_intersection_snapping: bool = True,
        friction_type: str = "Manning",  # what about constructing friction_defaults_fn?
        friction_value: float = 0.023,
        crosssections_fn: str = None,
        crosssections_type: str = None,
    ):
        """Prepares the 1D rivers and adds to 1D branches.

        1D rivers must contain valid geometry, friction and crosssections.

        The river geometry is read from ``rivers_fn``. If defaults attributes
        [material, friction_type, friction_value] are not present in ``rivers_fn``,
        they are added from defaults values in ``rivers_defaults_fn``.

        The river friction is read from attributes [friction_type, friction_value]. Friction attributes are either taken
        from ``rivers_fn`` or filled in using ``friction_type`` and ``friction_value`` arguments.
        Note for now only branch friction or global friction is supported.

        Crosssections are read from ``crosssections_fn`` based on the ``crosssections_type``.

        Adds/Updates model layers:

        * **rivers** geom: 1D rivers vector
        * **branches** geom: 1D branches vector
        * **crosssections** geom: 1D crosssection vector

        Parameters
        ----------
        rivers_fn : str
            Name of data source for branches parameters, see data/data_sources.yml.
            Note only the lines that are within the region polygon + 10m buffer will be used.
            * Required variables: [branchId, branchType]
            * Optional variables: [material, friction_type, friction_value, branchOrder]
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
        friction_value : float, optional.
            Units corresponding to [friction_type] are ["Chézy C [m 1/2 /s]", "Manning n [s/m 1/3 ]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Nikuradse k_n [m]", "Strickler k_s [m 1/3 /s]", "De Bos-Bijkerk γ [1/s]"]
            Friction value. By default 0.023.
        crosssections_fn : str Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyzpoints"
            * Required variables: crsId, order, z
            * Optional variables:
            If ``crosssections_type`` = "points"
            * Required variables: crsId, order, z
            * Optional variables:
            By default None, crosssections will be set from branches
        crosssections_type : str, optional
            Type of crosssections read from crosssections_fn. One of ["xyzpoints"].
            By default None.

        See Also
        ----------
        dflowfm._setup_branches
        """
        self.logger.info(f"Preparing 1D rivers.")

        # Read the rivers data
        gdf_riv = self.data_catalog.get_geodataframe(
            rivers_fn, geom=self.region, buffer=10, predicate="contains"
        )

        # check if feature and attributes exist
        if len(gdf_riv) == 0:
            self.logger.warning(
                f"No {rivers_fn} 1D river locations found within domain"
            )
            return None
        valid_attributes = self._check_gpd_attributes(
            gdf_riv, required_columns=["branchId", "branchType"], raise_error=False
        )
        if not valid_attributes:
            self.logger.error(
                f"Required attributes [branchId, branchType] do not exist"
            )
            return None

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
            "branchOrder" "material",
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
        assert {crosssections_type}.issubset({"xyzpoints", "branch"})
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
        self.add_branches(rivers, branchtype="river")

    def _setup_crosssections(
        self, branches, crosssections_fn: str = None, crosssections_type: str = "branch"
    ):
        """Prepares 1D crosssections.
        crosssections can be set from branchs, xyzpoints, # TODO to be extended also from dem data for rivers/channels?
        Crosssection must only be used after friction has been setup.

        Crosssections are read from ``crosssections_fn``.
        Crosssection types of this file is read from ``crosssections_type``
        If ``crosssections_type`` = "xyzpoints":
            * Required variables: crsId, order, z
            * Optional variables:

        If ``crosssections_fn`` is not defined, default method is ``crosssections_type`` = 'branch',
        meaning that branch attributes will be used to derive regular crosssections.
        The required attributes for this method are:
        for rivers: [branchId, branchType,branchOrder,shape,diameter,width,t_width,height,bedlev,closed,friction_type,friction_value]
        # TODO for pipes:

        Adds/Updates model layers:
        * **crosssections** geom: 1D crosssection vector

        Parameters
        ----------
        branches : gpd.GeoDataFrame
            geodataframe of the branches to apply crosssections.
            * Required variables: [branchId, branchType, branchOrder]
            * Optional variables: [material, friction_type, friction_value]
        crosssections_fn : str Path, optional
            Name of data source for crosssections, see data/data_sources.yml.
            If ``crosssections_type`` = "xyzpoints"
            Note that only points within the region + 1000m buffer will be read.
            * Required variables: crsId, order, z
            * Optional variables:
            If ``crosssections_type`` = "points"
            * Required variables: crsId, order, z
            * Optional variables:
            By default None, crosssections will be set from branches
        crosssections_type : str, optional
            Type of crosssections read from crosssections_fn. One of ["xyzpoints"].
            By default None.
        """

        # setup crosssections
        self.logger.info(f"Preparing 1D crosssections.")

        # TODO: allow multiple crosssection filenamess

        if crosssections_fn is None and crosssections_type == "branch":
            # TODO: set a seperate type for rivers because other branch types might require upstream/downstream

            # read crosssection from branches
            gdf_cs = set_branch_crosssections(branches)

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
            valid_attributes = self._check_gpd_attributes(
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
            gdf_cs = set_xyz_crosssections(branches, gdf_cs)

        elif crosssections_type == "point":
            # add setup point crosssections here
            raise NotImplementedError(
                f"Method {crosssections_type} is not implemented."
            )
        else:
            raise NotImplementedError(
                f"Method {crosssections_type} is not implemented."
            )

        # add crosssections to exisiting ones and update geoms
        self.logger.debug(f"Adding crosssections vector to geoms.")
        self.set_crosssections(gdf_cs)
        # TODO: sort out the crosssections, e.g. remove branch crosssections if point/xyz exist etc
        # TODO: setup river crosssections, set contrains based on branch types
    
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
        boundaries_geodataset_fn: str, Path
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
        boundaries_branch_type = select_boundary_type(
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
        da_out = compute_boundary_values(
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
        mesh2d_fn: str = None,
        geom_fn: str = None,
        bbox: list = None,
        res: float = 100.0,
    ):
        """Creates an 2D unstructured mesh or prepares an existing 2D mesh according UGRID conventions.

        An 2D unstructured mesh will be created as 2D rectangular grid from a geometry (geom_fn) or bbox. If an existing
        2D mesh is given, then no new mesh will be generated

        2D mesh contains ...

        Note that:
        (1) Refinement of the mesh is a seperate setup function, however a refined existing grid (mesh_fn) can already
        be read.
        (2) If no geometry, bbox or existing grid is specified for this setup function, then the self.region is used as
         mesh extent to generate the unstructured mesh.
        (3) Validation checks have been added to check if the mesh extent is within model region.
        (4) Only existing meshed with only 2D grid can be read.
        #FIXME: read existing 1D2D network file and extract 2D part.

        Adds/Updates model layers:

        * **1D2D links** geom: By any changes in 2D grid

        Parameters
        ----------
        mesh2D_fn : str Path, optional
            Name of data source for an existing unstructured 2D mesh
        geom_fn : str Path, optional
            Path to a polygon used to generate unstructured 2D mesh
        bbox: list, optional
            Describing the mesh extent of interest [xmin, ymin, xmax, ymax].
        res: float, optional
            Resolution used to generate 2D mesh. By default a value of 100 m is applied.

    	Raises
        ------
        ValueError
            If `mesh2d_fn`, `geom_fn` and `bbox` are all None.
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
            region = {"bbox": bbox}
        else:
            raise ValueError(
                "At least one argument of mesh2d_fn, geom_fn or bbox must be provided."
            )
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

    # ## I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
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
            type = da_dict["type"]
            _fn = join(auxroot, f"{name}.tif")
            if np.isnan(da.raster.nodata):
                da.raster.set_nodata(-999)
            da.raster.to_raster(_fn)
            # Prepare dict
            inidict = {
                "quantity": name,
                "dataFile": f"../auxmaps/{name}.tif",
                "dataFileType": "GeoTIFF",
                "interpolationMethod": "triangulation",
                # "averagingType": "nearestNb",
                "operand": "O",
                "locationType": "2d",
            }
            if type == "initial":
                inilist.append(inidict)
            elif type == "parameter":
                paramlist.append(inidict)

        # Only write auxmaps that are listed in self._AUXMAPS, rename tif on the fly
        for name, ds in self._auxmaps.items():
            if isinstance(ds, xr.DataArray):
                if name in self._AUXMAPS:
                    _prepare_inifields(self._AUXMAPS[name], ds)
            elif isinstance(ds, xr.Dataset):
                for v in ds.data_vars:
                    if v in self._AUXMAPS:
                        _prepare_inifields(self._AUXMAPS[v], ds[v])

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
            self.geoms[name].reset_index(drop=True).to_file(
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
        self._write_mesh2d()
        # TODO: create self._write_mesh2d() using hydrolib-core funcitonalities

        # write friction
        self._write_friction()  # FIXME: ask Rinske, add global section correctly

        # write crosssections
        self._write_crosssections()  # FIXME None handling, if there are no crosssections

        # save model
        self.dfmmodel.save(recurse=True)

    def _write_mesh1d(self):

        #
        branches = self._geoms["branches"]

        # FIXME: imporve the None handeling here, ref: crosssections

        # add mesh
        mesh.mesh1d_add_branch(
            self.dfmmodel.geometry.netfile.network,
            branches.geometry.to_list(),
            node_distance=40,
            branch_names=branches.branchId.to_list(),
            branch_orders=branches.branchOrder.to_list(),
        )

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

    def _write_friction(self):

        #
        frictions = self._geoms["branches"][
            ["frictionId", "frictionValue", "frictionType"]
        ]
        frictions = frictions.drop_duplicates(subset="frictionId")

        # create a new friction
        fric_model = FrictionModel(global_=frictions.to_dict("record"))
        self.dfmmodel.geometry.frictfile[0] = fric_model

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
        self._dfmmodel.geometry.frictfile = [FrictionModel()]
        self._dfmmodel.geometry.frictfile[0].filepath = outputdir.joinpath(
            "roughness.ini"
        )

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
        _ = self.set_branches_component(name="channel")
        _ = self.set_branches_component(name="pipe")

        # update geom
        self.logger.debug(f"Adding branches vector to geoms.")
        self.set_geoms(gpd.GeoDataFrame(branches, crs=self.crs), "branches")

        self.logger.debug(f"Updating branches in network.")
        mesh.mesh1d_add_branch(
            self.dfmmodel.geometry.netfile.network,
            self.staticgeoms["branches"].geometry.to_list(),
            node_distance=40,
            branch_names=self.staticgeoms["branches"].branchId.to_list(),
            branch_orders=self.staticgeoms["branches"].branchOrder.to_list(),
        )

    def add_branches(self, new_branches: gpd.GeoDataFrame, branchtype: str):
        """Add new branches of branchtype to the branches object"""
        branches = self.branches.copy()
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
            self.set_geoms(gdf_comp, name=f"{name}s")
        return gdf_comp

    @property
    def rivers(self):
        if "rivers" in self.staticgeoms:
            gdf = self.staticgeoms["rivers"]
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
        gdf = self.branches[self.branches["branchType"].isin(["river", "channel"])]
        return gdf

    @property
    def closedsystem(self):
        gdf = self.branches[self.branches["branchType"].isin(["pipe", "tunnel"])]
        return gdf

    @property
    def crosssections(self):
        """Quick accessor to crosssections geoms"""
        if "crosssections" in self.geoms:
            gdf = self.geoms["crosssections"]
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    def set_crosssections(self, crosssections: gpd.GeoDataFrame):
        """Updates crosssections in geoms with new ones"""
        if len(self.crosssections) > 0:
            crosssections = gpd.GeoDataFrame(
                pd.concat([self.crosssections, crosssections])
            )
        self.set_geoms(crosssections, name="crosssections")

    @property
    def boundaries(self):
        """Quick accessor to boundaries staticgeoms"""
        if "boundaries" in self.staticgeoms:
            gdf = self.staticgeoms["boundaries"]
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
        _boundaries = generate_boundaries_from_branches(self.branches, where="both")

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
        boundaries = hydromt.gis_utils.nearest_merge(
            _boundaries, _network1d_nodes, max_dist=0.1, overwrite=False
        )
        return boundaries

    def set_boundaries(self, boundaries: gpd.GeoDataFrame):
        """Updates boundaries in staticgeoms with new ones"""
        if len(self.boundaries) > 0:
            task_last = lambda s1, s2: s2
            boundaries = self.boundaries.combine(
                boundaries, func=task_last, overwrite=True
            )
        self.set_staticgeoms(boundaries, name="boundaries")

    def get_model_time(self):
        """Return (refdate, tstart, tstop) tuple with parsed model reference datem start and end time"""
        refdate = datetime.strptime(str(self.get_config("Time.refDate")), "%Y%m%d")
        tstart = refdate + timedelta(seconds=float(self.get_config("Time.tStart")))
        tstop = refdate + timedelta(seconds=float(self.get_config("Time.tStop")))
        return refdate, tstart, tstop
