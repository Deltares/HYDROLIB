"""Implement plugin model class"""

import glob
import logging
from os.path import basename, isfile, join
from pathlib import Path
from typing import Union

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import xugrid as xu
from hydromt import gis_utils, io, raster
from hydromt.models.model_api import Model
from rasterio.warp import transform_bounds
from shapely.geometry import box

from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.friction.models import *
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.net.models import *
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
)

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(Model):
    """General and basic API for models in HydroMT"""

    # FIXME
    _NAME = "dflowfm"
    _CONF = "DFlowFM.mdu"
    _DATADIR = DATADIR
    # TODO change below mapping table (hydrolib-core convention:shape file convention) to be read from data folder, maybe similar to _intbl for wflow
    # TODO: we also need one reverse table to read from static geom back. maybe a dictionary of data frame is better?
    # TODO: write static geom as geojson dataset, so that we dont get limitation for the 10 characters
    _GEOMS = {}  # FIXME Mapping from hydromt names to model specific names
    _MAPS = {}  # FIXME Mapping from hydromt names to model specific names
    _FOLDERS = ["dflowfm", "staticgeoms", "mesh"]

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
        self._dfmmodel = None
        self._branches = gpd.GeoDataFrame()
        self._mesh = None
        self._config_fn = (
            join("dflowfm", self._CONF) if config_fn is None else config_fn
        )

    def setup_basemaps(
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
        self.set_staticgeoms(geom, "region")

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

        # setup staticgeoms #TODO do we still need channels?
        self.logger.debug(f"Adding rivers and river_nodes vector to staticgeoms.")
        self.set_staticgeoms(rivers, "rivers")
        self.set_staticgeoms(river_nodes, "rivers_nodes")

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
        # if 'crosssections' in self.staticgeoms.keys():
        #    crosssections = self._staticgeoms['crosssections']
        # else:
        #    crosssections = gpd.GeoDataFrame()

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

        # add crosssections to exisiting ones and update staticgeoms
        self.logger.debug(f"Adding crosssections vector to staticgeoms.")
        self.set_crosssections(gdf_cs)
        # TODO: sort out the crosssections, e.g. remove branch crosssections if point/xyz exist etc
        # TODO: setup river crosssections, set contrains based on branch types

    def setup_mesh2d(
        self,
        mesh2d_fn : str = None,
        geom_fn : str = None,
        bbox : list = None,
        resolution : float = 100.0
    ):
        """ Creates an 2D unstructured mesh or prepares an existing 2D mesh according UGRID conventions.

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
        resolution: float, optional
            Resolution used to generate 2D mesh. By default a value of 100 m is applied.

        See Also
        ----------
        
        """
        self.logger.info(f"Preparing 2D mesh.")

        if mesh2d_fn != None:
            self.logger.info(f"An existing 2D grid is used to prepare 2D mesh.")
            try:
                mesh2d = xu.open_dataset(mesh2d_fn)
            except:
                raise NotImplementedError(f"{mesh2d_fn} cannot be opened. Please check if the existing grid is "
                                  f"an 2D mesh and not 1D2D mesh. This option is not yet available for 1D2D meshes.")
                # TODO: read existing 1D2D network file and extract 2D part.
            self.set_mesh(mesh2d)




            #TODO: deviate between the use of existing grid or grid generation
            #TODO: Check location of existing grid
            #TODO: geom_fn --> check if within region and with the same crs--> check if it is not empty after reading it
            #TODO: bbox  --> generatie geometry and check if within regionp--> check if it is not empty after reading it
            #TODO: Use region
            #TODO: generate grid
        #TODO: set as mesh
        a=1

    # ## I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.logger.info(f"Reading model data from {self.root}")
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_mesh()
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
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._mesh:
            self.write_mesh()
        if self.dfmmodel:
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
            self.staticgeoms[name].reset_index(drop=True).to_file(
                fn_out, driver="GeoJSON"
            )  # FIXME: does not work if does not reset index

    def read_mesh(self):
        """Read mesh at <root/?/> and parse to xarray Dataset """
        #FIXME: it can at the moment only read 2D/1D or network. Not all at once.
        if not self._write:
            # start fresh in read-only mode
            self._mesh = xu.Dataset()
        if isfile(join(self.root, "mesh","FlowFM_2D_net.nc")): #Change of file not implemented yet
            self._mesh = xu.open_dataset(join(self.root, "mesh","FlowFM_2D_net.nc"))

    def write_mesh(self):
        """Write grid at <root/?/> in xarray.Dataset """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        elif not self._mesh:
            self.logger.warning("No mesh to write - Exiting")
            return
        # filename
        #FIXME: change filename when xugrid can read combined networkfiles
        fn_default = join(self.root, "mesh","FlowFM_2D_net.nc")
        self.logger.info(f"Write mesh to {self.root}")
        ds_out = self.mesh
        #FIXME: remove pass if Huite has solution.
        ds_new = xu.UgridDataset(grid=ds_out.ugrid.grid)
        ds_new.ugrid.to_netcdf(fn_default)


        #ds_out.ugrid.to_netcdf(fn_default)

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

        # write 1D mesh
        self._write_mesh1d()  # FIXME None handling

        # write 2d mesh
        self._write_mesh2d()
        #TODO: create self._write_mesh2d() using hydrolib-core funcitonalities

        # write friction
        self._write_friction()  # FIXME: ask Rinske, add global section correctly

        # write crosssections
        self._write_crosssections()  # FIXME None handling, if there are no crosssections

        # save model
        self.dfmmodel.save(recurse=True)

    def _write_mesh1d(self):

        #
        branches = self._staticgeoms["branches"]

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
        self.dfmmodel.geometry.netfile.network._mesh2d._process(mesh2d)



    def _write_friction(self):

        #
        frictions = self._staticgeoms["branches"][
            ["frictionId", "frictionValue", "frictionType"]
        ]
        frictions = frictions.drop_duplicates(subset="frictionId")

        # create a new friction
        fric_model = FrictionModel(global_=frictions.to_dict("record"))
        self.dfmmodel.geometry.frictfile[0] = fric_model

    def _write_crosssections(self):
        """write crosssections into hydrolib-core crsloc and crsdef objects"""

        # preprocessing for crosssections from staticgeoms
        gpd_crs = self._staticgeoms["crosssections"]

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

        # update staticgeom #FIXME: do we need branches as staticgeom?
        self.logger.debug(f"Adding branches vector to staticgeoms.")
        self.set_staticgeoms(gpd.GeoDataFrame(branches, crs=self.crs), "branches")

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

    @property
    def crosssections(self):
        """Quick accessor to crosssections staticgeoms"""
        if "crosssections" in self.staticgeoms:
            gdf = self.staticgeoms["crosssections"]
        else:
            gdf = gpd.GeoDataFrame()
        return gdf

    def set_crosssections(self, crosssections: gpd.GeoDataFrame):
        """Updates crosssections in staticgeoms with new ones"""
        if len(self.crosssections) > 0:
            crosssections = gpd.GeoDataFrame(
                pd.concat([self.crosssections, crosssections])
            )
        self.set_staticgeoms(crosssections, name="crosssections")

    @property
    def mesh(self):
        """xarray.Dataset representation of all mesh"""
        if self._mesh is None:
            if self._read:
                self.read_mesh()
        return self._mesh

    def set_mesh(self, data, name=None):
        """Add data to mesh object.
        -Describe specifics of the mesh object (for example related to the network and ugrid representation)
        Parameters
        ----------
        data: xugrid.UgridDataArray or xugrid.UgridDataset
            new layer to add to mesh
        name: str, optional
            Name of new object layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        if name is None:
            if isinstance(data, xu.UgridDataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xu.UgridDataset):
                raise ValueError("Setting a mesh parameter requires a name")
        elif name is not None and isinstance(data, xu.UgridDataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError("Name not found in DataSet")
            else:
                data = data[[name]]
        if isinstance(data, xu.UgridDataArray):
            data.name = name
            data = data.to_dataset()
        if self._mesh is None:  # new data
            self._mesh = data
        else:
            for dvar in data.data_vars.keys():
                if dvar in self._mesh:
                    if self._read:
                        self.logger.warning(f"Replacing mesh parameter: {dvar}")
                self._mesh[dvar] = data[dvar]