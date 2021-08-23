"""Implement plugin model class"""

import glob
from os.path import join, basename
import logging
from rasterio.warp import transform_bounds
import pyproj
import geopandas as gpd
from shapely.geometry import box
import xarray as xr

import hydromt
from hydromt.models.model_api import Model
from hydromt import gis_utils, io
from hydromt import raster
from . import DATADIR

import delft3dfmpy.core.setup_functions as delft3dfmpy_setupfuncs
# TODO: replace all functions with delft3dfmpy_setupfuncs prefix

from pathlib import Path

__all__ = ["DFlowFMModel"]
logger = logging.getLogger(__name__)


class DFlowFMModel(Model):
    """General and basic API for models in HydroMT"""

    # FIXME
    _NAME = "dflowfm"
    _CONF = "FMmdu.txt"
    _DATADIR = DATADIR
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
        deltares_data=False,  # data from pdrive
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
        self._datamodel = None  # TODO: replace later? e.g. self._intbl in wflow?
        self._dfmmodel = None

    def setup_basemaps(
        self,
        region,
        **kwargs,
    ):
        """Define the model region.
        Adds model layer:
        * **region** geom: model region
        Parameters
        ----------
        region: dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}.
            See :py:meth:`~hydromt.workflows.parse_region()` for all options.
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

        # Set the model region geometry (to be accessed through the shortcut self.region).
        self.set_staticgeoms(geom, "region")

        # FIXME: how to deprecate WARNING:root:No staticmaps defined

    def setup_branches(
        self,
        branches: gpd.GeoDataFrame = None,
        branches_fn: str = None,
        branches_ini_fn: str = None,
        snap_offset: float = 0.0,
        id_col: str = None,
        rename_map: dict = None,
        required_columns: list = None,
        required_dtypes: list = None,
        pipe_query: str = None,
        channel_query: str = None,
    ):
        """ """
        self.logger.info(f"Preparing 1D branches.")
        branches = delft3dfmpy_setupfuncs.setup_branches(
            branches,
            branches_fn,
            branches_ini_fn,
            snap_offset,
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
            delft3dfmpy_setupfuncs.parse_arg(
                pipe_query
            ),  # TODO: replace with csv mapping
            delft3dfmpy_setupfuncs.parse_arg(
                channel_query
            ),  # TODO: replace with csv mapping
            logger=self.logger,
        )

        self.logger.debug(f"Adding branches vector to staticgeoms.")
        self.set_staticgeoms(
            branches, "branches"
        )


    def setup_roughness(self,
        generate_roughness_from_branches:bool = True,
        roughness_ini_fn: str = None,
                        ):
        """"""

        if generate_roughness_from_branches == True:

            # set roughness from branches (1D)
            self.logger.info(f"Preparing 1D roughness.")
            _branches = self.staticgeoms['branches']

            # roughness derived and written to branches
            branches=delft3dfmpy_setupfuncs.setup_roughness(_branches,
                                                            generate_roughness_from_branches = generate_roughness_from_branches,
                                                            roughness_ini_fn = roughness_ini_fn,
                                                                )

            # update new branches with roughness info to staticgeoms
            self.logger.debug(f"Updating branches vector to staticgeoms.")
            self.set_staticgeoms(
                branches, "branches"
            )

        else:
            pass
            # raise NotImplementedError()

    def setup_crosssections(self,
                            generate_crosssections_from_branches:bool = True,
                            crosssections_ini_fn: str = None,
                            ):
        """"""

        if generate_crosssections_from_branches == True:

            # set crosssections from branches (1D)
            self.logger.info(f"Preparing 1D crosssections.")
            _branches = self.staticgeoms['branches']

            # roughness derived and written to branches
            crsdefs, crslocs, branches = delft3dfmpy_setupfuncs.setup_crosssections(_branches,
                                                                 generate_crosssections_from_branches=generate_crosssections_from_branches,
                                                                 crosssections_ini_fn=crosssections_ini_fn,
                                                                 )

            # update new branches with crsdef info to staticgeoms
            self.logger.debug(f"Updating branches vector to staticgeoms.")
            self.set_staticgeoms(
                branches, "branches"
            )

            # add new crsdefs to staticgeoms
            self.logger.debug(f"Adding crsdefs vector to staticgeoms.")
            self.set_staticgeoms(
                gpd.GeoDataFrame(crsdefs, geometry=gpd.points_from_xy([0] * len(crsdefs), [0] * len(crsdefs))), "crsdefs"
            ) # FIXME: make crsdefs a vector to be add to static geoms. using dummy locations --> might cause issue for structures

            # add new crslocs to staticgeoms
            self.logger.debug(f"Adding crslocs vector to staticgeoms.")
            self.set_staticgeoms(
                crslocs, "crslocs"
            )

        else:
            pass
            # raise NotImplementedError()

    def setup_manholes(self,
                      manholes_ini_fn: str = None,
                      manholes_fn: str = None,
                      id_col: str = None,
                      snap_offset: float = 1,
                      rename_map: dict = None,
                      required_columns: list = None,
                      required_dtypes: list = None,
                      logger=logging):
        """"""

        self.logger.info(f"Preparing manholes.")
        _branches = self.staticgeoms['branches']

        # Setup of branches and manholes
        manholes, branches = delft3dfmpy_setupfuncs.setup_manholes(
            _branches,
            manholes_fn=delft3dfmpy_setupfuncs.parse_arg(
                manholes_fn
            ), # FIXME: hydromt config parser could not parse '' to None
            manholes_ini_fn=delft3dfmpy_setupfuncs.parse_arg(
                manholes_ini_fn
            ), # FIXME: hydromt config parser could not parse '' to None
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
            logger=logger
        )

        self.logger.debug(f"Adding manholes vector to staticgeoms.")
        self.set_staticgeoms(manholes, 'manholes')

        self.logger.debug(f"Updating branches vector to staticgeoms.")
        self.set_staticgeoms(branches, 'branches')


    def setup_bridges(self,
                      roughness_ini_fn: str = None,
                      bridges_ini_fn: str = None,
                      bridges_fn: str = None,
                      id_col: str = None,
                      branch_query: str = None,
                      snap_method: str = 'overall',
                      snap_offset: float = 1,
                      rename_map: dict = None,
                      required_columns: list = None,
                      required_dtypes: list = None,
                      logger=logging):
        """"""

        self.logger.info(f"Preparing bridges.")
        _branches = self.staticgeoms['branches']
        _crsdefs = self.staticgeoms['crsdefs']
        _crslocs = self.staticgeoms['crslocs']

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
            logger)


        self.logger.debug(f"Adding bridges vector to staticgeoms.")
        self.set_staticgeoms(bridges, 'bridges')

        self.logger.debug(f"Updating crsdefs vector to staticgeoms.")
        self.set_staticgeoms(crsdefs, 'crsdefs')

    def setup_gates(self,
                    roughness_ini_fn: str = None,
                    gates_ini_fn: str = None,
                    gates_fn: str = None,
                    id_col: str = None,
                    branch_query: str = None,
                    snap_method: str = 'overall',
                    snap_offset: float = 1,
                    rename_map: dict = None,
                    required_columns: list = None,
                    required_dtypes: list = None,
                    logger=logging):
        """"""

        self.logger.info(f"Preparing gates.")
        _branches = self.staticgeoms['branches']
        _crsdefs = self.staticgeoms['crsdefs']
        _crslocs = self.staticgeoms['crslocs']

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
            logger)

        self.logger.debug(f"Adding gates vector to staticgeoms.")
        self.set_staticgeoms(gates, 'gates')


    def setup_pumps(self,
                    roughness_ini_fn: str = None,
                    pumps_ini_fn: str = None,
                    pumps_fn: str = None,
                    id_col: str = None,
                    branch_query: str = None,
                    snap_method: str = 'overall',
                    snap_offset: float = 1,
                    rename_map: dict = None,
                    required_columns: list = None,
                    required_dtypes: list = None,
                    logger=logging):
        """"""

        self.logger.info(f"Preparing gates.")
        _branches = self.staticgeoms['branches']
        _crsdefs = self.staticgeoms['crsdefs']
        _crslocs = self.staticgeoms['crslocs']

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
            logger)


        self.logger.debug(f"Adding pumps vector to staticgeoms.")
        self.set_staticgeoms(pumps, 'pumps')

    def setup_culverts(self,
                    roughness_ini_fn: str = None,
                    culverts_ini_fn: str = None,
                    culverts_fn: str = None,
                    id_col: str = None,
                    branch_query: str = None,
                    snap_method: str = 'overall',
                    snap_offset: float = 1,
                    rename_map: dict = None,
                    required_columns: list = None,
                    required_dtypes: list = None,
                    logger=logging):
        """"""

        self.logger.info(f"Preparing culverts.")
        _branches = self.staticgeoms['branches']
        _crsdefs = self.staticgeoms['crsdefs']
        _crslocs = self.staticgeoms['crslocs']

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
            logger)

        self.logger.debug(f"Adding culverts vector to staticgeoms.")
        self.set_staticgeoms(culverts, 'culverts')

        self.logger.debug(f"Updating crsdefs vector to staticgeoms.")
        self.set_staticgeoms(crsdefs, 'crsdefs')


    def setup_compounds(self,
                    roughness_ini_fn:str = None,
                    compounds_ini_fn: str = None,
                    compounds_fn: str = None,
                    id_col: str = None,
                    branch_query: str = None,
                    snap_method: str = 'overall',
                    snap_offset: float = 1,
                    rename_map: dict = None,
                    required_columns: list = None,
                    required_dtypes: list = None,
                    logger=logging):
        """"""

        self.logger.info(f"Preparing compounds.")
        _structures = [self.staticgeoms[s] for s in ['bridges', 'gates', 'pumps', 'culverts'] if s in self.staticgeoms.keys()]

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
            logger)

        self.logger.debug(f"Adding compounds vector to staticgeoms.")
        self.set_staticgeoms(compounds, 'compounds')

    def setup_boundaries(self,
                         boundaries_fn: str = None,
                         boundaries_fn_ini: str = None,
                         id_col: str = None,
                         rename_map: dict = None,
                         required_columns: list = None,
                         required_dtypes: list = None,
                         logger=logging):
        """"""

        self.logger.info(f"Preparing boundaries.")
        _structures = [self.staticgeoms[s] for s in ['bridges', 'gates', 'pumps', 'culverts'] if s in self.staticgeoms.keys()]

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
            logger)

        self.logger.debug(f"Adding boundaries vector to staticgeoms.")
        self.set_staticgeoms(boundaries, 'boundaries')


    def _setup_datamodel(self):

        """setup data model using dfm and drr naming conventions"""
        if self._datamodel == None:
            self._datamodel = delft3dfmpy_setupfuncs.setup_dm(self.staticgeoms, logger=self.logger)

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
            self.write_config() # FIXME: config now isread from default, modified and saved temporaryly in the models folder --> being read by dfm and modify?
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
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        for name, gdf in self.staticgeoms.items():
            fn_out = join(self.root, "staticgeoms", f"{name}.shp")
            delft3dfmpy_setupfuncs.write_shp(gdf[[i for i in gdf.columns if i.endswith('_ID') or i == 'geometry']], fn_out) # only write geometry column
            # TODO: replace this function. Q: if output file columns need to be replaced? e.g. 10 char length limit

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
        return pyproj.CRS.from_epsg(self.get_config("global.epsg", fallback=4326))

    @property
    def dfmmodel(self):
        if self._dfmmodel == None:
            self.init_dfmmodel()
        return self._dfmmodel

    def init_dfmmodel(self):
        self._dfmmodel = delft3dfmpy_setupfuncs.DFlowFMModel()
