import logging
import os
from pathlib import Path
from typing import Union

import imod
import pandas as pd
import rasterio
from pydantic import validate_arguments
from rasterio.transform import from_origin

from hydrolib.dhydamo.io import drrreader

logger = logging.getLogger(__name__)


class DRRModel:
    """Main data structure for RR-model in DflowFM. Contains subclasses
    for unpaved, paved,greehouse and open water nodes and external forcings (seepage, precipitation, evaporation)
    """

    def __init__(self):
        """Initialize RR instances and arrays"""
        self.d3b_parameters = {}

        self.unpaved = Unpaved(self)

        self.paved = Paved(self)

        self.greenhouse = Greenhouse(self)

        self.openwater = Openwater(self)

        self.external_forcings = ExternalForcings(self)

        self.dimr_path = ""

    @validate_arguments
    def read_raster(self, file: Union[str, Path], static: bool = False) -> tuple:
        """
        Method to read a raster. All rasterio types are accepted, plus IDF: in that case the iMod-package is used to read the IDF raster (IDF is cusomary for MODFLOW/SIMGRO models.)

        Parameters
        ----------
        file : raster

        static : BOOL, optional
            If static than no time information needs to be deduced.

        Returns
        -------
        rasterio grid and an affine object.

        """
        if isinstance(file, str):
            file = Path(file)

        if not static:
            time = pd.Timestamp(os.path.split(file)[1].split("_")[1].split(".")[0])

        if file.suffix.lower() == ".idf":
            dataset = imod.idf.open(file)
            header = imod.idf.header(file, pattern=None)
            grid = dataset[0, 0, :, :].values
            affine = from_origin(
                header["xmin"], header["ymax"], header["dx"], header["dx"]
            )
        else:
            dataset = rasterio.open(file)
            affine = dataset.transform
            grid = dataset.read(1)

        if static:
            return grid, affine
        else:
            return grid, affine, time


class ExternalForcings:
    """
    Class for external forcings, which contains the boundary
    conditions and the initial conditions.
    """

    def __init__(self, drrmodel):
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel
        self.io = drrreader.ExternalForcingsIO(self)

        self.boundary_nodes = {}
        self.seepage = {}
        self.precip = {}
        self.evap = {}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_precip(self, id: str, series: pd.Series):
        self.precip[id] = {"precip": series}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_evap(self, id: str, series: pd.Series):
        self.evap[id] = {"evap": series}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_seepage(self, id: str, series: pd.Series):
        self.seepage[id] = {"seepage": series}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_boundary_node(self, id: str, px: str, py: str):
        self.boundary_nodes[id] = {"id": id, "px": px, "py": py}


class Unpaved:
    """
    Class for unpaved nodes
    """

    def __init__(self, drrmodel):
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel

        # initialize a dataframe for every type of nodes related to 'unpaved'
        self.unp_nodes = {}
        self.ernst_defs = {}
        # couple input class
        self.io = drrreader.UnpavedIO(self)

    @validate_arguments
    def add_unpaved(
        self,
        id: str,
        total_area: str,
        lu_areas: str,
        surface_level: str,
        soiltype: str,
        surface_storage: str,
        infiltration_capacity: str,
        initial_gwd: str,
        meteo_area: str,
        px: str,
        py: str,
        boundary_node: str,
    ) -> None:
        """Add elements of an unpaved node definition to a dataframe

        Args:
            id (str): catchment id
            total_area (str): total node area (m2)
            lu_areas (str): area per land use class (space sepaated; m2)
            surface_level (str): surface level (m)
            soiltype (str): soiltype (class id)
            surface_storage (str): surface storage (mm)
            infiltration_capacity (str): iniltration capacity (mm/d)
            initial_gwd (str): intial ground water level below surface (m, positve = below surface)
            meteo_area (str): id of meteo area to which a station in the meteo-file is assigned
            px (str): x-coordinate
            py (str): y-coordinante
            boundary_node (str): associated boundary node
        """
        self.unp_nodes[id] = {
            "id": "unp_" + id,
            "na": "16",
            "ar": lu_areas,
            "ga": total_area,
            "lv": surface_level,
            "co": "3",
            "su": "0",
            "sd": surface_storage,
            "sp": "sep_" + id,
            "ic": infiltration_capacity,
            "ed": "ernst_" + id,
            "bt": soiltype,
            "ig": initial_gwd,
            "mg": surface_level,
            "gl": "1.5",
            "is": "0",
            "ms": "ms_" + meteo_area,
            "px": px,
            "py": py,
            "boundary_node": boundary_node,
        }

    @validate_arguments
    def add_ernst_def(self, id: str, cvo: str, lv: str, cvi: str, cvs: str) -> None:
        """Add properties to a datafframe containing an Ernst definition.

        Args:
            id (str): catchment id
            cvo (str): Drainage resistance, space separated value per layer [d-1]
            lv (str): Layer thickness, space separated value per layer [m]
            cvi (str): Infiltration resistance [d-1]
            cvs (str): Surface runoff resistance [d-1]
        """
        self.ernst_defs[id] = {
            "id": "ernst_" + id,
            "cvi": cvi,
            "cvs": cvs,
            "cvo": cvo,
            "lv": lv,
        }


class Paved:
    """
    Class for paved nodes.
    """

    def __init__(self, drrmodel):
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel
        self.pav_nodes = {}

        # Create the io class
        self.io = drrreader.PavedIO(self)

        self.node_geom = {}
        self.link_geom = {}

    # PAVE id 'pav_Nde_n003' ar 16200 lv 1 sd '1' ss 0 qc 0 1.94E-05 0 qo 2 2 ms 'Station1' aaf 1 is 0 np 0 dw '1' ro 0 ru 0 qh '' pave#
    @validate_arguments
    def add_paved(
        self,
        id: str,
        area: str,
        surface_level: str,
        street_storage: str,
        sewer_storage: str,
        pump_capacity: str,
        meteo_area: str,
        px: str,
        py: str,
        boundary_node: str,
    ) -> None:
        """Add elements of a paved node definition to a dataframe

        Args:
            id (str): catchment id
            area (str): paved node area (m2)
            surface_level (str): surface level (m)
            street_storage (str): surface storage (mm)
            sewer_storage (str): sewer storage (mm)
            pump_capacity (str): pump capacity (mm/d)
            meteo_area (str): id of meteo area to which a station in the meteo-file is assigned
            px (str): x-coordinate
            py (str): y-coordinante
            boundary_node (str): associated boundary node
        """

        self.pav_nodes[id] = {
            "id": "pav_" + id,
            "ar": area,
            "lv": surface_level,
            "qc": pump_capacity,
            "strs": street_storage,
            "sews": sewer_storage,
            "ms": "ms_" + meteo_area,
            "is": "0",
            "np": "0",
            "ro": "0",
            "ru": "0",
            "px": px,
            "py": py,
            "boundary_node": boundary_node,
        }


class Greenhouse:
    """
    Class for greenhouse nodes
    """

    def __init__(self, drrmodel):

        self.drrmodel = drrmodel
        self.gh_nodes = {}

        # Create the io class
        self.io = drrreader.GreenhouseIO(self)

    #    GRHS id ’1’ na 10 ar 1000. 0. 0. 3000. 0. 0. 0. 0. 0. 0. sl 1.0 as 0. sd ’roofstor 1mm’ si
    #    ’silo typ1’ ms ’meteostat1’ is 50.0 grhs
    @validate_arguments
    def add_greenhouse(
        self,
        id: str,
        area: str,
        surface_level: str,
        roof_storage: str,
        meteo_area: str,
        px: str,
        py: str,
        boundary_node: str,
    ) -> None:
        """Add elements of a greenhouse node definition to a dataframe

        Args:
            id (str): catchment id
            area (str): greenhouse node area (m2)
            surface_level (str): surface level (m)
            roof_storage (str): roof storage (mm)
            meteo_area (str): id of meteo area to which a station in the meteo-file is assigned
            px (str): x-coordinate
            py (str): y-coordinante
            boundary_node (str): associated boundary node
        """
        self.gh_nodes[id] = {
            "id": "gh_" + id,
            "ar": area,
            "sl": surface_level,
            "sd": roof_storage,
            "ms": "ms_" + meteo_area,
            "is": "0",
            "px": px,
            "py": py,
            "boundary_node": boundary_node,
        }


class Openwater:
    """
    Class for open water nodes
    """

    def __init__(self, drrmodel):
        self.drrmodel = drrmodel
        self.ow_nodes = {}

        # Create the io class
        self.io = drrreader.OpenwaterIO(self)

    @validate_arguments
    def add_openwater(
        self, id: str, area: str, meteo_area: str, px: str, py: str, boundary_node: str
    ) -> None:
        """Add elements of an open water node definition to a dataframe

        Args:
            id (str): catchment id
            area (str): greenhouse node area (m2)
            meteo_area (str): id of meteo area to which a station in the meteo-file is assigned
            px (str): x-coordinate
            py (str): y-coordinante
            boundary_node (str): associated boundary node
        """
        self.ow_nodes[id] = {
            "id": "ow_" + id,
            "ar": area,
            "ms": "ms_" + meteo_area,
            "px": px,
            "py": py,
            "boundary_node": boundary_node,
        }
