"""The D-RR model: unpaved, paved, greenhouse and open water nodes, and their forcings."""

import logging
import os
from pathlib import Path

import pandas as pd
import rasterio
from pydantic import ConfigDict, validate_call

from hydrolib.dhydamo.geometry import zonal
from hydrolib.dhydamo.io import drrreader

logger = logging.getLogger(__name__)


class DRRModel:
    """The D-RR model.

    Holds the unpaved, paved, greenhouse and open water nodes, and the external forcings
    (seepage, precipitation, evaporation and boundary nodes). Write it with
    ``DRRWriter``.
    """

    def __init__(self) -> None:
        """Create an empty RR model."""
        self.d3b_parameters: dict[str, str | int | float] = {}

        # Land-use counts and surface levels shared by the RR node types.
        self.zonal_cache = zonal.ZonalCache()

        self.unpaved = Unpaved(self)

        self.paved = Paved(self)

        self.greenhouse = Greenhouse(self)

        self.openwater = Openwater(self)

        self.external_forcings = ExternalForcings(self)

        self.dimr_path = ""

    @staticmethod
    def _time_from_filename(path: str | Path) -> pd.Timestamp:
        """Return the time in a file name such as ``precip_20160601.tif``: the part after the first ``_``."""
        filename = os.path.split(path)[1]
        return pd.Timestamp(filename.split("_")[1].split(".")[0])

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def read_raster(self, file: str | Path, static: bool = False) -> tuple:
        """Read the first band of a raster with rasterio.

        Parameters
        ----------
        file : str or Path
            Raster file. Without ``static``, its name holds the time after the first
            ``_``, e.g. ``precip_20160601.tif``.
        static : bool, optional
            The raster has no time. Defaults to False.

        Returns
        -------
        tuple
            The array and its transform, and without ``static`` also the time.
        """
        if isinstance(file, str):
            filename = Path(file)
        else:
            filename = file

        if not static:
            time = self._time_from_filename(file)

        dataset = rasterio.open(filename)
        affine = dataset.transform
        grid = dataset.read(1)

        if static:
            return grid, affine
        else:
            return grid, affine, time


class ExternalForcings:
    """External forcings of the RR model: meteo, seepage and boundary nodes.

    Parameters
    ----------
    drrmodel : DRRModel
        The RR model they belong to.
    """

    def __init__(self, drrmodel: DRRModel) -> None:
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel
        self.io = drrreader.ExternalForcingsIO(self)

        self.boundary_nodes: dict[str, dict[str, str]] = {}
        self.seepage: dict[str, dict[str, pd.Series]] = {}
        # a series per meteo area, or the path of an existing meteo file
        self.precip: dict[str, dict[str, pd.Series]] | str = {}
        self.evap: dict[str, dict[str, pd.Series]] | str = {}

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_precip(self, id: str, series: pd.Series) -> None:
        """Add the precipitation time series of a meteo area.

        Parameters
        ----------
        id : str
            Id of the meteo area.
        series : pd.Series
            Precipitation, with a time index.

        Raises
        ------
        TypeError
            If the precipitation is already read from a meteo file.
        """
        if isinstance(self.precip, str):
            raise TypeError(f"The precipitation is already read from {self.precip}.")
        self.precip[id] = {"precip": series}

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_evap(self, id: str, series: pd.Series) -> None:
        """Add the evaporation time series of a meteo area.

        Parameters
        ----------
        id : str
            Id of the meteo area.
        series : pd.Series
            Evaporation, with a time index.

        Raises
        ------
        TypeError
            If the evaporation is already read from a meteo file.
        """
        if isinstance(self.evap, str):
            raise TypeError(f"The evaporation is already read from {self.evap}.")
        self.evap[id] = {"evap": series}

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_seepage(self, id: str, series: pd.Series) -> None:
        """Add the seepage time series of a catchment.

        Parameters
        ----------
        id : str
            Id of the catchment.
        series : pd.Series
            Seepage, with a time index.
        """
        self.seepage[id] = {"seepage": series}

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_boundary_node(self, id: str, px: str, py: str) -> None:
        """Add a boundary node, which couples RR nodes to a lateral of the FM model.

        Parameters
        ----------
        id : str
            Id of the boundary node.
        px, py : str
            Coordinates of the node.
        """
        self.boundary_nodes[id] = {"id": id, "px": px, "py": py}


class Unpaved:
    """Unpaved nodes of the RR model, with their Ernst definitions.

    Parameters
    ----------
    drrmodel : DRRModel
        The RR model they belong to.
    """

    def __init__(self, drrmodel: DRRModel) -> None:
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel

        # initialize a dataframe for every type of nodes related to 'unpaved'
        self.unp_nodes: dict[str, dict[str, str]] = {}
        self.ernst_defs: dict[str, dict[str, str]] = {}
        # couple input class
        self.io = drrreader.UnpavedIO(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
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
        """Add an unpaved node.

        Parameters
        ----------
        id : str
            Catchment id.
        total_area : str
            Total node area (m2).
        lu_areas : str
            Area per land use class (m2), space separated.
        surface_level : str
            Surface level (m).
        soiltype : str
            Soil type (class id).
        surface_storage : str
            Surface storage (mm).
        infiltration_capacity : str
            Infiltration capacity (mm/d).
        initial_gwd : str
            Initial groundwater depth below the surface (m, positive is below the surface).
        meteo_area : str
            Id of the meteo area, to which a station in the meteo file is assigned.
        px, py : str
            Coordinates of the node.
        boundary_node : str
            Boundary node to which the node drains.
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_ernst_def(self, id: str, cvo: str, lv: str, cvi: str, cvs: str) -> None:
        """Add an Ernst definition for the unpaved node of a catchment.

        Parameters
        ----------
        id : str
            Catchment id.
        cvo : str
            Drainage resistance per layer (d), space separated.
        lv : str
            Depth of each layer (m), space separated.
        cvi : str
            Infiltration resistance (d).
        cvs : str
            Surface runoff resistance (d).
        """
        self.ernst_defs[id] = {
            "id": "ernst_" + id,
            "cvi": cvi,
            "cvs": cvs,
            "cvo": cvo,
            "lv": lv,
        }


class Paved:
    """Paved nodes of the RR model.

    Parameters
    ----------
    drrmodel : DRRModel
        The RR model they belong to.
    """

    def __init__(self, drrmodel: DRRModel) -> None:
        # Point to relevant attributes from parent
        self.drrmodel = drrmodel
        self.pav_nodes: dict[str, dict[str, str]] = {}

        # Create the io class
        self.io = drrreader.PavedIO(self)

    # PAVE id 'pav_Nde_n003' ar 16200 lv 1 sd '1' ss 0 qc 0 1.94E-05 0 qo 2 2 ms 'Station1' aaf 1 is 0 np 0 dw '1' ro 0 ru 0 qh '' pave#
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
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
        """Add a paved node.

        Parameters
        ----------
        id : str
            Catchment or overflow id.
        area : str
            Paved area (m2).
        surface_level : str
            Surface level (m).
        street_storage : str
            Street storage (mm).
        sewer_storage : str
            Sewer storage (mm).
        pump_capacity : str
            Sewer pump capacity (m3/s); ``PavedIO`` computes it from mm/h and the area.
        meteo_area : str
            Id of the meteo area, to which a station in the meteo file is assigned.
        px, py : str
            Coordinates of the node.
        boundary_node : str
            Boundary node to which the node drains.
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
    """Greenhouse nodes of the RR model.

    Parameters
    ----------
    drrmodel : DRRModel
        The RR model they belong to.
    """

    def __init__(self, drrmodel: DRRModel) -> None:
        self.drrmodel = drrmodel
        self.gh_nodes: dict[str, dict[str, str]] = {}

        # Create the io class
        self.io = drrreader.GreenhouseIO(self)

    #    GRHS id ’1’ na 10 ar 1000. 0. 0. 3000. 0. 0. 0. 0. 0. 0. sl 1.0 as 0. sd ’roofstor 1mm’ si
    #    ’silo typ1’ ms ’meteostat1’ is 50.0 grhs
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_greenhouse(
        self,
        id: str,
        area: str,
        surface_level: str,
        roof_storage: str,
        basin_storage_class: str,
        meteo_area: str,
        px: str,
        py: str,
        boundary_node: str,
    ) -> None:
        """Add a greenhouse node.

        Parameters
        ----------
        id : str
            Catchment or greenhouse area id.
        area : str
            Greenhouse area (m2).
        surface_level : str
            Surface level (m).
        roof_storage : str
            Roof storage (mm).
        basin_storage_class : str
            Basin storage class.
        meteo_area : str
            Id of the meteo area, to which a station in the meteo file is assigned.
        px, py : str
            Coordinates of the node.
        boundary_node : str
            Boundary node to which the node drains.
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
            "basin_storage_class": basin_storage_class,
        }


class Openwater:
    """Open water nodes of the RR model.

    Parameters
    ----------
    drrmodel : DRRModel
        The RR model they belong to.
    """

    def __init__(self, drrmodel: DRRModel) -> None:
        self.drrmodel = drrmodel
        self.ow_nodes: dict[str, dict[str, str]] = {}

        # Create the io class
        self.io = drrreader.OpenwaterIO(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True))
    def add_openwater(self, id: str, area: str, meteo_area: str, px: str, py: str, boundary_node: str) -> None:
        """Add an open water node.

        Parameters
        ----------
        id : str
            Catchment id.
        area : str
            Open water area (m2).
        meteo_area : str
            Id of the meteo area, to which a station in the meteo file is assigned.
        px, py : str
            Coordinates of the node.
        boundary_node : str
            Boundary node to which the node drains.
        """
        self.ow_nodes[id] = {
            "id": "ow_" + id,
            "ar": area,
            "ms": "ms_" + meteo_area,
            "px": px,
            "py": py,
            "boundary_node": boundary_node,
        }
