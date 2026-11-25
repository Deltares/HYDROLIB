"""The HyDAMO data model: input data, cross sections, structures, forcings and storage nodes."""

from __future__ import annotations

import logging
import numbers
from collections.abc import Hashable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, no_type_check

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from netCDF4 import Dataset, Variable, chartostring
from pydantic import ConfigDict, Field, validate_call
from pyproj import CRS
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from hydrolib import dhydamo

# the network of hydrolib-core; Network in this module is the legacy one of delft3dfmpy
from hydrolib.core.dflowfm.net import models as netmodels
from hydrolib.dhydamo.converters.df2hydrolibmodel import level_or_depth
from hydrolib.dhydamo.converters.hydamo2df import (
    CrossSectionsIO,
    ExternalForcingsIO,
    RoughnessVariant,
    StorageNodesIO,
    StructuresIO,
)
from hydrolib.dhydamo.core.relations import build_cascade_plan
from hydrolib.dhydamo.geometry import zonal
from hydrolib.dhydamo.geometry.spatial import find_nearest_branch
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame
from hydrolib.dhydamo.io.damo_converters import (
    SourceLayer,
    load_layers,
    source_layers,
)
from hydrolib.dhydamo.io.damo_converters.data_model import (
    OWN_TABLES,
    TABLES,
    DamoRelations,
    Geotype,
)
from hydrolib.dhydamo.io.netcdfreader import time_index
from hydrolib.dhydamo.validation import (
    HydamoValidationResult,
    ValidationMode,
    validate_or_raise,
)

logger = logging.getLogger(__name__)


def _new_table(
    geotype: Geotype | None, required: tuple[str, ...]
) -> ExtendedGeoDataFrame | ExtendedDataFrame:
    """Return an empty table: spatial when it has a geometry type."""
    if geotype is None:
        return ExtendedDataFrame(required_columns=list(required))
    return ExtendedGeoDataFrame(geotype=geotype, required_columns=list(required))


class HyDAMO:
    """The HyDAMO input data and the model objects derived from it.

    Holds the DAMO tables (branches, structures, profiles, management, catchments, ...)
    and the cross sections, structures, observation points, storage nodes and external
    forcings of the model.
    """

    # The tables of the data model (damo_converters.data_model), created in __init__.
    # Only their types are declared here, for type checking.
    branches: ExtendedGeoDataFrame
    profile: ExtendedGeoDataFrame
    profile_roughness: ExtendedDataFrame
    profile_line: ExtendedGeoDataFrame
    profile_group: ExtendedDataFrame
    param_profile: ExtendedDataFrame
    param_profile_values: ExtendedDataFrame
    weirs: ExtendedGeoDataFrame
    opening: ExtendedDataFrame
    closing_device: ExtendedDataFrame
    management_device: ExtendedDataFrame
    bridges: ExtendedGeoDataFrame
    culverts: ExtendedGeoDataFrame
    pumpstations: ExtendedGeoDataFrame
    pumps: ExtendedDataFrame
    management: ExtendedDataFrame
    boundary_conditions: ExtendedGeoDataFrame
    catchments: ExtendedGeoDataFrame
    laterals: ExtendedGeoDataFrame
    overflows: ExtendedGeoDataFrame
    sewer_areas: ExtendedGeoDataFrame
    greenhouse_areas: ExtendedGeoDataFrame
    greenhouse_laterals: ExtendedGeoDataFrame
    storage_areas: ExtendedGeoDataFrame

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def __init__(self, extent_file: Path | str | None = None) -> None:
        """Create an empty HyDAMO object.

        Parameters
        ----------
        extent_file : Path or str, optional
            Model extent, used to clip the data. Defaults to None.
        """
        self.network = Network(self)

        self.structures = Structures(self)

        self.crosssections = CrossSections(self)  # Add all items

        self.observationpoints = ObservationPoints(self)

        self.external_forcings = ExternalForcings(self)

        self.storagenodes = StorageNodes(self)

        self.roughness_mapping = {
            "Chezy": "Chezy",
            "Manning": "Manning",
            "StricklerKn": "StricklerNikuradse",
            "StricklerKs": "Strickler",
            "White Colebrook": "WhiteColebrook",
            "Bos en Bijkerk": "deBosBijkerk",
            "Onbekend": "Strickler",
            "Overig": "Strickler",
        }

        # Dictionary for roughness definitions
        self.roughness_definitions: dict[str, dict] = {}

        # Read geometry to clip data
        self.clipgeo: BaseGeometry | None
        if extent_file is not None:
            self.clipgeo = gpd.read_file(extent_file).union_all()
        else:
            self.clipgeo = None

        # versioning info
        self.version = {
            "number": dhydamo.__version__,
            "date": datetime.strftime(datetime.now(UTC), "%Y-%m-%dT%H:%M:%S.%fZ"),
            "dimr_version": "Deltares, DIMR_EXE Version 2.00.00.140737 (Win64) (Win64)",
            "suite_version": "D-HYDRO Suite 2024.03 1D2D,",
        }
        # the DAMO version the data was loaded as; None until load_from_gpkg
        self.source_damo_version: str | None = None
        self.validation_result: HydamoValidationResult | None = None

        # The DAMO tables and HYDROLIB's own tables, from the data model
        for name, table in TABLES.items():
            setattr(self, name, _new_table(table.geotype, table.required_columns))
        for name, own_table in OWN_TABLES.items():
            setattr(
                self, name, _new_table(own_table.geotype, own_table.required_columns)
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def validate_gpkg(
        self,
        gpkg_path: Path | str,
        hydamo_version: str,
        validation_coverages: dict | None = None,
        validation_rules_path: Path | str | None = None,
    ) -> HydamoValidationResult:
        """Validate a GeoPackage without loading its layers.

        Requires Python 3.12 and the optional dependency installed with
        ``pip install "hydrolib[validation]"``.

        Parameters
        ----------
        gpkg_path : str or Path
            Path to the GeoPackage file.
        hydamo_version : str
            DAMO version to validate against, e.g. ``"2.6"``. Required: it selects
            the version; it does not detect the source version. Any version that the
            installed validatiemodule has a schema for is accepted, which can include
            versions that ``load_from_gpkg`` cannot load yet (see
            ``SUPPORTED_HYDAMO_VERSIONS``). For such a version no column counts as
            read by HYDROLIB, so every finding goes into the INFO line.
        validation_coverages : dict, optional
            Coverage dict forwarded to the validator.
        validation_rules_path : str or Path, optional
            Path to a validation-rules JSON file for the requested DAMO version.
            If omitted, generated rules check ID presence on recognized layers
            with an ID column. Supply custom rules for more comprehensive checks.

        Returns
        -------
        HydamoValidationResult
            Also stored as ``self.validation_result``. Inspect ``success``,
            ``status``, ``errors`` and ``syntax_findings`` to handle validation
            failures. Findings in columns HYDROLIB reads are logged as one
            warning per layer by the ``hydrolib.dhydamo.validation`` logger, the
            number of other findings as one INFO line. A reported failure is
            returned rather than raised as ``HyDAMOValidationError``. The
            validatiemodule's own diagnostics are logged by the
            ``hydamo_validation.validator`` logger.

        Raises
        ------
        ImportError
            If the optional validation dependency is unavailable.
        ValueError
            If generated rules are requested for a version unsupported by the
            installed validator's schemas.
        """
        # Without version steps to the version, no column counts as read.
        try:
            read_columns = self._read_columns(source_layers(hydamo_version))
        except ValueError:
            read_columns = {}
        result = validate_or_raise(
            gpkg_path=gpkg_path,
            hydamo_version=hydamo_version,
            validation_mode="warn",
            coverages=validation_coverages,
            validation_rules_path=validation_rules_path,
            read_columns=read_columns,
        )
        assert result is not None
        self.validation_result = result
        return result

    def _read_columns(self, sources: dict[str, SourceLayer]) -> dict[str, set[str]]:
        """Return, per lowercase DAMO layer, the columns HYDROLIB reads.

        The columns are the table's required columns, ``globalid`` and, for a
        spatial table, ``geometry``, under their names in the DAMO version of
        *sources* (from ``source_layers``).
        """
        read_columns: dict[str, set[str]] = {}
        for name, source in sources.items():
            table = getattr(self, name)
            columns = {*table.required_columns, "globalid"}
            if isinstance(table, ExtendedGeoDataFrame):
                columns.add("geometry")
            file_names = {new: old for old, new in source.renames.items()}
            read_columns[source.layer.lower()] = {
                file_names.get(column, column) for column in columns
            }
        return read_columns

    def load_from_gpkg(
        self,
        gpkg_path: Path | str,
        hydamo_version: str,
        clip_layers: dict | None = None,
        check_3d: bool = True,
        validation_mode: ValidationMode = "off",
        validation_coverages: dict | None = None,
        validation_rules_path: Path | str | None = None,
    ) -> HyDAMO:
        """Load HyDAMO data from a GeoPackage of a supported DAMO version.

        Parameters
        ----------
        gpkg_path : str or Path
            Path to the GeoPackage file.
        hydamo_version : str
            DAMO schema version, one of ``SUPPORTED_HYDAMO_VERSIONS``, e.g.
            ``"2.5"``. Required: it selects the renames back to DAMO 2.2, the
            names HYDROLIB uses; it does not detect or verify the source schema
            version.
        clip_layers : dict, optional
            Per-table spatial clip. Maps a HyDAMO table name to a
            ``(geometry, cliptype)`` tuple, e.g.
            ``{"branches": (gpd.read_file(extent).union_all(), "clip")}``.
            Layers absent from the dict are loaded without clipping.
        check_3d : bool
            Whether to require Z coordinates on profile points when grouping
            them into lines. Other layers are not checked for Z coordinates.
            Pass ``False`` to allow 2D profile points. Default is ``True``.
        validation_mode : str
            ``"off"`` skips validation (default). Both other modes act on the
            values the syntax check rejected in columns HYDROLIB reads (its
            required columns, ``globalid`` and, for spatial tables, the
            geometry): ``"warn"`` logs them as one warning per layer, by the
            ``hydrolib.dhydamo.validation`` logger, and continues; ``"strict"``
            raises ``HyDAMOValidationError`` before loading layers, as it does
            when the validation run fails. Rejected values in other columns
            only give one INFO line with their number. Inspect
            ``validation_result.syntax_findings`` for all of them. Validation
            setup errors can raise in either enabled mode.
        validation_coverages : dict, optional
            Coverage dict forwarded to the validator.
        validation_rules_path : str or Path, optional
            Path to a custom validation-rules JSON file for the source DAMO
            version. If omitted, generated rules check ID presence on recognized
            layers with an ID column. Strict mode changes failure handling,
            not the rules or their coverage.

        Returns
        -------
        HyDAMO
            This instance, to allow method chaining.

        Raises
        ------
        HyDAMOValidationError
            In ``"strict"`` mode, if the validation run fails or rejected values
            in a column HYDROLIB reads.
        ImportError
            If validation is enabled and its optional dependency is unavailable.
        ValueError
            If the DAMO version is unsupported, or generated validation
            rules require a version unsupported by the installed validator.

        Notes
        -----
        A load replaces every DAMO table, also those whose layer the GeoPackage
        lacks: nothing of an earlier load is left in them. HYDROLIB's own tables
        (sewer areas, overflows, greenhouses, storage areas) are kept, and so are
        model objects built from an earlier load, such as structures and cross
        sections. For a new model, use a new ``HyDAMO``.

        Validation requires Python 3.12 and ``pip install "hydrolib[validation]"``.
        After enabled validation returns normally, ``self.validation_result``
        contains ``success``, ``status``, ``errors`` and ``syntax_findings``
        (per layer, column and error code). It is ``None`` when
        validation is off. Use ``validate_gpkg()`` to inspect validation failures
        without loading data. Comprehensive schema, type, and relation checks
        require appropriate custom rules; selecting a DAMO version or using strict
        mode alone does not establish that the input is valid DAMO data.

        Examples
        --------
        Given a rules file for DAMO 2.2, load and inspect reported failures:

        >>> hydamo = HyDAMO().load_from_gpkg(
        ...     "model.gpkg",
        ...     hydamo_version="2.2",
        ...     validation_mode="warn",
        ...     validation_rules_path="validationrules.json",
        ... )
        >>> result = hydamo.validation_result
        >>> print(result.success, result.status, result.errors)

        Use ``validation_mode="strict"`` to raise ``HyDAMOValidationError``
        instead of loading after a reported validation failure.
        """
        # Resolve the layers first so an unsupported version fails before validation runs.
        sources = source_layers(hydamo_version)

        self.validation_result = None
        self.source_damo_version = hydamo_version
        self.validation_result = validate_or_raise(
            gpkg_path=gpkg_path,
            hydamo_version=hydamo_version,
            validation_mode=validation_mode,
            coverages=validation_coverages,
            validation_rules_path=validation_rules_path,
            read_columns=(
                self._read_columns(sources) if validation_mode != "off" else None
            ),
        )

        self._reset_damo_tables()
        load_layers(
            self,
            gpkg_path,
            sources,
            hydamo_version,
            clip_layers=clip_layers,
            check_3d=check_3d,
        )
        return self

    def _reset_damo_tables(self) -> None:
        """Empty every DAMO table in place, as it is after ``HyDAMO()``.

        In place, so references to the tables stay valid. HYDROLIB's own tables
        (``OWN_TABLES``) are left alone: they are filled from their own files.
        """
        for name, definition in TABLES.items():
            table = getattr(self, name)
            fresh = _new_table(definition.geotype, definition.required_columns)
            # inplace: the tables are replaced in place, so references stay valid
            table.drop(index=table.index, inplace=True)  # noqa: PD002
            table.drop(columns=list(table.columns), inplace=True)  # noqa: PD002
            table.index = fresh.index
            for column in fresh.columns:
                table[column] = fresh[column]

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def list_to_str(self, lst: list | np.ndarray) -> str:
        """Return numbers as a space separated string, with 3 decimals.

        Parameters
        ----------
        lst : list or np.ndarray
            Numbers.

        Returns
        -------
        str
            The numbers, or ``str(lst)`` for a single number.
        """
        if len(lst) == 1:
            string = str(lst)
        else:
            string = " ".join([f"{number:6.3f}" for number in lst])
        return string

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def dict_to_dataframe(self, dictionary: dict) -> pd.DataFrame:
        """Return a dictionary of dictionaries as a DataFrame, with the keys as rows.

        Parameters
        ----------
        dictionary : dict
            Input dictionary.

        Returns
        -------
        pd.DataFrame
            Output DataFrame.
        """
        return pd.DataFrame.from_dict(dictionary, orient="index")

    def snap_to_branch_and_drop(
        self,
        extendedgdf: ExtendedGeoDataFrame,
        branches: ExtendedGeoDataFrame,
        snap_method: str,
        maxdist: float = 5,
        drop_related: bool = True,
        source_table: str | None = None,
    ) -> None:
        """Snap geometries to branches and drop loose objects and cascades.

        Parameters
        ----------
        extendedgdf : ExtendedGeoDataFrame
            HyDAMO dataframe to snap and potentially prune.
        branches : ExtendedGeoDataFrame
            Branch geometries used for snapping.
        snap_method : str
            Method passed to :meth:`ExtendedGeoDataFrame.snap_to_branch`.
        maxdist : float, default=5
            Maximum snapping distance passed to ``snap_to_branch``.
        drop_related : bool, default=True
            Whether to drop rows reached through the configured cascade paths.
        source_table : str, optional
            Canonical HyDAMO table name.  If omitted, the name is determined
            from the dataframe object stored on this ``HyDAMO`` instance.
            Supply this when ``extendedgdf`` is a dataframe copy; the named
            table must still exist as an attribute on this instance.
        """
        # Snap the extended geodataframe to branches
        extendedgdf.snap_to_branch(branches, snap_method, maxdist=maxdist)

        # Determine which labels need to be drop for the first object based on
        # nan values for branch_offset.
        drop_idx = extendedgdf.index[pd.isna(extendedgdf.branch_offset)]
        logger.info("dropping objects with indices: %s", drop_idx)

        # Build the complete plan before mutating any dataframe.  This keeps
        # all relation traversals independent from the drops performed below.
        drop_plan: dict[str, pd.Index] = {}
        if drop_related:
            if source_table is None:
                for table_name, table in vars(self).items():
                    if table is extendedgdf:
                        source_table = table_name
                        break
            if source_table is None:
                raise ValueError(
                    "Could not determine the HyDAMO source table for the dataframe; pass source_table explicitly"
                )
            drop_plan = build_cascade_plan(self, source_table, drop_idx)

        # Drop the source and all related rows only after the plan is complete.
        # inplace: returning a copy of an ExtendedGeoDataFrame gives a plain GeoDataFrame
        extendedgdf.drop(labels=drop_idx, inplace=True)  # noqa: PD002
        for target_table, target_index in drop_plan.items():
            logger.info(
                "  - dropping objects from '%s' with indices: %s",
                target_table,
                target_index,
            )
            getattr(self, target_table).drop(labels=target_index, inplace=True)  # noqa: PD002

    def create_laterals(
        self,
        qspec_file: str | Path | None = None,
        *,
        min_coverage_frac: float = 0.0,
        coverage_weight: zonal.CoverageWeight = "fraction",
    ) -> None:
        """Create laterals at the catchments, with the mean specific discharge per catchment.

        Parameters
        ----------
        qspec_file : str, optional
            Raster of specific discharge (mm/d). Without it, the discharges are NaN.
        min_coverage_frac : float, optional
            Smallest fraction of a pixel a catchment must cover for the pixel to count in
            the zonal statistics. Defaults to 0: every touched pixel.
        coverage_weight : str, optional
            ``"fraction"`` counts pixels by the fraction of them the catchment covers,
            ``"none"`` counts them fully. Defaults to ``"fraction"``.
        """
        ## Afvoer per gebied bepalen met zonal stats; nodata telt niet mee
        if qspec_file is not None:
            afvoer_per_gebied = zonal.zonal_stats(
                gpd.GeoDataFrame(self.catchments),
                qspec_file,
                statistics=("mean",),
                strategy="feature-sequential",
                min_coverage_frac=min_coverage_frac,
                coverage_weight=coverage_weight,
            )

        self.laterals = self.catchments.copy()

        catchment_columns = zip(
            self.catchments.index,
            self.catchments.geometry,
            self.catchments["code"],
            self.catchments["lateraleknoopid"],
        )
        for num, (index, geometry, code, knoopid) in enumerate(catchment_columns):
            ## Koppelen van afvoeren met afwateringsgebieden
            if qspec_file is not None:
                q = afvoer_per_gebied.iloc[num]["mean"]  # mm/d
            else:
                q = np.nan

            area = geometry.area  # m2
            q_m3s = q * area / (1000 * 86400)  # van mm/d naar m3/s
            self.laterals.loc[index, "afvoer"] = q_m3s

            ## Afstand tussen watergang en afwateringsgebied. Als watergang in afwateringsgebied ligt is deze afstand 0.
            distances = self.branches.distance(geometry)
            ## EÃ©n of meer watergangen in afwateringsgebied:
            if sum(distances == 0) > 0:
                ## Watergangen intersecten met afwateringsgebied
                # wg = [watergang for watergang in watergangen.itertuples() if watergang.intersects(geometry]
                selectie = self.branches.intersection(geometry)
                ## Index van de watergang waarop je wilt snappen. Dit is de watergang die in het gebied ligt Ã©n het dichtst bij de centroid van het afwateringsgebied is.
                index_watergang = selectie.distance(geometry.centroid).idxmin()
                ## Combineer de ge intersecte watergangen met de index om het stuk watergang te vinden waarop mag worden gesnapt.
                watergang = selectie.loc[index_watergang]
            ## No intersects
            else:
                ## Vind de watergang die het dichtst bij de centroid van het gebied is
                index_watergang = self.branches.distance(geometry.centroid).idxmin()
                ## Selecteer deze watergang om op te mogen snappen
                watergang = self.branches.geometry.loc[index_watergang]

            ## Snap de centroid van het afwateringsgebied op de geselecteerde watergang
            lateral = watergang.interpolate(watergang.project(geometry.centroid))
            ## Schrijf de snap-locatie weg in de geodataframe
            # pandas-stubs does not accept a shapely geometry as value; pandas does
            self.laterals.loc[index, "geometry"] = lateral  # type: ignore[call-overload]
            self.laterals.loc[index, "globalid"] = knoopid
            self.laterals.loc[index, "code"] = f"lat_{code}"


class Network:
    """Legacy 1D network helpers, taken over from delft3dfmpy. Not functional.

    The methods use ``self.mesh1d``, ``self.branches``, ``self.schematised`` and
    ``self.str2chars``, which came from the gridgeom-based network in delft3dfmpy and
    are never set here: every method raises ``AttributeError``. Nothing in this package,
    its tests or notebooks uses ``HyDAMO.network``; the 1D network is the hydrolib-core
    ``Network`` (see ``hydrolib.dhydamo.geometry.mesh``).

    The methods are marked ``@no_type_check``, so mypy skips them, like the deprecated
    ``gridgeom`` module. Remove this class together with ``gridgeom``.
    """

    @no_type_check
    def __init__(self, hydamo: HyDAMO) -> None:
        """Create the legacy network helpers (not functional, see the class docstring)."""
        self.hydamo = hydamo

        # Mesh 1d offsets
        self.offsets = {}

    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def set_branch_order(self, branchids: list, idx: int | None = None) -> None:
        """Group branch ids so that the cross sections are interpolated along the branch.

        Parameters
        ----------
        branchids : list
            List of branches to group
        idx : int
            Order number with which to update a branch
        """
        # Get the ids (integers) of the branch names given by the user
        branchidx = np.isin(self.mesh1d.description1d["network_branch_ids"], branchids)
        # Get current order
        branchorder = self.mesh1d.get_values("nbranchorder", as_array=True)
        # Update
        if idx is None:
            branchorder[branchidx] = branchorder.max() + 1
        else:
            if not isinstance(idx, int):
                raise TypeError("Expected integer.")
            branchorder[branchidx] = idx
        # Save
        self.mesh1d.set_values("nbranchorder", branchorder)

    @no_type_check
    def set_branch_interpolation_modelwide(self) -> None:
        """Set cross-section interpolation over nodes on all branches model-wide.

        Note:
            - Interpolation will be set using branchorder property.
            - Branches are grouped between bifurcations where multiple branches meet.
            - No interpolation is applied over these type of bifurcations.
            - No branch order is set on branch groups consisting of 1 branch.
        """
        self.get_grouped_branches()
        for group in self.branch_groups.values():
            if len(group) > 1:
                self.set_branch_order(group)

    @no_type_check
    def make_nodes_to_branch_map(self) -> None:
        """Map nodes connected to each branch."""
        # Note: first node is upstream, second node is downstream
        self.nodes_to_branch_map = {
            b: [self.mesh1d.description1d["network_node_ids"][_idx - 1] for _idx in idx]
            for b, idx in zip(
                self.mesh1d.description1d["network_branch_ids"],
                self.mesh1d.get_values("nedge_nodes", as_array=True),
            )
        }

    @no_type_check
    def make_branches_to_node_map(self) -> None:
        """Map branches connected to each node."""
        self.make_nodes_to_branch_map()
        self.branches_to_node_map = {
            n: [k for k, v in self.nodes_to_branch_map.items() if n in v]
            for n in self.mesh1d.description1d["network_node_ids"]
        }

    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def generate_nodes_with_bedlevels(
        self,
        resolve_at_bifurcation_method: str = "min",
        return_reversed_branches: bool = False,
    ):
        """Generate nodes with upstream and downstream bed levels from the cross sections on a branch.

        It takes into account whether or not branch order is specified (so interpolation
        over nodes is set).

        Nodes in between cross-sections on same branch/branch-group are linearly interpolated. Outside are extrapolated
        constant (e.g. end points of branch).

        Branch groups which include branch with reversed direction compared to whole group will be taken into account.
        Use return_reversed_branches=True to return that information

        Specify with resolve_at_bifurcation_method how to resolve bedlevel at bifurcation of more than 2 branches using
        options 'min' (minimum), 'max' (maximum), 'mean' (average).
        """
        assert resolve_at_bifurcation_method in ["min", "max", "mean"], (
            "Incorrect value for 'resolve_at_bifurcation_method' supplied. Either use 'min', 'max' or 'mean'"
        )
        bedlevels_crs_branches = self.hydamo.crosssections.get_bottom_levels()
        branch_order = self.mesh1d.get_values("nbranchorder", as_array=True)
        self.make_branches_to_node_map()
        self.make_nodes_to_branch_map()
        nodes_dict = {n: {"up": [], "down": []} for n in self.branches_to_node_map}
        reserved_branches = []
        for order, (branch, nodes) in tqdm(
            zip(branch_order, self.nodes_to_branch_map.items()),
            total=len(branch_order),
            desc="Getting bedlevels",
        ):
            if order == -1:
                # No branch order so just get upstream and downstream levels
                branch_length = self.branches.loc[branch, "geometry"].length
                subset = bedlevels_crs_branches.loc[
                    bedlevels_crs_branches["branchid"] == branch
                ]
                if subset.empty:
                    continue  # if this happens, crs is not defined. This can be a problem.
                nodes_dict[nodes[0]]["up"].append(
                    np.interp(0.0, subset["chainage"], subset["minz"])
                )
                nodes_dict[nodes[1]]["down"].append(
                    np.interp(branch_length, subset["chainage"], subset["minz"])
                )
            else:
                # In case of branch order, first collect all branches and set in them in order of up- to downstream
                all_branches = [
                    self.mesh1d.description1d["network_branch_ids"][i]
                    for i in np.argwhere(order == branch_order).ravel()
                ]
                all_nodes = [self.nodes_to_branch_map[b] for b in all_branches]
                # First check if any of the branches has a bedlevel from a cross-section profile otherwise skip
                check = all(
                    bedlevels_crs_branches.loc[
                        bedlevels_crs_branches["branchid"] == b
                    ].empty
                    for b in all_branches
                )
                if check:
                    continue  # if this happens, cross-section is not defined. This can be a problem.

                # Check if every branch is from up to down direction. Otherwise fix by reversing
                n = 0
                n_length = len(all_nodes)
                direction = list(np.ones(n_length))
                max_tries = 0
                while n < n_length:
                    up = (
                        np.count_nonzero(
                            all_nodes[n][0] == np.array([x[0] for x in all_nodes])
                        )
                        == 1
                    )
                    down = (
                        np.count_nonzero(
                            all_nodes[n][1] == np.array([x[1] for x in all_nodes])
                        )
                        == 1
                    )
                    if (not up) or (not down):
                        # Reverse
                        all_nodes[n] = list(np.flip(all_nodes[n]))
                        direction[n] = direction[n] * -1
                    n += 1
                    # Check if indeed everything is now in proper direction. Otherwise try again
                    if n == n_length:
                        up = all(
                            np.count_nonzero(
                                node[0] == np.array([x[0] for x in all_nodes])
                            )
                            == 1
                            for node in all_nodes
                        )
                        down = all(
                            np.count_nonzero(
                                node[1] == np.array([x[1] for x in all_nodes])
                            )
                            == 1
                            for node in all_nodes
                        )
                        if (not up) or (not down):
                            n = 0
                            max_tries += 1
                    if max_tries > 500:
                        logger.warning(
                            "Can't fix correct directions branch groups %s",
                            all_branches,
                        )
                        break

                # Add reserved branches to return
                reserved_branches.extend(
                    [b for b, d in zip(all_branches, direction) if d == -1]
                )

                # Get most upstream node. Otherwise just pick i_upstream = 0 as starting point
                i_upstream = [
                    i
                    for i, n in enumerate([x[0] for x in all_nodes])
                    if n not in [x[1] for x in all_nodes]
                ]
                if len(i_upstream) == 1:
                    i_upstream = i_upstream[0]
                else:
                    # It could be that branch order group forms a ring. In this case check first which node has more
                    # than 2 branches (bifurcation) or just 1 branch (boundary) connected.
                    i_upstream = [
                        i
                        for i, n in enumerate([x[0] for x in all_nodes])
                        if (len(self.branches_to_node_map[n]) > 2)
                        or (len(self.branches_to_node_map[n]) == 1)
                    ]
                    if len(i_upstream) == 1:
                        i_upstream = i_upstream[0]
                    else:
                        raise ValueError(
                            f"Something is not right with the branch order group {all_branches}"
                        )

                # Now put branch list in correct order
                all_branches_sorted = []
                all_nodes_sorted = []
                direction_sorted = []
                for _ in range(len(all_branches)):
                    all_branches_sorted.append(all_branches[i_upstream])
                    all_nodes_sorted.append(all_nodes[i_upstream])
                    direction_sorted.append(direction[i_upstream])
                    next_upstream = next(
                        (
                            i
                            for i, n in enumerate([x[0] for x in all_nodes])
                            if [x[1] for x in all_nodes][i_upstream] == n
                        ),
                        None,
                    )
                    if next_upstream is None:
                        break
                    i_upstream = next_upstream
                # Stitch chainages and bedlevels together
                chainage, bedlevel = [], []
                branch_length = 0
                for b, d in zip(all_branches_sorted, direction_sorted):
                    subset = bedlevels_crs_branches.loc[
                        bedlevels_crs_branches["branchid"] == b
                    ]
                    chain, bed = subset["chainage"], subset["minz"]
                    # Reverse chainage and bedlevel arrays
                    if d == -1:
                        chain = np.flip(chain)
                        bed = np.flip(bed)
                    chainage.extend(chain + branch_length)
                    bedlevel.extend(bed)
                    branch_length = self.branches.loc[b, "geometry"].length
                # Get chainage of up- and downstream node of loop branch within the overall branch
                if len(all_branches_sorted) == 1:
                    up_node_chainage = 0
                    down_node_chainage = self.branches.loc[
                        all_branches_sorted[0], "geometry"
                    ].length
                else:
                    i = np.argmax(
                        [
                            1 if ((nodes == n) or (list(np.flip(nodes)) == n)) else 0
                            for n in all_nodes_sorted
                        ]
                    )
                    up_node_chainage = sum(
                        [0]
                        + [
                            self.branches.loc[b, "geometry"].length
                            for b, n in zip(
                                all_branches_sorted[:-1], all_nodes_sorted[:-1]
                            )
                        ][: i + 1]
                    )
                    down_node_chainage = sum(
                        [
                            self.branches.loc[b, "geometry"].length
                            for b, n in zip(all_branches_sorted, all_nodes_sorted)
                        ][: i + 1]
                    )
                # Finally interpolate
                nodes_dict[nodes[0]]["up"].append(
                    np.interp(up_node_chainage, chainage, bedlevel)
                )
                nodes_dict[nodes[1]]["down"].append(
                    np.interp(down_node_chainage, chainage, bedlevel)
                )

        # Summarize everything and save
        nodes = list(nodes_dict.keys())
        node_geom = [
            Point(x, y)
            for x, y in zip(
                self.mesh1d.get_values("nnodex"), self.mesh1d.get_values("nnodey")
            )
        ]
        if resolve_at_bifurcation_method == "min":
            upstream_bedlevel = [
                np.min(v["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.min(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        elif resolve_at_bifurcation_method == "max":
            upstream_bedlevel = [
                np.max(v["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.max(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        elif resolve_at_bifurcation_method == "mean":
            upstream_bedlevel = [
                np.average(["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.average(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        else:
            raise NotImplementedError

        self.nodes = gpd.GeoDataFrame(
            index=nodes_dict.keys(),
            data={
                "code": nodes_dict.keys(),
                "upstream_bedlevel": upstream_bedlevel,
                "downstream_bedlevel": downstream_bedlevel,
            },
            geometry=node_geom,
        )

        if return_reversed_branches:
            return list(np.unique(reserved_branches))

    @no_type_check
    def get_grouped_branches(self) -> None:
        """Get grouped branch ids to use in set_branch_order function."""
        # Get all network data
        branch_ids = self.mesh1d.description1d["network_branch_ids"]
        # node_ids = self.mesh1d.description1d["network_node_ids"]
        # branch_edge_nodes_idx = self.mesh1d.get_values("nedge_nodes", as_array=True)
        # Collect all node ids per branch and all branches per node id
        self.make_nodes_to_branch_map()
        self.make_branches_to_node_map()

        branch_ids_checked = []
        groups = {0: []}
        for branch_id in branch_ids:
            if branch_id in branch_ids_checked:
                continue

            connected_nodes = self.nodes_to_branch_map[branch_id]  # get connected nodes
            for n in connected_nodes:
                b = self.branches_to_node_map[n]  # get connected branches
                if len(b) > 2:
                    continue  # in this case there's a bifurcation so skip
                elif len(b) == 1:
                    groups[list(groups.keys())[-1] + 1] = b  # b is already a list
                    branch_ids_checked.extend(b)
                    continue  # in this case the branch is not connected to other branches. make separate group
                else:
                    # remove branch from connected_branches because we are not interested in it
                    b = next(_b for _b in b if _b != branch_id)
                    if b in branch_ids_checked:
                        # connected branch is already added to a group, so this means that branch should be added to
                        # that group
                        groups[next(k for k, v in groups.items() if b in v)].append(
                            branch_id
                        )
                        branch_ids_checked.extend([branch_id])
                    else:
                        groups[list(groups.keys())[-1] + 1] = [
                            branch_id
                        ]  # otherwise add to group
                        branch_ids_checked.extend([branch_id])
                branch_ids_checked = list(np.unique(branch_ids_checked))
        groups.pop(0)  # remove the 0th group because empty

        # The branches are grouped but not fully connected (although routine above should do the trick in theory).
        # Try merging groups if a branch is in multiple groups.
        # In that case we know that branch groups are connected to eachother and is in fact a bigger group
        for b in branch_ids:
            _groups = groups.copy()  # make copy to apply changes on
            _k = -1  # default index
            # Loop over groups
            for k, v in groups.items():
                if b in v:  # if branch in grouped branches
                    if _k == -1:
                        _k = k  # set index to first group found
                    else:
                        _groups[_k].extend(
                            v
                        )  # otherwise add group to first found group
                        _groups[_k] = list(
                            np.unique(_groups[_k])
                        )  # remove duplicates due to add
                        _groups.pop(k)  # and remove group from groups
            groups = _groups.copy()  # copy changed dict over original
        # One pass over all branches should be sufficient to group everything together. Otherwise raise error
        if (
            max(
                [
                    sum([1 if b in v else 0 for k, v in groups.items()])
                    for b in branch_ids
                ]
            )
            > 1
        ):
            raise ValueError(
                f"Still branches contained in multiple groups. Maximum number of groups where this "
                f"happens: {max([sum([1 if b in v else 0 for k, v in groups.items()]) for b in branch_ids])}"
            )

        # save
        self.branch_groups = groups.copy()

    @no_type_check
    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def get_node_idx_offset(
        self, branch_id: str, pt: shapely.geometry.Point, nnodes: int = 1
    ) -> tuple:
        """Get the index and offset of the nearest node on a 1d branch."""
        # Project the point on the branch
        dist = self.schematised[branch_id].project(pt)

        # Get the branch data from the networkdata
        branchidx = (
            self.mesh1d.description1d["network_branch_ids"].index(
                self.str2chars(branch_id, self.idstrlength)
            )
            + 1
        )
        pt_branch_id = self.mesh1d.get_values("branchidx", as_array=True)
        idx = np.nonzero(pt_branch_id == branchidx)

        # Find nearest offset
        offsets = self.mesh1d.get_values("branchoffset", as_array=True)[idx]
        isorted = np.argsort(np.absolute(offsets - dist))
        isorted = isorted[: min(nnodes, len(isorted))]

        # Get the offset
        offset = [offsets[imin] for imin in isorted]
        # Get the id of the node
        node_id = [idx[0][imin] + 1 for imin in isorted]

        return node_id, offset


class CrossSections:
    """Cross section definitions and locations of the model.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO object they belong to.
    """

    def __init__(self, hydamo: HyDAMO) -> None:
        """Create empty cross section definitions and locations."""
        self.hydamo = hydamo
        self.crosssections: list = []
        self.default_definition: str | None = None
        self.default_definition_shift = 0.0
        self.default_location = ""

        self.crosssection_loc: dict[str, dict] = {}
        self.crosssection_def: dict[str, dict] = {}

        self.get_roughnessname = self.get_roughness_description

        self.convert = CrossSectionsIO(self)

    def get_roughness_description(self, roughnesstype: str, value: float) -> str:
        """Return the name of a roughness definition, added if it is new.

        Parameters
        ----------
        roughnesstype : str
            HyDAMO roughness type, mapped to the D-HYDRO type.
        value : float
            Roughness value.

        Returns
        -------
        str
            Name of the definition, e.g. ``"Manning_0.03"``.

        Raises
        ------
        ValueError
            If the value is NaN.
        """
        if np.isnan(float(value)):
            raise ValueError("Roughness value should not be NaN.")

        # map HyDAMO definition to D-Hydro definition
        roughnesstype = self.hydamo.roughness_mapping[roughnesstype]

        # Get name
        name = f"{roughnesstype}_{float(value)}"

        # Check if the description is already known
        if name.lower() in map(str.lower, self.hydamo.roughness_definitions.keys()):
            return name

        # Add to dict
        self.hydamo.roughness_definitions[name] = {
            "frictionid": name,
            "frictiontype": roughnesstype,
            "frictionvalue": value,
        }

        return name

    def set_default_definition(self, definition: str, shift: float = 0.0) -> None:
        """Set the cross section definition for branches without a cross section.

        It is used by ``Df2HydrolibModel(assign_default_profiles=True)``.

        Parameters
        ----------
        definition : str
            Name of an existing cross section definition.
        shift : float, optional
            Vertical shift of the definition. Defaults to 0.0.

        Raises
        ------
        KeyError
            If the definition does not exist.
        """
        if definition not in self.crosssection_def:
            raise KeyError(f'Cross section definition "{definition}" not found."')

        self.default_definition = definition
        self.default_definition_shift = shift

    def set_default_locations(self, locations: list[str]) -> None:
        """Set the branches that get the default cross section definition.

        Parameters
        ----------
        locations : list[str]
            Branch ids.
        """
        self.default_locations = locations

    def add_yz_definition(
        self,
        yz: np.ndarray | None = None,
        thalweg: float | None = None,
        roughnesstype: str | None = None,
        roughnessvalue: float | None = None,
        name: str | None = None,
    ) -> str:
        """Add a yz cross section definition.

        Parameters
        ----------
        code : str
            Id of cross section
        branch : str
            Name of branch
        offset : float
            Position of cross section along branch. If not given, the position is determined
            from the branches in the network. These should thus be given in this case.
        crds : np.array
            Nx2 array with y, z coordinates
        """
        if (
            yz is None
            or thalweg is None
            or roughnesstype is None
            or roughnessvalue is None
        ):
            raise ValueError(
                "yz, thalweg, roughnesstype and roughnessvalue are required."
            )
        # get coordinates
        length, z = yz.T
        if name is None:
            name = f"yz_{yz}:08d"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "yz",
            "thalweg": np.round(thalweg, decimals=3),
            "yzcount": len(z),
            "ycoordinates": self.hydamo.list_to_str(length),
            "zcoordinates": self.hydamo.list_to_str(z),
            "sectioncount": 1,
            "frictionids": roughnessname,
            "frictionpositions": self.hydamo.list_to_str([length[0], length[-1]]),
        }

        return name

    def add_circle_definition(
        self,
        diameter: float,
        roughnesstype: str,
        roughnessvalue: float,
        name: str | None = None,
    ) -> str:
        """Add a circle cross section definition.

        Without a name, the name is derived from the shape only, so a definition with the
        same shape but another roughness replaces the earlier one.
        """
        # Get name if not given
        if name is None:
            name = f"circ_d{diameter:.3f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "circle",
            "thalweg": 0.0,
            "diameter": diameter,
            "frictionid": roughnessname,
        }

        return name

    def add_rectangle_definition(
        self,
        height: float,
        width: float,
        closed: bool | int,
        roughnesstype: str,
        roughnessvalue: float,
        name: str | None = None,
    ) -> str:
        """Add a rectangle cross section definition.

        Without a name, the name is derived from the shape only, so a definition with the
        same shape but another roughness replaces the earlier one.
        """
        # Get name if not given
        if name is None:
            name = f"rect_h{height:.3f}_w{width:.3f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "rectangle",
            "thalweg": 0.0,
            "height": height,
            "width": width,
            "closed": int(closed),
            "frictionid": roughnessname,
        }

        return name

    def add_trapezium_definition(
        self,
        slope: float,
        maximumflowwidth: float,
        bottomwidth: float,
        closed: bool | int,
        roughnesstype: str,
        roughnessvalue: float,
        bottomlevel: float | None = None,
        name: str | None = None,
    ) -> str:
        """Add a trapezium cross section definition.

        Without a name, the name is derived from the shape only, so a definition with the
        same shape but another roughness replaces the earlier one.
        """
        # Get name if not given
        if name is None:
            name = f"trapz_s{slope:.1f}_bw{bottomwidth:.1f}_bw{maximumflowwidth:.1f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        if bottomlevel is None:
            bottomlevel = 0.0

        if not closed:
            # an open profile ends at the bank top (maximumflowwidth); above it D-Flow FM continues vertically
            dh = (maximumflowwidth - bottomwidth) / (2.0 * slope)
            if np.isfinite(dh) and dh > 0:
                levels = f"{bottomlevel} {bottomlevel + dh:.3f}"
                flowwidths = f"{bottomwidth:.2f} {maximumflowwidth:.2f}"
            else:
                # bank height missing or below the bed: widen the profile up to a level of 100 m instead (as before)
                logger.warning(
                    f"Trapezium profile {name}: no valid height up to the bank top (maximum flow width "
                    f"{maximumflowwidth}, bottom width {bottomwidth}, slope {slope}). The profile is widened up to "
                    "a level of 100 m instead; check the bank heights (hoogte insteek) in the data."
                )
                levels = f"{bottomlevel} 100"
                flowwidths = f"{bottomwidth:.2f} {bottomwidth + 2.0 * ((100.0 - bottomlevel) * slope):.2f}"
        else:
            levels = f"0 {((maximumflowwidth - bottomwidth) / 2.0) / slope:.2f}"
            flowwidths = f"{bottomwidth:.2f} {maximumflowwidth:.2f}"

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "zw",
            "thalweg": 0.0,
            "numlevels": 2,
            "levels": levels,
            "flowwidths": flowwidths,
            "totalwidths": flowwidths,
            "frictionid": roughnessname,
        }

        return name

    def add_zw_definition(
        self,
        numLevels: int,
        levels: str,
        flowWidths: str,
        totalWidths: str,
        roughnesstype: str,
        roughnessvalue: float,
        name: str | None = None,
    ) -> str:
        """Add a zw cross section definition.

        Give a name: without one, building the name raises an error.
        """
        # Get name if not given
        if name is None:
            name = f"zw_h{levels.replace(' ', '_'):.1f}_w{flowWidths.replace(' ', '_'):.1f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "zw",
            "thalweg": 0.0,
            "numlevels": int(numLevels),
            "levels": levels,
            "flowwidths": flowWidths,
            "totalwidths": totalWidths,
            "frictionid": roughnessname,
        }

        return name

    def add_crosssection_location(
        self,
        branchid: str,
        chainage: float,
        definition: str,
        minz: float = np.nan,
        shift: float = 0.0,
    ) -> None:
        """Add a cross section location, with the id ``{branchid}_{chainage:.1f}``.

        Parameters
        ----------
        branchid : str
            Branch id.
        chainage : float
            Chainage on the branch.
        definition : str
            Name of the cross section definition.
        minz : float, optional
            Not used. Defaults to NaN.
        shift : float, optional
            Vertical shift of the definition. Defaults to 0.0.
        """
        descr = f"{branchid}_{chainage:.1f}"
        # Add cross section location
        self.crosssection_loc[descr] = {
            "id": descr,
            "branchid": branchid,
            "chainage": chainage,
            "shift": shift,
            "definitionId": definition,
        }

    def get_branches_without_crosssection(self) -> list[str]:
        """Return the ids of the branches without a cross section location."""
        # First find all branches that match a cross section
        branch_ids = {dct["branchid"] for _, dct in self.crosssection_loc.items()}
        # Select the branch-ids that do nog have a matching cross section
        branches = self.hydamo.branches
        no_crosssection = branches.index[~np.isin(branches.index, list(branch_ids))]

        return no_crosssection.tolist()

    def get_structures_without_crosssection(self) -> list[str]:
        """Return the cross section definition ids of bridges and culverts that do not exist."""
        csdef_ids = [dct["id"] for _, dct in self.crosssection_def.items()]
        no_crosssection: list[str] = []
        bridge_ids = [
            dct["csdefid"] for _, dct in self.hydamo.structures.bridges_df.iterrows()
        ]
        no_cross_bridge = np.asarray(bridge_ids)[
            ~np.isin(bridge_ids, csdef_ids)
        ].tolist()
        no_crosssection = no_crosssection + no_cross_bridge
        culvert_ids = [
            dct["csdefid"] for _, dct in self.hydamo.structures.culverts_df.iterrows()
        ]
        no_cross_culvert = np.asarray(culvert_ids)[
            ~np.isin(culvert_ids, csdef_ids)
        ].tolist()
        no_crosssection = no_crosssection + no_cross_culvert
        return no_crosssection

    # legacy from delft3dfmpy, not used and not working here (self.dflowfmmodel does not exist): mypy skips it
    @no_type_check
    def get_bottom_levels(self):
        """Determine bottom levels from cross sections (legacy from delft3dfmpy, not working)."""
        # Initialize lists
        data = []
        geometry = []

        for css in self.crosssection_loc.values():
            # Get location
            geometry.append(
                self.dflowfmmodel.network.schematised.loc[
                    css["branchid"], "geometry"
                ].interpolate(css["chainage"])
            )
            shift = css["shift"]

            # Get depth from definition if yz and shift
            definition = self.crosssection_def[css["definitionId"]]
            minz = shift
            if definition["type"] == "yz":
                minz += min(float(z) for z in definition["zCoordinates"].split())

            data.append([css["branchid"], css["chainage"], minz])

        # Add to geodataframe
        gdf = gpd.GeoDataFrame(
            data=data, columns=["branchid", "chainage", "minz"], geometry=geometry
        )
        return gdf

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def crosssection_to_yzprofiles(
        self,
        crosssections: gpd.GeoDataFrame | ExtendedGeoDataFrame,
        roughness: ExtendedDataFrame,
        branches: ExtendedGeoDataFrame | None,
        roughness_variant: RoughnessVariant | None = None,
    ) -> dict:
        """Convert the HyDAMO cross sections (dwarsprofiel) to D-Flow FM input.

        Parameters
        ----------
        crosssections : gpd.GeoDataFrame
            GeoDataFrame with x,y,z-coordinates of cross sections

        Returns
        -------
        dictionary
            Dictionary with attributes of cross sections, usable for dflowfm
        """
        cssdct = {}

        css_columns = zip(
            crosssections["code"],
            crosssections.geometry,
            crosssections["branch_id"],
            crosssections["branch_offset"],
            crosssections["globalid"],
        )
        for code, geometry, branch_id, branch_offset, globalid in css_columns:
            # The cross sections from hydamo are all yz profiles

            # Determine yz_values
            xyz = np.vstack(geometry.coords[:])
            length = np.r_[
                0, np.cumsum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1])))
            ]
            yz = np.c_[length, xyz[:, -1]]
            # the GUI cannot cope with identical y-coordinates. Add 1 cm to a 2nd duplicate.
            yz[:, 0] = np.round(yz[:, 0], 3)
            for i in range(1, yz.shape[0]):
                if yz[i, 0] <= yz[i - 1, 0]:
                    yz[i, 0] = yz[i - 1, 0] + 0.01

            # determine thalweg
            if branches is not None:
                branch_geom = branches.geometry[branches.code == branch_id].iloc[0]
                crossing = geometry.intersection(branch_geom)
                if crossing.geom_type == "MultiPoint":
                    crossing = crossing.geoms[0]
                thalweg_xyz = crossing.coords[0]
                # and the Y-coordinate of the thalweg
                thalweg = np.hypot(
                    thalweg_xyz[0] - xyz[0, 0], thalweg_xyz[1] - xyz[0, 1]
                )
            else:
                thalweg = 0.0

            if roughness_variant == RoughnessVariant.HIGH:
                ruwheid = roughness[roughness["profielpuntid"] == globalid].ruwheidhoog
            if roughness_variant == RoughnessVariant.LOW:
                ruwheid = roughness[roughness["profielpuntid"] == globalid].ruwheidlaag

            # Add to dictionary
            cssdct[code] = {
                "branchid": branch_id,
                "chainage": branch_offset,
                "yz": yz,
                "thalweg": thalweg,
                "typeruwheid": roughness[
                    roughness["profielpuntid"] == globalid
                ].typeruwheid.to_numpy()[0],
                "ruwheid": float(ruwheid.iloc[0]),
            }

        return cssdct

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def parametrised_to_profiles(
        self,
        parametrised: ExtendedDataFrame,
        parametrised_values: ExtendedDataFrame,
        branches: list,
        roughness_variant: RoughnessVariant | None = None,
    ) -> dict:
        """Generate parametrised cross sections for all branches, or those without one.

        Parameters
        ----------
        parametrised : pd.DataFrame
            GeoDataFrame with geometries and attributes of parametrised profiles.
        branches : list
            List of branches for which the parametrised profiles are derived

        Returns
        -------
        dictionary
            Dictionary with attributes of cross sections, usable for dflowfm
        """
        cssdct = {}
        branch_relation = DamoRelations.PARAM_PROFILE_BRANCH
        values_relation = DamoRelations.PARAM_PROFILE_VALUES

        # index label of the branch per branch id (the first matching branch)
        branch_index: dict[str, Hashable] = {}
        for branch in branches:
            branch_index.setdefault(
                getattr(branch, branch_relation.parent_key), branch.Index
            )

        for _, param in parametrised.iterrows():
            param_row = param.to_frame().T
            branch_id = param[branch_relation.foreign_key]
            index = branch_index[branch_id]
            values = values_relation.children_of(param_row, parametrised_values)

            # Drop profiles for which not enough data is available to write (as rectangle)
            # nulls = pd.isna(parambranches[['bodembreedte', 'bodemhoogtebenedenstrooms', 'bodemhoogtebovenstrooms']]).any(axis=1).to_numpy()
            # parambranches = parambranches.drop(ExtendedGeoDataFrame(geotype=LineString), parambranches.index[nulls], index_col='code',axis=0)
            # parambranches.drop(parambranches.index[nulls], inplace=True)

            if pd.isna(
                values[values.soortparameter == "bodemhoogte benedenstrooms"].waarde
            ).to_numpy()[0]:
                logger.warning(
                    "bodemhoogte benedenstrooms not available for profile %s.",
                    param.globalid,
                )
            if pd.isna(
                values[values.soortparameter == "bodembreedte"].waarde
            ).to_numpy()[0]:
                logger.warning(
                    "bodembreedte not available for profile %s.",
                    param.globalid,
                )
            if pd.isna(
                values[values.soortparameter == "bodemhoogte bovenstrooms"].waarde
            ).to_numpy()[0]:
                logger.warning(
                    "bodemhoogte bovenstrooms not available for profile %s.",
                    param.globalid,
                )

            # Determine characteristics
            botlev_upper = values[
                values.soortparameter == "bodemhoogte bovenstrooms"
            ].waarde.to_numpy()[0]
            botlev_lower = values[
                values.soortparameter == "bodemhoogte benedenstrooms"
            ].waarde.to_numpy()[0]

            if pd.isna(
                values[values.soortparameter == "taludhelling linkerzijde"].waarde
            ).to_numpy()[0]:
                css_type = "rectangle"
            else:
                css_type = "trapezium"
                dh1 = (
                    values[
                        values.soortparameter == "hoogte insteek linkerzijde"
                    ].waarde.to_numpy()[0]
                    - (botlev_upper + botlev_lower) / 2.0
                )
                dh2 = (
                    values[
                        values.soortparameter == "hoogte insteek rechterzijde"
                    ].waarde.to_numpy()[0]
                    - (botlev_upper + botlev_lower) / 2.0
                )
                # height = (dh1 + dh2) / 2.0
                # Determine maximum flow width and slope (both needed for output)
                maxflowwidth = (
                    values[values.soortparameter == "bodembreedte"].waarde.to_numpy()[0]
                    + values[
                        values.soortparameter == "taludhelling linkerzijde"
                    ].waarde.to_numpy()[0]
                    * dh1
                    + values[
                        values.soortparameter == "taludhelling rechterzijde"
                    ].waarde.to_numpy()[0]
                    * dh2
                )
                slope = (
                    values[
                        values.soortparameter == "taludhelling linkerzijde"
                    ].waarde.to_numpy()[0]
                    + values[
                        values.soortparameter == "taludhelling rechterzijde"
                    ].waarde.to_numpy()[0]
                ) / 2.0

            if roughness_variant == RoughnessVariant.LOW:
                roughness = values.ruwheidlaag.to_numpy()[0]
            elif roughness_variant == RoughnessVariant.HIGH:
                roughness = values.ruwheidhoog.to_numpy()[0]
            else:
                raise ValueError(
                    'Invalid value for roughness_variant; should be "High" or "Low".'
                )
            # Determine name for cross section
            if css_type == "trapezium":
                cssdct[index] = {
                    "type": css_type,
                    "slope": round(slope, 2),
                    "maximumflowwidth": round(maxflowwidth, 1),
                    "bottomwidth": round(
                        values[
                            values.soortparameter == "bodembreedte"
                        ].waarde.to_numpy()[0],
                        3,
                    ),
                    "closed": 0,
                    "thalweg": 0.0,
                    "typeruwheid": values.typeruwheid.to_numpy()[0],
                    "ruwheid": roughness,
                    "bottomlevel_upper": botlev_upper,
                    "bottomlevel_lower": botlev_lower,
                }
            elif css_type == "rectangle":
                cssdct[index] = {
                    "type": css_type,
                    "height": 5.0,
                    "width": round(
                        values[
                            values.soortparameter == "bodembreedte"
                        ].waarde.to_numpy()[0],
                        3,
                    ),
                    "closed": 0,
                    "thalweg": 0.0,
                    "typeruwheid": values.typeruwheid.to_numpy()[0],
                    "ruwheid": roughness,
                    "bottomlevel_upper": botlev_upper,
                    "bottomlevel_lower": botlev_lower,
                }

        return cssdct


def _boundary_crossings(line: LineString, polygons: gpd.GeoDataFrame) -> list[float]:
    """Chainages where a branch crosses the boundary of one of the polygons.

    Parameters
    ----------
    line : LineString
        Branch geometry.
    polygons : gpd.GeoDataFrame
        Polygons that intersect the branch.

    Returns
    -------
    list[float]
        Sorted, unique chainages of the crossings.
    """
    crossings: set[float] = set()
    for boundary in polygons.geometry.boundary:
        intersection = line.intersection(boundary)
        for geom in getattr(intersection, "geoms", [intersection]):
            if not geom.is_empty:
                crossings.update(line.project(Point(c)) for c in geom.coords)
    return sorted(crossings)


def _value_at_chainage(
    point: Point,
    chainage: float,
    polygons: gpd.GeoDataFrame,
    value_column: str,
    original: pd.DataFrame,
) -> float:
    """Value at a point on a branch: of the last polygon that contains it, otherwise of the model result.

    Parameters
    ----------
    point : Point
        Point on the branch.
    chainage : float
        Chainage of the point.
    polygons : gpd.GeoDataFrame
        Polygons with values that intersect the branch.
    value_column : str
        Column in polygons with the values.
    original : pd.DataFrame
        Model values on the branch, sorted by chainage, with columns 'chainage' and 'value'; interpolated
        linearly if the point is outside all polygons.

    Returns
    -------
    float
        The value at the point.
    """
    inside = polygons[polygons.geometry.intersects(point)]
    if len(inside):
        return float(inside[value_column].iloc[-1])
    return float(np.interp(chainage, original["chainage"], original["value"]))


class ExternalForcings:
    """Boundary conditions, laterals and initial and parameter fields of the model.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO object they belong to.
    """

    def __init__(self, hydamo: HyDAMO) -> None:
        self.hydamo = hydamo

        self.initial_waterlevel_polygons = gpd.GeoDataFrame(
            columns=["waterlevel", "geometry", "locationtype"]
        )
        self.initial_waterdepth_polygons = gpd.GeoDataFrame(
            columns=["waterdepth", "geometry", "locationtype"]
        )
        self.missing = None
        self.spatial_fields: list[dict] = []
        # number of the next spatial field, used in the generated file names
        self._next_field_number = 0

        self.boundary_nodes: dict[str, dict] = {}
        self.lateral_nodes: dict[str, dict] = {}
        self.pattern = "^[{]?[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}[}]?$"

        self.convert = ExternalForcingsIO(self)

    def set_initial_waterlevel(
        self,
        initial_values: float | gpd.GeoDataFrame,
        *,
        locationtype: str = "1d",
        value_column: str | None = None,
    ) -> None:
        """Set the initial water level: a uniform value, or a value per polygon.

        A uniform value for 1D (locationtype '1d') is written per branch for all branches, as a spatial field at
        the position of this call: fields added after it override it where they apply, fields added before it are
        overridden. Calling it again replaces the earlier uniform water level. For 2D or 'all', a uniform value is
        written as a [Global] value.

        Polygons (1D only) are passed on to set_spatial_field and written per branch. For other spatially varying
        input, such as a model result, use set_spatial_field directly.

        Parameters
        ----------
        initial_values : float or gpd.GeoDataFrame
            Uniform water level, or polygons with water levels.
        locationtype : str, optional
            ``"1d"``, ``"2d"`` or ``"all"``. Defaults to ``"1d"``.
        value_column : str, optional
            Column with the water levels, required for polygons. Defaults to None.
        """
        self._set_initial("waterlevel", initial_values, locationtype, value_column)

    # legacy from delft3dfmpy, not used and not working here (self.mdu_parameters does not exist): mypy skips it
    @no_type_check
    def set_missing_waterlevel(self, missing):
        """Set the missing value for the water level (legacy from delft3dfmpy, not working).

        This overwrites the water level at missing value in the mdu file.

        Parameters
        ----------
        missing : float
            Water depth
        """
        self.mdu_parameters["WaterLevIni"] = missing

    def set_initial_waterdepth(
        self,
        initial_values: float | gpd.GeoDataFrame,
        *,
        locationtype: str = "1d",
        value_column: str | None = None,
    ) -> None:
        """Set the initial water depth: a uniform value, or a value per polygon.

        A uniform value for 1D (locationtype '1d') is written per branch for all branches, as a spatial field at
        the position of this call: fields added after it override it where they apply, fields added before it are
        overridden. Calling it again replaces the earlier uniform water depth. For 2D or 'all', a uniform value is
        written as a [Global] value; D-HYDRO 2026.02 then lets it override spatial water levels.

        Polygons (1D only) are passed on to set_spatial_field and written per branch. For other spatially varying
        input, such as a model result, use set_spatial_field directly.

        Parameters
        ----------
        initial_values : float or gpd.GeoDataFrame
            Uniform water depth, or polygons with water depths.
        locationtype : str, optional
            ``"1d"``, ``"2d"`` or ``"all"``. Defaults to ``"1d"``.
        value_column : str, optional
            Column with the water depths, required for polygons. Defaults to None.
        """
        self._set_initial("waterdepth", initial_values, locationtype, value_column)

    def _set_initial(
        self,
        quantity: str,
        initial_values: float | gpd.GeoDataFrame,
        locationtype: str,
        value_column: str | None,
    ) -> None:
        """Shared implementation of set_initial_waterlevel and set_initial_waterdepth."""
        if isinstance(initial_values, numbers.Real):
            if locationtype == "1d":
                # a field per branch; it leaves a uniform value for 2D or 'all' untouched
                self._set_uniform_field(quantity, float(initial_values))
            else:
                setattr(
                    self,
                    f"initial_{quantity}_polygons",
                    gpd.GeoDataFrame(
                        {
                            quantity: [float(initial_values)],
                            "geometry": [None],
                            "locationtype": [locationtype],
                        },
                        index=["global"],
                    ),
                )
            return

        if locationtype != "1d":
            raise ValueError(
                f"Initial {quantity} with polygons is only supported for 1D here. "
                "Use hydamo.external_forcings.set_spatial_field() for 2D."
            )
        if value_column is None:
            raise ValueError(
                f"Provide value_column with the initial {quantity} per polygon."
            )
        if not isinstance(initial_values, gpd.GeoDataFrame):
            raise TypeError(
                f"initial_values must be a number or a GeoDataFrame, got {type(initial_values).__name__}."
            )
        self.set_spatial_field(
            initial_values, value_column, quantity=quantity, locationtype="1d"
        )

    def _set_uniform_field(self, quantity: str, value: float) -> None:
        """Add a uniform 1D value as a spatial field per branch over all branches, replacing an earlier one.

        The values per branch are derived from the branches when the model is converted.
        """
        self.spatial_fields = [
            sf
            for sf in self.spatial_fields
            if not (sf.get("uniform") is not None and sf["quantity"] == quantity)
        ]
        self.spatial_fields.append(
            {
                "gdf": None,
                "column": None,
                "uniform": value,
                "quantity": quantity,
                "kind": "initial",
                "filetype": "branch",
                "cellsize": None,
                # "_1d" keeps it apart from the [Global] file of a uniform 2D value (inifields_to_dhydro)
                "filename": f"initial{quantity}_1d.ini",
                "locationtype": "1d",
                "interpolationmethod": None,
                "averagingtype": None,
                "operand": "O",
                "nodata": -999.0,
                "max_snap_distance": None,
            }
        )

    def correct_values_from_modelrun(
        self,
        model_values: gpd.GeoDataFrame,
        correction_polygons: gpd.GeoDataFrame,
        value_column: str,
        boundary_step: float | None = 0.01,
    ) -> gpd.GeoDataFrame:
        """Replace values from a model result by the values of correction polygons.

        All points within the polygons get the value of the polygon.

        If the points have the columns 'branchid' and 'chainage' (as from read_values_from_modelrun) and the HyDAMO
        branches are available, two points are added where a polygon boundary crosses a branch, boundary_step apart:
        the point inside gets the polygon value, the point outside the model value interpolated along the branch.
        The value then changes at the polygon boundary instead of gradually between the nearest model points.

        Parameters
        ----------
        model_values : gpd.GeoDataFrame
            Points with a column ``value``, e.g. from `read_values_from_modelrun`.
        correction_polygons : gpd.GeoDataFrame
            Polygons with the corrected values.
        value_column : str
            Column in ``correction_polygons`` with the corrected values.
        boundary_step : float, optional
            Distance between the two points at a polygon boundary; None adds no points.
            Defaults to 0.01.

        Returns
        -------
        gpd.GeoDataFrame
            Copy of ``model_values`` with corrected values (and the points at the polygon
            boundaries). Where polygons overlap, the last polygon is used.
        """
        model_values_corrected = model_values.copy(deep=True)
        if model_values_corrected.crs is None and correction_polygons.crs is not None:
            model_values_corrected = model_values_corrected.set_crs(
                correction_polygons.crs
            )

        # use a plain GeoDataFrame, an ExtendedGeoDataFrame loses the value column in the spatial join
        joined = gpd.sjoin(
            model_values_corrected[["geometry"]],
            gpd.GeoDataFrame(correction_polygons[[value_column, "geometry"]]),
            how="inner",
            predicate="intersects",
        )
        joined = joined[~joined.index.duplicated(keep="last")]

        model_values_corrected.loc[joined.index, "value"] = joined[value_column]

        branches = getattr(self.hydamo, "branches", None)
        if boundary_step is None or not {"branchid", "chainage"} <= set(
            model_values.columns
        ):
            return model_values_corrected
        if branches is None or branches.empty:
            logger.info(
                "No branches available: no points are added at the boundaries of the correction polygons."
            )
            return model_values_corrected

        boundary_points = self._points_at_polygon_boundaries(
            model_values,
            gpd.GeoDataFrame(
                correction_polygons[[value_column, "geometry"]]
            ).reset_index(drop=True),
            value_column,
            branches,
            boundary_step,
        )
        if boundary_points.empty:
            return model_values_corrected
        logger.info(
            f"{len(boundary_points)} points added at the boundaries of the correction polygons."
        )
        combined = pd.concat(
            [model_values_corrected, boundary_points], ignore_index=True
        )
        return gpd.GeoDataFrame(
            combined, geometry="geometry", crs=model_values_corrected.crs
        )

    @staticmethod
    def _points_at_polygon_boundaries(
        model_values: gpd.GeoDataFrame,
        polygons: gpd.GeoDataFrame,
        value_column: str,
        branches: gpd.GeoDataFrame,
        step: float,
    ) -> gpd.GeoDataFrame:
        """Two points per crossing of a polygon boundary with a branch, step apart, with the value on each side."""
        rows = []
        for branch_id, original in model_values.groupby("branchid"):
            if branch_id not in branches.index:
                continue
            line = branches.geometry.loc[branch_id]
            if not isinstance(line, LineString):
                raise TypeError(
                    f"Branch {branch_id} is a {line.geom_type}, expected a LineString."
                )
            candidates = polygons.iloc[
                polygons.sindex.query(line, predicate="intersects")
            ]
            if candidates.empty:
                continue
            original = original.sort_values("chainage")

            # two chainages per crossing of a polygon boundary, half a step on either side, within the branch
            chainages = [
                chainage
                for crossing in _boundary_crossings(line, candidates)
                for chainage in (crossing - 0.5 * step, crossing + 0.5 * step)
                if 0.0 <= chainage <= line.length
            ]
            for chainage in chainages:
                point = line.interpolate(chainage)
                value = _value_at_chainage(
                    point, chainage, candidates, value_column, original
                )
                rows.append(
                    {
                        "x": point.x,
                        "y": point.y,
                        "value": value,
                        "branchid": branch_id,
                        "chainage": chainage,
                        "geometry": point,
                    }
                )
        if not rows:
            return gpd.GeoDataFrame(
                columns=["x", "y", "value", "branchid", "chainage", "geometry"],
                geometry="geometry",
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=model_values.crs)

    def read_values_from_modelrun(
        self,
        netcdf_path: str | Path,
        netcdf_field: str,
        netcdf_index: int | None = None,
        crs: CRS | str | int | None = None,
        time: datetime | pd.Timestamp | str | None = None,
    ) -> gpd.GeoDataFrame:
        """Read the values of a 1D variable from a model result (map or fou file).

        Variables on the 1D nodes (e.g. 'mesh1d_s1', 'mesh1d_waterdepth') are returned at the node locations,
        variables on the 1D edges/flow links (e.g. 'mesh1d_u1') at the edge locations.

        If 'mesh1d_s1' is requested but the file only contains the water depth, the water level is computed
        as bed level + water depth.

        Missing values (dry nodes or links in a map-file) are replaced by the bed level for water levels,
        and by 0 for water depths, velocities and discharges; for other variables they are left out.

        If the file contains the 1D network topology, the columns 'branchid' and 'chainage' are added, so the
        values can be written per branch with set_spatial_field(..., filetype="branch"). For edge variables,
        the column 'direction' (angle of the link in radians) is added, as used by set_initial_velocity.

        Parameters
        ----------
        netcdf_path : str or Path
            Path to the netCDF file.
        netcdf_field : str
            Variable to read, e.g. ``"mesh1d_s1"`` (map) or ``"mesh1d_fourier001_max"`` (fou).
        netcdf_index : int, optional
            Time index, e.g. 0 for the first time step. Defaults to None: without a
            time, the last time step.
        crs : CRS, str or int, optional
            Coordinate reference system of the model. Defaults to None.
        time : datetime, pd.Timestamp or str, optional
            Time for which the values are read, e.g. ``"2016-06-02 12:00"``; the nearest
            time step is used. Defaults to None.

        Returns
        -------
        gpd.GeoDataFrame
            Points with the columns x, y and value (and branchid, chainage and
            direction, if available).
        """
        logger.info(f"Reading {netcdf_field} from {netcdf_path} ...")
        with Dataset(netcdf_path, "r") as results:  # Open file in read mode
            variable, from_depth = self._select_netcdf_variable(
                results, netcdf_field, netcdf_path
            )
            netcdf_index = self._resolve_netcdf_index(
                results, variable, netcdf_field, netcdf_path, netcdf_index, time
            )
            masked = np.ma.masked_invalid(
                variable[:] if netcdf_index is None else variable[netcdf_index, :]
            )
            if from_depth:
                # dry nodes (missing depth) get the bed level
                bed_level = np.ma.filled(results["mesh1d_flowelem_bl"][:], np.nan)
                masked = np.ma.masked_invalid(bed_level + np.ma.filled(masked, 0.0))

            # location of the variable: 1D nodes or 1D edges (flow links)
            location = "edge" if variable.dimensions[-1] == "mesh1d_nEdges" else "node"
            x = np.array(results[f"mesh1d_{location}_x"][:])
            y = np.array(results[f"mesh1d_{location}_y"][:])
            if masked.ndim != 1 or len(x) != len(masked):
                raise ValueError(
                    f"For {netcdf_field} in {netcdf_path}, length of coordinates and values are not equal."
                )

            topology = self._read_1d_topology(results, location)
            value, missing = self._fill_missing_values(
                masked, results, netcdf_field, location
            )
            if missing.any():
                logger.warning(
                    f"{missing.sum()} {location}s without a value in {netcdf_field} are left out."
                )
                x, y, value = x[~missing], y[~missing], value[~missing]
                topology = {key: values[~missing] for key, values in topology.items()}

        return gpd.GeoDataFrame(
            {"x": x, "y": y, "value": value, **topology},
            geometry=gpd.points_from_xy(x, y),
            crs=crs,
        )

    @staticmethod
    def _select_netcdf_variable(
        results: Dataset, netcdf_field: str, netcdf_path: str | Path
    ) -> tuple[Variable, bool]:
        """Return the variable to read from a model result.

        If 'mesh1d_s1' is requested but the file only contains the water depth, the water depth is returned,
        to compute the water level as bed level + water depth.

        Parameters
        ----------
        results : Dataset
            The opened map- or fou-file.
        netcdf_field : str
            Requested variable, e.g. 'mesh1d_s1'.
        netcdf_path : str | Path
            Path of the file, for messages.

        Returns
        -------
        tuple[Variable, bool]
            The variable, and whether it is the water depth that replaces 'mesh1d_s1'.

        Raises
        ------
        KeyError
            If the variable is not in the file.
        """
        from_depth = (
            netcdf_field == "mesh1d_s1"
            and netcdf_field not in results.variables
            and {"mesh1d_waterdepth", "mesh1d_flowelem_bl"} <= set(results.variables)
        )
        if from_depth:
            logger.info(
                "mesh1d_s1 is not in the file; the water level is computed as bed level + water depth."
            )
            return results["mesh1d_waterdepth"], True
        if netcdf_field in results.variables:
            return results[netcdf_field], False
        raise KeyError(f"{netcdf_field} not found in {netcdf_path}.")

    @staticmethod
    def _resolve_netcdf_index(
        results: Dataset,
        variable: Variable,
        netcdf_field: str,
        netcdf_path: str | Path,
        netcdf_index: int | None,
        time: datetime | pd.Timestamp | str | None,
    ) -> int | None:
        """Time index to read, from netcdf_index or time; the last time step if neither is given.

        Parameters
        ----------
        results : Dataset
            The opened map- or fou-file.
        variable : Variable
            The variable to read.
        netcdf_field : str
            Requested variable, for messages.
        netcdf_path : str | Path
            Path of the file, for messages.
        netcdf_index : int | None
            Requested time index.
        time : datetime | pd.Timestamp | str | None
            Requested time; the nearest time step is used.

        Returns
        -------
        int | None
            The time index, or None for a variable without a time dimension.

        Raises
        ------
        ValueError
            If both netcdf_index and time are given, or one of them for a variable without a time dimension.
        """
        has_time = "time" in variable.dimensions
        if time is not None:
            if netcdf_index is not None:
                raise ValueError("Provide either netcdf_index or time, not both.")
            if not has_time:
                raise ValueError(
                    f"{netcdf_field} in {netcdf_path} has no time dimension."
                )
            return time_index(results, time)
        if netcdf_index is not None and not has_time:
            raise ValueError(
                f"{netcdf_field} in {netcdf_path} has no time dimension; do not provide netcdf_index."
            )
        if netcdf_index is None and has_time:
            n_time = results.dimensions["time"].size
            if n_time != 1:
                logger.info(
                    f"No time given; the last of the {n_time} time steps in {netcdf_path} is used."
                )
            return -1
        return netcdf_index

    @staticmethod
    def _read_1d_topology(results: Dataset, location: str) -> dict[str, np.ndarray]:
        """Branch id and chainage of the 1D nodes or edges, and the direction of the edges, if in the file.

        Parameters
        ----------
        results : Dataset
            The opened map- or fou-file.
        location : str
            'node' or 'edge'.

        Returns
        -------
        dict[str, np.ndarray]
            'branchid' and 'chainage' (if the file contains the 1D network topology), and for edges
            'direction' (angle of the link from its first to its second node, in radians).
        """
        topology = {}
        if all(
            v in results.variables
            for v in (
                f"mesh1d_{location}_branch",
                f"mesh1d_{location}_offset",
                "network1d_branch_id",
            )
        ):
            branch_ids = np.array(
                [b.strip() for b in chartostring(results["network1d_branch_id"][:])],
                dtype=object,
            )
            branch_index = np.array(results[f"mesh1d_{location}_branch"][:]).astype(int)
            branch_index -= int(
                getattr(results[f"mesh1d_{location}_branch"], "start_index", 0)
            )
            topology = {
                "branchid": branch_ids[branch_index],
                "chainage": np.array(results[f"mesh1d_{location}_offset"][:]),
            }
        if location == "edge" and "mesh1d_edge_nodes" in results.variables:
            # direction of the link (from its first to its second node), to decompose velocities in x and y
            edge_nodes = np.array(results["mesh1d_edge_nodes"][:]).astype(int)
            edge_nodes -= int(getattr(results["mesh1d_edge_nodes"], "start_index", 0))
            node_x, node_y = (
                np.array(results["mesh1d_node_x"][:]),
                np.array(results["mesh1d_node_y"][:]),
            )
            topology["direction"] = np.arctan2(
                node_y[edge_nodes[:, 1]] - node_y[edge_nodes[:, 0]],
                node_x[edge_nodes[:, 1]] - node_x[edge_nodes[:, 0]],
            )
        return topology

    @staticmethod
    def _fill_missing_values(
        masked: np.ma.MaskedArray, results: Dataset, netcdf_field: str, location: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill missing values (dry nodes or links) where a fill value is known.

        Water levels get the bed level, and water depths, velocities and discharges get 0. Other variables
        are not filled.

        Parameters
        ----------
        masked : np.ma.MaskedArray
            Values, with missing values masked.
        results : Dataset
            The opened map- or fou-file, for the bed level.
        netcdf_field : str
            Requested variable, which determines the fill value.
        location : str
            'node' or 'edge'; values on edges (velocities, discharges) are filled with 0.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The values (NaN where still missing), and a boolean array that is True where values are missing.
        """
        missing = np.ma.getmaskarray(masked)
        if not missing.any():
            return np.ma.filled(masked, np.nan), missing
        name = netcdf_field.lower()
        if "depth" in name or location == "edge" or "velocity" in name:
            fill, label = np.zeros(len(masked)), "0"
        elif (
            name.endswith("_s1") or "waterlevel" in name
        ) and "mesh1d_flowelem_bl" in results.variables:
            fill, label = (
                np.ma.filled(results["mesh1d_flowelem_bl"][:], np.nan),
                "the bed level",
            )
        else:
            return np.ma.filled(masked, np.nan), missing
        logger.info(
            f"{missing.sum()} {location}s without a value (dry) in {netcdf_field}, set to {label}."
        )
        value = np.where(missing, fill, np.ma.filled(masked, np.nan))
        return value, np.isnan(value)

    def set_initial_velocity(
        self, velocities: gpd.GeoDataFrame, column: str = "value"
    ) -> None:
        """Set the initial velocity on the 1D flow links from a model result.

        E.g. ``"mesh1d_u1"`` read with `read_values_from_modelrun`.

        D-Flow FM (2026.02) does not accept 'initialvelocity' in the initial fields file, only the components
        'initialvelocityx' and 'initialvelocityy'. The velocity along each link is therefore decomposed with
        the link direction and written as two sample files (triangulation). Tested with DIMR: on links that
        are in both models, the initial velocity equals the velocity of the model result within 1 cm/s.
        The effect of an initial velocity is short: the flow adapts within the first minutes.

        Parameters
        ----------
        velocities : gpd.GeoDataFrame
            Points at the link locations with the velocity and the column ``direction``
            (angle of the link in radians), as returned by `read_values_from_modelrun`.
        column : str, optional
            Column with the velocity along the link. Defaults to ``"value"``.
        """
        if "direction" not in velocities.columns:
            raise ValueError(
                "The column 'direction' (angle of the link) is required. Read the velocity with "
                "read_values_from_modelrun from a file with 'mesh1d_edge_nodes'."
            )
        speed = velocities[column].astype(float).to_numpy()
        angle = velocities["direction"].astype(float).to_numpy()
        for component, values in (
            ("x", speed * np.cos(angle)),
            ("y", speed * np.sin(angle)),
        ):
            self.set_spatial_field(
                velocities.assign(_component=values)[["_component", "geometry"]],
                "_component",
                quantity=f"initialvelocity{component}",
                filetype="sample",
                interpolationmethod="triangulation",
            )

    def set_spatial_field(
        self,
        gdf: gpd.GeoDataFrame | Path | str,
        column: str | None = None,
        quantity: str = "waterlevel",
        kind: str = "initial",
        filetype: str | None = None,
        cellsize: float | None = None,
        locationtype: str = "1d",
        interpolationmethod: str | None = None,
        averagingtype: str = "nearestNb",
        operand: str = "O",
        nodata: float = -999.0,
        max_snap_distance: float = 5.0,
        averagingrelsize: float | None = None,
        averagingnummin: int | None = None,
        averagingpercentile: float | None = None,
        extrapolationmethod: bool | None = None,
    ) -> None:
        """Add a spatially varying field to the initial fields file (initialFields.ini).

        By default, values are written per branch and chainage in a 1dField file (filetype 'branch'), for
        polygons and points. D-Flow FM interpolates them linearly along each branch only, and keeps them
        constant beyond the first and last chainage of a branch. For polygons, each part of a branch inside a
        polygon gets the polygon value, with a step at the polygon boundary. For points, the columns 'branchid'
        and 'chainage' are used if present (e.g. from read_values_from_modelrun); otherwise points are snapped to
        the nearest branch within max_snap_distance. In DIMR tests (D-HYDRO 2026.02) this gave the most reliable
        result for polygons, for model results on the same network and on a changed 1D grid, and for combining
        water level and water depth.

        Advanced: the other file types are kept for special cases (e.g. 2D, or if a future D-Flow FM version
        does not accept 1dField files), but are not recommended for 1D:

        - 'sample': points as samples (.xyz) with triangulation, which fills the convex hull of the points, so
          values spread to other branches.
        - 'polygon': .pol files with interpolationMethod=constant, one file and block per distinct value.
        - 'raster': polygons rasterized to a GeoTIFF (.tif) or ArcInfo grid (.asc), requires a cellsize. D-Flow FM
          (2026.02) uses the nodata value as data.

        Tested with D-HYDRO 2026.02 on 1D nodes: interpolationMethod=averaging assigns no values, so it is
        refused for locationtype '1d'; triangulation fills the convex hull of the samples; nodes that get no
        value from any field keep the default water level (WaterLevIni in the MDU), which leaves them dry when it
        is below the bed level, so cover all nodes. Fields are applied in the order they are added (the last
        one wins). A uniform 1D value from set_initial_waterlevel/-depth is a field per branch over all branches, at
        the position of that call, so use it first as a background. A water depth from polygons or samples removes earlier water levels from polygons
        or samples everywhere, also outside its own area; water levels per branch are kept. Add water depth
        fields before water level fields.

        For samples and rasters on 2D cells, choose the interpolation by the density of the data: 'averaging' for
        dense data (several values per cell, e.g. a point cloud or a fine raster), 'triangulation' for sparse data
        (fewer values than cells, e.g. measured points). Averaging combines the values inside each cell, as set by
        averagingtype and the averaging settings below; triangulation interpolates linearly between the values, and
        only within their convex hull unless extrapolationmethod is True. Settings left at None are not written, so
        D-Flow FM uses its defaults.

        The data files are written by Df2HydrolibModel.write_initial_fields().

        Parameters
        ----------
        gdf : gpd.GeoDataFrame or Path or str
            Polygons or points with values, or the path to an existing .xyz file (x, y
            and value per line). An .xyz file becomes a sample field (filetype
            ``"sample"``, no column) and is copied unchanged under the generated name
            when the model is written. For initial water levels and depths, only with
            locationtype ``"2d"``: D-HyDAMO does not read its values, so it cannot check
            them against other water level and depth fields on the 1D nodes.
        column : str, optional
            Column with the values. For points, None means the Z coordinate is used.
        quantity : str, optional
            D-Flow FM quantity, e.g. ``"waterlevel"`` or ``"waterdepth"``. For velocities,
            use `set_initial_velocity`. Defaults to ``"waterlevel"``.
        kind : str, optional
            ``"initial"`` ([Initial] block) or ``"parameter"`` ([Parameter] block).
            Defaults to ``"initial"``.
        filetype : str, optional
            ``"branch"`` (default), or the advanced types ``"sample"``, ``"polygon"`` or
            ``"raster"``.
        cellsize : float, optional
            Raster cell size, required for filetype ``"raster"``.
        locationtype : str, optional
            ``"1d"``, ``"2d"`` or ``"all"``. Defaults to ``"1d"``.
        interpolationmethod : str, optional
            ``"constant"`` (polygon), ``"triangulation"`` or ``"averaging"``. Not used for
            ``"branch"``. Defaults to ``"triangulation"`` for samples, ``"constant"`` for
            polygon files, and ``"averaging"`` for rasters (2D) or ``"triangulation"`` (1D).
        averagingtype : str, optional
            Averaging type, if interpolationmethod is ``"averaging"``. Defaults to
            ``"nearestNb"``.
        operand : str, optional
            How the field is combined with previous fields. Defaults to ``"O"`` (override).
        nodata : float, optional
            Nodata value for rasters, and the [Global] value of 1dField files (branch),
            which D-Flow FM treats as no value. Defaults to -999.0.
        max_snap_distance : float, optional
            Maximum distance to snap points to a branch for filetype ``"branch"``. Defaults
            to 5.0.
        averagingrelsize : float, optional
            With ``"averaging"``, the size of the area around a cell that is searched for
            values, relative to the cell size (D-Flow FM default 1.01). Larger values
            smooth more, and fill cells with few values. Defaults to None (not written).
        averagingnummin : int, optional
            With ``"averaging"``, the minimum number of values in that area; a cell with
            fewer gets no value from this field (D-Flow FM default 1). Defaults to None
            (not written).
        averagingpercentile : float, optional
            With ``"averaging"``, the percentile of the values to use instead of all
            values, e.g. a low percentile for the lowest part of a cell (D-Flow FM
            default 0, off). Defaults to None (not written).
        extrapolationmethod : bool, optional
            With ``"triangulation"``, True also gives values outside the convex hull of
            the data. Defaults to None: not set, which hydrolib-core writes as
            ``extrapolationMethod = 0``.
        """
        if kind not in ("initial", "parameter"):
            raise ValueError(f"kind should be 'initial' or 'parameter', not '{kind}'.")
        if quantity.lower() == "initialvelocity":
            raise ValueError(
                "D-Flow FM (2026.02) does not support 'initialvelocity' in the initial fields file. "
                "Use set_initial_velocity, which writes the x and y components."
            )

        xyz_file: Path | None = None
        field_gdf: gpd.GeoDataFrame | None = None
        if isinstance(gdf, (str, Path)):
            if column is not None:
                raise ValueError(
                    "An .xyz file has its values in the third column; do not pass a column."
                )
            if filetype not in {None, "sample"}:
                raise ValueError(
                    f"An .xyz file is a sample field, not filetype '{filetype}'."
                )
            if level_or_depth(quantity) is not None and locationtype != "2d":
                raise ValueError(
                    "An .xyz file for an initial water level or depth is only supported for locationtype '2d'. "
                    "D-HyDAMO does not read the file, so on 1D nodes it cannot check the field against other water "
                    "level and depth fields or export it with initial_conditions_at_nodes. Read the points into a "
                    "GeoDataFrame and pass that instead (filetype 'branch' is the default for 1D)."
                )
            xyz_file = self._xyz_file(gdf)
            filetype = "sample"
        else:
            if gdf.empty:
                raise ValueError("The provided GeoDataFrame is empty.")
            geomtypes = set(gdf.geom_type)
            filetype = self._resolve_filetype(geomtypes, filetype)
            if filetype == "branch" and locationtype != "1d":
                raise ValueError(
                    "Filetype 'branch' is only available for locationtype '1d'."
                )
            if filetype == "raster" and cellsize is None:
                raise ValueError("A cellsize is required to rasterize polygons.")
            self._warn_spatial_field_options(gdf, filetype, locationtype)
            gdf, column = self._value_column(gdf, column, geomtypes)
            columns = [column, "geometry"]
            if filetype == "branch" and {"branchid", "chainage"} <= set(gdf.columns):
                columns = ["branchid", "chainage"] + columns
            field_gdf = gpd.GeoDataFrame(gdf[columns].copy())

        interpolationmethod = self._resolve_interpolation(
            filetype, locationtype, interpolationmethod
        )
        sample_settings = {
            "averagingrelsize": averagingrelsize,
            "averagingnummin": averagingnummin,
            "averagingpercentile": averagingpercentile,
            "extrapolationmethod": extrapolationmethod,
        }
        self._warn_sample_settings(interpolationmethod, sample_settings)
        filename = self._next_field_filename(kind, quantity, filetype)

        self.spatial_fields.append(
            {
                "gdf": field_gdf,
                "xyz_file": xyz_file,
                "column": column,
                "quantity": quantity,
                "kind": kind,
                "filetype": filetype,
                "cellsize": cellsize,
                "filename": filename,
                "locationtype": locationtype,
                "interpolationmethod": interpolationmethod,
                "averagingtype": averagingtype,
                "operand": operand,
                "nodata": nodata,
                "max_snap_distance": max_snap_distance,
                "sample_settings": sample_settings,
            }
        )

    @staticmethod
    def _xyz_file(path: Path | str) -> Path:
        """Check an existing .xyz file for a spatial field, without reading the whole file.

        Parameters
        ----------
        path : Path or str
            Path to the .xyz file.

        Returns
        -------
        Path
            The absolute path to the file.

        Raises
        ------
        ValueError
            If the first data line does not start with three numbers.
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"The .xyz file {path} does not exist.")
        first_line = ""
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                # like hydrolib-core's XYZ parser: skip empty lines and comment lines
                # starting with '*'
                if line.strip() and not line.lstrip().startswith("*"):
                    first_line = line
                    break
        values = first_line.split()[:3]
        try:
            for value in values:
                float(value)
        except ValueError:
            values = []
        if len(values) < 3:
            raise ValueError(
                f"The first data line of {path} should start with three numbers (x, y, value), "
                f"not {first_line.strip()!r}."
            )
        return path.resolve()

    @staticmethod
    def _resolve_filetype(geomtypes: set[str], filetype: str | None) -> str:
        """Check the filetype for the geometry type of a spatial field; the default is 'branch'.

        Parameters
        ----------
        geomtypes : set[str]
            Geometry types in the GeoDataFrame.
        filetype : str | None
            Requested filetype, or None for the default.

        Returns
        -------
        str
            The filetype.

        Raises
        ------
        ValueError
            If the geometries are not all (multi)polygons or all points, or the filetype is not
            available for them.
        """
        if geomtypes <= {"Polygon", "MultiPolygon"}:
            filetype = "branch" if filetype is None else filetype
            if filetype not in ("branch", "polygon", "raster"):
                raise ValueError(
                    f"For polygons, filetype should be 'branch', 'polygon' or 'raster', not '{filetype}'."
                )
        elif geomtypes <= {"Point"}:
            filetype = "branch" if filetype is None else filetype
            if filetype not in ("sample", "branch"):
                raise ValueError(
                    f"For points, filetype should be 'sample' or 'branch', not '{filetype}'."
                )
        else:
            raise ValueError(
                f"Only (multi)polygons or points are supported, found: {geomtypes}."
            )
        return filetype

    @staticmethod
    def _resolve_interpolation(
        filetype: str, locationtype: str, interpolationmethod: str | None
    ) -> str | None:
        """Interpolation method of a spatial field, with the default per filetype and location type.

        Parameters
        ----------
        filetype : str
            'branch', 'polygon', 'raster' or 'sample'.
        locationtype : str
            '1d', '2d' or 'all'.
        interpolationmethod : str | None
            Requested interpolation method, or None for the default.

        Returns
        -------
        str | None
            None for 'branch' (not used for 1dField files); otherwise the requested method, or
            'constant' (polygon), 'averaging' (raster, not 1D) or 'triangulation'.

        Raises
        ------
        ValueError
            For polygon files with a method other than 'constant', and for 'averaging' on 1D nodes.
        """
        if filetype == "branch":
            return None
        if interpolationmethod is None:
            if filetype == "polygon":
                interpolationmethod = "constant"
            elif filetype == "raster" and locationtype != "1d":
                interpolationmethod = "averaging"
            else:
                interpolationmethod = "triangulation"
        if filetype == "polygon" and interpolationmethod != "constant":
            raise ValueError(
                "Polygon files only support interpolationmethod 'constant'."
            )
        if interpolationmethod == "averaging" and locationtype == "1d":
            raise ValueError(
                "interpolationmethod 'averaging' assigns no values to 1D nodes (D-HYDRO 2026.02). "
                "Use filetype 'branch' or 'polygon' for polygons, or 'triangulation' for samples."
            )
        return interpolationmethod

    @staticmethod
    def _warn_sample_settings(
        interpolationmethod: str | None, sample_settings: dict
    ) -> None:
        """Warn for sample settings that do not apply to the interpolation method.

        Such a setting is still written, but has no effect. The values themselves are
        checked by hydrolib-core when the field is converted.

        Parameters
        ----------
        interpolationmethod : str or None
            The resolved interpolation method, or None for ``"branch"``.
        sample_settings : dict
            averagingrelsize, averagingnummin, averagingpercentile and
            extrapolationmethod, None when not set.
        """
        if interpolationmethod != "averaging":
            for name in ["averagingrelsize", "averagingnummin", "averagingpercentile"]:
                if sample_settings[name] is not None:
                    logger.warning(
                        "averagingrelsize, averagingnummin and averagingpercentile only apply to "
                        f"interpolationmethod 'averaging', not {interpolationmethod!r}, and have no effect."
                    )
                    break
        if (
            sample_settings["extrapolationmethod"] is not None
            and interpolationmethod != "triangulation"
        ):
            logger.warning(
                "extrapolationmethod only applies to interpolationmethod 'triangulation', "
                f"not {interpolationmethod!r}, and has no effect."
            )

    @staticmethod
    def _warn_spatial_field_options(
        gdf: gpd.GeoDataFrame, filetype: str, locationtype: str
    ) -> None:
        """Warn for options that work, but may not give the intended result.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Polygons or points with values.
        filetype : str
            'branch', 'polygon', 'raster' or 'sample'.
        locationtype : str
            '1d', '2d' or 'all'.
        """
        if filetype == "raster" and locationtype in ("1d", "all"):
            logger.warning(
                "D-Flow FM uses the raster nodata value as data, which affects 1D nodes near polygon edges. "
                "Consider filetype 'branch' for 1D."
            )
        if (
            filetype == "polygon"
            and (
                shapely.get_num_interior_rings(gdf.geometry.explode().to_numpy()) > 0
            ).any()
        ):
            logger.warning(
                "Polygon files do not support holes; interiors of the polygons are ignored."
            )

    @staticmethod
    def _value_column(
        gdf: gpd.GeoDataFrame, column: str | None, geomtypes: set[str]
    ) -> tuple[gpd.GeoDataFrame, str]:
        """Column with the values of a spatial field; without a column, the Z-coordinate of points is used.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Polygons or points with values.
        column : str | None
            Column with values, or None to use the Z-coordinate of points.
        geomtypes : set[str]
            Geometry types in the GeoDataFrame.

        Returns
        -------
        tuple[gpd.GeoDataFrame, str]
            The GeoDataFrame (with a column '_z' if the Z-coordinate is used) and the value column.

        Raises
        ------
        ValueError
            If there is no column and no Z-coordinate, or the column has missing values.
        """
        if column is None:
            if "Point" not in geomtypes or not gdf.has_z.all():
                raise ValueError(
                    "Provide a value column, or use points with a Z-coordinate."
                )
            gdf = gdf.assign(_z=gdf.geometry.z)
            column = "_z"

        if gdf[column].isna().any():
            raise ValueError(
                f"Column {column} not filled for all features. Please provide valid values."
            )
        return gdf, column

    def _next_field_filename(self, kind: str, quantity: str, filetype: str) -> str:
        """Return a file name that is unique per field, e.g. initial_waterlevel_00.ini.

        Uniform values (set_initial_waterlevel/-depth) use other names.

        Parameters
        ----------
        kind : str
            'initial' or 'parameter'.
        quantity : str
            D-Flow FM quantity, e.g. 'waterlevel'.
        filetype : str
            'branch', 'polygon', 'raster' or 'sample'.

        Returns
        -------
        str
            The file name.
        """
        ext = {"branch": ".ini", "polygon": ".pol", "raster": ".tif", "sample": ".xyz"}[
            filetype
        ]
        name = quantity.lower().removeprefix(kind)
        filename = f"{kind}_{name}_{self._next_field_number:02d}{ext}"
        self._next_field_number += 1
        return filename

    # legacy from delft3dfmpy, not used and not working here (self.boundaries does not exist): mypy skips it
    @no_type_check
    def add_rainfall_2D(self, fName, bctype="rainfall"):
        """Add 2D rainfall from a netCDF file (legacy from delft3dfmpy, not working).

        Parameters
        ----------
        fName : str
            Location of netcdf file containing rainfall rasters
        bctype : str
            Type of boundary condition. Currently only rainfall is supported
        """
        assert bctype in ["rainfall"]

        # Add boundary condition
        self.boundaries["rainfall_2D"] = {
            "file_name": fName,
            "bctype": bctype + "bnd",
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_boundary_condition(
        self,
        name: str,
        pt: tuple[float, float] | Point,
        quantity: str,
        value: float | pd.Series | dict[str, list[float]],
        mesh1d: netmodels.Network | None = None,
    ) -> None:
        """Add a boundary condition at the 1D node nearest to a location.

        Parameters
        ----------
        name : str
            Id of the boundary condition.
        pt : tuple[float, float] or Point
            Location of the boundary condition.
        quantity : str
            ``"dischargebnd"``, ``"waterlevelbnd"`` or ``"qhbnd"``.
        value : float, pd.Series or dict[str, list[float]]
            A constant, a time series with a datetime index, or a Q-H table
            ``{"Q": [...], "H": [...]}``.
        mesh1d : Network, optional
            Network with the 1D mesh. Required; the default None raises a ValueError.

        Raises
        ------
        KeyError
            If a boundary condition with this name exists, or the network has no 1D mesh.
        ValueError
            If no network is given.
        """
        assert quantity in ["dischargebnd", "waterlevelbnd", "qhbnd"]

        if isinstance(value, pd.Series):
            vec1: list[float] | None = (
                (value.index - value.index[0]).total_seconds() / 60.0
            ).tolist()
            vec2: list[float] | float = value.to_numpy().tolist()
            startdate = value.index[0].strftime("%Y-%m-%d %H:%M:%S")
            unit1 = f"minutes since {startdate}"
            unit2 = "m3/s" if quantity == "dischargebnd" else "m"
        elif isinstance(value, float):
            vec1 = None
            vec2 = float(value)
            startdate = "0000-00-00 00:00:00"
            unit1 = f"minutes since {startdate}"
            unit2 = "m3/s" if quantity == "dischargebnd" else "m"
        else:
            vec1 = value["Q"]
            vec2 = value["H"]
            unit1 = "m3/s"
            unit2 = "m"

        if name in self.boundary_nodes:
            raise KeyError(
                f'A boundary condition with name "{name}" is already present.'
            )

        if isinstance(pt, tuple):
            pt = Point(*pt)

        # Find the nearest node
        if mesh1d is None:
            raise ValueError("Provide the network with the 1d mesh (mesh1d).")
        if len(mesh1d._mesh1d.mesh1d_node_id) == 0:
            raise KeyError(
                "To find the closest node a 1d mesh should be created first."
            )
        nodes1d = np.asarray(
            [
                n
                for n in zip(
                    mesh1d._mesh1d.mesh1d_node_x,
                    mesh1d._mesh1d.mesh1d_node_y,
                    mesh1d._mesh1d.mesh1d_node_id,
                )
            ]
        )
        get_nearest = KDTree(nodes1d[:, 0:2])
        _, idx_nearest = get_nearest.query(pt.coords[:])
        nodeid = f"{float(nodes1d[idx_nearest[0], 0]):12.6f}_{float(nodes1d[idx_nearest[0], 1]):12.6f}"

        # Add boundary condition
        self.boundary_nodes[name] = {
            "id": name,
            "quantity": quantity,
            "vec1": vec1,
            "vec2": vec2,
            "unit1": unit1,
            "unit2": unit2,
            "nodeid": nodeid,
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_rain_series(self, name: str, values: list, times: list) -> None:
        """Add a rain series as a boundary condition.

        Parameters
        ----------
        name : str
            ID of the condition
        values : list of floats
            Values of the rain intensity
        times : list of datetime
            Times for the values
        """
        # Add boundary condition
        self.boundary_nodes[name] = {
            "code": name,
            "bctype": "rainfall",
            "filetype": 1,
            "method": 1,
            "operand": "O",
            "value": values,
            "time": times,
            "geometry": None,
            "branchid": None,
        }

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_lateral(
        self,
        id: str,
        branchid: str,
        chainage: str,
        discharge: Annotated[
            pd.Series | float | str, Field(union_mode="left_to_right")
        ],
    ) -> None:
        """Add a lateral.

        Parameters
        ----------
        id : str
            Id of the lateral.
        branchid : str
            Branch on which the lateral lies.
        chainage : str
            Chainage on the branch.
        discharge : pd.Series, float or str
            ``"realtime"`` when coupled to RR, a constant discharge, or a time series with
            a datetime index.
        """
        # Convert time to minutes
        if isinstance(discharge, pd.Series):
            times = (
                (discharge.index - discharge.index[0]).total_seconds() / 60.0
            ).tolist()
            values = discharge.to_numpy().tolist()
            startdate = discharge.index[0].strftime("%Y-%m-%d %H:%M:%S")
        else:
            times = None
            values = None
            startdate = "0000-00-00 00:00:00"

        self.lateral_nodes[id] = {
            "id": id,
            "type": "discharge",
            "locationtype": "1d",
            "branchid": branchid,
            "chainage": chainage,
            "time": times,
            "time_unit": f"minutes since {startdate}",
            "value_unit": "m3/s",
            "value": values,
            "discharge": discharge,
        }


class Structures:
    """Structures of the model, one DataFrame per type.

    The ``add_*`` methods take keyword arguments only; they correspond to the keywords
    of the D-Flow FM structure file.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO object they belong to.
    """

    def __init__(self, hydamo: HyDAMO) -> None:
        self.hydamo = hydamo
        self.generalstructures_df = pd.DataFrame()
        self.rweirs_df = pd.DataFrame()
        self.orifices_df = pd.DataFrame()
        self.uweirs_df = pd.DataFrame()
        self.culverts_df = pd.DataFrame()
        self.bridges_df = pd.DataFrame()
        self.pumps_df = pd.DataFrame()
        self.compounds_df = pd.DataFrame()

        # Maps management_device.globalid -> generated D-FlowFM structure id,
        # for weirs/orifices that have more than one Kunstwerkopening.
        #
        # Background: a weir's structure id is normally just its HyDAMO code,
        # and is therefore derivable from the static FK chain
        # management_device.kunstwerkopeningid -> opening.globalid ->
        # opening.stuwid -> weirs.globalid (see DRTCModel._resolve_weir_via_opening
        # in core/drtc.py). But when a weir has multiple openings, StructuresIO
        # builds one compound structure per opening and invents the id as
        # f"{weir.code}_{opening_index+1}" - that id only exists once this
        # enumeration has run, so it cannot be derived from HyDAMO data alone.
        # StructuresIO.weirs() populates this dict as it assigns those ids, and
        # DRTCModel falls back to it when the static chain finds no match.
        # Consequently, DRTCModel.from_hydamo()/check_timeseries() only resolve
        # multi-opening weirs correctly if structure conversion (StructuresIO)
        # has already run on this `hydamo` instance.
        self.compound_weir_structure_ids: dict[str, str] = {}

        self.convert = StructuresIO(self)

    def check_branchid_chainage(self, branchid: str, chainage: float) -> None:
        """Raise a ValueError if the branch does not exist or the chainage is outside it.

        Parameters
        ----------
        branchid : str
            Branch id.
        chainage : float
            Chainage on the branch.
        """
        # Check if the ID exists
        if branchid not in self.hydamo.branches["code"]:
            raise ValueError(
                f"branchid {branchid} not present. Give an existing branch."
            )

        # Get the branch
        branch = self.hydamo.branches.geometry[branchid]

        # Check the limits
        if chainage < 0.0:
            raise ValueError(
                f"Chainage {chainage} is outside the branch range (0.0 - {branch.length})."
            )
        if chainage > branch.length:
            raise ValueError(
                f"Chainage {chainage} is outside the branch length (0.0 - {branch.length})."
            )

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_rweir(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        crestlevel: float | None = None,
        crestwidth: float | None = None,
        corrcoeff: float | None = None,
        usevelocityheight: bool = True,
        allowedflowdir: str = "both",
    ) -> None:
        """Add a regular weir.

        ``id``, ``branchid`` and ``chainage`` are required; the other arguments correspond
        to the D-Flow FM keywords of a weir.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "crestwidth": crestwidth,
                "corrcoeff": corrcoeff,
                "usevelocityheight": usevelocityheight,
                "allowedflowdir": allowedflowdir,
            },
            index=[id],
        )
        self.rweirs_df = pd.concat([self.rweirs_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_orifice(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        crestlevel: float | None = None,
        crestwidth: float | None = None,
        corrcoeff: float | None = None,
        usevelocityheight: bool = True,
        allowedflowdir: str = "both",
        gateloweredgelevel: float | None = None,
        uselimitflowpos: bool | None = None,
        limitflowpos: float | None = None,
        uselimitflowneg: bool | None = None,
        limitflowneg: float | None = None,
    ) -> None:
        """Add an orifice.

        ``id``, ``branchid`` and ``chainage`` are required; the other arguments correspond
        to the D-Flow FM keywords of an orifice.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "crestwidth": crestwidth,
                "corrcoeff": corrcoeff,
                "usevelocityheight": usevelocityheight,
                "allowedflowdir": allowedflowdir,
                "gateloweredgelevel": gateloweredgelevel,
                "uselimitflowpos": uselimitflowpos,
                "limitflowpos": limitflowpos,
                "uselimitflowneg": uselimitflowneg,
                "limitflowneg": limitflowneg,
            },
            index=[id],
        )
        self.orifices_df = pd.concat([self.orifices_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_uweir(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        crestlevel: float | None = None,
        crestwidth: float | None = None,
        dischargecoeff: float | None = None,
        usevelocityheight: bool = True,
        allowedflowdir: str = "both",
        numlevels: float | None = None,
        yvalues: str | None = None,
        zvalues: str | None = None,
    ) -> None:
        """Add a universal weir.

        ``id``, ``branchid`` and ``chainage`` are required; the other arguments correspond
        to the D-Flow FM keywords of a universal weir.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "dischargecoeff": dischargecoeff,
                "numlevels": numlevels,
                "allowedflowdir": allowedflowdir,
                "yvalues": yvalues,
                "zvalues": zvalues,
            },
            index=[id],
        )
        self.uweirs_df = pd.concat([self.uweirs_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_bridge(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        length: float | None = None,
        inletlosscoeff: float | None = None,
        outletlosscoeff: float | None = None,
        csdefid: str | None = None,
        shift: float | None = None,
        allowedflowdir: str = "both",
        frictiontype: str,
        friction: float | None = None,
    ) -> None:
        """Add a bridge.

        ``id``, ``branchid``, ``chainage`` and ``frictiontype`` (HyDAMO roughness type)
        are required; the other arguments correspond to the D-Flow FM keywords of a
        bridge. ``csdefid`` is the id of its cross section definition.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        # map HyDAMO definition to D-Hydro definition
        frictiontype = self.hydamo.roughness_mapping[frictiontype]

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "length": length,
                "inletlosscoeff": inletlosscoeff,
                "outletlosscoeff": outletlosscoeff,
                "csdefid": csdefid,
                "shift": shift,
                "allowedflowdir": allowedflowdir,
                "frictiontype": frictiontype,
                "friction": friction,
            },
            index=[id],
        )
        self.bridges_df = pd.concat([self.bridges_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_culvert(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        leftlevel: float | None = None,
        rightlevel: float | None = None,
        length: float | None = None,
        inletlosscoeff: float | None = None,
        outletlosscoeff: float | None = None,
        crosssection: dict,
        allowedflowdir: str = "both",
        valveonoff: int = 0,
        numlosscoeff: int | None = None,
        valveopeningheight: float = 0,
        relopening: list | None = None,
        losscoeff: list | None = None,
        bedfrictiontype: str,
        bedfriction: float,
    ) -> None:
        """Add a culvert, and a cross section definition named after it.

        ``id``, ``branchid``, ``chainage``, ``crosssection``, ``bedfrictiontype``
        (HyDAMO roughness type) and ``bedfriction`` are required; the other arguments
        correspond to the D-Flow FM keywords of a culvert. ``crosssection`` is
        ``{"shape": "circle", "diameter": ...}`` or
        ``{"shape": "rectangle", "height": ..., "width": ..., "closed": ...}``.

        Raises
        ------
        NotImplementedError
            For another cross section shape.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        if crosssection["shape"] == "circle":
            definition = self.hydamo.crosssections.add_circle_definition(
                crosssection["diameter"], bedfrictiontype, bedfriction, name=id
            )
        elif crosssection["shape"] == "rectangle":
            definition = self.hydamo.crosssections.add_rectangle_definition(
                crosssection["height"],
                crosssection["width"],
                crosssection["closed"],
                bedfrictiontype,
                bedfriction,
                name=id,
            )
        else:
            raise NotImplementedError(
                f'Cross section with shape "{crosssection["shape"]}" not implemented.'
            )

        bedfrictiontype = self.hydamo.roughness_mapping[bedfrictiontype]

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "rightlevel": rightlevel,
                "leftlevel": leftlevel,
                "length": length,
                "inletlosscoeff": inletlosscoeff,
                "outletlosscoeff": outletlosscoeff,
                "csdefid": definition,
                "bedfrictiontype": bedfrictiontype,
                "bedfriction": bedfriction,
                "allowedflowdir": allowedflowdir,
                "valveonoff": valveonoff,
                "numlosscoeff": numlosscoeff,
                "valveopeningheight": valveopeningheight,
                "relopening": [relopening],
                "losscoeff": [losscoeff],
            },
            index=[id],
        )

        self.culverts_df = pd.concat([self.culverts_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_pump(
        self,
        *,  # keyword-only, so required arguments can follow optional ones in the original order
        id: str,
        name: str | None = None,
        branchid: str,
        chainage: float,
        orientation: str = "positive",
        numstages: int = 1,
        controlside: str = "suctionSide",
        capacity: float | None = None,
        startlevelsuctionside: list | None = None,
        stoplevelsuctionside: list | None = None,
        startleveldeliveryside: list | None = None,
        stopleveldeliveryside: list | None = None,
    ) -> None:
        """Add a pump.

        ``id``, ``branchid`` and ``chainage`` are required; the other arguments correspond
        to the D-Flow FM keywords of a pump.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "orientation": orientation,
                "numstages": numstages,
                "controlside": controlside,
                "capacity": capacity,
                "startlevelsuctionside": [startlevelsuctionside],
                "stoplevelsuctionside": [stoplevelsuctionside],
                "startleveldeliveryside": [startleveldeliveryside],
                "stopleveldeliveryside": [stopleveldeliveryside],
            },
            index=[id],
        )
        self.pumps_df = pd.concat([self.pumps_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_compound(self, *, id: str, structureids: list) -> None:
        """Add a compound structure of existing structures.

        Parameters
        ----------
        id : str
            Id of the compound structure.
        structureids : list
            Ids of its structures.
        """
        structurestring = ";".join([f"{s}" for s in structureids])
        numstructures = len(structureids)
        dct = pd.DataFrame(
            {
                "id": id,
                "name": id,
                "numstructures": numstructures,
                "structureids": structurestring,
            },
            index=[id],
        )
        self.compounds_df = pd.concat([self.compounds_df, dct], ignore_index=True)

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def as_dataframe(
        self,
        generalstructures: bool = False,
        pumps: bool = False,
        rweirs: bool = False,
        bridges: bool = False,
        culverts: bool = False,
        uweirs: bool = False,
        orifices: bool = False,
        compounds: bool = False,
    ) -> pd.DataFrame | None:
        """Return the structures of the given types in one DataFrame, with a column ``structype``.

        Returns None if there are no such structures.
        """
        dfs = []
        for df, descr, add in zip(
            [
                self.generalstructures_df,
                self.culverts_df,
                self.rweirs_df,
                self.bridges_df,
                self.pumps_df,
                self.uweirs_df,
                self.orifices_df,
                self.compounds_df,
            ],
            [
                "generalstructures",
                "culvert",
                "weir",
                "bridge",
                "pump",
                "uweir",
                "orifice",
                "compound",
            ],
            [
                generalstructures,
                culverts,
                rweirs,
                bridges,
                pumps,
                uweirs,
                orifices,
                compounds,
            ],
        ):
            if any(df) and add:
                # Skip columns that are empty for this structure type, so they don't
                # affect the dtype of the combined column (pandas FutureWarning).
                df = df.dropna(axis=1, how="all").copy()
                df.insert(loc=0, column="structype", value=descr, allow_duplicates=True)
                dfs.append(df)

        if len(dfs) > 0:
            return pd.concat(dfs, sort=False, ignore_index=True)
        return None


class ObservationPoints:
    """Observation points of the model.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO object they belong to.
    """

    def __init__(self, hydamo: HyDAMO) -> None:
        self.hydamo = hydamo
        self.observation_points = gpd.GeoDataFrame().set_geometry([])

    @validate_call(
        config=ConfigDict(arbitrary_types_allowed=True, coerce_numbers_to_str=True)
    )
    def add_points(
        self,
        crds: list,
        names: list,
        locationTypes: str | list[str] | None = None,
        snap_distance: float = 5.0,
    ) -> None:
        """Add observation points, of type '1d' or '2d'; 1d points are snapped to a branch.

        Parameters
        ----------
        crds : Nx2 list or array
            x and y coordinates of observation points
        names : str or list
            names of the observation points
        locationTypes:  str or list
            type of the observationpoints: 1d or 2d
        snap_distance : float (default is 5 m)
            1d observation poinst within this distance to a branch will be snapped to it. Otherwise they are discarded.
        """
        if isinstance(names, str):
            names = [names]
            crds = [crds]

        if locationTypes is not None:
            if isinstance(locationTypes, str):
                locationTypes = [locationTypes]

            # split 1d and 2d points, as the first ones need to be snapped to branches
            obs2d = gpd.GeoDataFrame()
            obs2d["name"] = [
                n for nn, n in enumerate(names) if locationTypes[nn] == "2d"
            ]
            obs2d["locationtype"] = "2d"
            obs2d = obs2d.set_geometry(
                [
                    Point(*pt) if not isinstance(pt, Point) else pt
                    for ipt, pt in enumerate(crds)
                    if (locationTypes[ipt] == "2d")
                ]
            )
            obs2d["x"] = [pt.coords[0][0] for pt in obs2d["geometry"]]
            obs2d["y"] = [pt.coords[0][1] for pt in obs2d["geometry"]]
            names1d = [n for n_i, n in enumerate(names) if locationTypes[n_i] == "1d"]
            crds1d = [c for c_i, c in enumerate(crds) if locationTypes[c_i] == "1d"]
        else:
            names1d = names
            crds1d = crds

        obs1d = gpd.GeoDataFrame()
        obs1d["name"] = names1d
        obs1d = obs1d.set_geometry(
            [Point(*pt) if not isinstance(pt, Point) else pt for pt in crds1d]
        )
        obs1d["locationtype"] = "1d"
        find_nearest_branch(
            self.hydamo.branches, obs1d, method="overal", maxdist=snap_distance
        )
        obs1d = obs1d.rename(
            columns={"branch_id": "branchid", "branch_offset": "chainage"}
        )

        obs = (
            pd.concat([obs1d, obs2d], sort=True) if locationTypes is not None else obs1d
        )

        obs = obs.dropna(how="all", axis=1)

        # Add to dataframe
        if self.observation_points.empty:
            self.observation_points = gpd.GeoDataFrame(obs)
        else:
            self.observation_points = gpd.GeoDataFrame(
                pd.concat([self.observation_points, obs], ignore_index=True)
            )


class StorageNodes:
    """Storage nodes of the model.

    Parameters
    ----------
    hydamo : HyDAMO
        The HyDAMO object they belong to.
    """

    def __init__(self, hydamo: HyDAMO) -> None:
        self.storagenodes: dict[str, dict[str, object]] = {}

        self.hydamo = hydamo

        self.convert = StorageNodesIO(self)

    def add_storagenode(
        self,
        id: str,
        xy: tuple[float, float] | Point | None = None,
        branchid: str | None = None,
        chainage: float | None = None,
        nodeid: str | None = None,
        usestreetstorage: str | bool = "true",
        nodetype: str = "unspecified",
        manholeid: str | None = None,
        name: str | float = np.nan,
        usetable: str = "false",
        bedlevel: float = np.nan,
        area: float = np.nan,
        streetlevel: float = np.nan,
        streetstoragearea: float = np.nan,
        storagetype: str = "reservoir",
        levels: str | float = np.nan,
        storagearea: str | float = np.nan,
        interpolate: str = "linear",
        network: netmodels.Network | None = None,
    ) -> None:
        """Add a storage node, at a 1D node.

        The node is ``nodeid``, or the 1D node nearest to ``xy`` or to ``branchid`` and
        ``chainage``. The other arguments correspond to the D-Flow FM keywords of a
        storage node; with ``usetable="true"``, ``levels`` and ``storagearea`` are space
        separated numbers.

        Raises
        ------
        ValueError
            Without a ``nodeid``, a network or a location; or with a table but without
            ``levels`` and ``storagearea``.
        """
        if isinstance(xy, tuple):
            xy = Point(*xy)
        if xy is None and branchid is not None and chainage is not None:
            xy = self.hydamo.branches.geometry[branchid].interpolate(chainage)

        # Find the nearest node
        if nodeid is not None and not isinstance(nodeid, str):
            raise ValueError("If a nodeid is provided, it should be of type string.")
        if nodeid is None:
            if network is None or xy is None:
                raise ValueError(
                    "Without a nodeid, provide the network and a location: xy, or branchid and chainage."
                )
            if len(network._mesh1d.network1d_node_id) == 0:
                raise KeyError(
                    "To find the closest node a 1d mesh should be created first."
                )
            nodes1d = np.asarray(
                [
                    n
                    for n in zip(
                        network._mesh1d.network1d_node_x,
                        network._mesh1d.network1d_node_y,
                        network._mesh1d.network1d_node_id,
                    )
                ]
            )
            get_nearest = KDTree(nodes1d[:, 0:2])
            _, idx_nearest = get_nearest.query(xy.coords[:])
            nodeid = f"{float(nodes1d[idx_nearest[0], 0]):12.6f}_{float(nodes1d[idx_nearest[0], 1]):12.6f}"

        if manholeid is None:
            manholeid = nodeid

        base = {
            "type": "storageNode",
            "id": id,
            "name": id if name is None else name,
            "useStreetStorage": usestreetstorage,
            "nodeType": nodetype,
            "useTable": usetable,
            "manholeId": manholeid,
        }
        if nodeid is None:
            cds = {"branchid": branchid, "chainage": chainage}
        else:
            cds = {"nodeid": nodeid}

        if usetable == "false":
            out = {
                **base,
                **cds,
                "bedLevel": bedlevel,
                "area": area,
                "streetLevel": streetlevel,
                "streetStorageArea": streetstoragearea,
                "storageType": storagetype,
            }
        elif usetable == "true":
            if not isinstance(levels, str) or not isinstance(storagearea, str):
                raise ValueError(
                    "With usetable='true', provide levels and storagearea as strings of numbers."
                )
            assert len(levels.split()) == len(storagearea.split()), (
                "Number of levels does not equal number of storagearea"
            )
            out = {
                **base,
                **cds,
                "numLevels": len(levels.split()),
                "levels": levels,
                "storageArea": storagearea,
                "interpolate": interpolate,
                "bedLevel": -999.0,
                "area": -999.0,
                "streetLevel": -999.0,
                "streetStorageArea": -999.0,
                "storageType": storagetype,
            }
        else:
            raise ValueError(
                "Value of key 'usetable' is not supported. Either use 'true' or 'false"
            )
        self.storagenodes[id] = remove_nan_values(out)


def remove_nan_values(base: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of a mapping without the NaN values.

    Parameters
    ----------
    base : Mapping[str, object]
        Input data, with NaN values.

    Returns
    -------
    dict[str, object]
        Copy without the float NaN values.
    """
    base_copy = dict(base)
    for k, v in base.items():
        if isinstance(v, float) and np.isnan(v):
            base_copy.pop(k)
    return base_copy
