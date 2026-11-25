"""HYDROLIB's data model: the HyDAMO tables, under their DAMO 2.2 names, and HYDROLIB's extensions.

Every DAMO version is loaded into this one model: the version steps (`steps`) rename a
file's layers and columns back to DAMO 2.2. The model is plain data and independent of
any DAMO version. ``HyDAMO.__init__`` builds its tables from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import TypeAlias

import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

Geotype = type[BaseGeometry] | UnionType


@dataclass(frozen=True)
class Table:
    """A HyDAMO table: its DAMO 2.2 layer, the columns HYDROLIB reads, and how it is read.

    Parameters
    ----------
    layer : str
        DAMO 2.2 layer, matched case-insensitively.
    geotype : type[BaseGeometry] or UnionType or None
        Geometry type of a spatial table, e.g. ``Point`` or ``Polygon | MultiPolygon``;
        ``None`` for a table without geometry.
    required_columns : tuple[str, ...]
        Columns a loaded layer must have; a missing one is an error.
    optional_columns : tuple[str, ...]
        Other columns HYDROLIB reads. Loading doesn't need them: a missing one is
        absent after loading. Building the model may still need one for some
        objects, e.g. ``debiet`` for a discharge boundary; it then fails on the
        missing column.
    index : str, optional
        Column that becomes the index; duplicate values get a suffix.
    group_by : str, optional
        Column whose points are grouped into one line each.
    order_by : str, optional
        Column that orders the points of a line.
    optional_layer : bool
        Whether a file may lack the layer. Default is ``True``.
    check_geotype : bool
        Whether the geometries must have the table's geometry type. Default is
        ``True``.
    """

    layer: str
    geotype: Geotype | None
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    index: str | None = None
    group_by: str | None = None
    order_by: str | None = None
    optional_layer: bool = True
    check_geotype: bool = True


# The HyDAMO tables loaded from DAMO, by HyDAMO attribute, in loading order. Column
# names are DAMO 2.2 names; the required columns are listed in the order the
# missing-column error lists them.
TABLES: dict[str, Table] = {
    "branches": Table(
        "HydroObject",
        geotype=LineString,
        required_columns=("code", "geometry"),
        optional_columns=("globalid",),
        index="code",
    ),
    "profile": Table(
        "ProfielPunt",
        geotype=LineString,
        required_columns=("code", "geometry", "globalid", "profiellijnid"),
        group_by="profiellijnid",
        order_by="codevolgnummer",
    ),
    "profile_roughness": Table(
        "RuwheidProfiel",
        geotype=None,
        required_columns=("profielpuntid",),
        optional_columns=("ruwheidhoog", "ruwheidlaag", "typeruwheid"),
    ),
    "profile_line": Table(
        "profiellijn",
        geotype=LineString,
        required_columns=("globalid", "profielgroepid"),
    ),
    "profile_group": Table(
        "profielgroep",
        geotype=None,
        required_columns=(),
        optional_columns=("globalid", "brugid", "hydroobjectid", "stuwid"),
    ),
    "param_profile": Table(
        "NormgeparamProfiel",
        geotype=None,
        required_columns=("globalid", "normgeparamprofielid", "hydroobjectid"),
    ),
    "param_profile_values": Table(
        "NormgeparamProfielWaarde",
        geotype=None,
        required_columns=(
            "normgeparamprofielid",
            "soortparameter",
            "waarde",
            "ruwheidlaag",
            "ruwheidhoog",
            "typeruwheid",
        ),
    ),
    "weirs": Table(
        "Stuw",
        geotype=Point,
        required_columns=("code", "geometry", "globalid", "afvoercoefficient"),
        optional_columns=("naam", "laagstedoorstroomhoogte"),
        index="code",
    ),
    "opening": Table(
        "Kunstwerkopening",
        geotype=None,
        required_columns=(
            "globalid",
            "laagstedoorstroombreedte",
            "laagstedoorstroomhoogte",
            "afvoercoefficient",
        ),
        optional_columns=("stuwid", "regelmiddelid"),
    ),
    "closing_device": Table("Afsluitmiddel", geotype=None, required_columns=("code",)),
    "management_device": Table(
        "Regelmiddel",
        geotype=None,
        required_columns=("code", "overlaatonderlaat"),
        optional_columns=(
            "globalid",
            "kunstwerkopeningid",
            "duikersifonhevelid",
            "gemaalid",
            "soortregelmiddel",
            "hoogteopening",
            "afvoercoefficient",
        ),
    ),
    "bridges": Table(
        "Brug",
        geotype=Point,
        required_columns=(
            "code",
            "globalid",
            "geometry",
            "lengte",
            "intreeverlies",
            "uittreeverlies",
            "ruwheid",
            "typeruwheid",
        ),
        optional_columns=("naam",),
        index="code",
    ),
    "culverts": Table(
        "DuikerSifonHevel",
        geotype=LineString,
        required_columns=(
            "code",
            "geometry",
            "lengte",
            "hoogteopening",
            "breedteopening",
            "hoogtebinnenonderkantbene",
            "hoogtebinnenonderkantbov",
            "vormkoker",
            "intreeverlies",
            "uittreeverlies",
            "typeruwheid",
            "ruwheid",
        ),
        optional_columns=("globalid", "naam"),
        index="code",
    ),
    "pumpstations": Table(
        "Gemaal",
        geotype=Point,
        required_columns=("code", "globalid", "geometry"),
        optional_columns=("naam",),
        index="code",
    ),
    "pumps": Table(
        "Pomp",
        geotype=None,
        required_columns=("code", "globalid", "gemaalid", "maximalecapaciteit"),
        index="code",
    ),
    "management": Table(
        "Sturing",
        geotype=None,
        # DAMO defines no code for Sturing: see FILLED_FROM_GLOBALID.
        required_columns=("code", "globalid"),
        optional_columns=(
            "regelmiddelid",
            "pompid",
            "hydrologischerandvoorwaardeid",
            "typecontroller",
            "stuurvariabele",
            "doelvariabele",
            "meetlocatieid",
            "bovengrens",
            "ondergrens",
            "streefwaarde",
        ),
        index="code",
    ),
    "boundary_conditions": Table(
        "hydrologischerandvoorwaarde",
        geotype=Point,
        required_columns=("code", "typerandvoorwaarde", "geometry"),
        optional_columns=("globalid", "waterstand", "debiet"),
        index="code",
    ),
    "catchments": Table(
        "afvoergebiedaanvoergebied",
        geotype=Polygon | MultiPolygon,
        required_columns=("code", "geometry", "globalid", "lateraleknoopid"),
        index="code",
        check_geotype=False,
    ),
    "laterals": Table(
        "lateraleknoop",
        geotype=Point,
        required_columns=("code", "geometry", "globalid"),
    ),
}


@dataclass(frozen=True)
class OwnTable:
    """A table HYDROLIB defines itself: not DAMO, and loaded from its own files.

    Parameters
    ----------
    geotype : type[BaseGeometry]
        Geometry type of the table.
    required_columns : tuple[str, ...]
        Columns the table must have.
    reason : str
        Why HYDROLIB needs it.
    """

    geotype: type[BaseGeometry]
    required_columns: tuple[str, ...]
    reason: str


# HYDROLIB's own tables, by HyDAMO attribute. They aren't loaded by load_from_gpkg,
# and they relate on code (codegerelateerdobject), not on globalid.
OWN_TABLES: dict[str, OwnTable] = {
    "overflows": OwnTable(
        Point,
        ("code", "geometry", "codegerelateerdobject", "fractie"),
        "RR sewer overflows, each linked to its sewer area.",
    ),
    "sewer_areas": OwnTable(
        Polygon,
        ("code", "geometry"),
        "RR sewer areas, which drain through their overflows.",
    ),
    "greenhouse_areas": OwnTable(
        Polygon,
        ("code", "geometry"),
        "RR greenhouse areas.",
    ),
    "greenhouse_laterals": OwnTable(
        Point,
        ("code", "geometry", "codegerelateerdobject"),
        "RR greenhouse laterals, each linked to its greenhouse area.",
    ),
    "storage_areas": OwnTable(
        Polygon,
        ("code", "geometry"),
        "Storage areas that become D-Flow FM storage nodes.",
    ),
}


@dataclass(frozen=True, slots=True)
class Relation:
    """A foreign-key relation from a child table to a parent table.

    Following a relation is in `hydrolib.dhydamo.core.relations`; the two methods
    here only say how its keys match.

    Attributes
    ----------
    name : str
        Name of the relation, for messages.
    child_table : str
        HyDAMO attribute of the table with the foreign key.
    foreign_key : str
        Column of the child table that refers to the parent.
    parent_table : str
        HyDAMO attribute of the table that is referred to.
    parent_key : str
        Column of the parent table that is referred to. Defaults to ``"globalid"``.
    """

    name: str
    child_table: str
    foreign_key: str
    parent_table: str
    parent_key: str = "globalid"

    def parents_of(
        self,
        child_rows: pd.DataFrame,
        parent_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return parent rows referenced by one or more child rows.

        The ``of`` in ``parents_of`` refers to the first argument,
        ``child_rows``. This relation supplies the traversal metadata: values
        from its ``foreign_key`` on ``child_rows`` are matched against its
        ``parent_key`` on ``parent_table``.

        Parameters
        ----------
        child_rows : pandas.DataFrame
            One or more rows from this relation's ``child_table``. The
            relation's ``foreign_key`` must be present as a column.
        parent_table : pandas.DataFrame
            Candidate rows from this relation's ``parent_table``. The
            relation's ``parent_key`` must be present as a column.

        Returns
        -------
        pandas.DataFrame
            Rows from ``parent_table`` whose parent key is referenced by
            ``child_rows``. The original index, column order, and dataframe
            type are preserved. Null foreign-key values never match.

        Raises
        ------
        KeyError
            If ``foreign_key`` is missing from ``child_rows`` or ``parent_key``
            is missing from ``parent_table``.

        Notes
        -----
        This method does not enforce relation cardinality. It can return zero,
        one, or multiple parent rows; the caller decides how those cases are
        handled.
        """
        foreign_keys = child_rows[self.foreign_key].dropna()
        return parent_table.loc[parent_table[self.parent_key].isin(foreign_keys)]

    def children_of(
        self,
        parent_rows: pd.DataFrame,
        child_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return child rows that reference one or more parent rows.

        The ``of`` in ``children_of`` refers to the first argument,
        ``parent_rows``. This relation supplies the traversal metadata: values
        from its ``parent_key`` on ``parent_rows`` are matched against its
        ``foreign_key`` on ``child_table``.

        Parameters
        ----------
        parent_rows : pandas.DataFrame
            One or more rows from this relation's ``parent_table``. The
            relation's ``parent_key`` must be present as a column.
        child_table : pandas.DataFrame
            Candidate rows from this relation's ``child_table``. The
            relation's ``foreign_key`` must be present as a column.

        Returns
        -------
        pandas.DataFrame
            Rows from ``child_table`` whose foreign key references
            ``parent_rows``. The original index, column order, and dataframe
            type are preserved. Null parent-key values never match.

        Raises
        ------
        KeyError
            If ``parent_key`` is missing from ``parent_rows`` or
            ``foreign_key`` is missing from ``child_table``.

        Notes
        -----
        This method does not enforce relation cardinality. It can return zero,
        one, or multiple child rows; the caller decides how those cases are
        handled.
        """
        parent_keys = parent_rows[self.parent_key].dropna()
        return child_table.loc[child_table[self.foreign_key].isin(parent_keys)]


class DamoRelations:
    """The DAMO relations between the HyDAMO tables.

    A DAMO foreign key holds the parent's ``globalid``. The one exception is the
    parameterised-profile values, which share ``normgeparamprofielid`` with their
    profile, so that is the parent key.
    """

    PROFILE_LINE = Relation(
        name="profile_line",
        child_table="profile",
        foreign_key="profiellijnid",
        parent_table="profile_line",
    )
    PROFILE_ROUGHNESS = Relation(
        name="profile_roughness",
        child_table="profile_roughness",
        foreign_key="profielpuntid",
        parent_table="profile",
    )
    PROFILE_LINE_GROUP = Relation(
        name="profile_line_group",
        child_table="profile_line",
        foreign_key="profielgroepid",
        parent_table="profile_group",
    )
    PROFILE_GROUP_BRANCH = Relation(
        name="profile_group_branch",
        child_table="profile_group",
        foreign_key="hydroobjectid",
        parent_table="branches",
    )
    PROFILE_GROUP_WEIR = Relation(
        name="profile_group_weir",
        child_table="profile_group",
        foreign_key="stuwid",
        parent_table="weirs",
    )
    PROFILE_GROUP_BRIDGE = Relation(
        name="profile_group_bridge",
        child_table="profile_group",
        foreign_key="brugid",
        parent_table="bridges",
    )
    PARAM_PROFILE_BRANCH = Relation(
        name="param_profile_branch",
        child_table="param_profile",
        foreign_key="hydroobjectid",
        parent_table="branches",
    )
    PARAM_PROFILE_VALUES = Relation(
        name="param_profile_values",
        child_table="param_profile_values",
        foreign_key="normgeparamprofielid",
        parent_table="param_profile",
        parent_key="normgeparamprofielid",
    )
    OPENING_WEIR = Relation(
        name="opening_weir",
        child_table="opening",
        foreign_key="stuwid",
        parent_table="weirs",
    )
    OPENING_MANAGEMENT_DEVICE = Relation(
        name="opening_management_device",
        child_table="opening",
        foreign_key="regelmiddelid",
        parent_table="management_device",
    )
    MANAGEMENT_DEVICE_OPENING = Relation(
        name="management_device_opening",
        child_table="management_device",
        foreign_key="kunstwerkopeningid",
        parent_table="opening",
    )
    MANAGEMENT_DEVICE_CULVERT = Relation(
        name="management_device_culvert",
        child_table="management_device",
        foreign_key="duikersifonhevelid",
        parent_table="culverts",
    )
    MANAGEMENT_DEVICE_PUMPSTATION = Relation(
        name="management_device_pumpstation",
        child_table="management_device",
        foreign_key="gemaalid",
        parent_table="pumpstations",
    )
    PUMP_PUMPSTATION = Relation(
        name="pump_pumpstation",
        child_table="pumps",
        foreign_key="gemaalid",
        parent_table="pumpstations",
    )
    MANAGEMENT_PUMP = Relation(
        name="management_pump",
        child_table="management",
        foreign_key="pompid",
        parent_table="pumps",
    )
    MANAGEMENT_DEVICE = Relation(
        name="management_device",
        child_table="management",
        foreign_key="regelmiddelid",
        parent_table="management_device",
    )
    MANAGEMENT_BOUNDARY_CONDITION = Relation(
        name="management_boundary_condition",
        child_table="management",
        foreign_key="hydrologischerandvoorwaardeid",
        parent_table="boundary_conditions",
    )
    CATCHMENT_LATERAL = Relation(
        name="catchment_lateral",
        child_table="catchments",
        foreign_key="lateraleknoopid",
        parent_table="laterals",
    )


class OwnRelations:
    """The relations of HYDROLIB's own tables (`OWN_TABLES`), an extension.

    These tables aren't DAMO and have no ``globalid``: a child refers to its
    parent's ``code`` in ``codegerelateerdobject``.
    """

    OVERFLOW_SEWER_AREA = Relation(
        name="overflow_sewer_area",
        child_table="overflows",
        foreign_key="codegerelateerdobject",
        parent_table="sewer_areas",
        parent_key="code",
    )
    GREENHOUSE_LATERAL_AREA = Relation(
        name="greenhouse_lateral_area",
        child_table="greenhouse_laterals",
        foreign_key="codegerelateerdobject",
        parent_table="greenhouse_areas",
        parent_key="code",
    )


CascadePaths: TypeAlias = tuple[tuple[Relation, ...], ...]

# The rows dropped with a table's rows, by table: each path is followed from the
# dropped rows, relation by relation, in either direction.
CASCADE_PATHS: dict[str, CascadePaths] = {
    "profile": (
        (DamoRelations.PROFILE_ROUGHNESS,),
        (DamoRelations.PROFILE_LINE, DamoRelations.PROFILE_LINE_GROUP),
    ),
    "profile_line": (
        (DamoRelations.PROFILE_LINE_GROUP,),
        (DamoRelations.PROFILE_LINE, DamoRelations.PROFILE_ROUGHNESS),
    ),
    "weirs": ((DamoRelations.OPENING_WEIR, DamoRelations.MANAGEMENT_DEVICE_OPENING),),
    "pumpstations": ((DamoRelations.PUMP_PUMPSTATION, DamoRelations.MANAGEMENT_PUMP),),
    "culverts": ((DamoRelations.MANAGEMENT_DEVICE_CULVERT,),),
    "catchments": ((DamoRelations.CATCHMENT_LATERAL,),),
    "laterals": ((DamoRelations.CATCHMENT_LATERAL,),),
    "overflows": ((OwnRelations.OVERFLOW_SEWER_AREA,),),
    "sewer_areas": ((OwnRelations.OVERFLOW_SEWER_AREA,),),
    "greenhouse_areas": ((OwnRelations.GREENHOUSE_LATERAL_AREA,),),
    "greenhouse_laterals": ((OwnRelations.GREENHOUSE_LATERAL_AREA,),),
}


@dataclass(frozen=True)
class Extension:
    """A column HYDROLIB defines on top of DAMO 2.2.

    Parameters
    ----------
    table : str
        HyDAMO attribute of the table.
    column : str
        The column.
    reason : str
        Why HYDROLIB needs it.
    """

    table: str
    column: str
    reason: str


# Columns HYDROLIB reads on DAMO tables that no DAMO version defines. Each is
# optional: read when present.
OWN_COLUMNS: tuple[Extension, ...] = (
    Extension(
        "management_device",
        "maximaaldebiet",
        "The maximum flow through an orifice (an onderlaat). When given, the D-Flow FM "
        "orifice limits its flow to it in both directions.",
    ),
)


@dataclass(frozen=True)
class ValueRewrite:
    """A value HYDROLIB rewrites when loading, whatever the DAMO version.

    Only the exact value is rewritten, so no general stripping of whitespace or
    commas. A table without the column is left alone.

    Parameters
    ----------
    table : str
        HyDAMO attribute of the table.
    column : str
        The column, under its DAMO 2.2 name.
    old : str
        The value in the file.
    new : str
        The value HYDROLIB uses instead.
    reason : str
        Why HYDROLIB rewrites it.
    """

    table: str
    column: str
    old: str
    new: str
    reason: str


# Applied after the renames, before the derived columns.
VALUE_REWRITES: tuple[ValueRewrite, ...] = (
    ValueRewrite(
        "management",
        "stuurvariabele",
        "bovenkant afsluitmiddel,",
        "bovenkant afsluitmiddel",
        "Every validatiemodule schema from DAMO 2.2 to 2.6 allows only this steering "
        "variable with a trailing comma, a typo upstream. Files made under those "
        "versions keep it even after an upstream fix, so the rewrite is permanent.",
    ),
)

# Filled from ``globalid`` where the column is absent or empty, before duplicates
# get a suffix.
FILLED_FROM_GLOBALID: tuple[Extension, ...] = (
    Extension(
        "management",
        "code",
        "DAMO defines no code for Sturing, but HYDROLIB indexes Sturing by code and "
        "names its controllers after it.",
    ),
)

# Computed after the renames, whatever the DAMO version (see loader.load_layers).
DERIVED_COLUMNS: tuple[Extension, ...] = (
    Extension(
        "management_device",
        "soortafsluitmiddel",
        "The device type under the one name the code reads: a copy of "
        "soortregelmiddel when the file gives no soortafsluitmiddel.",
    ),
    Extension(
        "profile",
        "code",
        "The D-Flow FM cross-section id: the profiellijnid of the line the points "
        "are grouped into, instead of the code of one point.",
    ),
    Extension(
        "catchments",
        "boundary_node",
        "The RR boundary node id: the code of the lateral the catchment's "
        "lateraleknoopid points to.",
    ),
)
