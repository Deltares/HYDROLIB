"""HYDROLIB's data model: the HyDAMO tables, under their DAMO 2.2 names, and HYDROLIB's extensions.

Every DAMO version is loaded into this one model: the version steps
(`hydrolib.dhydamo.io.damo_versions.steps`) rename a file's layers and columns back to
DAMO 2.2. The model is plain data and independent of any DAMO version.
``HyDAMO.__init__`` builds its tables from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import Literal, TypeAlias

import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

Geotype = type[BaseGeometry] | UnionType


@dataclass(frozen=True)
class Column:
    """A declared column with something to say beyond its name.

    Ordinary columns are declared by name only; a column is declared as a `Column`
    when HYDROLIB has a reason to note, or treats it in a special way.

    Parameters
    ----------
    name : str
        The column, under its DAMO 2.2 name.
    reason : str
        Why HYDROLIB needs it, or treats it this way.
    hydrolib_only : bool
        Whether the column is HYDROLIB's own: no DAMO version defines it. Default is
        ``False``.
    fill_from_globalid : bool
        Whether the column is filled from ``globalid`` where it is absent or empty,
        before duplicates get a suffix. Default is ``False``.
    rewrite : tuple[str, str], optional
        ``(old, new)``: a value rewritten when loading, whatever the DAMO version.
        Only the exact value is rewritten, so no general stripping of whitespace or
        commas.
    identity : bool
        Whether the column is the table's identity: its index and the id of its model
        objects. An empty value raises an error, duplicates get a suffix. Default is
        ``False``.
    """

    name: str
    reason: str
    hydrolib_only: bool = False
    fill_from_globalid: bool = False
    rewrite: tuple[str, str] | None = None
    identity: bool = False


# The reason of the code identity, which tightens DAMO.
_IDENTITY = (
    "Tightens DAMO, where code is optional and not unique. It is the table's index, "
    "the id of its model objects and their name in logs and settings, so an empty "
    "code is an error and duplicates get a suffix."
)


def _column(column: str | Column) -> Column:
    """Return *column* as a `Column`; a plain name has no reason."""
    return column if isinstance(column, Column) else Column(column, reason="")


@dataclass(frozen=True)
class Table:
    """A HyDAMO table: its DAMO 2.2 layer, the columns HYDROLIB reads, and how it is read.

    A column is declared by its name, or as a `Column` when there is more to say.

    Parameters
    ----------
    layer : str
        DAMO 2.2 layer, matched case-insensitively.
    geotype : type[BaseGeometry] or UnionType or None
        Geometry type of a spatial table, e.g. ``Point`` or ``Polygon | MultiPolygon``;
        ``None`` for a table without geometry.
    required_columns : tuple[str or Column, ...]
        Columns a loaded layer must have; a missing one is an error.
    optional_columns : tuple[str or Column, ...]
        Other columns HYDROLIB reads. Loading doesn't need them: a missing one is
        absent after loading. Building the model may still need one for some
        objects, e.g. ``debiet`` for a discharge boundary; it then fails on the
        missing column.
    derived_columns : tuple[Column, ...]
        Columns computed after loading, whatever the DAMO version (see
        `hydrolib.dhydamo.io.damo_versions.loader`). A derived column may also be
        read; its reason then says how the two combine, e.g. overwritten or only
        filled where empty.
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
    required_columns: tuple[str | Column, ...]
    optional_columns: tuple[str | Column, ...] = ()
    derived_columns: tuple[Column, ...] = ()
    group_by: str | None = None
    order_by: str | None = None
    optional_layer: bool = True
    check_geotype: bool = True

    def __post_init__(self) -> None:
        """Check that at most one column is filled from ``globalid``.

        Raises
        ------
        ValueError
            If more than one column is: the loader fills one column per table.
        """
        filled = [column.name for column in self.columns if column.fill_from_globalid]
        if len(filled) > 1:
            raise ValueError(
                f"Table {self.layer} fills more than one column from globalid "
                f"({', '.join(filled)}); the loader fills one column per table."
            )

    @property
    def required_names(self) -> tuple[str, ...]:
        """The names of the required columns."""
        return tuple(_column(column).name for column in self.required_columns)

    @property
    def optional_names(self) -> tuple[str, ...]:
        """The names of the optional columns."""
        return tuple(_column(column).name for column in self.optional_columns)

    @property
    def columns(self) -> tuple[Column, ...]:
        """The read columns, required then optional, each as a `Column`."""
        return tuple(
            _column(column)
            for column in (*self.required_columns, *self.optional_columns)
        )

    @property
    def identity_column(self) -> str | None:
        """The read column that is the table's identity and index, if any.

        A derived identity, such as the profile's ``code``, isn't read: its
        derivation sets the index.
        """
        for column in self.columns:
            if column.identity:
                return column.name
        return None

    @property
    def column_filled_from_globalid(self) -> str | None:
        """The column filled from ``globalid``, if any."""
        for column in self.columns:
            if column.fill_from_globalid:
                return column.name
        return None


# The HyDAMO tables loaded from DAMO, by HyDAMO attribute, in loading order. Column
# names are DAMO 2.2 names; the required columns are listed in the order the
# missing-column error lists them.
TABLES: dict[str, Table] = {
    "branches": Table(
        "HydroObject",
        geotype=LineString,
        required_columns=(Column("code", _IDENTITY, identity=True), "geometry"),
        optional_columns=("globalid", "naam"),
    ),
    "profile": Table(
        "ProfielPunt",
        geotype=LineString,
        required_columns=("code", "geometry", "globalid", "profiellijnid"),
        derived_columns=(
            Column(
                "code",
                "The D-Flow FM cross-section id and the table's index: the "
                "profiellijnid of the line the points are grouped into, instead of "
                "the code of one point. Unique by construction.",
                identity=True,
            ),
        ),
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
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "geometry",
            "globalid",
            "afvoercoefficient",
        ),
        optional_columns=("naam", "laagstedoorstroomhoogte"),
        derived_columns=(
            Column(
                "laagstedoorstroomhoogte",
                "The crest level of a universal weir. DAMO moved it from Stuw to "
                "Kunstwerkopening in 2.4: where the weir has none, it is filled with "
                "the lowest crest of the weir's openings.",
            ),
        ),
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
    "management_device": Table(
        "Regelmiddel",
        geotype=None,
        # Joined on globalid; code is optional on the 2.5 Afsluitmiddel.
        required_columns=(),
        optional_columns=(
            "code",
            Column(
                "overlaatonderlaat",
                "Mandatory in DAMO up to 2.4, optional from 2.5, where the layer also "
                "holds the closing devices of locks, coupures and tunnels.",
            ),
            "globalid",
            "kunstwerkopeningid",
            "duikersifonhevelid",
            "gemaalid",
            "soortregelmiddel",
            "hoogteopening",
            "afvoercoefficient",
            Column(
                "maximaaldebiet",
                "The maximum flow through an orifice (an onderlaat). When given, the "
                "D-Flow FM orifice limits its flow to it in both directions.",
                hydrolib_only=True,
            ),
        ),
    ),
    "bridges": Table(
        "Brug",
        geotype=Point,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "globalid",
            "geometry",
            "lengte",
            "intreeverlies",
            "uittreeverlies",
            "ruwheid",
            "typeruwheid",
        ),
        optional_columns=("naam",),
    ),
    "culverts": Table(
        "DuikerSifonHevel",
        geotype=LineString,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
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
    ),
    "pumpstations": Table(
        "Gemaal",
        geotype=Point,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "globalid",
            "geometry",
        ),
        optional_columns=("naam",),
    ),
    "pumps": Table(
        "Pomp",
        geotype=None,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "globalid",
            "gemaalid",
            "maximalecapaciteit",
        ),
    ),
    "management": Table(
        "Sturing",
        geotype=None,
        required_columns=(
            Column(
                "code",
                "DAMO defines no code for Sturing, but HYDROLIB indexes Sturing by "
                "code and names its controllers after it.",
                fill_from_globalid=True,
                identity=True,
            ),
            "globalid",
        ),
        optional_columns=(
            "regelmiddelid",
            "pompid",
            "hydrologischerandvoorwaardeid",
            "typecontroller",
            Column(
                "stuurvariabele",
                "Every validatiemodule schema from DAMO 2.2 to 2.6 allows only this "
                "steering variable with a trailing comma, a typo upstream. Files made "
                "under those versions keep it even after an upstream fix, so the "
                "rewrite is permanent.",
                rewrite=("bovenkant afsluitmiddel,", "bovenkant afsluitmiddel"),
            ),
            "doelvariabele",
            "meetlocatieid",
            "bovengrens",
            "ondergrens",
            "streefwaarde",
        ),
    ),
    "boundary_conditions": Table(
        "hydrologischerandvoorwaarde",
        geotype=Point,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "typerandvoorwaarde",
            "geometry",
        ),
        optional_columns=("globalid", "waterstand", "debiet"),
    ),
    "catchments": Table(
        "afvoergebiedaanvoergebied",
        geotype=Polygon | MultiPolygon,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "geometry",
            "globalid",
            "lateraleknoopid",
        ),
        derived_columns=(
            Column(
                "boundary_node",
                "The RR boundary node id: the code of the lateral the catchment's "
                "lateraleknoopid points to.",
            ),
        ),
        check_geotype=False,
    ),
    "laterals": Table(
        "lateraleknoop",
        geotype=Point,
        required_columns=(
            Column("code", _IDENTITY, identity=True),
            "geometry",
            "globalid",
        ),
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
    optional_columns : tuple[Column, ...]
        Other columns HYDROLIB reads when present, each with its reason.
    """

    geotype: type[BaseGeometry]
    required_columns: tuple[str, ...]
    reason: str
    optional_columns: tuple[Column, ...] = ()

    @property
    def optional_names(self) -> tuple[str, ...]:
        """The names of the optional columns."""
        return tuple(column.name for column in self.optional_columns)


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
        optional_columns=(
            Column(
                "riool_berging_mm",
                "The sewer storage (mm) of the area. Without it, the storage given to "
                "the RR model setup (a value or a raster) is used.",
            ),
            Column(
                "riool_poc_m3s",
                "The pump capacity (m3/s) of the area. Without it, the capacity given "
                "to the RR model setup (a value or a raster, in mm/h) is used.",
            ),
        ),
    ),
    "greenhouse_areas": OwnTable(
        Polygon,
        ("code", "geometry"),
        "RR greenhouse areas.",
        optional_columns=(
            Column(
                "roof_storage_mm",
                "The roof storage (mm) of the area. Without it, the storage given to "
                "the RR model setup (a value or a raster) is used.",
            ),
            Column(
                "basin_storage_class",
                "The basin storage class of the area. Without it, the class given to "
                "the RR model setup is used.",
            ),
        ),
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


ExtensionKind: TypeAlias = Literal[
    "identity",
    "HYDROLIB-only column",
    "value rewrite",
    "filled from globalid",
    "derived column",
]


@dataclass(frozen=True)
class Extension:
    """A column HYDROLIB defines or treats on top of DAMO 2.2, from `extensions`.

    Parameters
    ----------
    table : str
        HyDAMO attribute of the table.
    column : str
        The column.
    kind : ExtensionKind
        What HYDROLIB does with it.
    reason : str
        Why HYDROLIB needs it.
    """

    table: str
    column: str
    kind: ExtensionKind
    reason: str


def extensions() -> list[Extension]:
    """Return HYDROLIB's extensions to DAMO 2.2, as declared on the tables' columns.

    Returns
    -------
    list[Extension]
        Per table in `TABLES` order: its identity, HYDROLIB-only columns, value
        rewrites and columns filled from ``globalid``, in column order, then its
        derived columns.
    """
    found = []
    for name, table in TABLES.items():
        for column in (*table.columns, *table.derived_columns):
            if column.identity:
                found.append(Extension(name, column.name, "identity", column.reason))
        for column in table.columns:
            if column.hydrolib_only:
                found.append(
                    Extension(name, column.name, "HYDROLIB-only column", column.reason)
                )
            if column.rewrite is not None:
                found.append(
                    Extension(name, column.name, "value rewrite", column.reason)
                )
            if column.fill_from_globalid:
                found.append(
                    Extension(name, column.name, "filled from globalid", column.reason)
                )
        for column in table.derived_columns:
            found.append(Extension(name, column.name, "derived column", column.reason))
    return found
