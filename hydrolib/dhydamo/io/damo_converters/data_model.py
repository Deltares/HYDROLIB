"""HYDROLIB's data model: the HyDAMO tables, under their DAMO 2.2 names, and HYDROLIB's extensions.

Every DAMO version is loaded into this one model: the version steps (`steps`) rename a
file's layers and columns back to DAMO 2.2. The model is plain data and independent of
any DAMO version. ``HyDAMO.__init__`` builds its tables from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType

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
