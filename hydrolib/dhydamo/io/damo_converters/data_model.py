"""HYDROLIB's data model: the HyDAMO tables, under their DAMO 2.2 names, and HYDROLIB's extensions.

Every DAMO version is loaded into this one model: the version steps (`steps`) rename a
file's layers and columns back to DAMO 2.2. The model is plain data and independent of
any DAMO version.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Table:
    """Where a HyDAMO table comes from in DAMO 2.2, and how it is read.

    Parameters
    ----------
    layer : str
        DAMO 2.2 layer, matched case-insensitively.
    index : str, optional
        Column that becomes the index; duplicate values get a suffix.
    group_by : str, optional
        Column whose points are grouped into one line each.
    order_by : str, optional
        Column that orders the points of a line.
    optional : bool
        Whether a file may lack the layer. Default is ``True``.
    check_geotype : bool
        Whether the geometries must have the table's geometry type. Default is
        ``True``.
    """

    layer: str
    index: str | None = None
    group_by: str | None = None
    order_by: str | None = None
    optional: bool = True
    check_geotype: bool = True


# The HyDAMO tables, by HyDAMO attribute, in loading order.
TABLES: dict[str, Table] = {
    "branches": Table("HydroObject", index="code"),
    "profile": Table(
        "ProfielPunt", group_by="profiellijnid", order_by="codevolgnummer"
    ),
    "profile_roughness": Table("RuwheidProfiel"),
    "profile_line": Table("profiellijn"),
    "profile_group": Table("profielgroep"),
    "param_profile": Table("NormgeparamProfiel"),
    "param_profile_values": Table("NormgeparamProfielWaarde"),
    "weirs": Table("Stuw", index="code"),
    "opening": Table("Kunstwerkopening"),
    "closing_device": Table("Afsluitmiddel"),
    "management_device": Table("Regelmiddel"),
    "bridges": Table("Brug", index="code"),
    "culverts": Table("DuikerSifonHevel", index="code"),
    "pumpstations": Table("Gemaal", index="code"),
    "pumps": Table("Pomp", index="code"),
    "management": Table("Sturing", index="code"),
    "boundary_conditions": Table("hydrologischerandvoorwaarde", index="code"),
    "catchments": Table("afvoergebiedaanvoergebied", index="code", check_geotype=False),
    "laterals": Table("lateraleknoop"),
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
