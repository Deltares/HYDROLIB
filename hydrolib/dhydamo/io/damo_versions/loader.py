"""Load a DAMO GeoPackage of any supported version into HYDROLIB's data model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from hydrolib.dhydamo.core.data_model import TABLES, DamoRelations, Table
from hydrolib.dhydamo.io.damo_versions.steps import SourceLayer

if TYPE_CHECKING:
    from hydrolib.dhydamo.core.hydamo import HyDAMO

logger = logging.getLogger(__name__)

# For the tests only: False drops the columns the data model doesn't declare, so that
# building a model fails on any column it reads without declaring it. Undeclared
# columns pass through under their own names otherwise, so such a read works on a
# DAMO 2.2 file and fails on a later version that renamed the column.
_KEEP_UNDECLARED_COLUMNS = True


def load_layers(
    hydamo: HyDAMO,
    gpkg_path: str | Path,
    sources: dict[str, SourceLayer],
    hydamo_version: str,
    clip_layers: dict | None = None,
    check_3d: bool = True,
) -> None:
    """Read the layers of a DAMO GeoPackage into *hydamo*, under their DAMO 2.2 names.

    Then rewrites values and computes the derived columns, as declared on the tables
    (`Column.rewrite`, `Table.derived_columns`).

    Parameters
    ----------
    hydamo : HyDAMO
        The ``HyDAMO`` instance whose tables are filled.
    gpkg_path : str or Path
        Path to the GeoPackage.
    sources : dict[str, SourceLayer]
        The layers and renames of the GeoPackage's DAMO version, from
        `source_layers`.
    hydamo_version : str
        DAMO version of the GeoPackage, for the messages.
    clip_layers : dict, optional
        Per-table spatial clip: maps a HyDAMO attribute to a ``(geometry, cliptype)``
        tuple. Tables absent from the dict are loaded without clipping.
    check_3d : bool
        Whether to require Z coordinates on profile points when grouping them into
        lines. Default is ``True``.

    Raises
    ------
    KeyError
        If a loaded layer lacks a required column (`data_model.TABLES`); the message
        names the layer and the column. Raised at once.
    ValueError
        If a layer that is not optional is missing, or a layer of points lacks the
        column that groups or orders them into lines; the message names the layer.
        Also if a table's identity, or the column that groups points into lines, is
        empty in some rows; the message names the layer and the first rows. These
        errors are collected over all layers and raised together, one line each.
    """
    gpkg_path = Path(gpkg_path)
    available_layers = {
        layer_name.lower(): layer_name
        for layer_name in gpd.list_layers(gpkg_path).name.tolist()
    }
    errors: list[str] = []
    for name, source in sources.items():
        table = TABLES[name]
        actual_layer = available_layers.get(source.layer.lower())
        if actual_layer is None:
            if table.optional_layer:
                logger.info(
                    'Skipping optional HyDAMO layer "%s" for DAMO %s.',
                    source.layer,
                    hydamo_version,
                )
                continue
            errors.append(
                f'Missing required HyDAMO layer "{source.layer}" for DAMO {hydamo_version}.'
            )
            continue

        kwargs: dict = {
            "gpkg_path": gpkg_path,
            "layer_name": actual_layer,
            "index_col": table.identity_column,
            "column_mapping": source.renames or None,
            "fill_from_globalid": table.column_filled_from_globalid,
            "identity_column": table.identity_column,
            "check_geotype": table.check_geotype,
            "check_3d": check_3d,
        }
        if table.group_by is not None:
            kwargs["groupby_column"] = table.group_by
        if table.order_by is not None:
            kwargs["order_column"] = table.order_by
        if clip_layers and name in clip_layers:
            kwargs["clip"], kwargs["cliptype"] = clip_layers[name]

        # The table's own messages don't name the layer.
        prefix = f'HyDAMO layer "{actual_layer}" (DAMO {hydamo_version}): '
        try:
            getattr(hydamo, name)._read_gpkg_layer(**kwargs)
        except KeyError as exc:
            # A missing column raises at once, as a KeyError, so callers that catch
            # it still do.
            raise KeyError(prefix + str(exc.args[0])) from exc
        except ValueError as exc:
            # e.g. empty codes: collected, so all layers are reported at once
            errors.append(prefix + str(exc))
            continue
        if not _KEEP_UNDECLARED_COLUMNS:
            _drop_undeclared_columns(hydamo, name, table)
    if errors:
        raise ValueError("\n".join(errors))

    _rewrite_values(hydamo)
    # The derived columns of the tables (Table.derived_columns)
    _derive_profile_code(hydamo)
    _derive_boundary_node(hydamo)
    _derive_weir_crest(hydamo)
    _derive_gate_height(hydamo)


def _derive_gate_height(hydamo: HyDAMO) -> None:
    """Set each device's ``hoogteopening`` from its ``maximalehoogteonderkant``, in place.

    From DAMO 2.5 a management device has the level of its gate's lower edge, not
    the gate's height. Where the level is given, the height is the level minus the
    crest of the device's opening, or minus the upstream invert of its culvert. Other
    devices keep their ``hoogteopening``, or none.
    """
    devices = hydamo.management_device
    if devices.empty or "maximalehoogteonderkant" not in devices.columns:
        return
    below = pd.Series(float("nan"), index=devices.index)
    for relation, level in (
        (DamoRelations.MANAGEMENT_DEVICE_OPENING, "laagstedoorstroomhoogte"),
        (DamoRelations.MANAGEMENT_DEVICE_CULVERT, "hoogtebinnenonderkantbov"),
    ):
        parent = getattr(hydamo, relation.parent_table)
        if relation.foreign_key not in devices.columns or level not in parent.columns:
            continue
        levels = pd.Series(
            parent[level].to_numpy(dtype=float), index=parent[relation.parent_key]
        )
        levels = levels[~levels.index.duplicated()]
        below = below.fillna(devices[relation.foreign_key].map(levels))
    height = pd.to_numeric(devices["maximalehoogteonderkant"]) - below
    if "hoogteopening" in devices.columns:
        # The level wins over a hoogteopening a 2.5 file still has from 2.4.
        devices["hoogteopening"] = height.fillna(devices["hoogteopening"])
    else:
        devices["hoogteopening"] = height


def _drop_undeclared_columns(hydamo: HyDAMO, name: str, definition: Table) -> None:
    """Drop, in place, the columns of table *name* that *definition* doesn't read.

    If the file supplies a column with the same name as a derived column that isn't
    also read, it is dropped and then derived. When the derivation can't run, e.g. ``boundary_node``
    without laterals, the column stays absent.
    """
    declared = {
        "geometry",
        *definition.required_names,
        *definition.optional_names,
        *(c for c in (definition.group_by, definition.order_by) if c),
    }
    table = getattr(hydamo, name)
    undeclared = [column for column in table.columns if column not in declared]
    # inplace: returning a copy of an Extended(Geo)DataFrame gives a plain (Geo)DataFrame
    table.drop(columns=undeclared, inplace=True)  # noqa: PD002


def _rewrite_values(hydamo: HyDAMO) -> None:
    """Apply the value rewrites declared on the tables' columns, in place."""
    for name, definition in TABLES.items():
        table = getattr(hydamo, name)
        for column in definition.columns:
            if column.rewrite is not None and column.name in table.columns:
                old, new = column.rewrite
                table[column.name] = table[column.name].replace(old, new)


def _derive_weir_crest(hydamo: HyDAMO) -> None:
    """Fill each weir's empty ``laagstedoorstroomhoogte`` from its openings, in place.

    A weir without a crest of its own (always from DAMO 2.4, which moved the crest
    to Kunstwerkopening) gets the lowest crest of its openings.
    """
    weirs, opening = hydamo.weirs, hydamo.opening
    opening_weir = DamoRelations.OPENING_WEIR
    if weirs.empty or opening.empty or opening_weir.foreign_key not in opening.columns:
        return
    lowest = opening.groupby(opening_weir.foreign_key)["laagstedoorstroomhoogte"].min()
    from_openings = weirs[opening_weir.parent_key].map(lowest)
    if "laagstedoorstroomhoogte" in weirs.columns:
        weirs["laagstedoorstroomhoogte"] = weirs["laagstedoorstroomhoogte"].fillna(
            from_openings
        )
    else:
        weirs["laagstedoorstroomhoogte"] = from_openings


def _derive_profile_code(hydamo: HyDAMO) -> None:
    """Set each profile's ``code`` to its ``profiellijnid``, and index by it, in place."""
    if not hydamo.profile.empty and "profiellijnid" in hydamo.profile.columns:
        # After grouping, each row is a profile line: replace the code of its first
        # point with the line's id.
        hydamo.profile["code"] = hydamo.profile["profiellijnid"]
        hydamo.profile.index = pd.Index(hydamo.profile["code"], name="code")


def _derive_boundary_node(hydamo: HyDAMO) -> None:
    """Set each catchment's ``boundary_node`` to the ``code`` of its lateral, in place."""
    if hydamo.catchments.empty or hydamo.laterals.empty:
        return
    catchment_lateral = DamoRelations.CATCHMENT_LATERAL
    mapping = hydamo.laterals.set_index(catchment_lateral.parent_key)["code"]
    hydamo.catchments["boundary_node"] = hydamo.catchments[
        catchment_lateral.foreign_key
    ].map(mapping)
