"""Load a DAMO GeoPackage of any supported version into HYDROLIB's data model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd

from hydrolib.dhydamo.core.relations import RelationsRR
from hydrolib.dhydamo.io.damo_converters.data_model import FILLED_FROM_GLOBALID, TABLES
from hydrolib.dhydamo.io.damo_converters.steps import SourceLayer

if TYPE_CHECKING:
    from hydrolib.dhydamo.core.hydamo import HyDAMO

logger = logging.getLogger(__name__)


def load_layers(
    hydamo: HyDAMO,
    gpkg_path: str | Path,
    sources: dict[str, SourceLayer],
    hydamo_version: str,
    clip_layers: dict | None = None,
    check_3d: bool = True,
) -> None:
    """Read the layers of a DAMO GeoPackage into *hydamo*, under their DAMO 2.2 names.

    Then computes the derived columns (`data_model.DERIVED_COLUMNS`).

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
    ValueError
        If a layer that is not optional is missing.
    """
    gpkg_path = Path(gpkg_path)
    available_layers = {
        layer_name.lower(): layer_name
        for layer_name in gpd.list_layers(gpkg_path).name.tolist()
    }
    filled_from_globalid = {ext.table: ext.column for ext in FILLED_FROM_GLOBALID}

    for name, source in sources.items():
        table = TABLES[name]
        actual_layer = available_layers.get(source.layer.lower())
        if actual_layer is None:
            if table.optional:
                logger.info(
                    'Skipping optional HyDAMO layer "%s" for DAMO %s.',
                    source.layer,
                    hydamo_version,
                )
                continue
            raise ValueError(
                f'Missing required HyDAMO layer "{source.layer}" for DAMO {hydamo_version}.'
            )

        kwargs: dict = {
            "gpkg_path": gpkg_path,
            "layer_name": actual_layer,
            "index_col": table.index,
            "column_mapping": source.renames or None,
            "fill_from_globalid": filled_from_globalid.get(name),
            "check_geotype": table.check_geotype,
            "check_3d": check_3d,
        }
        if table.group_by is not None:
            kwargs["groupby_column"] = table.group_by
        if table.order_by is not None:
            kwargs["order_column"] = table.order_by
        if clip_layers and name in clip_layers:
            kwargs["clip"], kwargs["cliptype"] = clip_layers[name]

        getattr(hydamo, name)._read_gpkg_layer(**kwargs)

    # The derived columns of data_model.DERIVED_COLUMNS
    _derive_soortafsluitmiddel(hydamo)
    _derive_profile_code(hydamo)
    _derive_boundary_node(hydamo)


def _derive_profile_code(hydamo: HyDAMO) -> None:
    """Set each profile's ``code`` to its ``profiellijnid``, in place."""
    if not hydamo.profile.empty and "profiellijnid" in hydamo.profile.columns:
        # After grouping, each row is a profile line: replace the code of its first
        # point with the line's id.
        hydamo.profile["code"] = hydamo.profile["profiellijnid"]


def _derive_boundary_node(hydamo: HyDAMO) -> None:
    """Set each catchment's ``boundary_node`` to the ``code`` of its lateral, in place."""
    if hydamo.catchments.empty or hydamo.laterals.empty:
        return
    catchment_lateral = RelationsRR.CATCHMENT_LATERAL
    mapping = hydamo.laterals.set_index(catchment_lateral.parent_key)["code"]
    hydamo.catchments["boundary_node"] = hydamo.catchments[
        catchment_lateral.foreign_key
    ].map(mapping)


def _derive_soortafsluitmiddel(hydamo: HyDAMO) -> None:
    """Copy ``soortregelmiddel`` to ``soortafsluitmiddel`` on the management devices, in place."""
    device = hydamo.management_device
    if (
        not device.empty
        and "soortregelmiddel" in device.columns
        and "soortafsluitmiddel" not in device.columns
    ):
        device["soortafsluitmiddel"] = device["soortregelmiddel"]
