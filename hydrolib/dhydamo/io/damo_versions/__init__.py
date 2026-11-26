"""Load a DAMO GeoPackage of a supported version into HYDROLIB's data model.

`steps` holds the renames of each DAMO version, and `loader` reads a GeoPackage into
HYDROLIB's data model (`hydrolib.dhydamo.core.data_model`): the HyDAMO tables under
their DAMO 2.2 names and HYDROLIB's extensions.
"""

from __future__ import annotations

from hydrolib.dhydamo.io.damo_versions.loader import load_layers
from hydrolib.dhydamo.io.damo_versions.steps import (
    SUPPORTED_HYDAMO_VERSIONS,
    SourceLayer,
    source_layers,
)

__all__ = ["SUPPORTED_HYDAMO_VERSIONS", "SourceLayer", "load_layers", "source_layers"]
