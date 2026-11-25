"""Load a DAMO GeoPackage of a supported version into HYDROLIB's data model.

`data_model` holds the HyDAMO tables under their DAMO 2.2 names and HYDROLIB's
extensions, `steps` the renames of each DAMO version, and `loader` reads a GeoPackage.
"""

from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.loader import load_layers
from hydrolib.dhydamo.io.damo_converters.steps import (
    SUPPORTED_HYDAMO_VERSIONS,
    SourceLayer,
    source_layers,
)

__all__ = ["SUPPORTED_HYDAMO_VERSIONS", "SourceLayer", "load_layers", "source_layers"]
