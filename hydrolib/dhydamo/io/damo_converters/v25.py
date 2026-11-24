"""Converter for DAMO 2.5."""

from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.base import LayerSpec, replace_layer_specs
from hydrolib.dhydamo.io.damo_converters.v24 import (
    ROUGHNESS_TYPE_MAPPING,
    Damo24Converter,
)


class Damo25Converter(Damo24Converter):
    """DAMO v2.5 converter.

    Schema changes relative to v2.4:
    - HydroObject: categorieoppervlaktewater → categorieoppwaterlichaam
    - Kunstwerkopening: afsluitmiddelid → regelmiddelid
    - Afsluitmiddel replaces Regelmiddel as the management device layer;
      typeafsluitmiddel → soortafsluitmiddel, typeregelbaarheid → soortregelbaarheid
    - Sturing: afsluitmiddelid → regelmiddelid, typestuurvariabele → stuurvariabele
    - NormgeparamProfielWaarde: soortparameter → typeparameter
    - DuikerSifonHevel: vormkoker → vormduikersifonhevel

    A spec overridden here replaces the 2.4 spec of that table, so it repeats the
    2.4 renames (``ROUGHNESS_TYPE_MAPPING``).
    """

    hydamo_version = "2.5"
    layer_specs = replace_layer_specs(
        Damo24Converter.layer_specs,
        LayerSpec(
            "branches",
            "HydroObject",
            index_col="code",
            column_mapping={"categorieoppervlaktewater": "categorieoppwaterlichaam"},
        ),
        LayerSpec(
            "opening",
            "Kunstwerkopening",
            column_mapping={"afsluitmiddelid": "regelmiddelid"},
        ),
        # Override the inherited Afsluitmiddel -> closing_device mapping:
        # in 2.5, Afsluitmiddel populates the internal management_device below.
        # The internal closing_device attribute is retained, but Regelmiddel
        # is absent from 2.5 input, so this optional spec is skipped.
        # Removing this override alone would load Afsluitmiddel into both tables.
        LayerSpec("closing_device", "Regelmiddel"),
        LayerSpec(
            "management_device",
            "Afsluitmiddel",
            column_mapping={
                "typeafsluitmiddel": "soortafsluitmiddel",
                "typeregelbaarheid": "soortregelbaarheid",
            },
        ),
        LayerSpec(
            "management",
            "Sturing",
            index_col="code",
            column_mapping={
                "afsluitmiddelid": "regelmiddelid",
                "typestuurvariabele": "stuurvariabele",
            },
        ),
        LayerSpec(
            "param_profile_values",
            "NormgeparamProfielWaarde",
            column_mapping={
                **ROUGHNESS_TYPE_MAPPING,
                "typeparameter": "soortparameter",
            },
        ),
        LayerSpec(
            "culverts",
            "DuikerSifonHevel",
            index_col="code",
            column_mapping={
                **ROUGHNESS_TYPE_MAPPING,
                "vormduikersifonhevel": "vormkoker",
            },
        ),
    )
