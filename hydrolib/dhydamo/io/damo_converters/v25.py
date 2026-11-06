from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.base import LayerSpec, replace_layer_specs
from hydrolib.dhydamo.io.damo_converters.v22 import Damo22Converter


class Damo25Converter(Damo22Converter):
    """DAMO v2.5 converter.

    Schema changes relative to v2.2:
    - HydroObject: categorieoppervlaktewater → categorieoppwaterlichaam
    - Kunstwerkopening: afsluitmiddelid → regelmiddelid
    - Afsluitmiddel replaces Regelmiddel as the management device layer;
      typeafsluitmiddel → soortafsluitmiddel, typeregelbaarheid → soortregelbaarheid
    - Sturing: afsluitmiddelid → regelmiddelid, typestuurvariabele → stuurvariabele
    """

    hydamo_version = "2.5"
    layer_specs = replace_layer_specs(
        Damo22Converter.layer_specs,
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
    )
