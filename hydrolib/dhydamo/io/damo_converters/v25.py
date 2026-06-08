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
        # In DAMO v2.5, Afsluitmiddel became the management device (previously
        # Regelmiddel in v2.2). There is no longer a separate closing device
        # layer. Redirecting to the old v2.2 source name "Regelmiddel" means
        # load_into() will not find the layer and skip it — all LayerSpecs are
        # optional=True by default, so a missing layer is logged and skipped
        # rather than treated as an error.
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
