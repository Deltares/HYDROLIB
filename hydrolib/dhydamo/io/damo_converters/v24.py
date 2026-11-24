"""Converter for DAMO 2.4."""

from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.base import LayerSpec, replace_layer_specs
from hydrolib.dhydamo.io.damo_converters.v22 import Damo22Converter

# DAMO 2.4 renamed typeruwheid to typemethoderuwheid.
ROUGHNESS_TYPE_MAPPING = {"typemethoderuwheid": "typeruwheid"}


class Damo24Converter(Damo22Converter):
    """DAMO v2.4 converter.

    Schema changes relative to v2.2:
    - NormgeparamProfielWaarde, Brug, DuikerSifonHevel, RuwheidProfiel:
      typeruwheid → typemethoderuwheid
    """

    hydamo_version = "2.4"
    layer_specs = replace_layer_specs(
        Damo22Converter.layer_specs,
        LayerSpec(
            "param_profile_values",
            "NormgeparamProfielWaarde",
            column_mapping=ROUGHNESS_TYPE_MAPPING,
        ),
        LayerSpec(
            "bridges",
            "Brug",
            index_col="code",
            column_mapping=ROUGHNESS_TYPE_MAPPING,
        ),
        LayerSpec(
            "culverts",
            "DuikerSifonHevel",
            index_col="code",
            column_mapping=ROUGHNESS_TYPE_MAPPING,
        ),
        LayerSpec(
            "profile_roughness",
            "RuwheidProfiel",
            column_mapping=ROUGHNESS_TYPE_MAPPING,
        ),
    )
