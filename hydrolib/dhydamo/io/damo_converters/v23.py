from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.v22 import Damo22Converter


class Damo23Converter(Damo22Converter):
    """DAMO v2.3 converter.

    Schema is identical to v2.2. This class exists as a version sentinel so that
    files tagged 2.3 load correctly without raising an unsupported-version error.
    If v2.3 introduces schema differences in the future, override layer_specs here.
    """

    hydamo_version = "2.3"
