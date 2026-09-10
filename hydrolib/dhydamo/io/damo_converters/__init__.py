from __future__ import annotations

from hydrolib.dhydamo.io.damo_converters.v22 import Damo22Converter
from hydrolib.dhydamo.io.damo_converters.v23 import Damo23Converter
from hydrolib.dhydamo.io.damo_converters.v24 import Damo24Converter
from hydrolib.dhydamo.io.damo_converters.v25 import Damo25Converter

_CONVERTERS = {
    "2.2": Damo22Converter,
    "2.3": Damo23Converter,
    "2.4": Damo24Converter,
    "2.5": Damo25Converter,
}

SUPPORTED_HYDAMO_VERSIONS = tuple(_CONVERTERS)


def get_damo_converter(version: str):
    try:
        return _CONVERTERS[version]()
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_HYDAMO_VERSIONS)
        raise ValueError(
            f'Unsupported HyDAMO DAMO version "{version}". Supported versions: {supported}.'
        ) from exc
