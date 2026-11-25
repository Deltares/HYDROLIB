"""The DAMO version steps: the renames DAMO made in each version, for what HYDROLIB reads.

A step is written forward, from the older version to the newer, like the DAMO release
notes. Loading applies the steps backward, from the file's version to DAMO 2.2, the
version of HYDROLIB's data model (`data_model`). Each rename is written once, in the
step where DAMO made it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydrolib.dhydamo.io.damo_converters.data_model import TABLES


@dataclass(frozen=True)
class Step:
    """The layer and column renames from DAMO version *old* to *new*.

    Parameters
    ----------
    old : str
        The older DAMO version, e.g. ``"2.4"``.
    new : str
        The newer DAMO version, e.g. ``"2.5"``.
    layers : dict[str, str]
        The tables whose layer changed: ``{HyDAMO attribute: layer in new}``.
    columns : dict[str, dict[str, str]]
        ``{layer in new: {column in old: column in new}}``.
    """

    old: str
    new: str
    layers: dict[str, str] = field(default_factory=dict)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)


STEPS: tuple[Step, ...] = (
    Step("2.2", "2.3"),
    Step(
        "2.3",
        "2.4",
        columns={
            "NormgeparamProfielWaarde": {"typeruwheid": "typemethoderuwheid"},
            "Brug": {"typeruwheid": "typemethoderuwheid"},
            "DuikerSifonHevel": {"typeruwheid": "typemethoderuwheid"},
            "RuwheidProfiel": {"typeruwheid": "typemethoderuwheid"},
        },
    ),
    Step(
        "2.4",
        "2.5",
        layers={
            # DAMO merged Regelmiddel into Afsluitmiddel.
            "management_device": "Afsluitmiddel",
            # Not a DAMO change: 2.5 has no Regelmiddel. Kept from the old 2.5
            # converter until closing_device is removed (ticket 12).
            "closing_device": "Regelmiddel",
        },
        columns={
            "HydroObject": {"categorieoppwaterlichaam": "categorieoppervlaktewater"},
            "Kunstwerkopening": {"regelmiddelid": "afsluitmiddelid"},
            # DAMO's own renames on Afsluitmiddel. Because management_device reads
            # Afsluitmiddel from 2.5 on, its device type arrives under Afsluitmiddel's
            # 2.4 name soortafsluitmiddel, not Regelmiddel's 2.2 name
            # soortregelmiddel. Ticket 12 maps it to soortregelmiddel.
            "Afsluitmiddel": {
                "soortafsluitmiddel": "typeafsluitmiddel",
                "soortregelbaarheid": "typeregelbaarheid",
            },
            "Sturing": {
                "regelmiddelid": "afsluitmiddelid",
                "stuurvariabele": "typestuurvariabele",
            },
            "NormgeparamProfielWaarde": {"soortparameter": "typeparameter"},
            "DuikerSifonHevel": {"vormkoker": "vormduikersifonhevel"},
        },
    ),
)

SUPPORTED_HYDAMO_VERSIONS: tuple[str, ...] = ("2.2", *(step.new for step in STEPS))


@dataclass(frozen=True)
class SourceLayer:
    """Where a HyDAMO table comes from in a file of one DAMO version.

    Parameters
    ----------
    layer : str
        The layer in that version.
    renames : dict[str, str]
        ``{column in that version: DAMO 2.2 name}``, for the renamed columns.
    """

    layer: str
    renames: dict[str, str]


def source_layers(version: str) -> dict[str, SourceLayer]:
    """Return, per HyDAMO table, its layer and column renames in a DAMO *version*.

    Chains the steps between DAMO 2.2 and *version*.

    Parameters
    ----------
    version : str
        DAMO version, one of `SUPPORTED_HYDAMO_VERSIONS`, e.g. ``"2.4"``.

    Returns
    -------
    dict[str, SourceLayer]
        By HyDAMO attribute, in the order of `data_model.TABLES`.

    Raises
    ------
    ValueError
        If the version is not supported.
    """
    if version not in SUPPORTED_HYDAMO_VERSIONS:
        supported = ", ".join(SUPPORTED_HYDAMO_VERSIONS)
        raise ValueError(
            f'Unsupported HyDAMO DAMO version "{version}". Supported versions: {supported}.'
        )
    layers = {name: table.layer for name, table in TABLES.items()}
    # {column in the current version: DAMO 2.2 name}, per table
    renames: dict[str, dict[str, str]] = {name: {} for name in TABLES}
    for step in STEPS[: SUPPORTED_HYDAMO_VERSIONS.index(version)]:
        layers.update(step.layers)
        for name, layer in layers.items():
            for old, new in step.columns.get(layer, {}).items():
                renames[name][new] = renames[name].pop(old, old)
    return {name: SourceLayer(layers[name], renames[name]) for name in TABLES}
