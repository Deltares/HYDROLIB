"""The DAMO version steps: the renames DAMO made in each version, for what HYDROLIB reads.

A step is written forward, from the older version to the newer, like the DAMO release
notes. Each rename is written once, in the step where DAMO made it.

Loading doesn't convert the data step by step. `source_layers` chains the steps into,
per HyDAMO table, the layer to read in the file's version and the renames from that
version's column names to DAMO 2.2, the version of HYDROLIB's data model
(`data_model`). The layer is then read and renamed once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydrolib.dhydamo.core.data_model import TABLES


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
        The tables whose columns were renamed:
        ``{HyDAMO attribute: {column in old: column in new}}``. For a table that moves
        to another layer (see *layers*), the old names are those of its old layer.
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
            "param_profile_values": {"typeruwheid": "typemethoderuwheid"},
            "bridges": {"typeruwheid": "typemethoderuwheid"},
            "culverts": {"typeruwheid": "typemethoderuwheid"},
            "profile_roughness": {"typeruwheid": "typemethoderuwheid"},
        },
    ),
    Step(
        "2.4",
        "2.5",
        layers={
            # DAMO merged Regelmiddel into Afsluitmiddel.
            "management_device": "Afsluitmiddel",
        },
        columns={
            "opening": {"regelmiddelid": "afsluitmiddelid"},
            # Not maximalehoogteonderkant -> hoogteopening: a level, not a height.
            # The loader derives the height from it.
            "management_device": {
                "soortregelmiddel": "typeafsluitmiddel",
                "richting": "stroomrichting",
            },
            "management": {
                "regelmiddelid": "afsluitmiddelid",
                "stuurvariabele": "typestuurvariabele",
            },
            "param_profile_values": {"soortparameter": "typeparameter"},
            "culverts": {"vormkoker": "vormduikersifonhevel"},
        },
    ),
    # DAMO 2.6 renamed none of the columns HYDROLIB reads.
    Step("2.5", "2.6"),
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
        If the version is not supported, a step names a table that isn't a HyDAMO
        table, or two tables read the same layer: the steps are keyed by table, so
        each table needs its own layer.
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
        _check_tables_exist(step)
        layers.update(step.layers)
        for name, columns in step.columns.items():
            # All renames of the step at once, from the names before it: a rename
            # may take the name another column had before the step.
            before = renames[name]
            kept = {
                current: original
                for current, original in before.items()
                if current not in columns
            }
            renamed = {new: before.get(old, old) for old, new in columns.items()}
            renames[name] = {**kept, **renamed}
    _check_layers_are_unique(layers, version)
    return {name: SourceLayer(layers[name], renames[name]) for name in TABLES}


def _check_tables_exist(step: Step) -> None:
    """Raise a ValueError if *step* names a table that isn't a HyDAMO table, e.g. a typo."""
    for field_name, tables in (("layers", step.layers), ("columns", step.columns)):
        for name in tables:
            if name not in TABLES:
                raise ValueError(
                    f'DAMO version step {step.old} -> {step.new} names "{name}" in its '
                    f"{field_name}, which is not a HyDAMO table."
                )


def _check_layers_are_unique(layers: dict[str, str], version: str) -> None:
    """Raise a ValueError if two tables read the same layer in DAMO *version*."""
    table_of_layer: dict[str, str] = {}
    for name, layer in layers.items():
        other = table_of_layer.setdefault(layer.lower(), name)
        if other != name:
            raise ValueError(
                f'HyDAMO tables "{other}" and "{name}" both read layer "{layer}" in '
                f"DAMO {version}; the version steps need one layer per table."
            )
