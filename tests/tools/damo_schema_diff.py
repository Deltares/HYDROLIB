"""Diff the HyDAMO schemas of two DAMO versions, as a help for writing a version step.

Uses the schemas of the installed ``hydamo_validation``. Run from the repo root::

    pixi run python tests/tools/damo_schema_diff.py

or call ``diff("2.5", "2.6")`` for a single comparison. The steps and the data model
are checked against the schemas by the contract test,
``tests/dhydamo/io/test_damo_contract.py``.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import hydamo_validation
from hydrolib.dhydamo.core.data_model import TABLES
from hydrolib.dhydamo.io.damo_versions import (
    SUPPORTED_HYDAMO_VERSIONS,
    source_layers,
)


def _schema_dir() -> Path:
    """Return the folder with the HyDAMO schemas of the installed validatiemodule."""
    return Path(hydamo_validation.__file__).parent / "schemas" / "hydamo"


def _schema_versions() -> list[str]:
    """Return the DAMO versions of the installed schemas, oldest first."""
    versions = []
    for path in _schema_dir().glob("HyDAMO_*.json"):
        versions.append(path.stem.removeprefix("HyDAMO_"))

    # Sort by number, so that a future 2.10 comes after 2.9.
    def version_number(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    return sorted(versions, key=version_number)


def _load_schema(version: str) -> dict[str, dict[str, Any]]:
    """Return the fields of each layer, ``{layer: {field: properties}}``."""
    path = _schema_dir() / f"HyDAMO_{version}.json"
    definitions = json.loads(path.read_text(encoding="utf-8"))["definitions"]
    fields = {}
    for layer, definition in definitions.items():
        fields[layer] = definition["properties"]
    return fields


def _field_signature(props: dict[str, Any]) -> dict[str, Any]:
    """Return the properties of a schema field that the diff compares.

    Only the properties that change what data is valid are compared: the type, the
    allowed values (``enum``), whether the field is mandatory (``minItems`` of 1) and
    the layer it refers to (``VerwijstNaar``, Dutch for "refers to"). Formatting
    limits such as ``maxlength`` and ``pattern`` are left out.
    """
    return {
        "type": props.get("type"),
        "enum": sorted(props.get("enum", [])),
        "mandatory": props.get("minItems", 0) >= 1,
        "references": props.get("VerwijstNaar"),
    }


def _declared_fields() -> dict[str, set[str]]:
    """Return, per layer HYDROLIB loads, the fields the data model declares.

    These are the declared columns and the group and order columns, under their
    DAMO 2.2 names and under the names the steps rename them from. Only these
    fields get their property changes reported.
    """
    fields: dict[str, set[str]] = {}
    for version in SUPPORTED_HYDAMO_VERSIONS:
        for name, source in source_layers(version).items():
            table = TABLES[name]
            layer_fields = fields.setdefault(source.layer.lower(), set())
            layer_fields.update(column.name for column in table.columns)
            for column in (table.group_by, table.order_by):
                if column is not None:
                    layer_fields.add(column)
            for file_name, target in source.renames.items():
                if target in layer_fields:
                    layer_fields.add(file_name)
    return fields


def _group_by_layers(layers_of_field: dict[str, set[str]]) -> list[str]:
    """Return one report line per set of layers, with the fields changed in all of them.

    A field removed from ten layers then takes one line instead of ten.
    """
    fields_of_layers: dict[tuple[str, ...], list[str]] = {}
    for field, layers in sorted(layers_of_field.items()):
        fields_of_layers.setdefault(tuple(sorted(layers)), []).append(field)
    lines = []
    for layer_names, fields in fields_of_layers.items():
        lines.append(f"  {', '.join(fields)}  (in {', '.join(layer_names)})")
    return lines


def _diff_schemas(
    old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]
) -> list[str]:
    """Return the report lines of the differences from *old* to *new*.

    Only layers that HYDROLIB loads are reported. Added and removed fields are all
    listed, because a removed field may have been renamed; property changes only for
    the fields the data model declares. Everything else is only counted.
    """
    required = _declared_fields()
    lines = []

    removed_layers = sorted((set(old) - set(new)) & set(required))
    if removed_layers:
        lines.append(f"Layers removed: {', '.join(removed_layers)}")
    added_layers = sorted((set(new) - set(old)) & set(required))
    if added_layers:
        lines.append(f"Layers added: {', '.join(added_layers)}")

    # Collect the changes of the loaded layers that are in both schemas.
    removed_fields: dict[str, set[str]] = {}
    added_fields: dict[str, set[str]] = {}
    property_lines = []
    hidden_properties = 0
    for layer in sorted(set(old) & set(new) & set(required)):
        old_fields, new_fields = old[layer], new[layer]
        for field in set(old_fields) - set(new_fields):
            removed_fields.setdefault(field, set()).add(layer)
        for field in set(new_fields) - set(old_fields):
            added_fields.setdefault(field, set()).add(layer)
        for field in sorted(set(old_fields) & set(new_fields)):
            before = _field_signature(old_fields[field])
            after = _field_signature(new_fields[field])
            for name, old_value in before.items():
                new_value = after[name]
                if old_value == new_value:
                    continue
                if field in required[layer]:
                    property_lines.append(
                        f"  {layer}.{field} {name}: {old_value} -> {new_value}"
                    )
                else:
                    hidden_properties += 1

    if removed_fields:
        lines.append("Fields removed:")
        lines.extend(_group_by_layers(removed_fields))
    if added_fields:
        lines.append("Fields added:")
        lines.extend(_group_by_layers(added_fields))
    if property_lines:
        lines.append("Changed properties of fields HYDROLIB reads:")
        lines.extend(property_lines)

    other_layers = 0
    for layer in set(old) | set(new):
        if layer not in required and old.get(layer) != new.get(layer):
            other_layers += 1
    lines.append(
        f"Not shown: {hidden_properties} property change(s) of fields HYDROLIB "
        f"doesn't read, {other_layers} changed layer(s) HYDROLIB doesn't load."
    )
    return lines


def diff(old: str, new: str) -> None:
    """Print the differences between the HyDAMO schemas of two DAMO versions.

    Prints the layers and fields added or removed from *old* to *new*, for the layers
    HYDROLIB loads, and the fields HYDROLIB reads whose type, allowed values,
    mandatory flag or reference changed. A removed field next to an added one may be
    a rename, for the step from *old* to *new*.

    Parameters
    ----------
    old : str
        The older DAMO version, e.g. ``"2.5"``.
    new : str
        The newer DAMO version, e.g. ``"2.6"``.

    Raises
    ------
    FileNotFoundError
        If the installed ``hydamo_validation`` has no schema for *old* or *new*.

    Examples
    --------
    >>> diff("2.5", "2.6")  # doctest: +SKIP
    """
    lines = _diff_schemas(_load_schema(old), _load_schema(new))
    print("\n".join(lines) if lines else "No differences.")


if __name__ == "__main__":
    # A step is two consecutive versions of the installed schemas, e.g. 2.5 and 2.6.
    for old, new in pairwise(_schema_versions()):
        print(f"===== DAMO {old} -> {new} =====")
        diff(old, new)
        print()
