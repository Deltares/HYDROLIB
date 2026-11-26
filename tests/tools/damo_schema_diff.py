"""Compare HyDAMO DAMO schemas and check the DAMO version steps against them.

Uses the schemas of the installed ``hydamo_validation``. Run from the repo root::

    pixi run python tests/tools/damo_schema_diff.py

or call ``diff("2.5", "2.6")`` or ``check_all()`` for a single comparison.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import hydamo_validation
from hydrolib.dhydamo.core.data_model import (
    FILLED_FROM_GLOBALID,
    TABLES,
)
from hydrolib.dhydamo.core.hydamo import HyDAMO
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


def _required_fields() -> dict[str, set[str]]:
    """Return, per layer HYDROLIB loads, the fields HYDROLIB requires.

    These are the HyDAMO required columns and the index, group and order columns,
    under their HYDROLIB names and under the names the steps rename them from.
    Only these fields get their property changes reported.
    """
    hydamo = HyDAMO()
    fields: dict[str, set[str]] = {}
    for version in SUPPORTED_HYDAMO_VERSIONS:
        for name, source in source_layers(version).items():
            table = TABLES[name]
            layer_fields = fields.setdefault(source.layer.lower(), set())
            layer_fields.update(getattr(hydamo, name).required_columns)
            for column in (table.index, table.group_by, table.order_by):
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

    Only layers that a converter loads are reported. Added and removed fields are
    all listed, because a removed field may have been renamed; property changes
    only for the fields HYDROLIB requires. Everything else is only counted.
    """
    required = _required_fields()
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
        lines.append("Changed properties of fields HYDROLIB requires:")
        lines.extend(property_lines)

    other_layers = 0
    for layer in set(old) | set(new):
        if layer not in required and old.get(layer) != new.get(layer):
            other_layers += 1
    lines.append(
        f"Not shown: {hidden_properties} property change(s) of fields HYDROLIB "
        f"does not require, {other_layers} changed layer(s) no converter loads."
    )
    return lines


def _check_converter(version: str, schema_version: str) -> tuple[list[str], list[str]]:
    """Return the problems and notes of loading a *schema_version* file as *version*.

    A note is something to be aware of that is not a problem, such as an optional
    layer that the schema does not have. The messages do not name the DAMO version,
    so that the messages of two versions can be compared.
    """
    schema = _load_schema(schema_version)
    hydamo = HyDAMO()
    filled_from_globalid = {ext.table: ext.column for ext in FILLED_FROM_GLOBALID}
    problems = []
    notes = []
    for name, source in source_layers(version).items():
        layer = source.layer
        table = f"HyDAMO.{name}"
        spec = TABLES[name]
        fields = schema.get(layer.lower())
        if fields is None:
            if spec.optional_layer:
                # An optional layer is skipped when absent, e.g. Regelmiddel in 2.5.
                notes.append(f"There is no layer {layer}, so {table} stays empty.")
            else:
                problems.append(
                    f"There is no layer {layer}, which the converter needs for {table}."
                )
            continue

        # Columns in the source layer. The schema does not list geometry as a
        # field, but every layer has it.
        available = set(fields) | {"geometry"}

        # The columns the steps rename must be in the source layer.
        for file_name, target in source.renames.items():
            if file_name.lower() not in available:
                problems.append(
                    f"{layer} has no column '{file_name}', which the steps rename "
                    f"to '{target}'."
                )
        if spec.group_by is not None and spec.group_by.lower() not in available:
            problems.append(
                f"{layer} has no column '{spec.group_by}', which loading uses to "
                "group points into lines."
            )
        if spec.order_by is not None and spec.order_by.lower() not in available:
            problems.append(
                f"{layer} has no column '{spec.order_by}', which loading uses to "
                "order the points of a line."
            )

        # Columns of the loaded table: the source columns after the renames of
        # the steps, plus a column filled from globalid.
        loaded = set()
        for column in available:
            loaded.add(source.renames.get(column, column))
        if name in filled_from_globalid and "globalid" in loaded:
            loaded.add(filled_from_globalid[name])

        # The loaded table must have HyDAMO's required columns and the index column.
        required = set(getattr(hydamo, name).required_columns)
        if spec.index is not None:
            required.add(spec.index)
        for column in sorted(required - loaded):
            purpose = " as its index" if column == spec.index else ""
            problems.append(
                f"{layer} has no column '{column}', but {table} requires it{purpose}."
            )
    return problems, notes


def _print_check(heading: str, problems: list[str], notes: list[str]) -> None:
    """Print a check result: the heading, then the problems, then the notes."""
    print(f"{heading}: ", end="")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    for problem in problems:
        print(f"  {problem}")
    for note in notes:
        print(f"  note: {note}")


def diff(old: str, new: str) -> bool:
    """Diff two HyDAMO schemas and show what the change breaks in the old converter.

    Prints the layers and fields added or removed from *old* to *new*, and the
    fields whose type, allowed values, mandatory flag or reference changed. Layers
    that the converters load are marked with ``*``.

    Then prints the problems that the converter of *old* gets on a *new* GeoPackage
    but not on an *old* one: the problems the schema change causes, which the new
    converter has to solve. Problems the converter already has on *old*, such as
    columns HYDROLIB requires that DAMO does not define, are counted but not
    listed; ``check_all`` lists them. Run this on a new schema before adding its
    converter.

    Parameters
    ----------
    old : str
        DAMO version of the current schema and converter, e.g. ``"2.5"``.
    new : str
        DAMO version of the new schema, e.g. ``"2.6"``.

    Returns
    -------
    bool
        ``True`` when the schema change causes a problem for the converter of *old*.

    Raises
    ------
    FileNotFoundError
        If the installed ``hydamo_validation`` has no schema for *old* or *new*.
    ValueError
        If there is no converter for *old*.

    Examples
    --------
    >>> diff("2.5", "2.6")  # doctest: +SKIP
    """
    lines = _diff_schemas(_load_schema(old), _load_schema(new))
    print("\n".join(lines) if lines else "No differences.", end="\n\n")

    # Keep only the problems and notes that the old schema does not have already.
    old_problems, old_notes = _check_converter(old, old)
    new_problems, new_notes = _check_converter(old, new)
    problems = [problem for problem in new_problems if problem not in old_problems]
    notes = [note for note in new_notes if note not in old_notes]
    existing = len(new_problems) - len(problems)

    heading = f"New problems of converter {old} reading a DAMO {new} GeoPackage"
    if existing:
        heading += f" ({existing} existing problem(s) of DAMO {old} not shown)"
    _print_check(heading, problems, notes)
    return bool(problems)


def check_all() -> bool:
    """Check every supported converter against the schema of its own version.

    For each version in ``SUPPORTED_HYDAMO_VERSIONS``, prints the layers,
    column-mapping sources and index/group/order columns of the converter that are
    missing from the schema, and the HyDAMO required columns that the loaded layer
    will not have.

    Returns
    -------
    bool
        ``True`` when the check of any converter finds a problem.

    Raises
    ------
    FileNotFoundError
        If the installed ``hydamo_validation`` has no schema for a supported version.

    Examples
    --------
    >>> check_all()  # doctest: +SKIP
    """
    failed = False
    for version in SUPPORTED_HYDAMO_VERSIONS:
        problems, notes = _check_converter(version, version)
        heading = f"Converter {version} reading a DAMO {version} GeoPackage"
        _print_check(heading, problems, notes)
        failed |= bool(problems)
    return failed


if __name__ == "__main__":
    # A step is two consecutive versions of the installed schemas, e.g. 2.5 and 2.6.
    # diff checks the converter of the older version, so steps without one are
    # skipped.
    for old, new in pairwise(_schema_versions()):
        if old not in SUPPORTED_HYDAMO_VERSIONS:
            print(f"Skipping DAMO {old} -> {new}: there is no converter for {old}.")
            continue
        print(f"===== DAMO {old} -> {new} =====")
        diff(old, new)
        print()

    print("===== Every converter on its own DAMO version =====")
    check_all()
