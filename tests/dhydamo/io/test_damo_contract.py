"""The contract test: HYDROLIB's data model and the DAMO version steps, against DAMO.

Checks against the HyDAMO JSON schemas of the installed ``hydamo_validation``:

- each step's renames: the old name in the older schema, the new one in the newer;
- per supported version, that every declared column reaches DAMO 2.2 through the
  steps, unless it is derived, filled from ``globalid`` or HYDROLIB's own;
- per supported version, that DAMO allows every declared value;
- that no extension takes a DAMO 2.2 name with another meaning.

The check is static, so it runs only where ``hydamo_validation`` is installed (Python
3.12) and is skipped elsewhere.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hydrolib.dhydamo.core.data_model import TABLES, Column, Table
from hydrolib.dhydamo.io.damo_versions import (
    SUPPORTED_HYDAMO_VERSIONS,
    source_layers,
    steps,
)

hydamo_validation = pytest.importorskip(
    "hydamo_validation", reason="hydamo-validation is only available for Python 3.12"
)

pytestmark = pytest.mark.requires_hydamo_validation

# {layer: {field: properties}}, layer names in lower case
Schema = dict[str, dict[str, dict[str, Any]]]

STEP_24_25 = next(step for step in steps.STEPS if step.new == "2.5")


@cache
def _schema(version: str) -> Schema:
    """Return the fields of each layer of the DAMO *version* schema."""
    folder = Path(hydamo_validation.__file__).parent / "schemas" / "hydamo"
    path = folder / f"HyDAMO_{version}.json"
    definitions = json.loads(path.read_text(encoding="utf-8"))["definitions"]
    schema = {}
    for layer, definition in definitions.items():
        # The schema doesn't list the geometry as a field.
        schema[layer.lower()] = {**definition["properties"], "geometry": {}}
    return schema


def _layers(version: str) -> dict[str, str]:
    """Return the layer of each HyDAMO table in DAMO *version*."""
    return {name: source.layer for name, source in source_layers(version).items()}


def _step_problems(step: steps.Step) -> list[str]:
    """Return the renames of *step* that its two schemas contradict."""
    old_schema, new_schema = _schema(step.old), _schema(step.new)
    old_layers = _layers(step.old)
    new_layers = {**old_layers, **step.layers}
    problems = []
    for name, layer in step.layers.items():
        if layer.lower() not in new_schema:
            problems.append(f"{name} moves to layer {layer}, which {step.new} lacks")
    for name, columns in step.columns.items():
        old_fields = old_schema.get(old_layers[name].lower(), {})
        new_fields = new_schema.get(new_layers[name].lower(), {})
        for old, new in columns.items():
            if old not in old_fields:
                problems.append(f"{name}: {old_layers[name]} {step.old} has no {old}")
            if new not in new_fields:
                problems.append(f"{name}: {new_layers[name]} {step.new} has no {new}")
    return problems


def _column_problems(version: str, tables: dict[str, Table] = TABLES) -> list[str]:
    """Return the declared columns a DAMO *version* file doesn't supply.

    A column is supplied when its layer has it under the name the steps rename to
    it. A derived column, a column filled from ``globalid`` and a HYDROLIB-only
    column needn't be supplied. The columns that group and order points into lines
    must be supplied too, declared or not. A column DAMO added (`Column.since`) is
    checked from that version on, and must be absent before it.
    """
    schema = _schema(version)
    problems = []
    for name, source in source_layers(version).items():
        table = tables[name]
        fields = schema.get(source.layer.lower())
        if fields is None:
            problems.append(f"{name}: DAMO {version} has no layer {source.layer}")
            continue
        supplied = {source.renames.get(field, field) for field in fields}
        derived = {column.name for column in table.derived_columns}
        for column in table.columns:
            if not column.in_version(version):
                if column.name in supplied:
                    problems.append(
                        f"{name} ({source.layer}) has {column.name} before "
                        f"{column.since}"
                    )
                continue
            if (
                column.name in supplied
                or column.name in derived
                or column.fill_from_globalid
                or column.hydrolib_only
            ):
                continue
            problems.append(f"{name} ({source.layer}) has no column {column.name}")
        for points_column in (table.group_by, table.order_by):
            if points_column is not None and points_column not in supplied:
                problems.append(
                    f"{name} ({source.layer}) has no column {points_column}"
                )
    return problems


def _value_problems(version: str, tables: dict[str, Table] = TABLES) -> list[str]:
    """Return the declared values that DAMO *version* doesn't allow.

    The allowed values are compared as the loader leaves them: a value the data
    model rewrites counts as the value it is rewritten to.
    """
    schema = _schema(version)
    problems = []
    for name, source in source_layers(version).items():
        fields = schema.get(source.layer.lower(), {})
        file_name = {target: field for field, target in source.renames.items()}
        for column in tables[name].columns:
            field = fields.get(file_name.get(column.name, column.name))
            if not column.values or field is None:
                # an absent column is a column problem
                continue
            rewrites = dict([column.rewrite]) if column.rewrite is not None else {}
            allowed = {rewrites.get(value, value) for value in field.get("enum", [])}
            for value in column.values:
                if value not in allowed:
                    problems.append(f"{name}.{column.name}: {value!r}")
    return problems


def _extension_problems(tables: dict[str, Table] = TABLES) -> list[str]:
    """Return the extensions that take a DAMO 2.2 name of their layer.

    A HYDROLIB-only column must not be a DAMO column in any supported version. A
    derived column with a DAMO 2.2 name must also be read, so that its reason says
    how it combines with DAMO's. An identity only tightens DAMO's rules for the
    column, and a rewrite or fill changes values of a column that is read, so
    neither redefines it.
    """
    problems = []
    for name, table in tables.items():
        read = {column.name for column in table.columns}
        for version in SUPPORTED_HYDAMO_VERSIONS:
            source = source_layers(version)[name]
            fields = _schema(version).get(source.layer.lower(), {})
            supplied = {source.renames.get(field, field) for field in fields}
            for column in table.columns:
                if column.hydrolib_only and column.name in supplied:
                    problems.append(
                        f"{name}.{column.name} is HYDROLIB-only, but DAMO {version} "
                        "defines it"
                    )
        fields_22 = _schema("2.2").get(table.layer.lower(), {})
        for column in table.derived_columns:
            if column.name in fields_22 and column.name not in read:
                problems.append(
                    f"{name}.{column.name} is derived under a DAMO 2.2 name, but not "
                    "read"
                )
    return sorted(set(problems))


def _with_column(
    name: str, column: Column, *, derived: bool = False
) -> dict[str, Table]:
    """Return `TABLES` with *column* in table *name*.

    As a derived column, or as an optional column that replaces its namesake.
    """
    table = TABLES[name]
    if derived:
        changed = dataclasses.replace(
            table, derived_columns=(*table.derived_columns, column)
        )
    else:
        others = tuple(c for c in table.columns if c.name != column.name)
        optional = tuple(c for c in others if c.name in table.optional_names)
        changed = dataclasses.replace(table, optional_columns=(*optional, column))
    return {**TABLES, name: changed}


@pytest.mark.parametrize("step", steps.STEPS, ids=lambda s: f"{s.old}-{s.new}")
def test_each_rename_of_a_step_is_in_its_two_schemas(step: steps.Step) -> None:
    assert _step_problems(step) == []


def test_step_check_fails_on_a_rename_damo_did_not_make() -> None:
    wrong = dataclasses.replace(
        STEP_24_25, columns={"culverts": {"vormkoker": "vormkokerr"}}
    )

    assert _step_problems(wrong) == ["culverts: DuikerSifonHevel 2.5 has no vormkokerr"]


def test_step_check_fails_on_a_layer_damo_does_not_have() -> None:
    wrong = dataclasses.replace(STEP_24_25, layers={"management_device": "Sluis2"})

    assert "management_device moves to layer Sluis2, which 2.5 lacks" in (
        _step_problems(wrong)
    )


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_every_declared_column_reaches_damo_22(version: str) -> None:
    assert _column_problems(version) == []


def test_column_check_fails_on_a_declared_column_without_its_rename() -> None:
    # DAMO 2.4 renamed soortstuw to typestuw, and no step says so.
    tables = _with_column("weirs", Column("soortstuw", "a test"))

    assert _column_problems("2.3", tables) == []
    assert _column_problems("2.4", tables) == ["weirs (Stuw) has no column soortstuw"]


def test_column_check_checks_a_column_only_from_the_version_that_added_it() -> None:
    # Stuw has typestuw from DAMO 2.4 (a rename DAMO made, here declared as added).
    tables = _with_column("weirs", Column("typestuw", "a test", since="2.4"))
    too_early = _with_column("weirs", Column("typestuw", "a test", since="2.5"))

    assert _column_problems("2.3", tables) == []
    assert _column_problems("2.4", tables) == []
    assert _column_problems("2.4", too_early) == [
        "weirs (Stuw) has typestuw before 2.5"
    ]


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_damo_allows_every_declared_value(version: str) -> None:
    assert _value_problems(version) == []


def test_value_check_fails_on_a_value_damo_does_not_allow() -> None:
    tables = _with_column(
        "management",
        Column("typecontroller", "a test", values=("PID", "fuzzy")),
    )

    assert _value_problems("2.2", tables) == ["management.typecontroller: 'fuzzy'"]


def test_value_check_allows_the_value_a_rewrite_makes() -> None:
    # Every schema allows only the typo, which the data model rewrites.
    without_rewrite = _with_column(
        "management",
        Column("stuurvariabele", "a test", values=("bovenkant afsluitmiddel",)),
    )

    assert _value_problems("2.5") == []
    assert _value_problems("2.5", without_rewrite) == [
        "management.stuurvariabele: 'bovenkant afsluitmiddel'"
    ]


def test_no_extension_takes_a_damo_name_with_another_meaning() -> None:
    assert _extension_problems() == []


@pytest.mark.parametrize(
    ("tables", "problem"),
    [
        (
            _with_column("weirs", Column("naam", "a test", hydrolib_only=True)),
            "weirs.naam is HYDROLIB-only, but DAMO 2.2 defines it",
        ),
        (
            _with_column("weirs", Column("kruinbreedte", "a test"), derived=True),
            "weirs.kruinbreedte is derived under a DAMO 2.2 name, but not read",
        ),
    ],
    ids=["hydrolib-only", "derived"],
)
def test_extension_check_fails_on_an_extension_that_clashes_with_damo(
    tables: dict[str, Table], problem: str
) -> None:
    assert problem in _extension_problems(tables)


@pytest.mark.parametrize("field", ["group_by", "order_by"])
def test_column_check_fails_on_a_group_or_order_column_damo_lacks(field: str) -> None:
    # The loader needs them to make lines of the points, though they needn't be
    # declared columns.
    if field == "group_by":
        profile = dataclasses.replace(TABLES["profile"], group_by="volgorde")
    else:
        profile = dataclasses.replace(TABLES["profile"], order_by="volgorde")
    tables = {**TABLES, "profile": profile}

    assert _column_problems("2.6", tables) == [
        "profile (ProfielPunt) has no column volgorde"
    ]


def test_value_check_fails_on_a_value_the_loader_rewrites_away() -> None:
    # The code never sees the typo: loading rewrites it.
    tables = _with_column(
        "management",
        Column(
            "stuurvariabele",
            "a test",
            rewrite=("bovenkant afsluitmiddel,", "bovenkant afsluitmiddel"),
            values=("bovenkant afsluitmiddel,",),
        ),
    )

    assert _value_problems("2.5", tables) == [
        "management.stuurvariabele: 'bovenkant afsluitmiddel,'"
    ]
