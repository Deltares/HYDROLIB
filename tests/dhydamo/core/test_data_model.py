"""HYDROLIB's data model: its tables, their columns, extensions and relations."""

from __future__ import annotations

import pytest
from hydrolib.dhydamo.core.data_model import (
    CASCADE_PATHS,
    OWN_TABLES,
    TABLES,
    Column,
    DamoRelations,
    OwnRelations,
    Relation,
    Table,
    extensions,
)


def test_a_table_gives_the_names_of_its_columns_whatever_their_declaration() -> None:
    table = Table(
        "Stuw",
        geotype=None,
        required_columns=("code", Column("afvoercoefficient", "A reason.")),
        optional_columns=(Column("naam", "Another reason."), "globalid"),
    )

    assert table.required_names == ("code", "afvoercoefficient")
    assert table.optional_names == ("naam", "globalid")


def test_extensions_lists_every_extension_with_its_kind_and_reason() -> None:
    listed = {(ext.table, ext.column, ext.kind) for ext in extensions()}

    identities = {
        (table, "code", "identity")
        for table in (
            "branches",
            "profile",
            "weirs",
            "bridges",
            "culverts",
            "pumpstations",
            "pumps",
            "management",
            "boundary_conditions",
            "catchments",
            "laterals",
        )
    }
    assert listed == identities | {
        ("management_device", "maximaaldebiet", "HYDROLIB-only column"),
        ("management", "stuurvariabele", "value rewrite"),
        ("management", "code", "filled from globalid"),
        ("weirs", "laagstedoorstroomhoogte", "derived column"),
        ("profile", "code", "derived column"),
        ("catchments", "boundary_node", "derived column"),
    }
    assert all(ext.reason for ext in extensions())


def test_the_steering_variable_rewrite_is_declared_on_its_column() -> None:
    (stuurvariabele,) = (
        column
        for column in TABLES["management"].columns
        if column.name == "stuurvariabele"
    )

    assert stuurvariabele.rewrite == (
        "bovenkant afsluitmiddel,",
        "bovenkant afsluitmiddel",
    )


def test_a_table_fills_at_most_one_column_from_globalid() -> None:
    # The loader fills one column per table, so a second would be reported but not
    # filled.
    with pytest.raises(ValueError, match="more than one column"):
        Table(
            "Sturing",
            geotype=None,
            required_columns=(
                Column("code", "A reason.", fill_from_globalid=True),
                Column("naam", "Another reason.", fill_from_globalid=True),
            ),
        )


def test_a_table_names_its_column_filled_from_globalid() -> None:
    assert TABLES["management"].column_filled_from_globalid == "code"
    assert TABLES["weirs"].column_filled_from_globalid is None


def test_own_tables_declare_the_optional_rr_inputs_they_read() -> None:
    assert OWN_TABLES["sewer_areas"].optional_names == (
        "riool_berging_mm",
        "riool_poc_m3s",
    )
    assert OWN_TABLES["greenhouse_areas"].optional_names == (
        "roof_storage_mm",
        "basin_storage_class",
    )


def _relations() -> list[Relation]:
    relations = []
    for namespace in (DamoRelations, OwnRelations):
        for value in vars(namespace).values():
            if isinstance(value, Relation):
                relations.append(value)
    return relations


def _unknown_tables(relations: list[Relation], cascade_tables: list[str]) -> set[str]:
    """Return the tables named by *relations* or *cascade_tables* that don't exist."""
    named = set(cascade_tables)
    for relation in relations:
        named.update((relation.child_table, relation.parent_table))
    return named - set(TABLES) - set(OWN_TABLES)


def test_relations_and_cascade_paths_name_only_existing_tables() -> None:
    cascade_relations: list[Relation] = []
    for paths in CASCADE_PATHS.values():
        for path in paths:
            cascade_relations.extend(path)

    # both namespaces are scanned
    assert DamoRelations.CATCHMENT_LATERAL in _relations()
    assert OwnRelations.OVERFLOW_SEWER_AREA in _relations()
    assert (
        _unknown_tables(_relations() + cascade_relations, list(CASCADE_PATHS)) == set()
    )


@pytest.mark.parametrize(
    ("relation", "cascade_table"),
    [
        (Relation("broken", "weir", "stuwid", "weirs"), "weirs"),
        (DamoRelations.OPENING_WEIR, "weir"),
    ],
)
def test_relations_check_fails_on_a_table_that_does_not_exist(
    relation: Relation, cascade_table: str
) -> None:
    assert _unknown_tables([relation], [cascade_table]) == {"weir"}
