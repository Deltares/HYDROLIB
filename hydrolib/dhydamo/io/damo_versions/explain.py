"""Report how a layer of a DAMO GeoPackage becomes a HyDAMO table.

The report is built from the data model (`hydrolib.dhydamo.core.data_model`) and the
version steps (`steps`) only, the same data loading uses, so it can't drift from the
loading code: the layers and column names come from `steps.source_layers`, as loading
gets them. It explains the contract, not one file: it doesn't open a GeoPackage, so it
can't say which columns a version's schema has.
"""

from __future__ import annotations

import textwrap
from itertools import pairwise

from hydrolib.dhydamo.core.data_model import (
    TABLES,
    Column,
    DamoRelations,
    Relation,
    Table,
    extensions,
)
from hydrolib.dhydamo.io.damo_versions.steps import (
    SUPPORTED_HYDAMO_VERSIONS,
    SourceLayer,
    source_layers,
)

_INDENT = "  "


def explain_table(table: str, version: str) -> str:
    """Return a text report of how HyDAMO *table* is loaded from a DAMO *version* file.

    The report gives the layer the table is read from and the steps that moved it;
    per declared column, whether the loaded table requires it, its name in *version*,
    the steps that renamed it, and whether the loader fills or derives it; HYDROLIB's
    extensions with their reasons; the values the code compares against; and the
    relations the table takes part in.

    Parameters
    ----------
    table : str
        HyDAMO attribute of the table, e.g. ``"management_device"``.
    version : str
        DAMO version, one of `SUPPORTED_HYDAMO_VERSIONS`, e.g. ``"2.5"``.

    Returns
    -------
    str
        The report, one table, several lines.

    Raises
    ------
    ValueError
        If *version* is not supported, or *table* isn't a HyDAMO table.

    Examples
    --------
    >>> print(explain_table("management_device", "2.5"))  # doctest: +SKIP
    """
    source = source_layers(version)
    if table not in TABLES:
        raise ValueError(
            f'"{table}" is not a HyDAMO table; the tables are {", ".join(TABLES)}.'
        )
    definition = TABLES[table]
    # The table's layer and renames in each version from 2.2 up to *version*
    versions = SUPPORTED_HYDAMO_VERSIONS[: SUPPORTED_HYDAMO_VERSIONS.index(version) + 1]
    history = {v: source_layers(v)[table] for v in versions}
    lines = [f'{table} <- layer "{source[table].layer}" (DAMO {version})']
    lines.append(_INDENT + _layer_history(history))
    if definition.group_by is not None:
        lines.append(
            _INDENT + f"Points grouped into lines by {definition.group_by}, ordered "
            f"by {definition.order_by}."
        )
    lines.append("")
    lines.extend(_column_lines(definition, history, version))
    lines.extend(_extension_lines(table))
    lines.extend(_value_lines(definition))
    lines.extend(_relation_lines(table))
    return "\n".join(lines)


def explain_tables(version: str) -> str:
    """Return the report of `explain_table` for every HyDAMO table, for DAMO *version*.

    Parameters
    ----------
    version : str
        DAMO version, one of `SUPPORTED_HYDAMO_VERSIONS`, e.g. ``"2.6"``.

    Returns
    -------
    str
        The reports, in the order of `data_model.TABLES`, separated by a blank line.

    Raises
    ------
    ValueError
        If *version* is not supported.
    """
    return "\n\n".join(explain_table(table, version) for table in TABLES)


def _layer_history(history: dict[str, SourceLayer]) -> str:
    """Return the line that says which layer the table reads, from DAMO 2.2 on."""
    versions = list(history)
    parts = [f"{history[versions[0]].layer} in {versions[0]}"]
    for old, new in pairwise(versions):
        if history[new].layer != history[old].layer:
            parts.append(f"{history[new].layer} from {new} (step {old} -> {new})")
    return f"Layer: {'; '.join(parts)}."


def _column_lines(
    definition: Table, history: dict[str, SourceLayer], version: str
) -> list[str]:
    """Return the column table: per declared column, its name in *version* and renames.

    The names come from the renames loading applies in each version of *history*;
    a column is renamed in the step after which its name differs.
    """
    # {DAMO 2.2 name: name in the file}, per version
    file_names = {
        v: {original: name for name, original in source.renames.items()}
        for v, source in history.items()
    }
    versions = list(history)
    derived = {column.name for column in definition.derived_columns}
    rows = [
        (
            "column (2.2)",
            "required after loading",
            f"DAMO {version} name",
            "renamed in",
            "note",
        )
    ]
    for column in definition.columns:
        renamed_in = []
        for old, new in pairwise(versions):
            old_name = file_names[old].get(column.name, column.name)
            if file_names[new].get(column.name, column.name) != old_name:
                renamed_in.append(f"{old} -> {new}")
        name = file_names[version].get(column.name, column.name)
        if column.hydrolib_only:
            name = "(HYDROLIB only)"
        elif not column.in_version(version):
            name = f"(from {column.since})"
        required = "yes" if column.name in definition.required_names else "no"
        rows.append(
            (column.name, required, name, ", ".join(renamed_in), _note(column, derived))
        )

    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = []
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=False)]
        lines.append(_INDENT + "  ".join([*cells, row[4]]).rstrip())
    lines.append(_INDENT + "Other columns pass through under their own names.")
    return lines


def _note(column: Column, derived: set[str]) -> str:
    """Return what the loader does with *column* when the file lacks it, if anything."""
    if column.fill_from_globalid:
        return "filled from globalid if absent"
    if column.name in derived:
        return "derived, see below"
    return ""


def _extension_lines(table: str) -> list[str]:
    """Return HYDROLIB's extensions of *table*, each with its reason."""
    lines = []
    for extension in extensions():
        if extension.table != table:
            continue
        heading = f"{extension.kind}: {extension.column}"
        column = _declared_column(table, extension.column)
        if extension.kind == "value rewrite" and column.rewrite is not None:
            old, new = column.rewrite
            heading += f", {old!r} -> {new!r}"
        lines.append("")
        lines.append(_INDENT + heading)
        lines.extend(
            textwrap.wrap(
                extension.reason,
                width=88,
                initial_indent=_INDENT * 2,
                subsequent_indent=_INDENT * 2,
            )
        )
    return lines


def _declared_column(table: str, name: str) -> Column:
    """Return the declared or derived `Column` *name* of *table*."""
    definition = TABLES[table]
    for column in (*definition.columns, *definition.derived_columns):
        if column.name == name:
            return column
    raise KeyError(name)


def _value_lines(definition: Table) -> list[str]:
    """Return the values the code compares the columns of *definition* against."""
    lines = []
    for column in definition.columns:
        if column.values:
            lines.append(_INDENT * 2 + f"{column.name}: {', '.join(column.values)}")
    if not lines:
        return []
    return ["", _INDENT + "Values the code compares against:", *lines]


def _relation_lines(table: str) -> list[str]:
    """Return the DAMO relations *table* takes part in, as child or as parent."""
    lines = []
    for relation in vars(DamoRelations).values():
        if not isinstance(relation, Relation):
            continue
        if table in (relation.child_table, relation.parent_table):
            lines.append(
                _INDENT
                * 2
                + f"{relation.name}: {relation.child_table}.{relation.foreign_key} -> "
                f"{relation.parent_table}.{relation.parent_key}"
            )
    if not lines:
        return []
    return ["", _INDENT + "Relations:", *lines]
