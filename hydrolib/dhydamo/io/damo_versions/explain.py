"""Report how a layer of a DAMO GeoPackage becomes a HyDAMO table.

The report is built from the data model (`hydrolib.dhydamo.core.data_model`) and the
version steps (`steps`) only, the same data loading uses, so it can't drift from the
loading code. It explains the contract, not one file: it doesn't open a GeoPackage.
"""

from __future__ import annotations

import textwrap

from hydrolib.dhydamo.core.data_model import (
    TABLES,
    Column,
    DamoRelations,
    Relation,
    Table,
    extensions,
)
from hydrolib.dhydamo.io.damo_versions.steps import (
    STEPS,
    SUPPORTED_HYDAMO_VERSIONS,
    Step,
    source_layers,
)

_INDENT = "  "


def explain_table(table: str, version: str) -> str:
    """Return a text report of how HyDAMO *table* is loaded from a DAMO *version* file.

    The report gives the layer the table is read from and the steps that moved it;
    per declared column, whether it's required, its name in *version* and the steps
    that renamed it; HYDROLIB's extensions with their reasons; the values the code
    compares against; and the relations the table takes part in.

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
    steps = STEPS[: SUPPORTED_HYDAMO_VERSIONS.index(version)]
    lines = [f'{table} <- layer "{source[table].layer}" (DAMO {version})']
    lines.append(_INDENT + _layer_history(table, steps))
    if definition.group_by is not None:
        lines.append(
            _INDENT + f"Points grouped into lines by {definition.group_by}, ordered "
            f"by {definition.order_by}."
        )
    lines.append("")
    lines.extend(_column_lines(table, definition, steps, version))
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


def _layer_history(table: str, steps: tuple[Step, ...]) -> str:
    """Return the line that says which layer *table* reads, from DAMO 2.2 on."""
    parts = [f"{TABLES[table].layer} in 2.2"]
    for step in steps:
        if table in step.layers:
            parts.append(
                f"{step.layers[table]} from {step.new} (step {step.old} -> {step.new})"
            )
    return f"Layer: {'; '.join(parts)}."


def _column_lines(
    table: str, definition: Table, steps: tuple[Step, ...], version: str
) -> list[str]:
    """Return the column table: per declared column, its name in *version* and renames."""
    rows = [("column (2.2)", "required", f"DAMO {version} name", "renamed in")]
    for column in definition.columns:
        name = column.name
        renamed_in = []
        for step in steps:
            new = step.columns.get(table, {}).get(name)
            if new is not None:
                name = new
                renamed_in.append(f"{step.old} -> {step.new}")
        if column.hydrolib_only:
            name = "(HYDROLIB only)"
        required = "yes" if column.name in definition.required_names else "no"
        rows.append((column.name, required, name, ", ".join(renamed_in)))

    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=False)]
        lines.append(_INDENT + "  ".join([*cells, row[3]]).rstrip())
    lines.append(_INDENT + "Other columns pass through under their own names.")
    return lines


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
