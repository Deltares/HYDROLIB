"""Follow the relations between the HyDAMO tables, and plan cascading drops.

The relations themselves, and the cascade paths, are in the data model
(`hydrolib.dhydamo.core.data_model`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from hydrolib.dhydamo.core.data_model import (
    CASCADE_PATHS,
    OWN_TABLES,
    TABLES,
    Relation,
)

if TYPE_CHECKING:
    from hydrolib.dhydamo.core.hydamo import HyDAMO


def follow_relation(
    hydamo: HyDAMO,
    relation: Relation,
    from_table: str,
    rows: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Return rows related to ``rows`` through a direct relation.

    This helper exists because a relation is defined once in foreign-key
    direction (child to parent), but callers may start from either endpoint.
    It selects the correct source and target columns for that direction, so
    callers do not need duplicate forward and reverse relation definitions.

    Matching is performed on relation key columns, not on dataframe indices.
    For example, following ``profile_line`` from ``profile`` matches
    ``profile.profiellijnid`` to ``profile_line.globalid``; following it in
    reverse matches ``profile_line.globalid`` to ``profile.profiellijnid``.

    Parameters
    ----------
    hydamo : HyDAMO
        Object containing the dataframe for the other relation endpoint.
    relation : Relation
        Direct relation to follow.  Its ``foreign_key`` belongs to
        ``child_table`` and its ``parent_key`` belongs to ``parent_table``.
    from_table : str
        Name of the endpoint represented by ``rows``.  It must equal either
        ``relation.child_table`` or ``relation.parent_table``.
    rows : pandas.DataFrame
        Selected rows from ``from_table``.  The relevant relation key must be
        present as a column.  An empty dataframe with that column returns an
        empty dataframe for the other endpoint.

    Returns
    -------
    tuple[str, pandas.DataFrame]
        The name of the other endpoint table and the rows whose relation key
        matches a key value in ``rows``.  The target dataframe's index and
        columns are preserved.

    Raises
    ------
    ValueError
        If ``from_table`` is not an endpoint of ``relation``.
    KeyError
        If the required source or target relation key is missing.
    """
    if from_table == relation.child_table:
        target_table = relation.parent_table
        target_rows = relation.parents_of(rows, getattr(hydamo, target_table))
    elif from_table == relation.parent_table:
        target_table = relation.child_table
        target_rows = relation.children_of(rows, getattr(hydamo, target_table))
    else:
        raise ValueError(
            f"Table {from_table!r} is not an endpoint of relation "
            f"{relation.name!r}; expected {relation.child_table!r} or "
            f"{relation.parent_table!r}"
        )

    return target_table, target_rows


def build_cascade_plan(
    hydamo: HyDAMO,
    source_table: str,
    source_index: pd.Index,
) -> dict[str, pd.Index]:
    """Return the configured, bounded cascade plan for selected rows.

    This helper exists to separate deciding *what may be deleted* from the
    subsequent dataframe mutation.  Cascade policy is expressed as explicit
    paths in the data model's ``CASCADE_PATHS``. Each configured path starts from the
    original ``source_index`` and is followed independently. Rows reached
    through the same table are merged by index, and this function never mutates
    a dataframe.

    Parameters
    ----------
    hydamo : HyDAMO
        Object containing the source and related table dataframes.
    source_table : str
        Name of the source dataframe, such as ``"profile"`` or
        ``"pumpstations"``.  Its cascade paths are looked up in
        ``CASCADE_PATHS``.
    source_index : pandas.Index
        Index labels identifying the source rows from which to build the
        plan.

    Returns
    -------
    dict[str, pandas.Index]
        Mapping from each related table name to the unique index labels that
        would be affected.  The source table itself is not included.  A
        table without cascade paths returns an empty dictionary.

    Raises
    ------
    ValueError
        If ``source_table`` is not a HyDAMO table (``TABLES`` or ``OWN_TABLES``),
        e.g. a typo: read as a table without cascade paths, it would let the
        caller drop the source rows and leave their related rows behind.
    """
    if source_table not in TABLES and source_table not in OWN_TABLES:
        raise ValueError(
            f"{source_table!r} is not a HyDAMO table, so its cascade paths are "
            "unknown. Use the table's attribute name, e.g. 'weirs'."
        )
    paths = CASCADE_PATHS.get(source_table, ())
    if not paths:
        return {}

    source = getattr(hydamo, source_table)
    current_source = source.loc[source_index]
    planned: dict[str, pd.Index] = {}

    for path in paths:
        current_table = source_table
        current_rows = current_source

        for relation in path:
            current_table, current_rows = follow_relation(
                hydamo, relation, current_table, current_rows
            )
            if current_table in planned:
                planned[current_table] = planned[current_table].union(
                    current_rows.index
                )
            else:
                planned[current_table] = current_rows.index.copy()

    return planned
