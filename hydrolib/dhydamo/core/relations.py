from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    from hydrolib.dhydamo.core.hydamo import HyDAMO


@dataclass(frozen=True, slots=True)
class Relation:
    name: str
    child_table: str
    foreign_key: str
    parent_table: str
    parent_key: str = "globalid"

    def parents_of(
        self,
        child_rows: pd.DataFrame,
        parent_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return parent rows referenced by one or more child rows.

        The ``of`` in ``parents_of`` refers to the first argument,
        ``child_rows``. This relation supplies the traversal metadata: values
        from its ``foreign_key`` on ``child_rows`` are matched against its
        ``parent_key`` on ``parent_table``.

        Parameters
        ----------
        child_rows : pandas.DataFrame
            One or more rows from this relation's ``child_table``. The
            relation's ``foreign_key`` must be present as a column.
        parent_table : pandas.DataFrame
            Candidate rows from this relation's ``parent_table``. The
            relation's ``parent_key`` must be present as a column.

        Returns
        -------
        pandas.DataFrame
            Rows from ``parent_table`` whose parent key is referenced by
            ``child_rows``. The original index, column order, and dataframe
            type are preserved. Null foreign-key values never match.

        Raises
        ------
        KeyError
            If ``foreign_key`` is missing from ``child_rows`` or ``parent_key``
            is missing from ``parent_table``.
        Notes
        -----
        This method does not enforce relation cardinality. It can return zero,
        one, or multiple parent rows; the caller decides how those cases are
        handled.
        """

        foreign_keys = child_rows[self.foreign_key].dropna()
        return parent_table.loc[parent_table[self.parent_key].isin(foreign_keys)]

    def children_of(
        self,
        parent_rows: pd.DataFrame,
        child_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return child rows that reference one or more parent rows.

        The ``of`` in ``children_of`` refers to the first argument,
        ``parent_rows``. This relation supplies the traversal metadata: values
        from its ``parent_key`` on ``parent_rows`` are matched against its
        ``foreign_key`` on ``child_table``.

        Parameters
        ----------
        parent_rows : pandas.DataFrame
            One or more rows from this relation's ``parent_table``. The
            relation's ``parent_key`` must be present as a column.
        child_table : pandas.DataFrame
            Candidate rows from this relation's ``child_table``. The
            relation's ``foreign_key`` must be present as a column.

        Returns
        -------
        pandas.DataFrame
            Rows from ``child_table`` whose foreign key references
            ``parent_rows``. The original index, column order, and dataframe
            type are preserved. Null parent-key values never match.

        Raises
        ------
        KeyError
            If ``parent_key`` is missing from ``parent_rows`` or
            ``foreign_key`` is missing from ``child_table``.
        Notes
        -----
        This method does not enforce relation cardinality. It can return zero,
        one, or multiple child rows; the caller decides how those cases are
        handled.
        """

        parent_keys = parent_rows[self.parent_key].dropna()
        return child_table.loc[child_table[self.foreign_key].isin(parent_keys)]


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
    paths in :class:`CascadeRelations`. Each configured path starts from the
    original ``source_index`` and is followed independently. Rows reached
    through the same table are merged by index, and this function never mutates
    a dataframe.

    Parameters
    ----------
    hydamo : HyDAMO
        Object containing the source and related table dataframes.
    source_table : str
        Name of the source dataframe, such as ``"profile"`` or
        ``"pumpstations"``.  The corresponding cascade configuration is
        looked up on :class:`CascadeRelations` using the upper-case table
        name.
    source_index : pandas.Index
        Index labels identifying the source rows from which to build the
        plan.

    Returns
    -------
    dict[str, pandas.Index]
        Mapping from each related table name to the unique index labels that
        would be affected.  The source table itself is not included.  An
        unconfigured source table returns an empty dictionary.

    """

    paths = getattr(CascadeRelations, source_table.upper(), ())
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


@dataclass(frozen=True, slots=True)
class RelationsHyDAMO:
    PROFILE_LINE = Relation(
        name="profile_line",
        child_table="profile",
        foreign_key="profiellijnid",
        parent_table="profile_line",
    )
    PROFILE_ROUGHNESS = Relation(
        name="profile_roughness",
        child_table="profile_roughness",
        foreign_key="profielpuntid",
        parent_table="profile",
    )
    PROFILE_LINE_GROUP = Relation(
        name="profile_line_group",
        child_table="profile_line",
        foreign_key="profielgroepid",
        parent_table="profile_group",
    )
    PROFILE_GROUP_BRANCH = Relation(
        name="profile_group_branch",
        child_table="profile_group",
        foreign_key="hydroobjectid",
        parent_table="branches",
    )
    PROFILE_GROUP_WEIR = Relation(
        name="profile_group_weir",
        child_table="profile_group",
        foreign_key="stuwid",
        parent_table="weirs",
    )
    PROFILE_GROUP_BRIDGE = Relation(
        name="profile_group_bridge",
        child_table="profile_group",
        foreign_key="brugid",
        parent_table="bridges",
    )
    PARAM_PROFILE_BRANCH = Relation(
        name="param_profile_branch",
        child_table="param_profile",
        foreign_key="hydroobjectid",
        parent_table="branches",
    )
    PARAM_PROFILE_VALUES = Relation(
        name="param_profile_values",
        child_table="param_profile_values",
        foreign_key="normgeparamprofielid",
        parent_table="param_profile",
        parent_key="normgeparamprofielid",
    )

    OPENING_WEIR = Relation(
        name="opening_weir",
        child_table="opening",
        foreign_key="stuwid",
        parent_table="weirs",
    )
    OPENING_MANAGEMENT_DEVICE = Relation(
        name="opening_management_device",
        child_table="opening",
        foreign_key="regelmiddelid",
        parent_table="management_device",
    )
    MANAGEMENT_DEVICE_OPENING = Relation(
        name="management_device_opening",
        child_table="management_device",
        foreign_key="kunstwerkopeningid",
        parent_table="opening",
    )
    MANAGEMENT_DEVICE_CULVERT = Relation(
        name="management_device_culvert",
        child_table="management_device",
        foreign_key="duikersifonhevelid",
        parent_table="culverts",
    )
    MANAGEMENT_DEVICE_PUMPSTATION = Relation(
        name="management_device_pumpstation",
        child_table="management_device",
        foreign_key="gemaalid",
        parent_table="pumpstations",
    )
    PUMP_PUMPSTATION = Relation(
        name="pump_pumpstation",
        child_table="pumps",
        foreign_key="gemaalid",
        parent_table="pumpstations",
    )
    MANAGEMENT_PUMP = Relation(
        name="management_pump",
        child_table="management",
        foreign_key="pompid",
        parent_table="pumps",
    )
    MANAGEMENT_DEVICE = Relation(
        name="management_device",
        child_table="management",
        foreign_key="regelmiddelid",
        parent_table="management_device",
    )
    MANAGEMENT_BOUNDARY_CONDITION = Relation(
        name="management_boundary_condition",
        child_table="management",
        foreign_key="hydrologischerandvoorwaardeid",
        parent_table="boundary_conditions",
    )


@dataclass(frozen=True, slots=True)
class RelationsRR:
    CATCHMENT_LATERAL = Relation(
        name="catchment_lateral",
        child_table="catchments",
        foreign_key="lateraleknoopid",
        parent_table="laterals",
    )
    OVERFLOW_SEWER_AREA = Relation(
        name="overflow_sewer_area",
        child_table="overflows",
        foreign_key="codegerelateerdobject",
        parent_table="sewer_areas",
        parent_key="code",
    )
    GREENHOUSE_LATERAL_AREA = Relation(
        name="greenhouse_lateral_area",
        child_table="greenhouse_laterals",
        foreign_key="codegerelateerdobject",
        parent_table="greenhouse_areas",
        parent_key="code",
    )


CascadePaths: TypeAlias = tuple[tuple[Relation, ...], ...]


@dataclass(frozen=True, slots=True)
class CascadeRelations:
    PROFILE: ClassVar[CascadePaths] = (
        (RelationsHyDAMO.PROFILE_ROUGHNESS,),
        (RelationsHyDAMO.PROFILE_LINE, RelationsHyDAMO.PROFILE_LINE_GROUP),
    )
    PROFILE_LINE: ClassVar[CascadePaths] = (
        (RelationsHyDAMO.PROFILE_LINE_GROUP,),
        (RelationsHyDAMO.PROFILE_LINE, RelationsHyDAMO.PROFILE_ROUGHNESS),
    )
    WEIRS: ClassVar[CascadePaths] = (
        (RelationsHyDAMO.OPENING_WEIR, RelationsHyDAMO.MANAGEMENT_DEVICE_OPENING),
    )
    PUMPSTATIONS: ClassVar[CascadePaths] = (
        (RelationsHyDAMO.PUMP_PUMPSTATION, RelationsHyDAMO.MANAGEMENT_PUMP),
    )
    CULVERTS: ClassVar[CascadePaths] = ((RelationsHyDAMO.MANAGEMENT_DEVICE_CULVERT,),)
    CATCHMENTS: ClassVar[CascadePaths] = ((RelationsRR.CATCHMENT_LATERAL,),)
    LATERALS: ClassVar[CascadePaths] = ((RelationsRR.CATCHMENT_LATERAL,),)
    OVERFLOWS: ClassVar[CascadePaths] = ((RelationsRR.OVERFLOW_SEWER_AREA,),)
    SEWER_AREAS: ClassVar[CascadePaths] = ((RelationsRR.OVERFLOW_SEWER_AREA,),)
    GREENHOUSE_AREAS: ClassVar[CascadePaths] = ((RelationsRR.GREENHOUSE_LATERAL_AREA,),)
    GREENHOUSE_LATERALS: ClassVar[CascadePaths] = (
        (RelationsRR.GREENHOUSE_LATERAL_AREA,),
    )
