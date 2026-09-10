from types import SimpleNamespace

import pandas as pd
import pytest
from hydrolib.dhydamo.core.relations import (
    RelationsHyDAMO,
    build_cascade_plan,
    follow_relation,
)


def _profile_hydamo():
    return SimpleNamespace(
        profile=pd.DataFrame(
            {
                "globalid": ["profile-1", "profile-2"],
                "profiellijnid": ["line-1", "line-2"],
            },
            index=["p1", "p2"],
        ),
        profile_roughness=pd.DataFrame(
            {"profielpuntid": ["profile-1", "profile-2"]},
            index=["roughness-1", "roughness-2"],
        ),
        profile_line=pd.DataFrame(
            {
                "globalid": ["line-1", "line-2"],
                "profielgroepid": ["group-1", "group-2"],
            },
            index=["l1", "l2"],
        ),
        profile_group=pd.DataFrame(
            {"globalid": ["group-1", "group-2"]},
            index=["g1", "g2"],
        ),
    )


def _structure_hydamo():
    return SimpleNamespace(
        weirs=pd.DataFrame({"globalid": ["weir-1", "weir-2"]}, index=["w1", "w2"]),
        opening=pd.DataFrame(
            {
                "globalid": ["opening-1", "opening-2"],
                "stuwid": ["weir-1", "weir-2"],
            },
            index=["o1", "o2"],
        ),
        management_device=pd.DataFrame(
            {
                "kunstwerkopeningid": ["opening-1", None, None],
                "duikersifonhevelid": [None, "culvert-1", None],
            },
            index=["d1", "d2", "d3"],
        ),
        culverts=pd.DataFrame(
            {"globalid": ["culvert-1", "culvert-2"]}, index=["c1", "c2"]
        ),
        pumpstations=pd.DataFrame(
            {"globalid": ["station-1", "station-2"]}, index=["ps1", "ps2"]
        ),
        pumps=pd.DataFrame(
            {
                "globalid": ["pump-1", "pump-2"],
                "gemaalid": ["station-1", "station-2"],
            },
            index=["pump1", "pump2"],
        ),
        management=pd.DataFrame({"pompid": ["pump-1", "pump-2"]}, index=["m1", "m2"]),
    )


def _rr_hydamo():
    return SimpleNamespace(
        catchments=pd.DataFrame(
            {"lateraleknoopid": ["lateral-1", "lateral-2"]},
            index=["catchment1", "catchment2"],
        ),
        laterals=pd.DataFrame(
            {"globalid": ["lateral-1", "lateral-2"]},
            index=["lateral1", "lateral2"],
        ),
        overflows=pd.DataFrame(
            {"codegerelateerdobject": ["sewer-1", "sewer-2"]},
            index=["overflow1", "overflow2"],
        ),
        sewer_areas=pd.DataFrame(
            {"code": ["sewer-1", "sewer-2"]}, index=["sewer1", "sewer2"]
        ),
        greenhouse_laterals=pd.DataFrame(
            {"codegerelateerdobject": ["greenhouse-1", "greenhouse-2"]},
            index=["greenhouse_lateral1", "greenhouse_lateral2"],
        ),
        greenhouse_areas=pd.DataFrame(
            {"code": ["greenhouse-1", "greenhouse-2"]},
            index=["greenhouse1", "greenhouse2"],
        ),
    )


def test_parents_of_selects_rows_referenced_by_children():
    hydamo = _profile_hydamo()

    parents = RelationsHyDAMO.PROFILE_LINE.parents_of(
        hydamo.profile.loc[["p1"]], hydamo.profile_line
    )

    assert parents.index.tolist() == ["l1"]


def test_children_of_selects_rows_referencing_parents():
    hydamo = _profile_hydamo()

    children = RelationsHyDAMO.PROFILE_LINE.children_of(
        hydamo.profile_line.loc[["l2"]], hydamo.profile
    )

    assert children.index.tolist() == ["p2"]


def test_relation_methods_ignore_null_keys():
    relation = RelationsHyDAMO.PROFILE_LINE
    profiles = pd.DataFrame({"profiellijnid": [None]}, index=["profile-without-line"])
    profile_lines = pd.DataFrame({"globalid": [None]}, index=["line-without-id"])

    assert relation.parents_of(profiles, profile_lines).empty
    assert relation.children_of(profile_lines, profiles).empty


def test_relation_methods_require_source_and_target_keys():
    relation = RelationsHyDAMO.PROFILE_LINE
    hydamo = _profile_hydamo()

    with pytest.raises(KeyError, match="profiellijnid"):
        relation.parents_of(pd.DataFrame(index=["p1"]), hydamo.profile_line)
    with pytest.raises(KeyError, match="globalid"):
        relation.parents_of(
            hydamo.profile.loc[["p1"]],
            hydamo.profile_line.drop(columns="globalid"),
        )
    with pytest.raises(KeyError, match="globalid"):
        relation.children_of(pd.DataFrame(index=["l1"]), hydamo.profile)
    with pytest.raises(KeyError, match="profiellijnid"):
        relation.children_of(
            hydamo.profile_line.loc[["l1"]],
            hydamo.profile.drop(columns="profiellijnid"),
        )


def test_follow_relation_dispatches_in_both_directions():
    hydamo = _profile_hydamo()
    relation = RelationsHyDAMO.PROFILE_LINE

    target_table, target_rows = follow_relation(
        hydamo, relation, "profile", hydamo.profile.loc[["p1"]]
    )
    assert target_table == "profile_line"
    assert target_rows.index.tolist() == ["l1"]

    target_table, target_rows = follow_relation(
        hydamo, relation, "profile_line", hydamo.profile_line.loc[["l2"]]
    )
    assert target_table == "profile"
    assert target_rows.index.tolist() == ["p2"]


def test_follow_relation_supports_empty_rows():
    hydamo = _profile_hydamo()

    target_table, target_rows = follow_relation(
        hydamo,
        RelationsHyDAMO.PROFILE_LINE,
        "profile",
        hydamo.profile.iloc[0:0],
    )

    assert target_table == "profile_line"
    assert target_rows.empty
    assert target_rows.columns.tolist() == hydamo.profile_line.columns.tolist()


def test_follow_relation_rejects_unrelated_source_table():
    hydamo = _profile_hydamo()

    with pytest.raises(ValueError, match="not an endpoint"):
        follow_relation(
            hydamo,
            RelationsHyDAMO.PROFILE_LINE,
            "branches",
            hydamo.profile,
        )


def test_build_cascade_plan_follows_profile_paths():
    hydamo = _profile_hydamo()

    plan = build_cascade_plan(hydamo, "profile", pd.Index(["p1", "p2"]))

    assert {table: index.tolist() for table, index in plan.items()} == {
        "profile_roughness": ["roughness-1", "roughness-2"],
        "profile_line": ["l1", "l2"],
        "profile_group": ["g1", "g2"],
    }


def test_build_cascade_plan_follows_profile_line_paths_in_reverse():
    hydamo = _profile_hydamo()

    plan = build_cascade_plan(hydamo, "profile_line", pd.Index(["l2"]))

    assert {table: index.tolist() for table, index in plan.items()} == {
        "profile_group": ["g2"],
        "profile": ["p2"],
        "profile_roughness": ["roughness-2"],
    }


@pytest.mark.parametrize(
    ("source_table", "source_index", "expected"),
    [
        (
            "weirs",
            "w1",
            {"opening": ["o1"], "management_device": ["d1"]},
        ),
        ("culverts", "c1", {"management_device": ["d2"]}),
        (
            "pumpstations",
            "ps1",
            {"pumps": ["pump1"], "management": ["m1"]},
        ),
    ],
)
def test_build_cascade_plan_follows_structure_paths(
    source_table, source_index, expected
):
    plan = build_cascade_plan(
        _structure_hydamo(), source_table, pd.Index([source_index])
    )

    assert {table: index.tolist() for table, index in plan.items()} == expected


@pytest.mark.parametrize(
    ("source_table", "source_index", "expected"),
    [
        ("catchments", "catchment1", {"laterals": ["lateral1"]}),
        ("laterals", "lateral1", {"catchments": ["catchment1"]}),
        ("overflows", "overflow1", {"sewer_areas": ["sewer1"]}),
        ("sewer_areas", "sewer1", {"overflows": ["overflow1"]}),
        (
            "greenhouse_laterals",
            "greenhouse_lateral1",
            {"greenhouse_areas": ["greenhouse1"]},
        ),
        (
            "greenhouse_areas",
            "greenhouse1",
            {"greenhouse_laterals": ["greenhouse_lateral1"]},
        ),
    ],
)
def test_build_cascade_plan_follows_rr_paths_in_both_directions(
    source_table, source_index, expected
):
    plan = build_cascade_plan(_rr_hydamo(), source_table, pd.Index([source_index]))

    assert {table: index.tolist() for table, index in plan.items()} == expected


def test_build_cascade_plan_returns_empty_for_unconfigured_source():
    hydamo = SimpleNamespace(branches=pd.DataFrame(index=["branch-1"]))

    assert build_cascade_plan(hydamo, "branches", pd.Index(["branch-1"])) == {}
