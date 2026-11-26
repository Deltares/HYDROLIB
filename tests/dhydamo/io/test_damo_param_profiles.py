"""Parameterised profiles loaded from DAMO."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import geopandas as gpd
import pytest
from hydrolib.dhydamo.converters.hydamo2df import RoughnessVariant
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.io.damo_versions import loader
from shapely.geometry import LineString, Point

from tests.dhydamo.io.damo_gpkg import (
    _write_layer,
)


def _write_param_profiles(
    gpkg_path: Path,
    branches: dict[str, str],
    links: list[tuple[str, str]],
    widths: dict[str, float],
    omit: tuple[str, ...] = (),
) -> None:
    """Write branches, their links to parameterised profiles, and the profiles' values.

    *branches* maps a code to a ``globalid``; each branch is a 100 m line. *links* are
    ``(hydroobjectid, normgeparamprofielid)`` rows of ``hydroobject_normgp``. Each
    profile in *widths* gets a rectangular profile of that bottom width: its slope
    row is empty. The parameters in *omit* get no row at all.
    """
    _write_layer(
        gpkg_path,
        "HydroObject",
        gpd.GeoDataFrame(
            {
                "code": list(branches),
                "globalid": list(branches.values()),
                "geometry": [
                    LineString([(200 * i, 0), (200 * i + 100, 0)])
                    for i in range(len(branches))
                ],
            },
            crs="EPSG:28992",
        ),
    )
    _write_layer(
        gpkg_path,
        "hydroobject_normgp",
        gpd.GeoDataFrame(
            {
                "hydroobjectid": [branch for branch, _ in links],
                "normgeparamprofielid": [profile for _, profile in links],
                # undeclared, so dropped by the tests
                "nen3610id": [f"NORMGP.{i}" for i in range(len(links))],
                "geometry": [Point(0, 0)] * len(links),
            },
            crs="EPSG:28992",
        ),
    )
    rows = []
    for profile, width in widths.items():
        parameters = {
            "bodembreedte": width,
            "bodemhoogte bovenstrooms": 1.0,
            "bodemhoogte benedenstrooms": 0.5,
            # no slope: a rectangle
            "taludhelling linkerzijde": float("nan"),
        }
        for parameter, value in parameters.items():
            if parameter in omit:
                continue
            rows.append(
                {
                    "normgeparamprofielid": profile,
                    "soortparameter": parameter,
                    "waarde": value,
                    "ruwheidlaag": 30.0,
                    "ruwheidhoog": 20.0,
                    "typeruwheid": "Manning",
                    "geometry": Point(0, 0),
                }
            )
    _write_layer(
        gpkg_path,
        "NormgeparamProfielWaarde",
        gpd.GeoDataFrame(rows, crs="EPSG:28992"),
    )


def _param_profile_widths(gpkg_path: Path) -> dict[str, float]:
    """Load *gpkg_path* without undeclared columns and build its cross sections.

    Returns the bottom width of the cross sections of each branch.
    """
    hydamo = HyDAMO()
    # Without undeclared columns, so building fails on a column read undeclared.
    with mock.patch.object(loader, "_KEEP_UNDECLARED_COLUMNS", False):
        hydamo.load_from_gpkg(gpkg_path, hydamo_version="2.2")
    hydamo.crosssections.convert.profiles(
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
    )
    widths = {}
    for location in hydamo.crosssections.crosssection_loc.values():
        definition = hydamo.crosssections.crosssection_def[location["definitionId"]]
        widths[location["branchid"]] = definition["width"]
    return widths


def test_load_from_gpkg_reads_param_profiles_from_hydroobject_normgp(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file, {"W1": "w1-guid"}, [("w1-guid", "np-guid")], {"np-guid": 4.0}
    )

    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")

    assert hydamo.param_profile[
        ["hydroobjectid", "normgeparamprofielid"]
    ].to_numpy().tolist() == [["w1-guid", "np-guid"]]
    assert _param_profile_widths(gpkg_file) == {"W1": 4.0}


def test_a_param_profile_linked_to_two_branches_applies_to_both(
    tmp_path: Path,
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid", "W2": "w2-guid"},
        [("w1-guid", "np-guid"), ("w2-guid", "np-guid")],
        {"np-guid": 4.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1": 4.0, "W2": 4.0}


def test_a_branch_linked_to_two_param_profiles_gets_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid"},
        [("w1-guid", "np-1"), ("w1-guid", "np-2"), ("w1-guid", "np-3")],
        {"np-1": 4.0, "np-2": 6.0, "np-3": 8.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1": 4.0}
    assert (
        "Branch W1 is linked to 3 parameterised profiles; using np-1 and ignoring "
        "np-2, np-3." in caplog.text
    )


def test_a_param_profile_link_matching_two_branches_goes_to_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Split parts sharing a globalid: invalid DAMO, as in the example model.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1_0": "w1-guid", "W1_1": "w1-guid"},
        [("w1-guid", "np-guid")],
        {"np-guid": 4.0},
    )

    assert _param_profile_widths(gpkg_file) == {"W1_0": 4.0}
    assert (
        "Parameterised profile np-guid: hydroobjectid w1-guid matches branches "
        "W1_0, W1_1; only W1_0 gets the profile." in caplog.text
    )


def test_a_param_profile_link_to_a_branch_with_a_cross_section_is_skipped(
    tmp_path: Path,
) -> None:
    # Measured profiles come first; before, this link raised a KeyError.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid", "W2": "w2-guid"},
        [("w1-guid", "np-1"), ("w2-guid", "np-2")],
        {"np-1": 4.0, "np-2": 6.0},
    )
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")
    measured = hydamo.crosssections.add_rectangle_definition(
        height=2.0,
        width=1.0,
        closed=False,
        roughnesstype="Manning",
        roughnessvalue=0.03,
        name="measured",
    )
    hydamo.crosssections.add_crosssection_location(
        branchid="W1", chainage=50.0, definition=measured
    )

    hydamo.crosssections.convert.profiles(
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
    )

    widths: dict[str, set[float]] = {}
    for location in hydamo.crosssections.crosssection_loc.values():
        definition = hydamo.crosssections.crosssection_def[location["definitionId"]]
        widths.setdefault(location["branchid"], set()).add(definition["width"])
    assert widths == {"W1": {1.0}, "W2": {6.0}}


def test_a_param_profile_without_slope_rows_is_a_rectangle(tmp_path: Path) -> None:
    # Before, an absent slope row raised an IndexError.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid"},
        [("w1-guid", "np-guid")],
        {"np-guid": 4.0},
        omit=("taludhelling linkerzijde",),
    )

    assert _param_profile_widths(gpkg_file) == {"W1": 4.0}


@pytest.mark.parametrize("omit", ["row", "value"])
def test_a_param_profile_without_its_bottom_width_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, omit: str
) -> None:
    # Before, an absent row raised an IndexError, and an empty value went on as NaN.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1": "w1-guid", "W2": "w2-guid"},
        [("w1-guid", "np-1"), ("w2-guid", "np-2")],
        {"np-1": float("nan"), "np-2": 6.0},
        omit=("bodembreedte",) if omit == "row" else (),
    )

    widths = _param_profile_widths(gpkg_file)

    if omit == "row":
        # both profiles lack the row
        assert widths == {}
    else:
        assert widths == {"W2": 6.0}
    assert (
        "Parameterised profile np-1 has no bodembreedte; branch W1 gets no "
        "parameterised profile." in caplog.text
    )


def test_a_trapezium_param_profile_needs_both_slopes_and_insteek_heights(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file, {"W1": "w1-guid"}, [("w1-guid", "np-guid")], {"np-guid": 4.0}
    )
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")
    values = hydamo.param_profile_values
    values.loc[values["soortparameter"] == "taludhelling linkerzijde", "waarde"] = 2.0

    profiles = hydamo.crosssections.parametrised_to_profiles(
        hydamo.param_profile,
        values,
        list(hydamo.branches.itertuples()),
        roughness_variant=RoughnessVariant.HIGH,
    )

    assert profiles == {}
    assert (
        "Parameterised profile np-guid has no taludhelling rechterzijde, hoogte "
        "insteek linkerzijde, hoogte insteek rechterzijde; branch W1 gets no "
        "parameterised profile." in caplog.text
    )


def test_a_param_profile_link_resolves_against_all_branches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The first part, W1_0, has a measured profile. Before, the link was resolved
    # among the branches without one only, so W1_1 got the profile, without warning.
    gpkg_file = tmp_path / "param.gpkg"
    _write_param_profiles(
        gpkg_file,
        {"W1_0": "w1-guid", "W1_1": "w1-guid"},
        [("w1-guid", "np-guid")],
        {"np-guid": 4.0},
    )
    hydamo = HyDAMO().load_from_gpkg(gpkg_file, hydamo_version="2.2")
    measured = hydamo.crosssections.add_rectangle_definition(
        height=2.0,
        width=1.0,
        closed=False,
        roughnesstype="Manning",
        roughnessvalue=0.03,
        name="measured",
    )
    hydamo.crosssections.add_crosssection_location(
        branchid="W1_0", chainage=50.0, definition=measured
    )

    hydamo.crosssections.convert.profiles(
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
    )

    branches = {
        location["branchid"]
        for location in hydamo.crosssections.crosssection_loc.values()
    }
    assert branches == {"W1_0"}
    assert (
        "Parameterised profile np-guid: hydroobjectid w1-guid matches branches "
        "W1_0, W1_1; only W1_0 gets the profile." in caplog.text
    )
