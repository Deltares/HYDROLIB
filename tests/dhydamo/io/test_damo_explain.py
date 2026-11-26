"""The report of how a GeoPackage layer becomes a HyDAMO table."""

from __future__ import annotations

import dataclasses

import pytest
from hydrolib.dhydamo.core.data_model import TABLES
from hydrolib.dhydamo.io.damo_versions import (
    SUPPORTED_HYDAMO_VERSIONS,
    explain_table,
    explain_tables,
    steps,
)


def _lines(report: str) -> list[str]:
    """Return the lines of *report*, with runs of spaces collapsed."""
    return [" ".join(line.split()) for line in report.splitlines()]


def test_explains_a_table_of_damo_22_without_renames() -> None:
    lines = _lines(explain_table("management_device", "2.2"))

    assert lines[0] == 'management_device <- layer "Regelmiddel" (DAMO 2.2)'
    assert "soortregelmiddel no soortregelmiddel" in lines
    assert "hoogteopening no hoogteopening" in lines
    assert "Other columns pass through under their own names." in lines


def test_explains_the_layer_move_and_the_renames_of_damo_25() -> None:
    lines = _lines(explain_table("management_device", "2.5"))

    assert lines[0] == 'management_device <- layer "Afsluitmiddel" (DAMO 2.5)'
    assert "Layer: Regelmiddel in 2.2; Afsluitmiddel from 2.5 (step 2.4 -> 2.5)." in (
        lines
    )
    assert "soortregelmiddel no typeafsluitmiddel 2.4 -> 2.5" in lines
    assert "hoogteopening no maximalehoogteonderkant 2.4 -> 2.5" in lines
    # values and a HYDROLIB-only column
    assert "soortregelmiddel: schuif, terugslagklep" in lines
    assert "maximaaldebiet no (HYDROLIB only)" in lines
    # relations, as child and as parent
    assert (
        "management_device_opening: management_device.kunstwerkopeningid -> "
        "opening.globalid"
    ) in lines
    assert (
        "management_device: management.regelmiddelid -> management_device.globalid"
    ) in lines


def test_explains_a_rename_over_several_steps() -> None:
    lines = _lines(explain_table("culverts", "2.6"))

    assert "typeruwheid yes typemethoderuwheid 2.3 -> 2.4" in lines
    assert "vormkoker yes vormduikersifonhevel 2.4 -> 2.5" in lines


def test_explains_the_rewrite_and_the_fill_of_management() -> None:
    lines = _lines(explain_table("management", "2.5"))

    assert "stuurvariabele no typestuurvariabele 2.4 -> 2.5" in lines
    assert (
        "value rewrite: stuurvariabele, 'bovenkant afsluitmiddel,' -> "
        "'bovenkant afsluitmiddel'"
    ) in lines
    assert "filled from globalid: code" in lines
    assert "identity: code" in lines


def test_explains_the_derived_code_of_profile() -> None:
    lines = _lines(explain_table("profile", "2.2"))

    assert "Points grouped into lines by profiellijnid, ordered by codevolgnummer." in (
        lines
    )
    assert "derived column: code" in lines
    # The reason follows, indented.
    index = lines.index("derived column: code")
    assert lines[index + 1].startswith("The D-Flow FM cross-section id")


def test_explain_rejects_an_unsupported_version() -> None:
    with pytest.raises(ValueError, match='Unsupported HyDAMO DAMO version "9.9"'):
        explain_table("weirs", "9.9")


def test_explain_rejects_an_unknown_table() -> None:
    with pytest.raises(ValueError, match='"weir" is not a HyDAMO table'):
        explain_table("weir", "2.2")


@pytest.mark.parametrize("version", SUPPORTED_HYDAMO_VERSIONS)
def test_explains_every_table(version: str) -> None:
    report = explain_tables(version)

    for name in TABLES:
        assert f"{name} <- layer" in report


def test_explain_labels_the_columns_the_loader_fills_or_derives() -> None:
    # Sturing has no code in DAMO, and Stuw has no crest from 2.4.
    management = _lines(explain_table("management", "2.6"))
    weirs = _lines(explain_table("weirs", "2.6"))

    assert management[3] == (
        "column (2.2) required after loading DAMO 2.6 name renamed in note"
    )
    assert "code yes code filled from globalid if absent" in management
    assert "laagstedoorstroomhoogte no laagstedoorstroomhoogte derived, see below" in (
        weirs
    )


def test_explain_shows_the_renames_the_loader_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In one step, old naam becomes code and old code becomes nummer.
    chained = dataclasses.replace(
        steps.STEPS[-1], columns={"weirs": {"naam": "code", "code": "nummer"}}
    )
    monkeypatch.setattr(steps, "STEPS", (*steps.STEPS[:-1], chained))
    lines = _lines(explain_table("weirs", "2.6"))

    assert "code yes nummer 2.5 -> 2.6" in lines
    assert "naam no code 2.5 -> 2.6" in lines
