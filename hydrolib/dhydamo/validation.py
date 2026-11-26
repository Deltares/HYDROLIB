"""Validate HyDAMO GeoPackages with the hydamo_validation package."""

from __future__ import annotations

import importlib
import json
import logging
import re
import shutil
import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

ValidationMode = Literal["off", "warn", "strict"]

logger = logging.getLogger(__name__)


class HyDAMOValidationError(RuntimeError):
    """A HyDAMO GeoPackage failed validation in strict mode."""


# A syntax result is "" when valid, "<value> -> NULL (fouten: 1,4)" for a rejected
# value, "(fouten: 4)" for a missing column (no value) and "<type> -> NULL (7)" for an
# invalid geometry. The value shown is the one after emptying: an unconvertible value
# and an empty one both read "nan". The error marker is matched at the end, so a
# value that contains marker-like text isn't read as codes.
_SYNTAX_RESULT = re.compile(
    r"^(?:(?P<value>.*?) ?-> NULL ?)?"
    r"\((?:fouten: (?P<codes>[\d,]+)|(?P<geometry_code>7))\)$"
)
# The validatiemodule's error codes that make it reject a row's value, and what they
# mean; 2 ("converted") rejects nothing. hydamo_validation exposes no constants for
# them: they're literals in hydamo_validation.syntax_validation.fields_syntax (1.7.0),
# like the result format above. Check both there before raising its upper limit.
_ERROR_MEANINGS = {
    1: "not convertible",
    3: "not unique",
    4: "missing",
    5: "fill value",
    6: "not an allowed value",
    7: "invalid geometry",
}
_MAX_EXAMPLES = 3
# How the validatiemodule writes an empty value in front of " -> NULL".
_EMPTY_VALUES = {"", "nan", "None", "NaT", "<NA>"}


@dataclass
class SyntaxFinding:
    """Rows of one layer whose value in one column failed one syntax check.

    Attributes
    ----------
    layer : str
        Layer name, lowercase, as the validatiemodule reports it.
    column : str
        Column name.
    code : int
        The validatiemodule's error code.
    meaning : str
        What the error code means, e.g. ``"not unique"``.
    row_count : int
        Number of rows with this error.
    examples : list[str]
        Up to three rejected values. Where the validatiemodule shows no value
        (an empty or unconvertible value, or a missing column), the row's
        ``code``.
    read_by_hydrolib : bool
        Whether HYDROLIB reads this column, so that the rejected values affect
        the model. Only these findings make ``"strict"`` validation fail.
    """

    layer: str
    column: str
    code: int
    meaning: str
    row_count: int
    examples: list[str]
    read_by_hydrolib: bool = False

    def add_row(self, example: str) -> None:
        """Count one more rejected row; keep *example* if there's room for it."""
        self.row_count += 1
        is_new = example and example not in self.examples
        if is_new and len(self.examples) < _MAX_EXAMPLES:
            self.examples.append(example)

    def describe(self) -> str:
        """Describe the finding in one line, e.g. ``globalid: 2 rows not unique``."""
        examples = f" (e.g. {', '.join(self.examples)})" if self.examples else ""
        return f"{self.column}: {self.row_count} rows {self.meaning}{examples}"


@dataclass
class HydamoValidationResult:
    """Result of validating a HyDAMO GeoPackage.

    Attributes
    ----------
    success : bool
        Whether the validation succeeded.
    status : str or None
        Status reported by hydamo_validation.
    errors : list[str]
        Error messages.
    raw_result : Any
        The result object of hydamo_validation.
    syntax_findings : list[SyntaxFinding]
        The rows the syntax check rejected, per layer, column and error code.
    """

    success: bool
    status: str | None
    errors: list[str]
    raw_result: Any = None
    syntax_findings: list[SyntaxFinding] = field(default_factory=list)


def validate_hydamo_package(
    gpkg_path: str | Path,
    hydamo_version: str,
    coverages: dict | None = None,
    validation_rules_path: str | Path | None = None,
) -> HydamoValidationResult:
    """Validate a HyDAMO GeoPackage using the hydamo_validation library.

    When no *validation_rules_path* is supplied, minimal auto-generated rules are
    used that only check whether each layer's ID column (globalid/code/nen3610id)
    is non-null. Full schema, type, and referential-integrity checks require an
    explicit rules file.
    """
    try:
        _hydamo_validation = importlib.import_module("hydamo_validation")
    except ImportError:
        raise ImportError(
            "HyDAMO validation requires the 'hydamo-validation' package (Python 3.12 only). "
            "Install it with: pip install hydrolib[validation]"
        )
    validator_factory = _hydamo_validation.validator

    gpkg_path = Path(gpkg_path)
    with tempfile.TemporaryDirectory(prefix="hydrolib-hydamo-validation-") as tmpdir:
        task_dir = Path(tmpdir)
        datasets_dir = task_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gpkg_path, datasets_dir / gpkg_path.name)

        rules_target = task_dir / "validationrules.json"
        if validation_rules_path is not None:
            shutil.copy2(validation_rules_path, rules_target)
        else:
            schema_version, valid_objects = _schema_version_for_damo(
                _hydamo_validation, hydamo_version
            )
            rules_target.write_text(
                json.dumps(
                    _build_minimal_validation_rules(
                        gpkg_path=gpkg_path,
                        hydamo_version=hydamo_version,
                        schema_version=schema_version,
                        valid_objects=valid_objects,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

        # output_types=[] — no artefacts written to disk; results are read in
        # memory, the syntax findings from layers_summary. Callers that want
        # CSV/GeoJSON outputs must call hydamo_validation.validator directly.
        hydamo_validator = validator_factory(
            output_types=[],
            coverages=coverages or {},
            log_level="INFO",
        )
        result = hydamo_validator(directory=task_dir, raise_error=False)
        if result is None:
            return HydamoValidationResult(
                success=False,
                status="validator-error",
                errors=["Validator returned no result."],
                raw_result=None,
            )

        _, layers_summary, result_summary = result
        success = bool(getattr(result_summary, "success", False))
        return HydamoValidationResult(
            success=success,
            status=getattr(result_summary, "status", None),
            errors=_summary_errors(result_summary, success),
            raw_result=result_summary,
            syntax_findings=_syntax_findings(layers_summary),
        )


def _parse_syntax_result(result: str) -> tuple[str | None, list[int]]:
    """Split one syntax result into the value shown and its error codes.

    The value is None where the validatiemodule shows none, or shows an empty
    one. A valid result gives no codes.
    """
    match = _SYNTAX_RESULT.match(result)
    if match is None:
        return None, []
    codes = match["codes"] or match["geometry_code"]
    value = match["value"]
    if value in _EMPTY_VALUES:
        value = None
    return value, [int(code) for code in codes.split(",")]


def _syntax_findings(layers_summary: Any) -> list[SyntaxFinding]:
    """Count the rejected rows per layer, column and error code."""
    if layers_summary is None:
        return []
    # The layers summary has one attribute per validated layer, listed in geo_types.
    findings = []
    for layer in getattr(layers_summary, "geo_types", {}):
        results = getattr(layers_summary, layer)
        findings.extend(_layer_findings(layer, results))
    return findings


def _layer_findings(layer: str, results: pd.DataFrame) -> list[SyntaxFinding]:
    """Return the findings of one layer, from its ``syntax_<column>`` columns."""
    # The row's code names a row whose rejected value the validatiemodule doesn't show.
    if "code" in results.columns:
        row_codes = results["code"].astype(str)
    else:
        row_codes = pd.Series("", index=results.index)

    # syntax_oordeel is the row's overall verdict, not a column's result.
    result_columns = []
    for column in results.columns:
        if column.startswith("syntax_") and column != "syntax_oordeel":
            result_columns.append(column)
    findings = []
    for result_column in result_columns:
        column = result_column.removeprefix("syntax_")
        findings.extend(
            _column_findings(layer, column, results[result_column], row_codes)
        )
    return findings


def _column_findings(
    layer: str, column: str, results: pd.Series, row_codes: pd.Series
) -> list[SyntaxFinding]:
    """Return one finding per error code in one column's results, ordered by code."""
    per_code: dict[int, SyntaxFinding] = {}
    for result, row_code in zip(results.astype(str), row_codes, strict=True):
        value, codes = _parse_syntax_result(result)
        for code in codes:
            # Code 2 ("converted") and unknown codes reject nothing.
            if code not in _ERROR_MEANINGS:
                continue
            if code not in per_code:
                per_code[code] = SyntaxFinding(
                    layer=layer,
                    column=column,
                    code=code,
                    meaning=_ERROR_MEANINGS[code],
                    row_count=0,
                    examples=[],
                )
            per_code[code].add_row(example=value or row_code)

    findings = []
    for code in sorted(per_code):
        findings.append(per_code[code])
    return findings


def _summary_errors(result_summary: Any, success: bool) -> list[str]:
    """Collect error messages from a hydamo_validation result summary."""
    # hydamo_validation uses .error for unhandled exceptions and .errors for
    # errors appended during rule processing — read both to capture all messages.
    errors = []
    for attr in ("error", "errors"):
        val = getattr(result_summary, attr, None)
        if val:
            errors.extend([str(e) for e in val if e])

    # When validation rules fail (success=False) but no exception-level error
    # messages were raised, surface which layers failed so that validate_or_raise
    # produces a useful message instead of "unknown error".
    if not success and not errors:
        error_layers = getattr(result_summary, "error_layers", None) or []
        missing_layers = getattr(result_summary, "missing_layers", None) or []
        if error_layers:
            errors.append(f"Layers with errors: {', '.join(error_layers)}")
        if missing_layers:
            errors.append(f"Missing layers: {', '.join(missing_layers)}")
    return errors


def validate_or_raise(
    gpkg_path: str | Path,
    hydamo_version: str,
    validation_mode: ValidationMode,
    coverages: dict | None = None,
    validation_rules_path: str | Path | None = None,
    read_columns: Mapping[str, Collection[str]] | None = None,
) -> HydamoValidationResult | None:
    """Validate a HyDAMO GeoPackage, depending on the validation mode.

    Parameters
    ----------
    gpkg_path : str or Path
        The GeoPackage.
    hydamo_version : str
        DAMO version, e.g. ``"2.4"``.
    validation_mode : {"off", "warn", "strict"}
        ``"off"`` skips the validation; ``"strict"`` raises when it fails.
    coverages : dict, optional
        Coverages for hydamo_validation. Defaults to None.
    validation_rules_path : str or Path, optional
        Rules file; see `validate_hydamo_package`. Defaults to None.
    read_columns : mapping of str to collection of str, optional
        Per lowercase layer name, the columns HYDROLIB reads, under the names
        of the validated DAMO version. Syntax findings in these columns are
        marked ``read_by_hydrolib``. Defaults to None: no column is read.

    Returns
    -------
    HydamoValidationResult or None
        The result, or None with ``"off"``.

    Raises
    ------
    HyDAMOValidationError
        With ``"strict"``, if the validation run fails, or if the syntax check
        rejected values in a column HYDROLIB reads.
    """
    if validation_mode == "off":
        return None

    result = validate_hydamo_package(
        gpkg_path=gpkg_path,
        hydamo_version=hydamo_version,
        coverages=coverages,
        validation_rules_path=validation_rules_path,
    )
    for finding in result.syntax_findings:
        finding.read_by_hydrolib = finding.column in (read_columns or {}).get(
            finding.layer, ()
        )
    _log_syntax_findings(result.syntax_findings, validation_mode)
    if validation_mode == "strict" and not result.success:
        joined_errors = "; ".join(result.errors) if result.errors else "unknown error"
        raise HyDAMOValidationError(
            f"HyDAMO validation failed for DAMO {hydamo_version}: {joined_errors}"
        )
    blocking = [f for f in result.syntax_findings if f.read_by_hydrolib]
    if validation_mode == "strict" and blocking:
        listed = "; ".join(f"{f.layer}.{f.describe()}" for f in blocking)
        raise HyDAMOValidationError(
            f"HyDAMO validation rejected data HYDROLIB reads, for DAMO "
            f"{hydamo_version}: {listed}"
        )
    return result


def _log_syntax_findings(
    findings: list[SyntaxFinding], validation_mode: ValidationMode
) -> None:
    """Log the syntax findings that matter for the model, and count the rest.

    With ``"warn"``, findings in columns HYDROLIB reads are logged as one warning
    per layer; with ``"strict"``, they are raised instead. Other findings get one
    INFO line in both modes.
    """
    if validation_mode == "warn":
        per_layer: dict[str, list[SyntaxFinding]] = {}
        for finding in findings:
            if finding.read_by_hydrolib:
                per_layer.setdefault(finding.layer, []).append(finding)
        for layer, layer_findings in per_layer.items():
            logger.warning(
                "HyDAMO validation rejected values in layer %s: %s",
                layer,
                "; ".join(f.describe() for f in layer_findings),
            )
    other = sum(not finding.read_by_hydrolib for finding in findings)
    if other:
        logger.info(
            "HyDAMO validation: %d other findings in columns HYDROLIB doesn't "
            "read; see validation_result.syntax_findings.",
            other,
        )


def _schema_version_for_damo(
    hv_module: Any, hydamo_version: str
) -> tuple[str, list[str]]:
    """Return ``(schema_version, valid_object_names)`` for *hydamo_version*.

    Scans the schema files shipped with the installed package to find the highest
    schema version that declares *hydamo_version* as a supported value, and also
    returns the ``hydamo_objects`` enum so callers can filter out non-standard layers.

    DAMO version support depends on the schemas shipped with the installed
    hydamo-validation package. Raises ``ValueError`` with upgrade guidance when
    none of those schemas supports the requested version.
    """
    schema_dir = Path(hv_module.__file__).parent / "schemas" / "rules"
    best_version: str | None = None
    best_schema: dict | None = None
    for schema_file in sorted(schema_dir.glob("rules_*.json")):
        schema = json.loads(schema_file.read_text())
        supported = (
            schema.get("properties", {}).get("hydamo_version", {}).get("enum", [])
        )
        if hydamo_version in supported:
            best_version = schema_file.stem.split("_")[1]
            best_schema = schema
    if best_version is not None and best_schema is not None:
        valid_objects: list[str] = (
            best_schema.get("$defs", {}).get("hydamo_objects", {}).get("enum", [])
        )
        return best_version, valid_objects
    raise ValueError(
        f"The installed hydamo-validation package does not support DAMO version "
        f"'{hydamo_version}'. Upgrade hydamo-validation for DAMO {hydamo_version} support."
    )


def _build_minimal_validation_rules(
    gpkg_path: Path,
    hydamo_version: str,
    schema_version: str = "1.4",
    valid_objects: list[str] | None = None,
) -> dict[str, Any]:
    objects = []
    for layer_name in gpd.list_layers(gpkg_path).name.tolist():
        object_name = layer_name.lower()
        sample = gpd.read_file(gpkg_path, layer=layer_name, engine="pyogrio")
        sample.columns = [col.lower() for col in sample.columns]

        parameter = next(
            (
                candidate
                for candidate in ("globalid", "code", "nen3610id")
                if candidate in sample.columns
            ),
            None,
        )
        if parameter is None:
            continue
        if valid_objects and object_name not in valid_objects:
            continue

        objects.append(
            {
                "object": object_name,
                "validation_rules": [
                    {
                        "id": 0,
                        "name": f"{object_name} {parameter} aanwezig",
                        "type": "logic",
                        "error_type": "non-critical",
                        "result_variable": f"{parameter}_present",
                        "error_message": f"{parameter} ontbreekt",
                        "active": True,
                        "function": {"NOTNA": {"parameter": parameter}},
                    }
                ],
            }
        )

    return {
        "schema": schema_version,
        "hydamo_version": hydamo_version,
        "objects": objects,
    }
