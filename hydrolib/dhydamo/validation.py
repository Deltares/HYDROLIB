from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd

ValidationMode = Literal["off", "warn", "strict"]


class HyDAMOValidationError(RuntimeError):
    pass


@dataclass
class HydamoValidationResult:
    success: bool
    status: str | None
    errors: list[str]
    raw_result: Any = None


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
    hydamo_validation = importlib.import_module("hydamo_validation")
    validator_factory = getattr(hydamo_validation, "validator")

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
            rules_target.write_text(
                json.dumps(
                    _build_minimal_validation_rules(
                        gpkg_path=gpkg_path,
                        hydamo_version=hydamo_version,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

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

        _, _, result_summary = result
        errors = []
        if getattr(result_summary, "error", None):
            errors.extend([str(err) for err in result_summary.error if err])

        return HydamoValidationResult(
            success=bool(getattr(result_summary, "success", False)),
            status=getattr(result_summary, "status", None),
            errors=errors,
            raw_result=result_summary,
        )


def validate_or_raise(
    gpkg_path: str | Path,
    hydamo_version: str,
    validation_mode: ValidationMode,
    coverages: dict | None = None,
    validation_rules_path: str | Path | None = None,
) -> HydamoValidationResult | None:
    if validation_mode == "off":
        return None

    result = validate_hydamo_package(
        gpkg_path=gpkg_path,
        hydamo_version=hydamo_version,
        coverages=coverages,
        validation_rules_path=validation_rules_path,
    )
    if validation_mode == "strict" and not result.success:
        joined_errors = "; ".join(result.errors) if result.errors else "unknown error"
        raise HyDAMOValidationError(
            f"HyDAMO validation failed for DAMO {hydamo_version}: {joined_errors}"
        )
    return result


def _build_minimal_validation_rules(
    gpkg_path: Path, hydamo_version: str
) -> dict[str, Any]:
    objects = []
    for layer_name in gpd.list_layers(gpkg_path).name.tolist():
        object_name = layer_name.lower()
        sample = gpd.read_file(gpkg_path, layer=layer_name, engine="pyogrio")
        sample.columns = [col.lower() for col in sample.columns]

        parameter = next(
            (candidate for candidate in ("globalid", "code", "nen3610id") if candidate in sample.columns),
            None,
        )
        if parameter is None:
            continue

        objects.append(
            {
                "object": object_name,
                "validation_rules": [
                    {
                        "id": 0,
                        "name": f"{object_name} {parameter} aanwezig",
                        "type": "logic",
                        "validation_rule_set": "hydrolib-minimal",
                        "error_type": "non-critical",
                        "result_variable": f"{parameter}_present",
                        "error_message": f"{parameter} ontbreekt",
                        "active": True,
                        "function": {"NOTNA": {"parameter": parameter}},
                    }
                ],
            }
        )

    return {"schema": "1.5", "hydamo_version": hydamo_version, "objects": objects}
