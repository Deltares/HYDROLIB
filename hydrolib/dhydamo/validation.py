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
            schema_version, valid_objects = _schema_version_for_damo(_hydamo_validation, hydamo_version)
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

        # output_types=[] — no artefacts written to disk; results are read from
        # result_summary in memory. Callers that want CSV/GeoJSON outputs must
        # call hydamo_validation.validator directly.
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
        success = bool(getattr(result_summary, "success", False))
        if not success and not errors:
            error_layers = getattr(result_summary, "error_layers", None) or []
            missing_layers = getattr(result_summary, "missing_layers", None) or []
            if error_layers:
                errors.append(f"Layers with errors: {', '.join(error_layers)}")
            if missing_layers:
                errors.append(f"Missing layers: {', '.join(missing_layers)}")

        return HydamoValidationResult(
            success=success,
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


def _schema_version_for_damo(hv_module: Any, hydamo_version: str) -> tuple[str, list[str]]:
    """Return ``(schema_version, valid_object_names)`` for *hydamo_version*.

    Scans the schema files shipped with the installed package to find the highest
    schema version that declares *hydamo_version* as a supported value, and also
    returns the ``hydamo_objects`` enum so callers can filter out non-standard layers.

    Raises ``ValueError`` when the installed package does not support the requested
    DAMO version (e.g. DAMO 2.5 requires schema 1.5 which ships with a future
    hydamo-validation release).  Falls back to ``("1.4", [])`` when the module is a
    test fake with no ``__file__`` attribute.
    """
    try:
        schema_dir = Path(hv_module.__file__).parent / "schemas" / "rules"
        best_version: str | None = None
        best_schema: dict | None = None
        for schema_file in sorted(schema_dir.glob("rules_*.json")):
            schema = json.loads(schema_file.read_text())
            supported = schema.get("properties", {}).get("hydamo_version", {}).get("enum", [])
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
    except (AttributeError, OSError):
        # Fake module injected in tests — return safe fallbacks.
        return "1.4", []


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
            (candidate for candidate in ("globalid", "code", "nen3610id") if candidate in sample.columns),
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

    return {"schema": schema_version, "hydamo_version": hydamo_version, "objects": objects}
