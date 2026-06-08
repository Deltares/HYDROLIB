from __future__ import annotations

import inspect
import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSpec:
    target_attr: str
    source_layer: str
    index_col: str | None = None
    groupby_column: str | None = None
    order_column: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    check_geotype: bool = True
    optional: bool = True


class BaseDamoConverter(ABC):
    hydamo_version: str
    layer_specs: tuple[LayerSpec, ...]

    def load_into(self, hydamo, gpkg_path: str | Path) -> None:
        gpkg_path = Path(gpkg_path)
        available_layers = {
            layer_name.lower(): layer_name
            for layer_name in gpd.list_layers(gpkg_path).name.tolist()
        }

        for spec in self.layer_specs:
            actual_layer = available_layers.get(spec.source_layer.lower())
            if actual_layer is None:
                if spec.optional:
                    logger.info(
                        'Skipping optional HyDAMO layer "%s" for DAMO %s.',
                        spec.source_layer,
                        self.hydamo_version,
                    )
                    continue
                raise ValueError(
                    f'Missing required HyDAMO layer "{spec.source_layer}" for DAMO {self.hydamo_version}.'
                )

            target = getattr(hydamo, spec.target_attr)
            kwargs = {
                "gpkg_path": gpkg_path,
                "layer_name": actual_layer,
                "index_col": spec.index_col,
                "column_mapping": spec.column_mapping if spec.column_mapping else None,
            }
            if spec.groupby_column is not None:
                kwargs["groupby_column"] = spec.groupby_column
            if spec.order_column is not None:
                kwargs["order_column"] = spec.order_column
            if hasattr(target, "read_gpkg_layer") and "check_geotype" in inspect.signature(target.read_gpkg_layer).parameters:
                kwargs["check_geotype"] = spec.check_geotype

            target.read_gpkg_layer(**kwargs)

        self.postprocess(hydamo)

    def postprocess(self, hydamo) -> None:
        """Hook for version-specific canonicalization after layer loading."""

    @staticmethod
    def _rename_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        if source in df.columns and target not in df.columns:
            return df.rename(columns={source: target})
        return df

    @staticmethod
    def _copy_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        if source in df.columns and target not in df.columns:
            df = df.copy()
            df[target] = df[source]
        return df


def replace_layer_specs(
    base_specs: tuple[LayerSpec, ...], *overrides: LayerSpec
) -> tuple[LayerSpec, ...]:
    override_map = {spec.target_attr: spec for spec in overrides}
    merged_specs = [override_map.pop(spec.target_attr, spec) for spec in base_specs]
    merged_specs.extend(override_map.values())
    return tuple(merged_specs)
