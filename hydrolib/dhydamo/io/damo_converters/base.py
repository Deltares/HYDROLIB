from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSpec:
    """Schema descriptor for a single HyDAMO GeoPackage layer.

    Parameters
    ----------
    target_attr : str
        Name of the ``HyDAMO`` instance attribute to populate.
    source_layer : str
        GeoPackage layer name to read (case-insensitive match).
    index_col : str, optional
        Column to set as the DataFrame index after loading.
    groupby_column : str, optional
        Column used to group point rows into line geometries.
    order_column : str, optional
        Column that defines point order within each group.
    column_mapping : dict[str, str], optional
        ``{source_col: target_col}`` renames applied before writing to the
        attribute.
    check_geotype : bool
        Whether to validate that geometries match the expected type declared
        on the target ``ExtendedGeoDataFrame``. Default is ``True``.
    optional : bool
        When ``True``, a missing GeoPackage layer is logged and skipped
        rather than raising an error. Default is ``True``.
    """

    target_attr: str
    source_layer: str
    index_col: str | None = None
    groupby_column: str | None = None
    order_column: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    check_geotype: bool = True
    optional: bool = True


class BaseDamoConverter(ABC):
    """Abstract base class for versioned HyDAMO GeoPackage converters.

    Subclasses declare ``hydamo_version`` and ``layer_specs``, and optionally
    override ``postprocess`` for any column-level canonicalization required
    after all layers are loaded.
    """

    hydamo_version: str
    layer_specs: tuple[LayerSpec, ...]

    def load_into(
        self,
        hydamo,
        gpkg_path: str | Path,
        clip_layers: dict | None = None,
        check_3d: bool = True,
    ) -> None:
        """Read all declared layer specs from *gpkg_path* into *hydamo*.

        Parameters
        ----------
        hydamo : HyDAMO
            The ``HyDAMO`` instance whose attributes are populated.
        gpkg_path : str or Path
            Path to the source GeoPackage.
        clip_layers : dict, optional
            Per-layer spatial clip. Maps ``target_attr`` to a
            ``(geometry, cliptype)`` tuple. Layers absent from the dict are
            loaded without clipping.
        check_3d : bool
            Passed to ``read_gpkg_layer`` on geo layers. Set ``False`` for
            flat (2D-only) GeoPackages. Default is ``True``.
        """
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
                "check_geotype": spec.check_geotype,
                "check_3d": check_3d,
            }
            if spec.groupby_column is not None:
                kwargs["groupby_column"] = spec.groupby_column
            if spec.order_column is not None:
                kwargs["order_column"] = spec.order_column
            if clip_layers and spec.target_attr in clip_layers:
                kwargs["clip"], kwargs["cliptype"] = clip_layers[spec.target_attr]

            target._read_gpkg_layer(**kwargs)

        self.postprocess(hydamo)

    def postprocess(self, hydamo) -> None:
        """Hook for version-specific canonicalization after all layers are loaded.

        Override in subclasses to rename, copy, or derive columns on any
        ``HyDAMO`` attribute that requires post-load normalisation.
        """

    @staticmethod
    def _rename_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        """Return *df* with column *source* renamed to *target*.

        No-ops silently if *source* is absent or *target* already exists.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        source : str
            Existing column name.
        target : str
            Desired column name.

        Returns
        -------
        pd.DataFrame
            DataFrame with the column renamed, or the original if no-op.
        """
        if source in df.columns and target not in df.columns:
            return df.rename(columns={source: target})
        return df

    @staticmethod
    def _copy_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        """Add column *target* to *df* as a copy of *source*, in-place.

        No-ops silently if *source* is absent or *target* already exists.
        Mutates *df* directly to preserve the original subclass type
        (e.g. ``ExtendedDataFrame``).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        source : str
            Column to copy from.
        target : str
            New column name to copy into.

        Returns
        -------
        pd.DataFrame
            The same *df* object with the new column added, or unchanged if
            no-op.
        """
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
        return df


def replace_layer_specs(
    base_specs: tuple[LayerSpec, ...], *overrides: LayerSpec
) -> tuple[LayerSpec, ...]:
    """Merge *overrides* into *base_specs*, replacing entries by ``target_attr``.

    Specs present in *base_specs* are replaced in-place by the matching
    override if one exists. Overrides with no matching base spec are
    appended at the end.

    Parameters
    ----------
    base_specs : tuple[LayerSpec, ...]
        The canonical spec tuple from the parent converter.
    *overrides : LayerSpec
        Replacement ``LayerSpec`` objects keyed by ``target_attr``.

    Returns
    -------
    tuple[LayerSpec, ...]
        New tuple with overrides applied.
    """
    override_map = {spec.target_attr: spec for spec in overrides}
    merged_specs = [override_map.pop(spec.target_attr, spec) for spec in base_specs]
    merged_specs.extend(override_map.values())
    return tuple(merged_specs)
