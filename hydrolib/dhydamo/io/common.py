import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Union
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon

from hydrolib.dhydamo.geometry import spatial

logger = logging.getLogger()


class ExtendedGeoDataFrame(gpd.GeoDataFrame):
    # normal properties
    _metadata = ["required_columns", "geotype", "related"] + gpd.GeoDataFrame._metadata

    def __init__(self, geotype, required_columns=None, related=None, logger=logging, *args, **kwargs):
        # Check type
        if required_columns is None:
            required_columns = []
        elif not isinstance(required_columns, list):
            required_columns = [required_columns]

        # Add required columns to column list
        if "columns" in kwargs.keys():
            kwargs["columns"] += required_columns
        else:
            kwargs["columns"] = required_columns

        super(ExtendedGeoDataFrame, self).__init__(*args, **kwargs)

        self.required_columns = required_columns[:]
        self.geotype = geotype
        self.related = deepcopy(related)

    def copy(self, deep=True):
        """
        Create a copy
        """

        index = self.index.tolist() if not deep else deepcopy(self.index.tolist())
        columns = self.columns.tolist() if not deep else deepcopy(self.columns.tolist())

        edf = ExtendedGeoDataFrame(
            geotype=self.geotype,
            required_columns=[],
            index=index,
            columns=columns,
        )
        edf.required_columns.extend(self.required_columns[:])
        edf.loc[:, :] = self.values if not deep else deepcopy(self.values)

        return edf

    def delete_all(self):
        """
        Empty the dataframe
        """
        if not self.empty:
            self.iloc[:, 0] = np.nan
            self.dropna(inplace=True)

    def read_shp(
        self,
        path: Union[str, Path],
        index_col: str = None,
        column_mapping: dict = None,
        check_columns: bool = True,
        proj_crs=None,
        clip: Union[Polygon, MultiPolygon] = None,
        check_geotype: bool = True,
        id_col: str = "code",
        filter_cols: bool = False,
        filter_rows=None,
    ):
        """
        Import function, extended with type checks. Does not destroy reference to object.
        """
        # Read GeoDataFrame
        gdf = gpd.read_file(path)

        # Only keep required columns
        if filter_cols:
            logger.info("Filtering required column keys")
            gdf.drop(
                columns=gdf.columns[~gdf.columns.isin(self.required_columns)],
                inplace=True,
            )

        # filter out rows on key/value pairs if required
        if filter_rows is not None:
            logger.info("Filter rows using key value pairs")
            filtered = (gdf[list(filter_rows)] == pd.Series(filter_rows)).all(axis=1)
            gdf = gdf[filtered]

        # Drop features without geometry
        total_features = len(gdf)
        missing_features = len(gdf.index[gdf.geometry.isnull()])
        gdf.drop(gdf.index[gdf.geometry.isnull()], inplace=True)  # temporary fix
        logger.debug(
            f"{missing_features} out of {total_features} do not have a geometry"
        )

        # Rename columns:
        if column_mapping is not None:
            gdf.rename(columns=column_mapping, inplace=True)

        # Check number of entries
        if gdf.empty:
            raise IOError("Imported shapefile contains no rows.")

        # Add data to class GeoDataFrame
        self.set_data(
            gdf,
            index_col=index_col,
            check_columns=check_columns,
            check_geotype=check_geotype,
        )

        # Clip if extent is provided
        if clip is not None:
            self.clip(clip)

        # To re-project CRS system to projected CRS, first all empty geometries should be dropped
        if proj_crs is not None:
            self.check_projection(proj_crs)
        else:
            logger.debug("No projected CRS is given in ini-file")

    def set_data(self, gdf, index_col=None, check_columns=True, check_geotype=True):
        if not self.empty:
            self.delete_all()

        # Check columns
        if check_columns:
            self._check_columns(gdf)

        # Copy content
        for col, values in gdf.items():
            if str(values.dtype) == "geometry":
                self.set_geometry(values.values, inplace=True)
            else:
                self[col] = values.values

        if index_col is None:
            self.index = gdf.index
            self.index.name = gdf.index.name

        else:
            self.index = gdf[index_col]
            self.index.name = index_col

        # Check geometry types
        if check_geotype:
            self._check_geotype()

    def _check_columns(self, gdf):
        """
        Check presence of columns in geodataframe
        """
        present_columns = gdf.columns.tolist()
        for column in self.required_columns:
            if column not in present_columns:
                # geopackages don't support very long names. Only give an error if the first 25 characters don't occur, otherwise lengthen the column name
                if column not in present_columns:
                    raise KeyError(
                        'Column "{}" not found. Got {}, Expected at least {}'.format(
                            column,
                            ", ".join(present_columns),
                            ", ".join(self.required_columns),
                        )
                    )

    def _check_geotype(self):
        """
        Check geometry type
        """
        if not all(isinstance(geo, self.geotype) for geo in self.geometry):
            raise TypeError(
                'Geometrytype "{}" required. The input shapefile has geometry type(s) {}.'.format(
                    re.findall("([A-Z].*)'", repr(self.geotype))[0],
                    self.geometry.type.unique().tolist(),
                )
            )

    def show_gpkg(self, gpkg_path: Union[str, Path]):
        if not Path(gpkg_path).exists():
            raise OSError(f'File not found: "{gpkg_path}"')

        if isinstance(gpkg_path, Path):
            gpkg_path = str(gpkg_path)

        layerlist = gpd.list_layers(gpkg_path).name.tolist()
        print(f"Content of gpkg-file {gpkg_path}, containing {len(layerlist)} layers:")
        print(
            "\tINDEX\t|\tNAME                        \t|\tGEOM_TYPE      \t|\t NFEATURES\t|\t   NFIELDS"
        )
        for laynum, layer_name in enumerate(layerlist):
            layer = gpd.read_file(gpkg_path, layer=layer_name)
            if layer.empty:
                logger.warning(f'Layer "{layer_name}" is empty.')
                continue

            nfields = len(layer.columns)
            nfeatures = layer.shape[0]
            
            if isinstance(layer, gpd.GeoDataFrame):
                geom_type = layer.geom_type.iloc[0]
                if geom_type is None:
                    geom_type = "None"                
            else:
                geom_type = "None"
                
            print(
                f"\t{laynum:5d}\t|\t{layer_name:30s}\t|\t{geom_type}\t|\t{nfeatures:10d}\t|\t{nfields:10d}"
            )

    def read_gpkg_layer(
        self,
        gpkg_path: Union[str, Path],
        layer_name: str,
        index_col: str = None,
        groupby_column: str = None,
        order_column: str = None,
        id_col: str = "code",
        column_mapping: dict = None,
        check_columns: bool = True,
        check_geotype: bool = True,
        clip: Union[Polygon, MultiPolygon] = None,
        check_3d: bool = True
    ):
        if not Path(gpkg_path).exists():
            raise OSError(f'File not found: "{gpkg_path}"')

        if isinstance(gpkg_path, Path):
            gpkg_path = str(gpkg_path)

        if layer_name.lower() not in map(str.lower, gpd.list_layers(gpkg_path).name.tolist()):
            raise ValueError(f'Layer "{layer_name}" does not exist in: "{gpkg_path}"')

        layer = gpd.read_file(gpkg_path, layer=layer_name, engine='pyogrio')
        columns = [col.lower() for col in layer.columns]
        layer.columns = columns

        # Get group by columns
        if groupby_column is not None:
            # Check if the group by column is found
            if groupby_column not in columns:
                raise ValueError("Groupby column not found in feature list.")

            if order_column is None or order_column not in columns:
                raise ValueError("Order column not found in feature list.")

            # Check if the geometry is as expected
            geom_types = layer.geometry.geom_type.unique()
            if len(geom_types) != 1 or geom_types[0] != "Point":
                raise ValueError("Can only group Points to LineString")
            if check_3d and np.isnan(layer.geometry.z.to_numpy()).any():
                raise ValueError("All geometries need to have a Z coordinate")


            # Group geometries to lines
            geometries = []
            fields = []
            for groupname, group in layer.groupby(groupby_column, sort=False):
                # Filter branches with too few points
                if len(group) < 2:
                    logger.warning(f'Ignoring {groupby_column} "{groupname}": contains less than two points.')
                    continue

                # Determine relative order of points in profile
                group = group.sort_values(order_column)
                geometries.append(LineString(group.geometry.tolist()))
                fields.append(group.iloc[0, :]) 

        else:
            fields = layer
            geometries = layer.geometry

        # Create geodataframe
        gdf = gpd.GeoDataFrame(fields, columns=columns, geometry=geometries)

        if column_mapping is not None:
            gdf.rename(columns=column_mapping, inplace=True)

        # Enforce a unique index column
        if index_col is not None:
            dupes = gdf[gdf.duplicated(subset=index_col, keep="first")].copy()
            if len(dupes) > 0:
                logger.warning(f"Index column '{index_col}' contains duplicates ({list(gdf[gdf[index_col].duplicated()].code.unique())}). Adding a suffix to make it unique.")
                for dupe_id, group in dupes.groupby(by=index_col, sort=False):
                    gdf.loc[group.index, index_col] = [f"{dupe_id}_{i+1}" for i in range(len(group))]

        # Add data to class GeoDataFrame
        self.set_data(
            gdf,
            index_col=index_col,
            check_columns=check_columns,
            check_geotype=check_geotype,
        )

        if clip is not None:
            self.clip(geometry=clip)

    def clip(self, geometry: Union[Polygon, MultiPolygon]):
        """
        Clip geometry
        """
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise TypeError("Expected geometry of type Polygon or MultiPolygon")

        # Clip if needed
        gdf = self.loc[self.intersects(geometry).values]
        if gdf.empty:
            raise ValueError("Found no features within extent geometry.")

        self.set_data(gdf)

    def check_projection(self, crs_out):
        """
        Check if reprojection is required
        """
        if crs_out != self.crs:
            self.to_crs(crs_out, inplace=True)
        else:
            logger.info("OSM data has same projection as projected crs in ini-file")

    def branch_to_prof(
        self, offset=0.0, vertex_end=False, rename_col=None, prefix="", suffix=""
    ):
        """Create profiles on branches from branch data"""

        gdf_out = self.copy()

        # interpolate over feature geometries
        if vertex_end:
            chainage = self.length - offset
            p = self.interpolate(chainage)
        else:
            chainage = offset
            p = self.interpolate(chainage)
        gdf_out.geometry = p
        gdf_out["offset"] = chainage

        if rename_col is not None:
            try:
                gdf_out["branch_id"] = gdf_out[rename_col]
                gdf_out[rename_col] = [
                    f"{prefix}{g[1][rename_col]}{suffix}" for g in self.iterrows()
                ]
            except Exception:
                raise ValueError(f"Column rename with '{rename_col}' did not succeed.")

        return gdf_out

    def merge_columns(self, col1, col2, rename_col):
        """merge columns"""

        if col1 or col2 in self.columns.values:
            try:
                self[rename_col] = self[col1] + self[col2]
            except Exception:
                raise ValueError(
                    f"Merge of two profile columns'{col1}' and '{col2}' did not succeed."
                )

    def snap_to_branch(self, branches, snap_method, maxdist=5):
        """Snap the geometries to the branch"""
        spatial.find_nearest_branch(
            branches=branches, geometries=self, method=snap_method, maxdist=maxdist
        )


class ExtendedDataFrame(pd.DataFrame):
    _metadata = ["required_columns"] + pd.DataFrame._metadata

    def __init__(self, required_columns=None, *args, **kwargs):
        super(ExtendedDataFrame, self).__init__(*args, **kwargs)

        if required_columns is None:
            required_columns = []

        self.required_columns = (
            required_columns[:]
            if isinstance(required_columns, list)
            else [required_columns]
        )

    def delete_all(self):
        """
        Empty the dataframe
        """
        if not self.empty:
            self.iloc[:, 0] = np.nan
            self.dropna(inplace=True)

    def set_data(self, df, index_col):
        if not self.empty:
            self.delete_all()

        # Copy content
        for col, values in df.items():
            self[col] = values.values

        if index_col is None:
            self.index = df.index
            self.index.name = df.index.name

        else:
            self.index = df[index_col]
            self.index.name = index_col

        # Check columns and types
        self._check_columns()

    def add_data(self, df):
        if not np.in1d(df.columns, self.columns).all():
            raise KeyError(
                "The new df contains columns that are not present in the current df."
            )

        # Concatenate data
        current = pd.DataFrame(self.values, index=self.index, columns=self.columns)
        newdf = pd.concat([current, df], ignore_index=False, sort=False)

        # Empty current df
        self.delete_all()

        # Add values again
        self.set_data(newdf, index_col=self.index.name)

    def _check_columns(self):
        """
        Check presence of columns in geodataframe
        """
        present_columns = self.columns.tolist()
        for i, column in enumerate(self.required_columns):
            if column not in present_columns:
                raise KeyError(
                    'Column "{}" not found. Got {}, Expected at least {}'.format(
                        column,
                        ", ".join(present_columns),
                        ", ".join(self.required_columns),
                    )
                )

    def read_gpkg_layer(
        self,
        gpkg_path: Union[str, Path],
        layer_name: str = None,
        column_mapping: dict = None,
        index_col: str = None,
    ):
        """
        Read GML file to GeoDataFrame.

        This function has the option to group Points into LineStrings. To do so,
        specify the groupby column (which the set has in common) and the order_column,
        the column which indicates the order of the grouping.

        A mask file can be specified to clip the selection.

        Parameters
        ----------
        gml_path : str
            Path to the GML file
        layer_name: str
            Layer name of the desired layer in the package
        index_col : str
            Optional, column to be set as index
        """

        if not Path(gpkg_path).exists():
            raise OSError(f'File not found: "{gpkg_path}"')

        if isinstance(gpkg_path, Path):
            gpkg_path = str(gpkg_path)

        if layer_name.lower() not in map(str.lower, gpd.list_layers(gpkg_path).name.tolist()):
            raise ValueError(f'Layer "{layer_name}" does not exist in: "{gpkg_path}"')

        layer = gpd.read_file(gpkg_path, layer=layer_name, engine='pyogrio')
        columns = [col.lower() for col in layer.columns]
        layer.columns = columns
        if 'geometry' in layer.columns:
            layer.drop("geometry", axis=1, inplace=True)

        df = layer
        if column_mapping is not None:
            df.rename(columns=column_mapping, inplace=True)

        # Add data to class GeoDataFrame
        self.set_data(df, index_col=index_col)
