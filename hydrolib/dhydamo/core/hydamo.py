import logging
import numbers
from datetime import datetime, timezone
from pathlib import Path

from dask.dot import name
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pydantic.v1 import ConfigDict, validate_arguments
from rasterstats import zonal_stats
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from tqdm.auto import tqdm
from netCDF4 import Dataset, chartostring

from hydrolib.dhydamo.io.netcdfreader import time_index
from hydrolib import dhydamo
from hydrolib.dhydamo.converters.hydamo2df import (
    CrossSectionsIO,
    ExternalForcingsIO,
    RoughnessVariant,
    StorageNodesIO,
    StructuresIO,
)
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.geometry import common
from hydrolib.dhydamo.geometry.spatial import find_nearest_branch
from hydrolib.dhydamo.io.common import ExtendedDataFrame, ExtendedGeoDataFrame

logger = logging.getLogger(__name__)


class HyDAMO:
    """Main data structure for both the HyDAMO input data and the intermediate dataframes. Contains subclasses
    for network, structures, cross sections, observation points, storage nodes and external forcings.
    """

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, extent_file: Path | str = None) -> None:
        """Initiate subclasses and IO-methods

        Args:
            extent_file (Union[Path, str], optional): model extent, use to clip datsaets. Defaults to None.
        """
        self.network = Network(self)

        self.structures = Structures(self)

        self.crosssections = CrossSections(self)  # Add all items

        self.observationpoints = ObservationPoints(self)

        self.external_forcings = ExternalForcings(self)

        self.storagenodes = StorageNodes(self)

        self.roughness_mapping = {
            "Chezy": "Chezy",
            "Manning": "Manning",
            "StricklerKn": "StricklerNikuradse",
            "StricklerKs": "Strickler",
            "White Colebrook": "WhiteColebrook",
            "Bos en Bijkerk": "deBosBijkerk",
            "Onbekend": "Strickler",
            "Overig": "Strickler",
        }

        # Dictionary for roughness definitions
        self.roughness_definitions = {}

        # Read geometry to clip data
        if extent_file is not None:
            self.clipgeo = gpd.read_file(extent_file).union_all()
        else:
            self.clipgeo = None

        # versioning info
        self.version = {
            "number": dhydamo.__version__,
            "date": datetime.strftime(datetime.now(timezone.utc), "%Y-%m-%dT%H:%M:%S.%fZ"),
            "dimr_version": "Deltares, DIMR_EXE Version 2.00.00.140737 (Win64) (Win64)",
            "suite_version": "D-HYDRO Suite 2024.03 1D2D,",
        }

        # Create standard dataframe for network, crosssections, orifices, weirs
        self.branches = ExtendedGeoDataFrame(
            geotype=LineString,
            required_columns=[
                "code",
                "geometry"
            ],
            related=None
        )

        self.profile = ExtendedGeoDataFrame(
            geotype=LineString,
            required_columns=["code", "geometry", "globalid", "profiellijnid"],
            related={
                "profile_roughness": {
                    "via": "globalid",
                    "on": "profielpuntid",
                    "coupled_to": None
                },
                "profile_line": {
                    "via": "profiellijnid",
                    "on": "globalid",
                    "coupled_to": {
                        "profile_group": {
                            "via": "profielgroepid",
                            "on": "globalid",
                            "coupled_to": None
                        }
                    }
                }
            }
        )
        self.profile_roughness = ExtendedDataFrame(
            required_columns=["profielpuntid"]
        )

        self.profile_line = ExtendedGeoDataFrame(
            geotype=LineString,
            required_columns=["globalid", "profielgroepid"],
            related={
                "profile_group": {
                    "via": "profielgroepid",
                    "on": "globalid",
                    "coupled_to": None
                },
                "profile": {
                    "via": "globalid",
                    "on": "profiellijnid",
                    "coupled_to": {
                        "profile_roughness": {
                            "via": "globalid",
                            "on": "profielpuntid",
                            "coupled_to": None
                        }
                    }
                }
            }
        )

        self.profile_group = ExtendedDataFrame(
            required_columns=[]
        )

        self.param_profile = ExtendedDataFrame(
            required_columns=["globalid", "normgeparamprofielid", "hydroobjectid"]
        )

        self.param_profile_values = ExtendedDataFrame(
            required_columns=[
                "normgeparamprofielid",
                "soortparameter",
                "waarde",
                "ruwheidlaag",
                "ruwheidhoog",
                "typeruwheid",
            ]
        )

        # Weirs
        self.weirs = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=[
                "code",
                "geometry",
                "globalid",                
                "afvoercoefficient",
            ],
            related={
                "opening": {
                    "via": "globalid",
                    "on": "stuwid",
                    "coupled_to": {
                        "management_device": {
                            "via": "globalid",
                            "on": "kunstwerkopeningid",
                            "coupled_to": None
                        }
                    }
                }
            }
        )

        # opening
        self.opening = ExtendedDataFrame(
            required_columns=[            
                "globalid",
                "laagstedoorstroombreedte",
                "laagstedoorstroomhoogte",
                "afvoercoefficient",
            ]
        )

        # opening
        self.closing_device = ExtendedDataFrame(
            required_columns=["code"]
        )

        # opening
        self.management_device = ExtendedDataFrame(
            required_columns=["code", "overlaatonderlaat"]
        )

        # Bridges
        self.bridges = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=[
                "code",
                "globalid",
                "geometry",
                "lengte",
                "intreeverlies",
                "uittreeverlies",
                "ruwheid",
                "typeruwheid",
            ],
            related=None
        )

        # Culverts
        self.culverts = ExtendedGeoDataFrame(
            geotype=LineString,
            required_columns=[
                "code",
                "geometry",
                "lengte",
                "hoogteopening",
                "breedteopening",
                "hoogtebinnenonderkantbene",
                "hoogtebinnenonderkantbov",
                "vormkoker",
                "intreeverlies",
                "uittreeverlies",
                "typeruwheid",
                "ruwheid",
            ],
            related={
                "management_device": {
                    "via": "globalid",
                    "on": "duikersifonhevelid",
                    "coupled_to": None
                }
            }
        )

        # Gemalen
        self.pumpstations = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=[
                "code",
                "globalid",
                "geometry",
            ],
            related={
                "pumps": {
                    "via": "globalid",
                    "on": "gemaalid",
                    "coupled_to": {
                        "management": {
                            "via": "globalid",
                            "on": "pompid",
                            "coupled_to": None
                        }
                    }
                }
            }
        )
        self.pumps = ExtendedDataFrame(
            required_columns=["code", "globalid", "gemaalid", "maximalecapaciteit"]
        )
        self.management = ExtendedDataFrame(
            required_columns=["code", "globalid"]
        )

        # Hydraulische randvoorwaarden
        self.boundary_conditions = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=["code", "typerandvoorwaarde", "geometry"],
            related=None
        )

        # RR catchments
        self.catchments = ExtendedGeoDataFrame(
            geotype=Polygon | MultiPolygon,
            required_columns=["code", "geometry", "globalid", "lateraleknoopid"],
            related={
                "laterals": {
                    "via": "lateraleknoopid",
                    "on": "globalid",
                    "coupled_to": None
                }
            }
        )

        # Laterals
        self.laterals = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=["code", "geometry", "globalid"],
            related={
                "catchments": {
                    "via": "globalid",
                    "on": "lateraleknoopid",
                    "coupled_to": None
                }
            }
        )

        # RR overflows
        self.overflows = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=["code", "geometry", "codegerelateerdobject", "fractie"],
            related={
                "sewer_areas": {
                    "via": "codegerelateerdobject",
                    "on": "code",
                    "coupled_to": None
                }
            }
        )

        # RR sewer areas
        self.sewer_areas = ExtendedGeoDataFrame(
            geotype=Polygon,
            required_columns=["code", "geometry"],
            related={
                "overflows": {
                    "via": "code",
                    "on": "codegerelateerdobject",
                    "coupled_to": None
                }
            }
        )

        # RR overflows
        self.overflows = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=["code", "geometry", "codegerelateerdobject", "fractie"],
            related={
                "sewer_areas": {
                    "via": "codegerelateerdobject",
                    "on": "code",
                    "coupled_to": None
                }
            }
        )

        # RR greenhouse areas
        self.greenhouse_areas = ExtendedGeoDataFrame(
            geotype=Polygon,
            required_columns=["code", "geometry"],
            related={
                "greenhouse_laterals": {
                    "via": "code",
                    "on": "codegerelateerdobject",
                    "coupled_to": None
                }
            }
        )

         # RR overflows
        self.greenhouse_laterals = ExtendedGeoDataFrame(
            geotype=Point,
            required_columns=["code", "geometry", "codegerelateerdobject"],
            related={
                "greenhouse_areas": {
                    "via": "codegerelateerdobject",
                    "on": "code",
                    "coupled_to": None
                }
            }
        )

        # RR overflows
        self.storage_areas = ExtendedGeoDataFrame(
            geotype=Polygon,
            required_columns=["code", "geometry"],            
        )

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def list_to_str(self, lst: list | np.ndarray) -> str:
        """Converts list to string

        Args:
            lst (list): The list

        Returns:
            str: The output string
        """
        if len(lst) == 1:
            string = str(lst)
        else:
            string = " ".join([f"{number:6.3f}" for number in lst])
        return string

    @validate_arguments
    def dict_to_dataframe(self, dictionary: dict) -> pd.DataFrame:
        """Converts a dictionary to dataframe, using index as rows

        Args:
            dictionary (dict): Input dictionary

        Returns:
            pd.DataFrame: Output dataframe
        """

        return pd.DataFrame.from_dict(dictionary, orient="index")

    def snap_to_branch_and_drop(self, extendedgdf, branches, snap_method: str, maxdist=5, drop_related=True):
        """Snap the geometries to the branch and drop loose objects"""

        # Snap the extended geodataframe to branches
        extendedgdf.snap_to_branch(branches, snap_method, maxdist=maxdist)

        # Determine which labels need to be drop for the first object based on
        # nan values for branch_offset.
        drop_idx = extendedgdf[pd.isna(extendedgdf.branch_offset)].index.to_numpy()
        drop_list = [(extendedgdf, drop_idx)]
        logger.info("dropping objects with indices: %s", drop_idx)

        # Find out which labels need to be dropped from related objects
        if drop_related and extendedgdf.related is not None:
            for target_str, relation in extendedgdf.related.items():
                self._recursive_drop_related(drop_list, extendedgdf, drop_idx, target_str, **relation)

        # Drop the relevant rows with the list of labels
        for source, drop_idx in drop_list:
            source.drop(labels=drop_idx, inplace=True)

    def _recursive_drop_related(self, drop_list, source, drop_idx, target_str, via, on, coupled_to):
        target = getattr(self, target_str)
        drop_related = source.loc[drop_idx, via].to_numpy()
        drop_idx = target[target[on].isin(drop_related)].index.to_numpy()
        drop_list.append((target, drop_idx))
        logger.info(
            "  - dropping objects from '%s' with indices: %s",
            target_str,
            drop_idx,
        )

        if coupled_to is not None:
            for next_target_str, next_relation in coupled_to.items():
                return self._recursive_drop_related(drop_list, target, drop_idx, next_target_str, **next_relation)
            
    def create_laterals(self, qspec_file=None):            
        ## Specifieke afvoeren inlezen
        if qspec_file is not None:
            rr = DRRModel()
            qspec, affine = rr.read_raster(qspec_file, static=True)
            fill_value_specifieke_afvoeren = 0
            qspec = np.where(qspec<0, fill_value_specifieke_afvoeren, qspec)

            ## Afvoer per gebied bepalen met zonal stats
            afvoer_per_gebied = zonal_stats(self.catchments, qspec, affine=affine, stats="mean", all_touched=True, nodata=-2147483647)   
        
        self.laterals = self.catchments.copy()

        for num, cat in enumerate(self.catchments.itertuples()):
            ## Koppelen van afvoeren met afwateringsgebieden
            if qspec_file is not None:
                q = afvoer_per_gebied[num]['mean'] # mm/d
            else:
                q = np.nan

            area = cat.geometry.area           # m2
            q_m3s = q*area/(1000*86400)        # van mm/d naar m3/s
            self.laterals.at[cat.Index, 'afvoer'] = q_m3s          
            knoopid = cat.lateraleknoopid
        
        
            ## Afstand tussen watergang en afwateringsgebied. Als watergang in afwateringsgebied ligt is deze afstand 0.
            distances = self.branches.distance(cat.geometry)
            ## EÃ©n of meer watergangen in afwateringsgebied:
            if sum(distances == 0) > 0:
                ## Watergangen intersecten met afwateringsgebied
                #wg = [watergang for watergang in watergangen.itertuples() if watergang.intersects(cat.geometry]
                selectie = self.branches.intersection(cat.geometry)
                ## Index van de watergang waarop je wilt snappen. Dit is de watergang die in het gebied ligt Ã©n het dichtst bij de centroid van het afwateringsgebied is.
                index_watergang = selectie.distance(cat.geometry.centroid).idxmin() 
                ## Combineer de ge intersecte watergangen met de index om het stuk watergang te vinden waarop mag worden gesnapt.
                watergang = selectie.at[index_watergang]
            ## No intersects 
            else:
                ## Vind de watergang die het dichtst bij de centroid van het gebied is
                index_watergang = self.branches.distance(cat.geometry.centroid).idxmin()
                ## Selecteer deze watergang om op te mogen snappen
                watergang = self.branches.at[index_watergang, 'geometry']
        
            ## Snap de centroid van het afwateringsgebied op de geselecteerde watergang
            lateral = watergang.interpolate(watergang.project(cat.geometry.centroid))
            ## Schrijf de snap-locatie weg in de geodataframe
            self.laterals.at[cat.Index, 'geometry'] = lateral
            self.laterals.at[cat.Index, 'globalid'] = knoopid
            self.laterals.at[cat.Index, 'code']= f'lat_{cat.code}'

class Network:
    def __init__(self, hydamo: HyDAMO) -> None:
        """Set class variables

        Args:
            hydamo (HyDAMO): HyDAMO object containign all input data
        """
        self.hydamo = hydamo

        # Mesh 1d offsets
        self.offsets = {}

    @validate_arguments
    def set_branch_order(self, branchids: list, idx: int = None) -> None:
        """
        Group branch ids so that the cross sections are
        interpolated along the branch.

        Parameters
        ----------
        branchids : list
            List of branches to group
        idx : int
            Order number with which to update a branch
        """
        # Get the ids (integers) of the branch names given by the user
        branchidx = np.isin(self.mesh1d.description1d["network_branch_ids"], branchids)
        # Get current order
        branchorder = self.mesh1d.get_values("nbranchorder", as_array=True)
        # Update
        if idx is None:
            branchorder[branchidx] = branchorder.max() + 1
        else:
            if not isinstance(idx, int):
                raise TypeError("Expected integer.")
            branchorder[branchidx] = idx
        # Save
        self.mesh1d.set_values("nbranchorder", branchorder)

    def set_branch_interpolation_modelwide(self) -> None:
        """
        Set cross-section interpolation over nodes on all branches model-wide. I

        Note:
            - Interpolation will be set using branchorder property.
            - Branches are grouped between bifurcations where multiple branches meet.
            - No interpolation is applied over these type of bifurcations.
            - No branch order is set on branch groups consisting of 1 branch.
        """
        self.get_grouped_branches()
        for group in self.branch_groups.values():
            if len(group) > 1:
                self.set_branch_order(group)

    def make_nodes_to_branch_map(self) -> None:
        """Map nodes connected to each branch"""
        # Note: first node is upstream, second node is downstream
        self.nodes_to_branch_map = {
            b: [self.mesh1d.description1d["network_node_ids"][_idx - 1] for _idx in idx]
            for b, idx in zip(
                self.mesh1d.description1d["network_branch_ids"],
                self.mesh1d.get_values("nedge_nodes", as_array=True),
            )
        }

    def make_branches_to_node_map(self) -> None:
        """Map branches connected to each node"""
        self.make_nodes_to_branch_map()
        self.branches_to_node_map = {
            n: [k for k, v in self.nodes_to_branch_map.items() if n in v]
            for n in self.mesh1d.description1d["network_node_ids"]
        }

    @validate_arguments
    def generate_nodes_with_bedlevels(
        self,
        resolve_at_bifurcation_method: str = "min",
        return_reversed_branches: bool = False,
    ):
        """
        Generate nodes with upstream and downstream bedlevels derived from set cross-sections on branch. It takes into
        account whether or not branch order is specified (so interpolation over nodes is set).

        Nodes in between cross-sections on same branch/branch-group are linearly interpolated. Outside are extrapolated
        constant (e.g. end points of branch).

        Branch groups which include branch with reversed direction compared to whole group will be taken into account.
        Use return_reversed_branches=True to return that information

        Specify with resolve_at_bifurcation_method how to resolve bedlevel at bifurcation of more than 2 branches using
        options 'min' (minimum), 'max' (maximum), 'mean' (average).
        """
        assert resolve_at_bifurcation_method in ["min", "max", "mean"], (
            "Incorrect value for "
            "'resolve_at_bifurcation_method' supplied. "
            "Either use 'min', 'max' or 'mean'"
        )
        bedlevels_crs_branches = self.hydamo.crosssections.get_bottom_levels()
        branch_order = self.mesh1d.get_values("nbranchorder", as_array=True)
        self.make_branches_to_node_map()
        self.make_nodes_to_branch_map()
        nodes_dict = {
            n: {"up": [], "down": []} for n in self.branches_to_node_map.keys()
        }
        reserved_branches = []
        for order, (branch, nodes) in tqdm(
            zip(branch_order, self.nodes_to_branch_map.items()),
            total=len(branch_order),
            desc="Getting bedlevels",
        ):
            if order == -1:
                # No branch order so just get upstream and downstream levels
                branch_length = self.branches.loc[branch, "geometry"].length
                subset = bedlevels_crs_branches.loc[
                    bedlevels_crs_branches["branchid"] == branch
                ]
                if subset.empty:
                    continue  # if this happens, crs is not defined. This can be a problem.
                nodes_dict[nodes[0]]["up"].append(
                    np.interp(0.0, subset["chainage"], subset["minz"])
                )
                nodes_dict[nodes[1]]["down"].append(
                    np.interp(branch_length, subset["chainage"], subset["minz"])
                )
            else:
                # In case of branch order, first collect all branches and set in them in order of up- to downstream
                all_branches = [
                    self.mesh1d.description1d["network_branch_ids"][i]
                    for i in np.argwhere(order == branch_order).ravel()
                ]
                all_nodes = [self.nodes_to_branch_map[b] for b in all_branches]
                # First check if any of the branches has a bedlevel from a cross-section profile otherwise skip
                check = all(
                    [
                        bedlevels_crs_branches.loc[
                            bedlevels_crs_branches["branchid"] == b
                        ].empty
                        for b in all_branches
                    ]
                )
                if check:
                    continue  # if this happens, cross-section is not defined. This can be a problem.

                # Check if every branch is from up to down direction. Otherwise fix by reversing
                n = 0
                n_length = len(all_nodes)
                direction = list(np.ones(n_length))
                max_tries = 0
                while n < n_length:
                    up = (
                        np.count_nonzero(
                            all_nodes[n][0] == np.array([x[0] for x in all_nodes])
                        )
                        == 1
                    )
                    down = (
                        np.count_nonzero(
                            all_nodes[n][1] == np.array([x[1] for x in all_nodes])
                        )
                        == 1
                    )
                    if (not up) or (not down):
                        # Reverse
                        all_nodes[n] = list(np.flip(all_nodes[n]))
                        direction[n] = direction[n] * -1
                    n += 1
                    # Check if indeed everything is now in proper direction. Otherwise try again
                    if n == n_length:
                        up = all(
                            [
                                np.count_nonzero(
                                    node[0] == np.array([x[0] for x in all_nodes])
                                )
                                == 1
                                for node in all_nodes
                            ]
                        )
                        down = all(
                            [
                                np.count_nonzero(
                                    node[1] == np.array([x[1] for x in all_nodes])
                                )
                                == 1
                                for node in all_nodes
                            ]
                        )
                        if (not up) or (not down):
                            n = 0
                            max_tries += 1
                    if max_tries > 500:
                        logger.warning(
                            "Can't fix correct directions branch groups %s",
                            all_branches,
                        )
                        break

                # Add reserved branches to return
                reserved_branches.extend(
                    [b for b, d in zip(all_branches, direction) if d == -1]
                )

                # Get most upstream node. Otherwise just pick i_upstream = 0 as starting point
                i_upstream = [
                    i
                    for i, n in enumerate([x[0] for x in all_nodes])
                    if n not in [x[1] for x in all_nodes]
                ]
                if len(i_upstream) == 1:
                    i_upstream = i_upstream[0]
                else:
                    # It could be that branch order group forms a ring. In this case check first which node has more
                    # than 2 branches (bifurcation) or just 1 branch (boundary) connected.
                    i_upstream = [
                        i
                        for i, n in enumerate([x[0] for x in all_nodes])
                        if (len(self.branches_to_node_map[n]) > 2)
                        or (len(self.branches_to_node_map[n]) == 1)
                    ]
                    if len(i_upstream) == 1:
                        i_upstream = i_upstream[0]
                    else:
                        raise ValueError(
                            f"Something is not right with the branch order group {all_branches}"
                        )

                # Now put branch list in correct order
                all_branches_sorted = []
                all_nodes_sorted = []
                direction_sorted = []
                for _ in range(len(all_branches)):
                    all_branches_sorted.append(all_branches[i_upstream])
                    all_nodes_sorted.append(all_nodes[i_upstream])
                    direction_sorted.append(direction[i_upstream])
                    try:
                        i_upstream = [
                            i
                            for i, n in enumerate([x[0] for x in all_nodes])
                            if [x[1] for x in all_nodes][i_upstream] == n
                        ][0]
                    except IndexError:
                        break
                # Stitch chainages and bedlevels together
                chainage, bedlevel = [], []
                branch_length = 0
                for b, d in zip(all_branches_sorted, direction_sorted):
                    subset = bedlevels_crs_branches.loc[
                        bedlevels_crs_branches["branchid"] == b
                    ]
                    chain, bed = subset["chainage"], subset["minz"]
                    # Reverse chainage and bedlevel arrays
                    if d == -1:
                        chain = np.flip(chain)
                        bed = np.flip(bed)
                    chainage.extend(chain + branch_length)
                    bedlevel.extend(bed)
                    branch_length = self.branches.loc[b, "geometry"].length
                # Get chainage of up- and downstream node of loop branch within the overall branch
                if len(all_branches_sorted) == 1:
                    up_node_chainage = 0
                    down_node_chainage = self.branches.loc[
                        all_branches_sorted[0], "geometry"
                    ].length
                else:
                    i = np.argmax(
                        [
                            1 if ((nodes == n) or (list(np.flip(nodes)) == n)) else 0
                            for n in all_nodes_sorted
                        ]
                    )
                    up_node_chainage = sum(
                        [0]
                        + [
                            self.branches.loc[b, "geometry"].length
                            for b, n in zip(
                                all_branches_sorted[:-1], all_nodes_sorted[:-1]
                            )
                        ][: i + 1]
                    )
                    down_node_chainage = sum(
                        [
                            self.branches.loc[b, "geometry"].length
                            for b, n in zip(all_branches_sorted, all_nodes_sorted)
                        ][: i + 1]
                    )
                # Finally interpolate
                nodes_dict[nodes[0]]["up"].append(
                    np.interp(up_node_chainage, chainage, bedlevel)
                )
                nodes_dict[nodes[1]]["down"].append(
                    np.interp(down_node_chainage, chainage, bedlevel)
                )

        # Summarize everything and save
        nodes = list(nodes_dict.keys())
        node_geom = [
            Point(x, y)
            for x, y in zip(
                self.mesh1d.get_values("nnodex"), self.mesh1d.get_values("nnodey")
            )
        ]
        if resolve_at_bifurcation_method == "min":
            upstream_bedlevel = [
                np.min(v["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.min(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        elif resolve_at_bifurcation_method == "max":
            upstream_bedlevel = [
                np.max(v["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.max(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        elif resolve_at_bifurcation_method == "mean":
            upstream_bedlevel = [
                np.average(["up"]) if len(v["up"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
            downstream_bedlevel = [
                np.average(v["down"]) if len(v["down"]) > 0 else np.nan
                for v in nodes_dict.values()
            ]
        else:
            raise NotImplementedError

        self.nodes = gpd.GeoDataFrame(
            index=nodes_dict.keys(),
            data={
                "code": nodes_dict.keys(),
                "upstream_bedlevel": upstream_bedlevel,
                "downstream_bedlevel": downstream_bedlevel,
            },
            geometry=node_geom,
        )

        if return_reversed_branches:
            return list(np.unique(reserved_branches))

    def get_grouped_branches(self) -> None:
        """
        Get grouped branch ids to use in set_branch_order function
        """
        # Get all network data
        branch_ids = self.mesh1d.description1d["network_branch_ids"]
        # node_ids = self.mesh1d.description1d["network_node_ids"]
        # branch_edge_nodes_idx = self.mesh1d.get_values("nedge_nodes", as_array=True)
        # Collect all node ids per branch and all branches per node id
        self.make_nodes_to_branch_map()
        self.make_branches_to_node_map()

        branch_ids_checked = []
        groups = {0: []}
        for branch_id in branch_ids:
            if branch_id in branch_ids_checked:
                continue

            connected_nodes = self.nodes_to_branch_map[branch_id]  # get connected nodes
            for n in connected_nodes:
                b = self.branches_to_node_map[n]  # get connected branches
                if len(b) > 2:
                    continue  # in this case there's a bifurcation so skip
                elif len(b) == 1:
                    groups[list(groups.keys())[-1] + 1] = b  # b is already a list
                    branch_ids_checked.extend(b)
                    continue  # in this case the branch is not connected to other branches. make separate group
                else:
                    # remove branch from connected_branches because we are not interested in it
                    b = [_b for _b in b if _b != branch_id][0]
                    if b in branch_ids_checked:
                        # connected branch is already added to a group, so this means that branch should be added to
                        # that group
                        groups[[k for k, v in groups.items() if b in v][0]].append(
                            branch_id
                        )
                        branch_ids_checked.extend([branch_id])
                    else:
                        groups[list(groups.keys())[-1] + 1] = [
                            branch_id
                        ]  # otherwise add to group
                        branch_ids_checked.extend([branch_id])
                branch_ids_checked = list(np.unique(branch_ids_checked))
        groups.pop(0)  # remove the 0th group because empty

        # The branches are grouped but not fully connected (although routine above should do the trick in theory).
        # Try merging groups if a branch is in multiple groups.
        # In that case we know that branch groups are connected to eachother and is in fact a bigger group
        for b in branch_ids:
            _groups = groups.copy()  # make copy to apply changes on
            _k = -1  # default index
            # Loop over groups
            for k, v in groups.items():
                if b in v:  # if branch in grouped branches
                    if _k == -1:
                        _k = k  # set index to first group found
                    else:
                        _groups[_k].extend(
                            v
                        )  # otherwise add group to first found group
                        _groups[_k] = list(
                            np.unique(_groups[_k])
                        )  # remove duplicates due to add
                        _groups.pop(k)  # and remove group from groups
            groups = _groups.copy()  # copy changed dict over original
        # One pass over all branches should be sufficient to group everything together. Otherwise raise error
        if (
            max(
                [
                    sum([1 if b in v else 0 for k, v in groups.items()])
                    for b in branch_ids
                ]
            )
            > 1
        ):
            raise ValueError(
                f"Still branches contained in multiple groups. Maximum number of groups where this "
                f"happens: {max([sum([1 if b in v else 0 for k, v in groups.items()]) for b in branch_ids])}"
            )

        # save
        self.branch_groups = groups.copy()

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def get_node_idx_offset(
        self, branch_id: str, pt: shapely.geometry.Point, nnodes: int = 1
    ) -> tuple:
        """
        Get the index and offset of a node on a 1d branch.
        The nearest node is looked for.
        """

        # Project the point on the branch
        dist = self.schematised[branch_id].project(pt)

        # Get the branch data from the networkdata
        branchidx = (
            self.mesh1d.description1d["network_branch_ids"].index(
                self.str2chars(branch_id, self.idstrlength)
            )
            + 1
        )
        pt_branch_id = self.mesh1d.get_values("branchidx", as_array=True)
        idx = np.nonzero(pt_branch_id == branchidx)

        # Find nearest offset
        offsets = self.mesh1d.get_values("branchoffset", as_array=True)[idx]
        isorted = np.argsort(np.absolute(offsets - dist))
        isorted = isorted[: min(nnodes, len(isorted))]

        # Get the offset
        offset = [offsets[imin] for imin in isorted]
        # Get the id of the node
        node_id = [idx[0][imin] + 1 for imin in isorted]

        return node_id, offset


class CrossSections:
    def __init__(self, hydamo: HyDAMO) -> None:
        """Initiate class variables

        Args:
            hydamo (HyDAMO): input data structure
        """
        self.hydamo = hydamo
        self.crosssections = []
        self.default_definition = None
        self.default_definition_shift = 0.0
        self.default_location = ""

        self.crosssection_loc = {}
        self.crosssection_def = {}

        self.get_roughnessname = self.get_roughness_description

        self.convert = CrossSectionsIO(self)

    def get_roughness_description(self, roughnesstype, value):
        if np.isnan(float(value)):
            raise ValueError("Roughness value should not be NaN.")

        # map HyDAMO definition to D-Hydro definition
        roughnesstype = self.hydamo.roughness_mapping[roughnesstype]

        # Get name
        name = f"{roughnesstype}_{float(value)}"

        # Check if the description is already known
        if name.lower() in map(str.lower, self.hydamo.roughness_definitions.keys()):
            return name

        # Add to dict
        self.hydamo.roughness_definitions[name] = {
            "frictionid": name,
            "frictiontype": roughnesstype,
            "frictionvalue": value,
        }

        return name

    def set_default_definition(self, definition, shift=0.0):
        if definition not in self.crosssection_def.keys():
            raise KeyError(f'Cross section definition "{definition}" not found."')

        self.default_definition = definition
        self.default_definition_shift = shift

    def set_default_locations(self, locations):
        """
        Add default profile locations
        """
        self.default_locations = locations

    def add_yz_definition(
        self, yz=None, thalweg=None, roughnesstype=None, roughnessvalue=None, name=None
    ):
        """
        Add xyz crosssection

        Parameters
        ----------
        code : str
            Id of cross section
        branch : str
            Name of branch
        offset : float
            Position of cross section along branch. If not given, the position is determined
            from the branches in the network. These should thus be given in this case.
        crds : np.array
            Nx2 array with y, z coordinates
        """

        # get coordinates
        length, z = yz.T
        if name is None:
            name = f"yz_{yz}:08d"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "yz",
            "thalweg": np.round(thalweg, decimals=3),
            "yzcount": len(z),
            "ycoordinates": self.hydamo.list_to_str(length),
            "zcoordinates": self.hydamo.list_to_str(z),
            "sectioncount": 1,
            "frictionids": roughnessname,
            "frictionpositions": self.hydamo.list_to_str([length[0], length[-1]]),
        }

        return name

    def add_circle_definition(self, diameter, roughnesstype, roughnessvalue, name=None):
        """
        Add circle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """
        # Get name if not given
        if name is None:
            name = f"circ_d{diameter:.3f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "circle",
            "thalweg": 0.0,
            "diameter": diameter,
            "frictionid": roughnessname,
        }

        return name

    def add_rectangle_definition(
        self, height, width, closed, roughnesstype, roughnessvalue, name=None
    ):
        """
        Add rectangle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """
        # Get name if not given
        if name is None:
            name = f"rect_h{height:.3f}_w{width:.3f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "rectangle",
            "thalweg": 0.0,
            "height": height,
            "width": width,
            "closed": int(closed),
            "frictionid": roughnessname,
        }

        return name

    def add_trapezium_definition(
        self,
        slope,
        maximumflowwidth,
        bottomwidth,
        closed,
        roughnesstype,
        roughnessvalue,
        bottomlevel=None,
        name=None,
    ):
        """
        Add rectangle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """
        # Get name if not given
        if name is None:
            name = f"trapz_s{slope:.1f}_bw{bottomwidth:.1f}_bw{maximumflowwidth:.1f}"

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        if bottomlevel is None:
            bottomlevel = 0.0

        if not closed:
            dh = (maximumflowwidth - bottomwidth) / (2.0 * slope)   # hoogte tot insteekhoogte
            levels = f"{bottomlevel} {bottomlevel + dh:.3f}"
            flowwidths = f"{bottomwidth:.2f} {maximumflowwidth:.2f}"
        else:
            levels = f"0 {((maximumflowwidth - bottomwidth)/2.0) / slope:.2f}"
            flowwidths = f"{bottomwidth:.2f} {maximumflowwidth:.2f}"

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "zw",
            "thalweg": 0.0,
            "numlevels": 2,
            "levels": levels,
            "flowwidths": flowwidths,
            "totalwidths": flowwidths,
            "frictionid": roughnessname,
        }

        return name

    def add_zw_definition(
        self,
        numLevels,
        levels,
        flowWidths,
        totalWidths,
        roughnesstype,
        roughnessvalue,
        name=None,
    ):
        """
        Add zw cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """
        # Get name if not given
        if name is None:
            name = (
                f'zw_h{levels.replace(" ","_"):.1f}_w{flowWidths.replace(" ","_"):.1f}'
            )

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            "id": name,
            "type": "zw",
            "thalweg": 0.0,
            "numlevels": int(numLevels),
            "levels": levels,
            "flowwidths": flowWidths,
            "totalwidths": totalWidths,
            "frictionid": roughnessname,
        }

        return name

    def add_crosssection_location(
        self, branchid, chainage, definition, minz=np.nan, shift=0.0
    ):
        descr = f"{branchid}_{chainage:.1f}"
        # Add cross section location
        self.crosssection_loc[descr] = {
            "id": descr,
            "branchid": branchid,
            "chainage": chainage,
            "shift": shift,
            "definitionId": definition,
        }

    def get_branches_without_crosssection(self):
        # First find all branches that match a cross section
        branch_ids = {dct["branchid"] for _, dct in self.crosssection_loc.items()}
        # Select the branch-ids that do nog have a matching cross section
        branches = self.hydamo.branches
        no_crosssection = branches.index[~np.isin(branches.index, list(branch_ids))]

        return no_crosssection.tolist()

    def get_structures_without_crosssection(self):
        csdef_ids = [dct["id"] for _, dct in self.crosssection_def.items()]
        no_crosssection = []
        bridge_ids = [
            dct["csdefid"] for _, dct in self.hydamo.structures.bridges_df.iterrows()
        ]
        no_cross_bridge = np.asarray(bridge_ids)[
            ~np.isin(bridge_ids, csdef_ids)
        ].tolist()
        no_crosssection = no_crosssection + no_cross_bridge
        culvert_ids = [
            dct["csdefid"] for _, dct in self.hydamo.structures.culverts_df.iterrows()
        ]
        no_cross_culvert = np.asarray(culvert_ids)[
            ~np.isin(culvert_ids, csdef_ids)
        ].tolist()
        no_crosssection = no_crosssection + no_cross_culvert
        return no_crosssection

    def get_bottom_levels(self):
        """Method to determine bottom levels from cross sections"""

        # Initialize lists
        data = []
        geometry = []

        for key, css in self.crosssection_loc.items():
            # Get location
            geometry.append(
                self.dflowfmmodel.network.schematised.at[
                    css["branchid"], "geometry"
                ].interpolate(css["chainage"])
            )
            shift = css["shift"]

            # Get depth from definition if yz and shift
            definition = self.crosssection_def[css["definitionId"]]
            minz = shift
            if definition["type"] == "yz":
                minz += min(float(z) for z in definition["zCoordinates"].split())

            data.append([css["branchid"], css["chainage"], minz])

        # Add to geodataframe
        gdf = gpd.GeoDataFrame(
            data=data, columns=["branchid", "chainage", "minz"], geometry=geometry
        )
        return gdf

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def crosssection_to_yzprofiles(
        self,
        crosssections: gpd.GeoDataFrame | ExtendedGeoDataFrame,
        roughness: ExtendedDataFrame,
        branches: ExtendedGeoDataFrame | None,
        roughness_variant: RoughnessVariant = None,
    ) -> dict:
        """
        Function to convert hydamo cross sections 'dwarsprofiel' to
        dflowfm input.

        Parameters
        ----------
        crosssections : gpd.GeoDataFrame
            GeoDataFrame with x,y,z-coordinates of cross sections

        Returns
        -------
        dictionary
            Dictionary with attributes of cross sections, usable for dflowfm
        """
        cssdct = {}

        for css in crosssections.itertuples():
            # The cross sections from hydamo are all yz profiles

            # Determine yz_values
            xyz = np.vstack(css.geometry.coords[:])
            length = np.r_[
                0, np.cumsum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1])))
            ]
            yz = np.c_[length, xyz[:, -1]]
            # the GUI cannot cope with identical y-coordinates. Add 1 cm to a 2nd duplicate.
            yz[:, 0] = np.round(yz[:, 0], 3)
            for i in range(1, yz.shape[0]):
                if yz[i, 0] <= yz[i - 1, 0]:
                    yz[i, 0] = yz[i-1,0] + 0.01

            # determine thalweg
            if branches is not None:
                branche_geom = branches[branches.code == css.branch_id].geometry.to_numpy()

                if css.geometry.intersection(branche_geom[0]).geom_type == "MultiPoint":
                    thalweg_xyz = css.geometry.intersection(branche_geom[0]).geoms[0].coords[
                        :
                    ][0]
                else:
                    thalweg_xyz = css.geometry.intersection(branche_geom[0]).coords[:][
                        0
                    ]
                # and the Y-coordinate of the thalweg
                thalweg = np.hypot(
                    thalweg_xyz[0] - xyz[0, 0], thalweg_xyz[1] - xyz[0, 1]
                )
            else:
                thalweg = 0.0

            if roughness_variant == RoughnessVariant.HIGH:
                ruwheid = roughness[
                    roughness["profielpuntid"] == css.globalid
                ].ruwheidhoog
            if roughness_variant == RoughnessVariant.LOW:
                ruwheid = roughness[
                    roughness["profielpuntid"] == css.globalid
                ].ruwheidlaag

            # Add to dictionary
            cssdct[css.code] = {
                "branchid": css.branch_id,
                "chainage": css.branch_offset,
                "yz": yz,
                "thalweg": thalweg,
                "typeruwheid": roughness[
                    roughness["profielpuntid"] == css.globalid
                ].typeruwheid.to_numpy()[0],
                "ruwheid": float(ruwheid.iloc[0]),
            }

        return cssdct

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def parametrised_to_profiles(
        self,
        parametrised: ExtendedDataFrame,
        parametrised_values: ExtendedDataFrame,
        branches: list,
        roughness_variant: RoughnessVariant = None,
    ) -> dict:
        """
        Generate parametrised cross sections for all branches,
        or the branches missing a cross section.

        Parameters
        ----------
        parametrised : pd.DataFrame
            GeoDataFrame with geometries and attributes of parametrised profiles.
        branches : list
            List of branches for which the parametrised profiles are derived

        Returns
        -------
        dictionary
            Dictionary with attributes of cross sections, usable for dflowfm
        """

        cssdct = {}
        for param in parametrised.itertuples():
            branch = [
                branch for branch in branches if branch.globalid == param.hydroobjectid
            ]

            values = parametrised_values[
                parametrised_values.normgeparamprofielid == param.normgeparamprofielid
            ]

            # Drop profiles for which not enough data is available to write (as rectangle)
            # nulls = pd.isna(parambranches[['bodembreedte', 'bodemhoogtebenedenstrooms', 'bodemhoogtebovenstrooms']]).any(axis=1).to_numpy()
            # parambranches = parambranches.drop(ExtendedGeoDataFrame(geotype=LineString), parambranches.index[nulls], index_col='code',axis=0)
            # parambranches.drop(parambranches.index[nulls], inplace=True)

            if pd.isna(
                values[values.soortparameter == "bodemhoogte benedenstrooms"].waarde
            ).to_numpy()[0]:
                logger.warning(
                    "bodemhoogte benedenstrooms not available for profile %s.",
                    param.globalid,
                )
            if pd.isna(values[values.soortparameter == "bodembreedte"].waarde).to_numpy()[
                0
            ]:
                logger.warning(
                    "bodembreedte not available for profile %s.",
                    param.globalid,
                )
            if pd.isna(
                values[values.soortparameter == "bodemhoogte bovenstrooms"].waarde
            ).to_numpy()[0]:
                logger.warning(
                    "bodemhoogte bovenstrooms not available for profile %s.",
                    param.globalid,
                )

            # Determine characteristics
            botlev_upper = values[ values.soortparameter == "bodemhoogte bovenstrooms" ].waarde.to_numpy()[0]
            botlev_lower = values[ values.soortparameter == "bodemhoogte benedenstrooms" ].waarde.to_numpy()[0]            

            if pd.isna(
                values[values.soortparameter == "taludhelling linkerzijde"].waarde
            ).to_numpy()[0]:
                css_type = "rectangle"
            else:
                css_type = "trapezium"
                dh1 = (
                    values[
                        values.soortparameter == "hoogte insteek linkerzijde"
                    ].waarde.to_numpy()[0]
                    - (botlev_upper + botlev_lower)/2.
                )
                dh2 = (
                    values[
                        values.soortparameter == "hoogte insteek rechterzijde"
                    ].waarde.to_numpy()[0]
                    - (botlev_upper + botlev_lower)/2.
                )
                # height = (dh1 + dh2) / 2.0
                # Determine maximum flow width and slope (both needed for output)
                maxflowwidth = (
                    values[values.soortparameter == "bodembreedte"].waarde.to_numpy()[0]
                    + values[
                        values.soortparameter == "taludhelling linkerzijde"
                    ].waarde.to_numpy()[0]
                    * dh1
                    + values[
                        values.soortparameter == "taludhelling rechterzijde"
                    ].waarde.to_numpy()[0]
                    * dh2
                )
                slope = (
                    values[
                        values.soortparameter == "taludhelling linkerzijde"
                    ].waarde.to_numpy()[0]
                    + values[
                        values.soortparameter == "taludhelling rechterzijde"
                    ].waarde.to_numpy()[0]
                ) / 2.0

            if roughness_variant == RoughnessVariant.LOW:
                roughness = values.ruwheidlaag.to_numpy()[0]
            elif roughness_variant == RoughnessVariant.HIGH:
                roughness = values.ruwheidhoog.to_numpy()[0]
            else:
                raise ValueError(
                    'Invalid value for roughness_variant; should be "High" or "Low".'
                )
            # Determine name for cross section
            if css_type == "trapezium":
                cssdct[branch[0].Index] = {
                    "type": css_type,
                    "slope": round(slope, 2),
                    "maximumflowwidth": round(maxflowwidth, 1),
                    "bottomwidth": round(
                        values[values.soortparameter == "bodembreedte"].waarde.to_numpy()[
                            0
                        ],
                        3,
                    ),
                    "closed": 0,
                    "thalweg": 0.0,
                    "typeruwheid": values.typeruwheid.to_numpy()[0],
                    "ruwheid": roughness,
                    "bottomlevel_upper": botlev_upper,
                    "bottomlevel_lower": botlev_lower,
                }
            elif css_type == "rectangle":
                cssdct[branch[0].Index] = {
                    "type": css_type,
                    "height": 5.0,
                    "width": round(
                        values[values.soortparameter == "bodembreedte"].waarde.to_numpy()[
                            0
                        ],
                        3,
                    ),
                    "closed": 0,
                    "thalweg": 0.0,
                    "typeruwheid": values.typeruwheid.to_numpy()[0],
                    "ruwheid": roughness,
                    "bottomlevel_upper": botlev_upper,
                    "bottomlevel_lower": botlev_lower,
                }

        return cssdct


class ExternalForcings:
    def __init__(self, hydamo):
        self.hydamo = hydamo

        self.initial_waterlevel_polygons = gpd.GeoDataFrame(
            columns=["waterlevel", "geometry", "locationtype"]
        )
        self.initial_waterdepth_polygons = gpd.GeoDataFrame(
            columns=["waterdepth", "geometry", "locationtype"]
        )
        self.missing = None
        self.spatial_fields = []

        self.boundary_nodes = {}
        self.lateral_nodes = {}
        self.pattern = "^[{]?[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}[}]?$"

        self.convert = ExternalForcingsIO(self)


    def set_initial_waterlevel(self, initial_values: float | gpd.GeoDataFrame, locationtype : str="1d", name_column: str=None, value_column: str=None) -> None:
        """
        Method to set a uniform initial water level. For spatially varying water levels
        (polygons from a shapefile or values from a model result), use set_spatial_field.
        """

        if isinstance(initial_values, (int, float)):
            self.initial_waterlevel_polygons = gpd.GeoDataFrame(
                {"waterlevel": [float(initial_values)], "geometry": [None], "locationtype": [locationtype]},
                index=["global"],
            )
        else:

            if initial_values[name_column].isnull().any():
                raise ValueError(f"Column {name_column} not filled for all of the initial water level polygons. Please provide valid values.")
            if initial_values[value_column].isnull().any():
                raise ValueError(f"Column {value_column} not filled for all of the initial water level polygons. Please provide valid values.")
            self.initial_waterlevel_polygons['waterlevel'] = initial_values[value_column]
            self.initial_waterlevel_polygons['geometry'] = initial_values['geometry']
            self.initial_waterlevel_polygons['locationtype'] = locationtype
            self.initial_waterlevel_polygons.index = initial_values[name_column]     
                            
    def set_missing_waterlevel(self, missing):
        """
        Method to set the missing value for the water level.
        this overwrites the water level at missing value in the mdu file.

        Parameters
        ----------
        missing : float
            Water depth
        """
        self.mdu_parameters["WaterLevIni"] = missing

    def set_initial_waterdepth(self, initial_values: float | gpd.GeoDataFrame, locationtype : str="1d", name_column: str=None, value_column: str=None) -> None:
           """
           Method to set a uniform initial water depth. For spatially varying water depths
           (polygons from a shapefile or values from a model result), use set_spatial_field.
           """
   
           if isinstance(initial_values, (int, float)):
               self.initial_waterdepth_polygons = gpd.GeoDataFrame(
                   {"waterdepth": [float(initial_values)], "geometry": [None], "locationtype": [locationtype]},
                   index=["global"],
               )
           else:   
               if initial_values[name_column].isnull().any():
                   raise ValueError(f"Column {name_column} not filled for all of the initial water depth polygons. Please provide valid values.")
               if initial_values[value_column].isnull().any():
                   raise ValueError(f"Column {value_column} not filled for all of the initial water depth polygons. Please provide valid values.")
               self.initial_waterdepth_polygons['waterdepth'] = initial_values[value_column]
               self.initial_waterdepth_polygons['geometry'] = initial_values['geometry']
               self.initial_waterdepth_polygons['locationtype'] = locationtype
               self.initial_waterdepth_polygons.index = initial_values[name_column]    

    def correct_values_from_modelrun(self, model_values: gpd.GeoDataFrame, correction_polygons: gpd.GeoDataFrame, value_column: str) -> gpd.GeoDataFrame:
        """
        Replace values from a model result by the values of correction polygons, for all points within these polygons.

        Args:
            model_values (gpd.GeoDataFrame): points with a column 'value', e.g. from read_values_from_modelrun
            correction_polygons (gpd.GeoDataFrame): polygons with the corrected values
            value_column (str): column in correction_polygons with the corrected values

        Returns:
            gpd.GeoDataFrame: copy of model_values with corrected values. Where polygons overlap, the last polygon is used.
        """
        model_values_corrected = model_values.copy(deep=True)
        if model_values_corrected.crs is None and correction_polygons.crs is not None:
            model_values_corrected = model_values_corrected.set_crs(correction_polygons.crs)

        # use a plain GeoDataFrame, an ExtendedGeoDataFrame loses the value column in the spatial join
        joined = gpd.sjoin(
            model_values_corrected[["geometry"]],
            gpd.GeoDataFrame(correction_polygons[[value_column, "geometry"]]),
            how="inner",
            predicate="intersects",
        )
        joined = joined[~joined.index.duplicated(keep="last")]

        model_values_corrected.loc[joined.index, 'value'] = joined[value_column]
        return model_values_corrected


    def read_values_from_modelrun(
        self,
        netcdf_path: str | Path,
        netcdf_field: str,
        netcdf_index: int = None,
        crs=None,
        time: datetime | pd.Timestamp | str = None,
    ) -> gpd.GeoDataFrame:
        """
        Read the values of a 1D variable from a model result (map- or fou-file).

        Variables on the 1D nodes (e.g. 'mesh1d_s1', 'mesh1d_waterdepth') are returned at the node locations,
        variables on the 1D edges/flow links (e.g. 'mesh1d_u1') at the edge locations.

        Args:
            netcdf_path (str | Path): path to the netcdf file
            netcdf_field (str): variable to read, e.g. 'mesh1d_s1' (map) or 'mesh1d_fourier001_max' (fou)
            netcdf_index (int, optional): time index, e.g. 0 for the first time step. If neither netcdf_index
                nor time is given, the last time step is used. Defaults to None.
            crs (optional): coordinate reference system of the model. Defaults to None.
            time (datetime | pd.Timestamp | str, optional): time for which the values are read, e.g.
                "2016-06-02 12:00"; the nearest time step is used. Defaults to None.

        If 'mesh1d_s1' is requested but the file only contains the water depth, the water level is computed
        as bed level + water depth.

        Missing values (dry nodes or links in a map-file) are replaced by the bed level for water levels,
        and by 0 for water depths, velocities and discharges; for other variables they are left out.

        If the file contains the 1D network topology, the columns 'branchid' and 'chainage' are added, so the
        values can be written per branch with set_spatial_field(..., filetype="branch"). For edge variables,
        the column 'direction' (angle of the link in radians) is added, as used by set_initial_velocity.

        Returns:
            gpd.GeoDataFrame: points with columns x, y and value (and branchid, chainage and direction, if available)
        """
        logging.info(f"Reading {netcdf_field} from {netcdf_path} ...")
        with Dataset(netcdf_path, "r") as results:  # Open file in read mode
            from_depth = (
                netcdf_field == "mesh1d_s1"
                and netcdf_field not in results.variables
                and {"mesh1d_waterdepth", "mesh1d_flowelem_bl"} <= set(results.variables)
            )
            if from_depth:
                logging.info("mesh1d_s1 is not in the file; the water level is computed as bed level + water depth.")
                variable = results["mesh1d_waterdepth"]
            elif netcdf_field in results.variables:
                variable = results[netcdf_field]
            else:
                raise KeyError(f"{netcdf_field} not found in {netcdf_path}.")

            if time is not None:
                if netcdf_index is not None:
                    raise ValueError("Provide either netcdf_index or time, not both.")
                if "time" not in variable.dimensions:
                    raise ValueError(f"{netcdf_field} in {netcdf_path} has no time dimension.")
                netcdf_index = time_index(results, time)
            if netcdf_index is None and "time" in variable.dimensions:
                n_time = results.dimensions["time"].size
                if n_time != 1:
                    logging.info(f"No time given; the last of the {n_time} time steps in {netcdf_path} is used.")
                netcdf_index = -1
            if netcdf_index is None:
                value = np.ma.masked_invalid(variable[:])
            else:
                value = np.ma.masked_invalid(variable[netcdf_index, :])
            if from_depth:
                # dry nodes (missing depth) get the bed level
                bed_level = np.ma.filled(results["mesh1d_flowelem_bl"][:], np.nan)
                value = np.ma.masked_invalid(bed_level + np.ma.filled(value, 0.0))

            # location of the variable: 1D nodes or 1D edges (flow links)
            location = "edge" if variable.dimensions[-1] == "mesh1d_nEdges" else "node"
            x = np.array(results[f"mesh1d_{location}_x"][:])
            y = np.array(results[f"mesh1d_{location}_y"][:])
            if value.ndim != 1 or len(x) != len(value):
                raise ValueError(f'For {netcdf_field} in {netcdf_path}, length of coordinates and values are not equal.')

            topology = {}
            if all(v in results.variables for v in (f"mesh1d_{location}_branch", f"mesh1d_{location}_offset", "network1d_branch_id")):
                branch_ids = np.array([b.strip() for b in chartostring(results["network1d_branch_id"][:])], dtype=object)
                branch_index = np.array(results[f"mesh1d_{location}_branch"][:]).astype(int)
                branch_index -= int(getattr(results[f"mesh1d_{location}_branch"], "start_index", 0))
                topology = {"branchid": branch_ids[branch_index], "chainage": np.array(results[f"mesh1d_{location}_offset"][:])}
            if location == "edge" and "mesh1d_edge_nodes" in results.variables:
                # direction of the link (from its first to its second node), to decompose velocities in x and y
                edge_nodes = np.array(results["mesh1d_edge_nodes"][:]).astype(int)
                edge_nodes -= int(getattr(results["mesh1d_edge_nodes"], "start_index", 0))
                node_x, node_y = np.array(results["mesh1d_node_x"][:]), np.array(results["mesh1d_node_y"][:])
                topology["direction"] = np.arctan2(
                    node_y[edge_nodes[:, 1]] - node_y[edge_nodes[:, 0]], node_x[edge_nodes[:, 1]] - node_x[edge_nodes[:, 0]]
                )

            missing = np.ma.getmaskarray(value)
            if missing.any():
                name = netcdf_field.lower()
                if "depth" in name or location == "edge" or "velocity" in name:
                    fill, label = np.zeros(len(value)), "0"
                elif "s1" in netcdf_field or "waterlevel" in name:
                    fill = np.ma.filled(results["mesh1d_flowelem_bl"][:], np.nan) if "mesh1d_flowelem_bl" in results.variables else None
                    label = "the bed level"
                else:
                    fill = None
                if fill is not None:
                    logging.info(f"{missing.sum()} {location}s without a value (dry) in {netcdf_field}, set to {label}.")
                    value = np.where(missing, fill, np.ma.filled(value, np.nan))
                    missing = np.isnan(value)
                else:
                    value = np.ma.filled(value, np.nan)
                if missing.any():
                    logging.warning(f"{missing.sum()} {location}s without a value in {netcdf_field} are left out.")
                    x, y, value = x[~missing], y[~missing], value[~missing]
                    topology = {key: values[~missing] for key, values in topology.items()}
            else:
                value = np.ma.filled(value, np.nan)

        return gpd.GeoDataFrame(
            {"x": x, "y": y, "value": value, **topology},
            geometry=gpd.points_from_xy(x, y),
            crs=crs,
        )

    def set_initial_velocity(self, velocities: gpd.GeoDataFrame, column: str = "value", filename: str = "initial_velocity") -> None:
        """
        Set the initial velocity on the 1D flow links from a model result, e.g. 'mesh1d_u1' read with
        read_values_from_modelrun.

        D-Flow FM (2026.02) does not accept 'initialvelocity' in the initial fields file, only the components
        'initialvelocityx' and 'initialvelocityy'. The velocity along each link is therefore decomposed with
        the link direction and written as two sample files (triangulation). Tested with DIMR: on links that
        are in both models, the initial velocity equals the velocity of the model result within 1 cm/s.
        The effect of an initial velocity is short: the flow adapts within the first minutes.

        Args:
            velocities (gpd.GeoDataFrame): points at the link locations with the velocity and the column
                'direction' (angle of the link in radians), as returned by read_values_from_modelrun.
            column (str, optional): column with the velocity along the link. Defaults to "value".
            filename (str, optional): base name of the sample files; '_x.xyz' and '_y.xyz' are added.
                Defaults to "initial_velocity".
        """
        if "direction" not in velocities.columns:
            raise ValueError(
                "The column 'direction' (angle of the link) is required. Read the velocity with "
                "read_values_from_modelrun from a file with 'mesh1d_edge_nodes'."
            )
        speed = velocities[column].astype(float).to_numpy()
        angle = velocities["direction"].astype(float).to_numpy()
        for component, values in (("x", speed * np.cos(angle)), ("y", speed * np.sin(angle))):
            self.set_spatial_field(
                velocities.assign(_component=values)[["_component", "geometry"]],
                "_component",
                quantity=f"initialvelocity{component}",
                filetype="sample",
                filename=f"{filename}_{component}.xyz",
                interpolationmethod="triangulation",
            )

    def set_spatial_field(
        self,
        gdf: gpd.GeoDataFrame,
        column: str = None,
        quantity: str = "waterlevel",
        kind: str = "initial",
        filetype: str = None,
        cellsize: float = None,
        filename: str = None,
        locationtype: str = "1d",
        interpolationmethod: str = None,
        averagingtype: str = "nearestNb",
        operand: str = "O",
        nodata: float = -999.0,
        max_snap_distance: float = 5.0,
    ) -> None:
        """
        Add a spatially varying field to the initial fields file (initialFields.ini).

        Four file types are supported:

        - 'branch': values per branch and chainage in a 1dField file. Values are interpolated linearly
          along each branch only, and kept constant beyond the first and last chainage of a branch.
          For polygons, each part of a branch inside a polygon gets the polygon value, with a step at the
          polygon boundary. For points, the columns 'branchid' and 'chainage' are used if present (e.g. from
          read_values_from_modelrun on the same network); otherwise points are snapped to the nearest branch
          within max_snap_distance. Default for polygons; independent of the 1D computational grid.
        - 'sample': points are written as samples (.xyz) with triangulation. Default for points: values
          from a model result on the same network are reproduced exactly at the nodes.
        - 'polygon': polygons are written as .pol files with interpolationMethod=constant. Polygons with
          the same value share one file and one block.
        - 'raster': polygons are rasterized to a GeoTIFF (.tif) or ArcInfo grid (.asc). Requires a cellsize.
          Note that D-Flow FM (2026.02) uses the nodata value as data.

        Tested with D-HYDRO 2026.02 on 1D nodes: interpolationMethod=averaging assigns no values, so it is
        refused for locationtype '1d'; triangulation fills the convex hull of the samples; nodes that get no
        value from any field keep the default water level (WaterLevIni in the MDU), which leaves them dry when it
        is below the bed level, so cover all nodes. Fields are applied in the order they are added (the last
        one wins), after the uniform values of set_initial_waterlevel/-depth. A uniform water depth overrides
        spatial water levels, and a water depth field removes water levels set before it: add water depth
        fields before water level fields.

        The data files are written by Df2HydrolibModel.write_initial_fields().

        Args:
            gdf (gpd.GeoDataFrame): polygons or points with values
            column (str, optional): column with values. For points, None means the Z-coordinate is used.
            quantity (str, optional): D-Flow FM quantity, e.g. 'waterlevel' or 'waterdepth'. For velocities, use
                set_initial_velocity. Defaults to "waterlevel".
            kind (str, optional): 'initial' ([Initial] block) or 'parameter' ([Parameter] block). Defaults to "initial".
            filetype (str, optional): 'branch', 'sample', 'polygon' or 'raster'. Defaults to 'branch' for polygons
                and 'sample' for points.
            cellsize (float, optional): raster cell size, required for filetype 'raster'.
            filename (str, optional): name of the data file, relative to the fm-folder. Without an extension, the
                extension of the file type is added (.ini, .xyz, .pol or .tif). For 'polygon', a number is added per
                distinct value. Defaults to '<quantity>_<nr>' with that extension.
            locationtype (str, optional): '1d', '2d' or 'all'. Defaults to "1d".
            interpolationmethod (str, optional): 'constant' (polygon), 'triangulation' or 'averaging'. Not used for
                'branch'. Defaults to 'triangulation' for samples, 'constant' for polygon files, and 'averaging'
                for rasters (2D) or 'triangulation' (1D).
            averagingtype (str, optional): averaging type, if interpolationmethod is 'averaging'. Defaults to "nearestNb".
            operand (str, optional): how the field is combined with previous fields. Defaults to "O" (override).
            nodata (float, optional): nodata value for rasters, and the [Global] value of 1dField files (branch), which
                D-Flow FM treats as no value. Defaults to -999.0.
            max_snap_distance (float, optional): maximum distance to snap points to a branch for filetype 'branch'.
                Defaults to 5.0.
        """
        if kind not in ("initial", "parameter"):
            raise ValueError(f"kind should be 'initial' or 'parameter', not '{kind}'.")
        if gdf.empty:
            raise ValueError("The provided GeoDataFrame is empty.")
        if quantity.lower() == "initialvelocity":
            raise ValueError(
                "D-Flow FM (2026.02) does not support 'initialvelocity' in the initial fields file. "
                "Use set_initial_velocity, which writes the x and y components."
            )

        geomtypes = set(gdf.geom_type)
        if geomtypes <= {"Polygon", "MultiPolygon"}:
            filetype = "branch" if filetype is None else filetype
            if filetype not in ("branch", "polygon", "raster"):
                raise ValueError(f"For polygons, filetype should be 'branch', 'polygon' or 'raster', not '{filetype}'.")
        elif geomtypes <= {"Point"}:
            filetype = "sample" if filetype is None else filetype
            if filetype not in ("sample", "branch"):
                raise ValueError(f"For points, filetype should be 'sample' or 'branch', not '{filetype}'.")
        else:
            raise ValueError(
                f"Only (multi)polygons or points are supported, found: {geomtypes}."
            )

        if filetype == "branch" and locationtype != "1d":
            raise ValueError("Filetype 'branch' is only available for locationtype '1d'.")
        if filetype == "raster" and cellsize is None:
            raise ValueError("A cellsize is required to rasterize polygons.")

        if filetype == "branch":
            interpolationmethod = None
        elif interpolationmethod is None:
            if filetype == "polygon":
                interpolationmethod = "constant"
            elif filetype == "raster" and locationtype != "1d":
                interpolationmethod = "averaging"
            else:
                interpolationmethod = "triangulation"
        if filetype == "polygon" and interpolationmethod != "constant":
            raise ValueError("Polygon files only support interpolationmethod 'constant'.")
        if interpolationmethod == "averaging" and locationtype == "1d":
            raise ValueError(
                "interpolationmethod 'averaging' assigns no values to 1D nodes (D-HYDRO 2026.02). "
                "Use filetype 'branch' or 'polygon' for polygons, or 'triangulation' for samples."
            )
        if filetype == "raster" and locationtype in ("1d", "all"):
            logger.warning(
                "D-Flow FM uses the raster nodata value as data, which affects 1D nodes near polygon edges. "
                "Consider filetype 'branch' for 1D."
            )
        if filetype == "polygon" and any(
            len(p.interiors) > 0 for geom in gdf.geometry for p in common.as_polygon_list(geom)
        ):
            logger.warning("Polygon files do not support holes; interiors of the polygons are ignored.")

        if column is None:
            if "Point" not in geomtypes or not gdf.has_z.all():
                raise ValueError(
                    "Provide a value column, or use points with a Z-coordinate."
                )
            gdf = gdf.assign(_z=gdf.geometry.z)
            column = "_z"

        if gdf[column].isnull().any():
            raise ValueError(
                f"Column {column} not filled for all features. Please provide valid values."
            )

        ext = {"branch": ".ini", "polygon": ".pol", "raster": ".tif", "sample": ".xyz"}[filetype]
        if filename is None:
            filename = f"{quantity}_{len(self.spatial_fields):02d}{ext}"
        elif Path(filename).suffix == "":
            # without an extension, the extension follows the file type
            filename = f"{filename}{ext}"
        if filename in [sf["filename"] for sf in self.spatial_fields]:
            raise ValueError(f"Filename {filename} is already used for another spatial field.")

        columns = [column, "geometry"]
        if filetype == "branch" and {"branchid", "chainage"} <= set(gdf.columns):
            columns = ["branchid", "chainage"] + columns

        self.spatial_fields.append(
            {
                "gdf": gpd.GeoDataFrame(gdf[columns].copy()),
                "column": column,
                "quantity": quantity,
                "kind": kind,
                "filetype": filetype,
                "cellsize": cellsize,
                "filename": filename,
                "locationtype": locationtype,
                "interpolationmethod": interpolationmethod,
                "averagingtype": averagingtype,
                "operand": operand,
                "nodata": nodata,
                "max_snap_distance": max_snap_distance,
            }
        )


    def add_rainfall_2D(self, fName, bctype="rainfall"):
        """
        Parameters
        ----------
        fName : str
            Location of netcdf file containing rainfall rasters
        bctype : str
            Type of boundary condition. Currently only rainfall is supported
        """
        assert bctype in ["rainfall"]

        # Add boundary condition
        self.boundaries["rainfall_2D"] = {
            "file_name": fName,
            "bctype": bctype + "bnd",
        }

    @validate_arguments
    def add_boundary_condition(
        self, name: str, pt, quantity: str, value, mesh1d=None
    ) -> None:
        """
        Add boundary conditions to model:
        - The boundary condition can be discharge or waterlevel, or a Q-H relation
        - Is specified by a geographical location (pt) and a branchid
        - If no branchid is given, the nearest is searched
        - The boundary condition is added to the end of the given or nearest branch.
        Value can be one of three options:
        -  timeseries: a Pandas series with time as index should be given
        -  constant:  a float should be given
        -  Q-H boundary: a dictionary with Q-H 
        
        Parameters
        ----------
        name : str
            ID of the boundary condition
        pt : tuple or shapely.geometry.Point
            Location of the boundary condition
        bctype : str
            Type of boundary condition. Currently only discharge and waterlevel are supported
        series : pd.Series dict, or float
            If a float, a constant in time boundary condition is used. If a pandas series,
            the values per time step are used. Index should be in datetime format, if dict a Q-H table is assumed
        branchid : str, optional
            ID of the branch. If None, the branch nearest to the given location (pt) is
            searched, by default None
        """

        assert quantity in ["dischargebnd", "waterlevelbnd", 'qhbnd']


        if isinstance(value, pd.Series):
            vec1 = ((value.index - value.index[0]).total_seconds() / 60.0).tolist()
            vec2 = value.to_numpy().tolist()
            startdate = value.index[0].strftime("%Y-%m-%d %H:%M:%S")
            unit1 =  f"minutes since {startdate}"
            unit2 = "m3/s" if quantity == "dischargebnd" else "m"            
        elif isinstance(value, numbers.Real):
            vec1 = None
            vec2 = float(value)
            startdate = "0000-00-00 00:00:00"
            unit1 =  f"minutes since {startdate}"
            unit2 = "m3/s" if quantity == "dischargebnd" else "m"
        else:
            vec1 = value['Q']
            vec2 = value['H']            
            unit1 = 'm3/s'
            unit2 = 'm'
        
        if name in self.boundary_nodes.keys():
            raise KeyError(
                f'A boundary condition with name "{name}" is already present.'
            )

        if isinstance(pt, tuple):
            pt = Point(*pt)

        # Find the nearest node
        if len(mesh1d._mesh1d.mesh1d_node_id) == 0:
            raise KeyError(
                "To find the closest node a 1d mesh should be created first."
            )
        nodes1d = np.asarray(
            [
                n
                for n in zip(
                    mesh1d._mesh1d.mesh1d_node_x,
                    mesh1d._mesh1d.mesh1d_node_y,
                    mesh1d._mesh1d.mesh1d_node_id,
                )
            ]
        )
        get_nearest = KDTree(nodes1d[:, 0:2])
        _, idx_nearest = get_nearest.query(pt.coords[:])
        nodeid = f"{float(nodes1d[idx_nearest[0],0]):12.6f}_{float(nodes1d[idx_nearest[0],1]):12.6f}"

        # Add boundary condition
        self.boundary_nodes[name] = {
            "id": name,
            "quantity": quantity,
            "vec1": vec1,
            "vec2": vec2,
            "unit1": unit1,
            "unit2": unit2,
            "nodeid": nodeid,
        }

        
    @validate_arguments
    def add_rain_series(self, name: str, values: list, times: list) -> None:
        """
        Adds a rain series a boundary condition.
        Specify name, values, and times

        Parameters
        ----------
        name : str
            ID of the condition
        values : list of floats
            Values of the rain intensity
        times : list of datetime
            Times for the values
        """
        # Add boundary condition
        self.boundary_nodes[name] = {
            "code": name,
            "bctype": "rainfall",
            "filetype": 1,
            "method": 1,
            "operand": "O",
            "value": values,
            "time": times,
            "geometry": None,
            "branchid": None,
        }

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def add_lateral(
        self,
        id: str,
        branchid: str,
        chainage: str,
        discharge: pd.Series | float | str,
    ) -> None:
        """Add a lateral to an FM model

        Args:
            id (str): Id of th lateral node
            name (str): name of the node
            branchid (str): branchid it is snapped to
            chainage (str): chainage on the branch
            discharge (str, float, or pd.Series): discharge type: REALTIME when linked to RR, or float (constant value) or a pd.Series with time index
        """
        # Convert time to minutes
        if isinstance(discharge, pd.Series):
            times = (
                (discharge.index - discharge.index[0]).total_seconds() / 60.0
            ).tolist()
            values = discharge.to_numpy().tolist()
            startdate = discharge.index[0].strftime("%Y-%m-%d %H:%M:%S")
        else:
            times = None
            values = None
            startdate = "0000-00-00 00:00:00"

        self.lateral_nodes[id] = {
            "id": id,
            "type": "discharge",
            "locationtype": "1d",
            "branchid": branchid,
            "chainage": chainage,
            "time": times,
            "time_unit": f"minutes since {startdate}",
            "value_unit": "m3/s",
            "value": values,
            "discharge": discharge,
        }


class Structures:
    def __init__(self, hydamo):
        self.hydamo = hydamo
        self.generalstructures_df = pd.DataFrame()
        self.rweirs_df = pd.DataFrame()
        self.orifices_df = pd.DataFrame()
        self.uweirs_df = pd.DataFrame()
        self.culverts_df = pd.DataFrame()
        self.bridges_df = pd.DataFrame()
        self.pumps_df = pd.DataFrame()
        self.compounds_df = pd.DataFrame()

        self.convert = StructuresIO(self)

    def check_branchid_chainage(self, branchid, chainage):
        # Check if the ID exists
        if branchid not in self.hydamo.branches["code"]:
            raise ValueError(
                f"branchid {branchid} not present. Give an existing branch."
            )

        # Get the branch
        branch = self.hydamo.branches.at[branchid, "geometry"]

        # Check the limits
        if chainage < 0.0:
            raise ValueError(
                f"Chainage {chainage} is outside the branch range (0.0 - {branch.length})."
            )
        if chainage > branch.length:
            raise ValueError(
                f"Chainage {chainage} is outside the branch length (0.0 - {branch.length})."
            )

    @validate_arguments
    def add_rweir(
        self,
        id: str = None,
        name: str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        crestlevel: float = None,
        crestwidth: float = None,
        corrcoeff: float = None,
        usevelocityheight: str = "true",
        allowedflowdir: str = "both",
    ) -> None:
        """
        Function to add a regular weir. Arguments correspond to the required input of DFlowFM.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "crestwidth": crestwidth,
                "corrcoeff": corrcoeff,
                "usevelocityheight": usevelocityheight,
                "allowedflowdir": allowedflowdir,
            },
            index=[id],
        )
        self.rweirs_df = pd.concat([self.rweirs_df, dct], ignore_index=True)

    @validate_arguments
    def add_orifice(
        self,
        id: str = None,
        name:  str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        crestlevel: float = None,
        crestwidth: float = None,
        corrcoeff: float = None,
        usevelocityheight: str = "true",
        allowedflowdir: str = "both",
        gateloweredgelevel: float = None,
        uselimitflowpos: str = None,
        limitflowpos: float = None,
        uselimitflowneg: str = None,
        limitflowneg: float = None,
    ) -> None:
        """
        Function to add a orifice. Arguments correspond to the required input of DFlowFM.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "crestwidth": crestwidth,
                "corrcoeff": corrcoeff,
                "usevelocityheight": usevelocityheight,
                "allowedflowdir": allowedflowdir,
                "gateloweredgelevel": gateloweredgelevel,
                "uselimitflowpos": uselimitflowpos,
                "limitflowpos": limitflowpos,
                "uselimitflowneg": uselimitflowneg,
                "limitflowneg": limitflowneg,
            },
            index=[id],
        )
        self.orifices_df = pd.concat([self.orifices_df, dct], ignore_index=True)

    @validate_arguments
    def add_uweir(
        self,
        id: str = None,
        name:  str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        crestlevel: float = None,
        crestwidth: float = None,
        dischargecoeff: float = None,
        usevelocityheight: str = "true",
        allowedflowdir: str = "both",
        numlevels: float = None,
        yvalues: str = None,
        zvalues: str = None,
    ) -> None:
        """
        Function to add a universalweir. Arguments correspond to the required input of DFlowFM.
        """
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "crestlevel": crestlevel,
                "crestwidth": crestwidth,
                "dischargecoeff": dischargecoeff,
                "usevelocityheight": usevelocityheight,
                "numlevels": numlevels,
                "allowedflowdir": allowedflowdir,
                "yvalues": yvalues,
                "zvalues": zvalues,
            },
            index=[id],
        )
        self.uweirs_df = pd.concat([self.uweirs_df, dct], ignore_index=True)

    @validate_arguments
    def add_bridge(
        self,
        id: str = None,
        name:  str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        length: float = None,
        inletlosscoeff: float = None,
        outletlosscoeff: float = None,
        csdefid: str = None,
        shift: float = None,
        allowedflowdir: str = "both",
        frictiontype: str = None,
        friction: float = None,
    ) -> None:
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        # map HyDAMO definition to D-Hydro definition
        frictiontype = self.hydamo.roughness_mapping[frictiontype]

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "length": length,
                "inletlosscoeff": inletlosscoeff,
                "outletlosscoeff": outletlosscoeff,
                "csdefid": csdefid,
                "shift": shift,
                "allowedflowdir": allowedflowdir,
                "frictiontype": frictiontype,
                "friction": friction,
            },
            index=[id],
        )
        self.bridges_df = pd.concat([self.bridges_df, dct], ignore_index=True)

    @validate_arguments
    def add_culvert(
        self,
        id: str = None,
        name:  str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        leftlevel: float = None,
        rightlevel: float = None,
        length: float = None,
        inletlosscoeff: float = None,
        outletlosscoeff: float = None,
        crosssection: dict = None,
        allowedflowdir: str = "both",
        valveonoff: int = 0,
        numlosscoeff: int = None,
        valveopeningheight: float = 0,
        relopening: list = None,
        losscoeff: list = None,
        bedfrictiontype: str = None,
        bedfriction: float = None,
    ) -> None:
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        if crosssection["shape"] == "circle":
            definition = self.hydamo.crosssections.add_circle_definition(
                crosssection["diameter"], bedfrictiontype, bedfriction, name=id
            )
        elif crosssection["shape"] == "rectangle":
            definition = self.hydamo.crosssections.add_rectangle_definition(
                crosssection["height"],
                crosssection["width"],
                crosssection["closed"],
                bedfrictiontype,
                bedfriction,
                name=id,
            )
        else:
            raise NotImplementedError(
                f'Cross section with shape "{crosssection["shape"]}" not implemented.'
            )

        bedfrictiontype = self.hydamo.roughness_mapping[bedfrictiontype]

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "rightlevel": rightlevel,
                "leftlevel": leftlevel,
                "length": length,
                "inletlosscoeff": inletlosscoeff,
                "outletlosscoeff": outletlosscoeff,
                "csdefid": definition,
                "bedfrictiontype": bedfrictiontype,
                "bedfriction": bedfriction,
                "allowedflowdir": allowedflowdir,
                "valveonoff": valveonoff,
                "numlosscoeff": numlosscoeff,
                "valveopeningheight": valveopeningheight,
                "relopening": [relopening],
                "losscoeff": [losscoeff],
            },
            index=[id],
        )

        self.culverts_df = pd.concat([self.culverts_df, dct], ignore_index=True)

    @validate_arguments
    def add_pump(
        self,
        id: str = None,
        name:  str | float | None = None,
        branchid: str = None,
        chainage: float = None,
        orientation: str = "positive",
        numstages: int = 1,
        controlside: str = "suctionSide",
        capacity: float = None,
        startlevelsuctionside: list = None,
        stoplevelsuctionside: list = None,
        startleveldeliveryside: list = None,
        stopleveldeliveryside: list = None,
    ) -> None:
        # Check branchid chainage
        self.check_branchid_chainage(branchid, chainage)

        dct = pd.DataFrame(
            {
                "id": id,
                "name": name,
                "branchid": branchid,
                "chainage": chainage,
                "orientation": orientation,
                "numstages": numstages,
                "controlside": controlside,
                "capacity": capacity,
                "startlevelsuctionside": [startlevelsuctionside],
                "stoplevelsuctionside": [stoplevelsuctionside],
                "startleveldeliveryside": [startleveldeliveryside],
                "stopleveldeliveryside": [stopleveldeliveryside],
            },
            index=[id],
        )
        self.pumps_df = pd.concat([self.pumps_df, dct], ignore_index=True)

    @validate_arguments
    def add_compound(self, id:  str | float | None = None, structureids: list = None) -> None:
        structurestring = ";".join([f"{s}" for s in structureids])
        numstructures = len(structureids)
        dct = pd.DataFrame(
            {
                "id": id,
                "name": id,
                "numstructures": numstructures,
                "structureids": structurestring,
            },
            index=[id],
        )
        self.compounds_df = pd.concat([self.compounds_df, dct], ignore_index=True)

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def as_dataframe(
        self,
        generalstructures: bool = False,
        pumps: bool = False,
        rweirs: bool = False,
        bridges: bool = False,
        culverts: bool = False,
        uweirs: bool = False,
        orifices: bool = False,
        compounds: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a dataframe with the structures. Specify with the keyword arguments what structure types need to be returned.
        """
        dfs = []
        for df, descr, add in zip(
            [
                self.generalstructures_df,
                self.culverts_df,
                self.rweirs_df,
                self.bridges_df,
                self.pumps_df,
                self.uweirs_df,
                self.orifices_df,
                self.compounds_df,
            ],
            [
                "generalstructures",
                "culvert",
                "weir",
                "bridge",
                "pump",
                "uweir",
                "orifice",
                "compound",
            ],
            [
                generalstructures,
                culverts,
                rweirs,
                bridges,
                pumps,
                uweirs,
                orifices,
                compounds,
            ],
        ):
            if any(df) and add:
                df = df.copy()
                df.insert(loc=0, column="structype", value=descr, allow_duplicates=True)
                dfs.append(df)

        if len(dfs) > 0:
            return pd.concat(dfs, sort=False, ignore_index=True)


class ObservationPoints:
    def __init__(self, hydamo):
        self.hydamo = hydamo
        self.observation_points = gpd.GeoDataFrame().set_geometry([])

    @validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def add_points(
        self, crds: list, names: list, locationTypes=None, snap_distance: float = 5.0
    ) -> None:
        """
        Method to add observation points to schematisation. Observation points can be of type '1d' or '2d'. 1d-points are snapped to the branch.

        Parameters
        ----------
        crds : Nx2 list or array
            x and y coordinates of observation points
        names : str or list
            names of the observation points
        locationTypes:  str or list
            type of the observationpoints: 1d or 2d
        snap_distance : float (default is 5 m)
            1d observation poinst within this distance to a branch will be snapped to it. Otherwise they are discarded.
        """
        if isinstance(names, str):
            names = [names]
            crds = [crds]

        if locationTypes is not None:
            if isinstance(locationTypes, str):
                locationTypes = [locationTypes]

            # split 1d and 2d points, as the first ones need to be snapped to branches
            obs2d = gpd.GeoDataFrame()
            obs2d["name"] = [
                n for nn, n in enumerate(names) if locationTypes[nn] == "2d"
            ]
            obs2d["locationtype"] = "2d"
            obs2d = obs2d.set_geometry([
                Point(*pt) if not isinstance(pt, Point) else pt
                for ipt, pt in enumerate(crds)
                if (locationTypes[ipt] == "2d")
            ])
            obs2d["x"] = [pt.coords[0][0] for pt in obs2d["geometry"]]
            obs2d["y"] = [pt.coords[0][1] for pt in obs2d["geometry"]]
            names1d = [n for n_i, n in enumerate(names) if locationTypes[n_i] == "1d"]
            crds1d = [c for c_i, c in enumerate(crds) if locationTypes[c_i] == "1d"]
        else:
            names1d = names
            crds1d = crds

        obs1d = gpd.GeoDataFrame()
        obs1d["name"] = names1d
        obs1d = obs1d.set_geometry([
            Point(*pt) if not isinstance(pt, Point) else pt for pt in crds1d
        ])
        obs1d["locationtype"] = "1d"
        find_nearest_branch(
            self.hydamo.branches, obs1d, method="overal", maxdist=snap_distance
        )
        obs1d.rename(
            columns={"branch_id": "branchid", "branch_offset": "chainage"}, inplace=True
        )

        obs = pd.concat([obs1d, obs2d], sort=True) if locationTypes is not None else obs1d

        obs.dropna(how="all", axis=1, inplace=True)

        # Add to dataframe
        if self.observation_points.empty:
            self.observation_points = obs
        else:
            self.observation_points = pd.concat([self.observation_points, obs], ignore_index=True)


class StorageNodes:
    def __init__(self, hydamo):
        self.storagenodes = {}

        self.hydamo = hydamo
        
        self.convert = StorageNodesIO(self)
    
    def add_storagenode(
        self,
        id,
        xy=None,
        branchid=None,
        chainage=None,
        nodeid=None,
        usestreetstorage="true",
        nodetype="unspecified",
        manholeid=None,
        name=np.nan,
        usetable="false",
        bedlevel=np.nan,
        area=np.nan,
        streetlevel=np.nan,
        streetstoragearea=np.nan,
        storagetype="reservoir",
        levels=np.nan,
        storagearea=np.nan,
        interpolate="linear",
        network=None
    ):

        if isinstance(xy, tuple):
            xy = Point(*xy)
        if xy is None and chainage is not None:
            xy = self.hydamo.branches.loc[branchid].geometry.interpolate(chainage)            
        
        # Find the nearest node
        if nodeid is not None and not isinstance(nodeid, str):
            raise ValueError('If a nodeid is provided, it should be of type string.')
        if nodeid is None:
            if len(network._mesh1d.network1d_node_id) == 0:
                raise KeyError(
                    "To find the closest node a 1d mesh should be created first."
                )
            nodes1d = np.asarray(
                [
                    n
                    for n in zip(
                        network._mesh1d.network1d_node_x,
                        network._mesh1d.network1d_node_y,
                        network._mesh1d.network1d_node_id,
                    )
                ]
            )
            get_nearest = KDTree(nodes1d[:, 0:2])
            _, idx_nearest = get_nearest.query(xy.coords[:])
            nodeid = f"{float(nodes1d[idx_nearest[0],0]):12.6f}_{float(nodes1d[idx_nearest[0],1]):12.6f}"

        if manholeid is None:
            manholeid = nodeid
        
        base = {
            "type": "storageNode",
            "id": id,
            "name": id if name is None else name,
            "useStreetStorage": usestreetstorage,
            "nodeType": nodetype,            
            "useTable": usetable,
            "manholeId": manholeid
        }
        if nodeid is None:
            cds = {'branchid':branchid, 
                   'chainage': chainage}
        else:
            cds = {'nodeid':nodeid}

        if usetable == "false":
            out = {
                **base,
                **cds,
                "bedLevel": bedlevel,
                "area": area,
                "streetLevel": streetlevel,
                "streetStorageArea": streetstoragearea,
                "storageType": storagetype,
            }
        elif usetable == "true":
            assert len(levels.split()) == len(
                storagearea.split()
            ), "Number of levels does not equal number of storagearea"
            out = {
                **base,
                **cds,
                "numLevels": len(levels.split()),
                "levels": levels,
                "storageArea": storagearea,
                "interpolate": interpolate,
                "bedLevel": -999.,
                "area":-999.,
                "streetLevel": -999.,
                "streetStorageArea":-999.,
                "storageType": storagetype,
            }
        else:
            raise ValueError(
                "Value of key 'usetable' is not supported. Either use 'true' or 'false"
            )
        self.storagenodes[id] = remove_nan_values(out)


def remove_nan_values(base):
    """Remove nan values from object

    Args:
        base (_type_): input data, containig nans

    Returns:
       base_copy : output data with nans-filtered
    """
    base_copy = base.copy()
    for k, v in base.items():
        if isinstance(v, float):
            if np.isnan(v):
                base_copy.pop(k)
    return base_copy
