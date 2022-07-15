from ast import Or
from calendar import c
import logging

from typing import List, Union
from wsgiref import validate

from tqdm.auto import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
import sys
import numpy as np
import rstr
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io import fmconverter
from datetime import datetime
from enum import Enum
from pydantic import validate_arguments
# from hydrolib.core.io.structure.models import *
# from hydrolib.core.io.crosssection.models import *
# from hydrolib.core.io.ext.models import *
# from hydrolib.core.io.net.models import *
from hydrolib.dhydamo.io.fmconverter import RoughnessVariant
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame, ExtendedDataFrame
from hydrolib.dhydamo.geometry.geometry import find_nearest_branch
from hydrolib.dhydamo.geometry.mesh import *
from hydrolib import dhydamo

logger = logging.getLogger(__name__)



class HyDAMO:
    """Main data structure for dflowfm model. Contains subclasses
    for network, structures, cross sections, observation points
    and external forcings.
    """
    def __init__(self, extent_file=None):

        self.network = Network(self)

        self.structures = Structures(self)
        
        self.crosssections = CrossSections(self) # Add all items
      
        self.observationpoints = ObservationPoints(self)

        self.external_forcings = ExternalForcings(self) 

        self.storagenodes = StorageNodes(self)

        # Dictionary for roughness definitions
        self.roughness_definitions = {}     
    
        # Read geometry to clip data
        if extent_file is not None:
            self.clipgeo = gpd.read_file(extent_file).unary_union
        else:
            self.clipgeo = None
        
         # versioning info
        self.version = { 'number' : dhydamo.__version__,
                         'date'   :  datetime.strftime(datetime.utcnow(),'%Y-%m-%dT%H:%M:%S.%fZ'),
                         'dfm_version'   : 'Deltares, D-Flow FM Version 5.00.024.74498M',
                         'dimr_version'  : 'Deltares, DIMR_EXE Version 2.00.00.140737 (Win64) (Win64)',
                         'suite_version' : 'D-HYDRO Suite 2022.03 1D2D,'} 

        # Create standard dataframe for network, crosssections, orifices, weirs
        self.branches = ExtendedGeoDataFrame(geotype=LineString, required_columns=[
            'code',
            'geometry',
            'typeruwheid',            
        ])
        
        self.profile = ExtendedGeoDataFrame(geotype=LineString, required_columns=[
            'code',
            'geometry',
            'globalid',
            'profiellijnid'            
            
        ])
        self.profile_roughness = ExtendedDataFrame(required_columns=[
            'code',
            'globalid',
            'profielpuntid'        
        ])
        
        self.profile_line = ExtendedGeoDataFrame(geotype=LineString, required_columns=[            
            'globalid',
            'profielgroepid'        
        ])
        
        self.profile_group = ExtendedDataFrame(required_columns=[            
            'globalid'            
        ])     
             

        self.param_profile = ExtendedDataFrame(required_columns=[
            'globalid',
            'normgeparamprofielid',
            'hydroobjectid'
        ])
        
        self.param_profile_values = ExtendedDataFrame(required_columns=[
            'normgeparamprofielid',
            'soortparameter',
            'waarde',            
            'ruwheidlaag',
            'ruwheidhoog',
            'typeruwheid'
        ])
        
        # Weirs
        self.weirs = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',
            'geometry',
            'globalid',
            'soortstuw',                          
            'afvoercoefficient'
        ])         
        
        # opening
        self.opening = ExtendedDataFrame(required_columns=[            
            'vormopening',            
            'globalid',
            'hoogstedoorstroombreedte',
            'hoogstedoorstroomhoogte',
            'laagstedoorstroombreedte',
            'laagstedoorstroomhoogte',            
            'vormopening',
            'afvoercoefficient'
         ])
        
        # opening
        self.closing_device = ExtendedDataFrame(required_columns=[
            'code'    
            
        ])
        
        # opening
        self.management_device = ExtendedDataFrame(required_columns=[
            'code',
            'soortregelbaarheid',                      
            'overlaatonderlaat'
         ])
        
        # Bridges
        self.bridges = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',                        
            'globalid',
            'geometry',
            'lengte',
            'intreeverlies',
            'uittreeverlies',
            'ruwheid',
            'typeruwheid'            
        ])       
     
        # Culverts
        self.culverts = ExtendedGeoDataFrame(geotype=LineString, required_columns=[
            'code',            
            'geometry',
            'lengte',
            'hoogteopening',
            'breedteopening',
            'hoogtebinnenonderkantbene',
            'hoogtebinnenonderkantbov',
            'vormkoker',
            'intreeverlies',
            'uittreeverlies',
            'typeruwheid',
            'ruwheid'
        ])
        
        # Laterals
        self.laterals = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'globalid',            
            'geometry'
        ])

        # Gemalen
        self.pumpstations = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',            
            'globalid',
            'geometry',
        ])
        self.pumps = ExtendedDataFrame(required_columns=[
            'code',
            'globalid',
            'gemaalid',
            'maximalecapaciteit'            
        ])
        self.management = ExtendedDataFrame(required_columns=[
            'code',
            'globalid',
            'pompid',
            'doelvariabele'            
        ])

        # self.afsluitmiddel = ExtendedDataFrame(required_columns=[
        #     'code',
        #     'soortafsluitmiddelcode',
        #     'codegerelateerdobject'
        # ])

        # Hydraulische randvoorwaarden
        self.boundary_conditions = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',
            'typerandvoorwaarde',
            'geometry'
        ])
        
        # RR catchments
        self.catchments = ExtendedGeoDataFrame(geotype=Polygon, required_columns=[
            'code',
            'geometry',
            'globalid',
            'lateraleknoopid'            
        ])
        
        # Laterals
        self.laterals = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',
            'geometry',
            'globalid'            
        ])
        
        # RR overflows
        self.overflows = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            'code',
            'geometry',
            'codegerelateerdobject',
            'fractie'
            
        ])

        # RR sewer areas
        self.sewer_areas = ExtendedGeoDataFrame(geotype=Polygon, required_columns=[
            'code',
            'geometry'            
        ])
     
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def list_to_str(self, lst:Union[list, np.ndarray]) -> str:
        """Converts list to string

        Args:
            lst (list): The list

        Returns:
            str: The output string
        """
        if len(lst)==1:
            string = str(lst)
        else:
            string = ' '.join([f'{number:6.3f}' for number in lst])
        return string
    
    @validate_arguments
    def dict_to_dataframe(self, dictionary:dict) -> pd.DataFrame:
        """Converts a dictionary to dataframe, using index as rows

        Args:
            dictionary (dict): Input dictionary

        Returns:
            pd.DataFrame: Output dataframe
        """
       
        return(pd.DataFrame.from_dict(dictionary, orient='index'))

class Network:
    
    def __init__(self, hydamo):

        self.hydamo = hydamo
        
        # Mesh 1d offsets
        self.offsets = {}
     
    @validate_arguments
    def set_branch_order(self, branchids:list, idx:int=None) -> None:
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
        branchidx = np.isin(self.mesh1d.description1d['network_branch_ids'], branchids)
        # Get current order
        branchorder = self.mesh1d.get_values('nbranchorder', as_array=True)
        # Update
        if idx is None:
            branchorder[branchidx] = branchorder.max() + 1
        else:
            if not isinstance(idx, int):
                raise TypeError('Expected integer.')
            branchorder[branchidx] = idx
        # Save
        self.mesh1d.set_values('nbranchorder', branchorder)

    def set_branch_interpolation_modelwide(self):
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

    def make_nodes_to_branch_map(self):
        # Note: first node is upstream, second node is downstream
        self.nodes_to_branch_map = {b: [self.mesh1d.description1d['network_node_ids'][_idx - 1] for _idx in idx]
                                    for b, idx in zip(self.mesh1d.description1d['network_branch_ids'],
                                                      self.mesh1d.get_values('nedge_nodes', as_array=True))}

    def make_branches_to_node_map(self):
        self.make_nodes_to_branch_map()
        self.branches_to_node_map = {n: [k for k, v in self.nodes_to_branch_map.items() if n in v]
                                     for n in self.mesh1d.description1d['network_node_ids']}

    def generate_nodes_with_bedlevels(self,
                                      resolve_at_bifurcation_method='min',
                                      return_reversed_branches=False):
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
        assert resolve_at_bifurcation_method in ['min', 'max', 'mean'], f"Incorrect value for " \
                                                                        f"'resolve_at_bifurcation_method' supplied. " \
                                                                        f"Either use 'min', 'max' or 'mean'"
        bedlevels_crs_branches = self.hydamo.crosssections.get_bottom_levels()
        branch_order = self.mesh1d.get_values('nbranchorder',  as_array=True)
        self.make_branches_to_node_map(), self.make_nodes_to_branch_map()
        nodes_dict = {n: {'up': [], 'down': []} for n in self.branches_to_node_map.keys()}
        reserved_branches = []
        for order, (branch, nodes) in tqdm(zip(branch_order, self.nodes_to_branch_map.items()),
                                           total=len(branch_order), desc='Getting bedlevels'):
            if order == -1:
                # No branch order so just get upstream and downstream levels
                branch_length = self.branches.loc[branch, 'geometry'].length
                subset = bedlevels_crs_branches.loc[bedlevels_crs_branches['branchid'] == branch]
                if subset.empty:
                    continue  # if this happens, crs is not defined. This can be a problem.
                nodes_dict[nodes[0]]['up'].append(np.interp(0.0, subset['chainage'], subset['minz']))
                nodes_dict[nodes[1]]['down'].append(np.interp(branch_length, subset['chainage'], subset['minz']))
            else:
                # In case of branch order, first collect all branches and set in them in order of up- to downstream
                all_branches = [self.mesh1d.description1d['network_branch_ids'][i]
                                for i in np.argwhere(order == branch_order).ravel()]
                all_nodes = [self.nodes_to_branch_map[b] for b in all_branches]
                # First check if any of the branches has a bedlevel from a cross-section profile otherwise skip
                check = all([bedlevels_crs_branches.loc[bedlevels_crs_branches['branchid'] == b].empty
                             for b in all_branches])
                if check:
                   continue  # if this happens, cross-section is not defined. This can be a problem.

                # Check if every branch is from up to down direction. Otherwise fix by reversing
                n = 0
                n_length = len(all_nodes)
                direction = list(np.ones(n_length))
                max_tries = 0
                while n < n_length:
                    up = np.count_nonzero(all_nodes[n][0] == np.array([x[0] for x in all_nodes])) == 1
                    down = np.count_nonzero(all_nodes[n][1] == np.array([x[1] for x in all_nodes])) == 1
                    if (not up) or (not down):
                        # Reverse
                        all_nodes[n] = list(np.flip(all_nodes[n]))
                        direction[n] = direction[n] * -1
                    n += 1
                    # Check if indeed everything is now in proper direction. Otherwise try again
                    if n == n_length:
                        up = all([np.count_nonzero(node[0] == np.array([x[0] for x in all_nodes])) == 1 for node in all_nodes])
                        down = all([np.count_nonzero(node[1] == np.array([x[1] for x in all_nodes])) == 1 for node in all_nodes])
                        if (not up) or (not down):
                            n = 0
                            max_tries += 1
                    if max_tries > 500:
                        print(f"Can't fix correct directions branch groups {all_branches}")
                        break

                # Add reserved branches to return
                reserved_branches.extend([b for b, d in zip(all_branches, direction) if d == -1])

                # Get most upstream node. Otherwise just pick i_upstream = 0 as starting point
                i_upstream = [i for i, n in enumerate([x[0] for x in all_nodes])
                              if n not in [x[1] for x in all_nodes]]
                if len(i_upstream) == 1:
                    i_upstream = i_upstream[0]
                else:
                    # It could be that branch order group forms a ring. In this case check first which node has more
                    # than 2 branches (bifurcation) or just 1 branch (boundary) connected.
                    i_upstream = [i for i, n in enumerate([x[0] for x in all_nodes])
                                  if (len(self.branches_to_node_map[n]) > 2) or (len(self.branches_to_node_map[n]) == 1)]
                    if len(i_upstream) == 1:
                        i_upstream = i_upstream[0]
                    else:
                        raise ValueError(f"Something is not right with the branch order group {all_branches}")

                # Now put branch list in correct order
                all_branches_sorted = []
                all_nodes_sorted = []
                direction_sorted = []
                for ii in range(len(all_branches)):
                    all_branches_sorted.append(all_branches[i_upstream])
                    all_nodes_sorted.append(all_nodes[i_upstream])
                    direction_sorted.append(direction[i_upstream])
                    try:
                        i_upstream = [i for i, n in enumerate([x[0] for x in all_nodes])
                                      if [x[1] for x in all_nodes][i_upstream] == n][0]
                    except IndexError:
                        break
                # Stitch chainages and bedlevels together
                chainage, bedlevel = [], []
                branch_length = 0
                for b, d in zip(all_branches_sorted, direction_sorted):
                    subset = bedlevels_crs_branches.loc[bedlevels_crs_branches['branchid'] == b]
                    chain, bed = subset['chainage'], subset['minz']
                    # Reverse chainage and bedlevel arrays
                    if d == -1:
                        chain = np.flip(chain)
                        bed = np.flip(bed)
                    chainage.extend(chain + branch_length)
                    bedlevel.extend(bed)
                    branch_length = self.branches.loc[b, 'geometry'].length
                # Get chainage of up- and downstream node of loop branch within the overall branch
                if len(all_branches_sorted) == 1:
                    up_node_chainage = 0
                    down_node_chainage = self.branches.loc[all_branches_sorted[0], 'geometry'].length
                else:
                    i = np.argmax([1 if ((nodes == n) or (list(np.flip(nodes)) == n)) else 0 for n in all_nodes_sorted])
                    up_node_chainage = sum([0] + [self.branches.loc[b, 'geometry'].length
                                                  for b, n in zip(all_branches_sorted[:-1], all_nodes_sorted[:-1])][:i+1])
                    down_node_chainage = sum([self.branches.loc[b, 'geometry'].length
                                              for b, n in zip(all_branches_sorted, all_nodes_sorted)][:i+1])
                # Finally interpolate
                nodes_dict[nodes[0]]['up'].append(np.interp(up_node_chainage, chainage, bedlevel))
                nodes_dict[nodes[1]]['down'].append(np.interp(down_node_chainage, chainage, bedlevel))

        # Summarize everything and save
        nodes = list(nodes_dict.keys())
        node_geom = [Point(x,y) for x, y in zip(self.mesh1d.get_values('nnodex'), self.mesh1d.get_values('nnodey'))]
        if resolve_at_bifurcation_method == 'min':
            upstream_bedlevel = [np.min(v['up']) if len(v['up']) > 0 else np.nan for v in nodes_dict.values()]
            downstream_bedlevel = [np.min(v['down']) if len(v['down']) > 0 else np.nan for v in nodes_dict.values()]
        elif resolve_at_bifurcation_method == 'max':
            upstream_bedlevel = [np.max(v['up']) if len(v['up']) > 0 else np.nan for v in nodes_dict.values()]
            downstream_bedlevel = [np.max(v['down']) if len(v['down']) > 0 else np.nan for v in nodes_dict.values()]
        elif resolve_at_bifurcation_method == 'mean':
            upstream_bedlevel = [np.average(['up']) if len(v['up']) > 0 else np.nan for v in nodes_dict.values()]
            downstream_bedlevel = [np.average(v['down']) if len(v['down']) > 0 else np.nan for v in nodes_dict.values()]
        else:
            raise NotImplementedError

        self.nodes = gpd.GeoDataFrame(index=nodes_dict.keys(),
                                      data={'code': nodes_dict.keys(),
                                            'upstream_bedlevel': upstream_bedlevel,
                                            'downstream_bedlevel': downstream_bedlevel},
                                      geometry=node_geom)

        if return_reversed_branches:
            return list(np.unique(reserved_branches))

    def get_grouped_branches(self):
        """
        Get grouped branch ids to use in set_branch_order function
        """
        # Get all network data
        branch_ids = self.mesh1d.description1d['network_branch_ids']
        node_ids = self.mesh1d.description1d['network_node_ids']
        branch_edge_nodes_idx = self.mesh1d.get_values('nedge_nodes', as_array=True)
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
                        groups[[k for k, v in groups.items() if b in v][0]].append(branch_id)
                        branch_ids_checked.extend([branch_id])
                    else:
                        groups[list(groups.keys())[-1] + 1] = [branch_id]  # otherwise add to group
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
                        _groups[_k].extend(v)  # otherwise add group to first found group
                        _groups[_k] = list(np.unique(_groups[_k]))  # remove duplicates due to add
                        _groups.pop(k)  # and remove group from groups
            groups = _groups.copy()  # copy changed dict over original
        # One pass over all branches should be sufficient to group everything together. Otherwise raise error
        if max([sum([1 if b in v else 0 for k, v in groups.items()]) for b in branch_ids]) > 1:
            raise ValueError(f"Still branches contained in multiple groups. Maximum number of groups where this "
                             f"happens: {max([sum([1 if b in v else 0 for k, v in groups.items()]) for b in branch_ids])}")

        # save
        self.branch_groups = groups.copy()




    # generate network and 1d mesh
    def generate_1dnetwork_old(self, one_d_mesh_distance:float=40.0, seperate_structures:bool=True, max_dist_to_struc:float=None, urban_branches:list=[]) -> None:
        """
        Parameters
        ----------
        one_d_mesh_distance : float
            Distance between nodes
        separate_strucures: bool
            whether or not to put nodes between structures
        nax_dist_to_struc: float
        
        urban_branches: list
            list of urban branches (closed profiles)
        """

        if self.branches.empty:
            raise ValueError('Branches should be added before 1d network can be generated.')

        # Temporary dictionary to store the id number of the nodes and branches
        nodes = []
        edge_nodes_dict = {}

        # Check if any structures present (if not, structures will be None)
        structures = self.hydamo.structures.as_dataframe(generalstructures=False,
                                                               rweirs=True,
                                                               bridges=True,
                                                               culverts=True,
                                                               pumps=True,
                                                               uweirs=True,
                                                               orifices=True,
                                                               compounds=False)                                                               

        # If offsets are not predefined, generate them basd on one_d_mesh_distance
        if not self.offsets:
            self.generate_offsets(one_d_mesh_distance, structures=structures, max_dist_to_struc=max_dist_to_struc,urban_branches=urban_branches)

        # Add the network data to the 1d mesh structure
        sorted_branches = self.branches.iloc[self.branches.length.argsort().values]

        # Add network branch data
        dimensions = self.mesh1d.meshgeomdim
        dimensions.nbranches = len(sorted_branches)
        self.mesh1d.set_values('nbranchorder', (np.ones(dimensions.nbranches, dtype=int) * -1).tolist())
        self.mesh1d.set_values('nbranchlengths', sorted_branches.geometry.length + 1e-12)
        self.mesh1d.description1d['network_branch_ids'] = sorted_branches.index.astype(str).tolist()
        self.mesh1d.description1d['network_branch_long_names'] = sorted_branches.index.astype(str).tolist()
        
        # Add network branch geometry
        import itertools
        coords = [line.coords[:] for line in sorted_branches.geometry]
        geomx, geomy = list(zip(*list(itertools.chain(*coords))))[:2]
        dimensions.ngeometry = len(geomx)
        self.mesh1d.set_values('nbranchgeometrynodes', [len(lst) for lst in coords])
        self.mesh1d.set_values('ngeopointx', geomx)
        self.mesh1d.set_values('ngeopointy', geomy)

        branch_names = sorted_branches.index.astype(str).tolist()
        branch_longnames = ['long_' + s for s in branch_names]

        network_edge_nodes = []
        mesh1d_edge_nodes = []
        mesh1d_node_branchidx = []
        mesh1d_node_branchoffset = []        
        mesh1d_edge_branchidx = []
        mesh1d_edge_branchoffset = []
        mesh1d_edge_x = []
        mesh1d_edge_y = []       
        mesh1d_node_names = []
                   
        # For each branch
        for i_branch, branch in enumerate(sorted_branches.itertuples()):

            # Get branch coordinates. Round to 6 decimals to make sure that, due to precision errors in coordinates,
            # already existing first and/or end nodes are not recognized
            points = shapely.wkt.loads(shapely.wkt.dumps(branch.geometry, rounding_precision=6)).coords[:]

            # Network edge node administration
            # -------------------------------
            first_point = points[0]
            last_point = points[-1]
            
            # Get offsets from dictionary
            offsets = self.offsets[branch.Index]
            # The number of links on the branch
            nlinks = len(offsets) - 1
            
            # also get the offsets of the edge nodes, halfway the segments
            edge_offsets = [(offsets[i]+offsets[i+1])/2. for i in range(np.max([1,len(offsets)-1]))]

            # Check if the first and last point of the branch are already in the set            
            if (first_point not in nodes):
                first_present = False
                nodes.append(first_point)
            else:
                first_present = True
                offsets = offsets[1:]
                
            if (last_point not in nodes):
                last_present = False
                nodes.append(last_point)
            else:
                last_present = True
                offsets = offsets[:-1]
            
            # If no points remain, add an extra halfway: each branch should have at least 1 node
            if len(offsets) == 0:
                if branch.Index not in urban_branches:
                    offsets = np.array([branch.geometry.length / 2.])
                    edge_offsets = np.array([i * branch.geometry.length for i in [0.25, 0.75]])
                    nlinks += 1
                else:
                    offsets = self.offsets[branch.Index]
                                
            # Get the index of the first and last node in the dictionary (1 based, so +1)
            i_from = nodes.index(first_point) + 1
            i_to = nodes.index(last_point) + 1
            if i_from == i_to:                
                raise ValueError(f'For {branch.Index} a ring geometry was found: start and end node are the same. Ring geometries are not accepted.')

            network_edge_nodes.append([i_from, i_to])

            # Mesh1d edge node administration
            # -------------------------------
            # First determine the start index. This is equal to the number of already present points (+1, since 1 based)
            start_index = len(mesh1d_node_branchidx) + 1
            # For each link, create a new edge node connection
            if first_present:
                start_index -= 1
            new_edge_nodes = [[start_index + i, start_index + i + 1] for i in range(nlinks)]
            # If the first node is present, change the first point of the first edge to the existing point
            if first_present:
                new_edge_nodes[0][0] = edge_nodes_dict[first_point]
            else:
                edge_nodes_dict[first_point] = new_edge_nodes[0][0]
            # If the last node is present, change the last point of the last edge too
            if last_present:
                new_edge_nodes[-1][1] = edge_nodes_dict[last_point]
            else:
                edge_nodes_dict[last_point] = new_edge_nodes[-1][1]
            # Add to edge_nodes
            mesh1d_edge_nodes.extend(new_edge_nodes)
            
            # Update number of nodes
            mesh_point_names = [f'{branch.Index}_{offset:.2f}' for offset in offsets]
            
            # Append ids, longnames, branch and offset
            self.mesh1d.description1d['mesh1d_node_ids'].extend(mesh_point_names)
            self.mesh1d.description1d['mesh1d_node_long_names'].extend(mesh_point_names)
            mesh1d_node_branchidx.extend([i_branch + 1] * len(offsets))
            mesh1d_node_branchoffset.extend(offsets.tolist())
          
            lengths  = np.r_[0.,np.cumsum(np.hypot(np.diff(coords[i_branch],axis=0)[:,0], np.diff(coords[i_branch],axis=0)[:,1]))]
            edge_x = []
            edge_y = []
            for i_edge, edge in enumerate(edge_offsets):
                closest = (np.abs(lengths-edge)).argmin()
                if lengths[closest] > edge:
                    closest = closest - 1
                edge_x.append(edge/(lengths[closest+1]-lengths[closest])*(coords[i_branch][closest+1][0]-coords[i_branch][closest][0])+coords[i_branch][closest][0])                
                edge_y.append(edge/(lengths[closest+1]-lengths[closest])*(coords[i_branch][closest+1][1]-coords[i_branch][closest][1])+coords[i_branch][closest][1])             
               
            mesh1d_edge_branchidx.extend([i_branch+1] * len(edge_offsets))
            mesh1d_edge_branchoffset.extend(edge_offsets)
            mesh1d_edge_x.extend(edge_x)
            mesh1d_edge_y.extend(edge_y)
            
        # Parse nodes
        dimensions.nnodes = len(nodes)
        nodex, nodey = list(zip(*nodes))[:2]
        self.mesh1d.set_values('nnodex', nodex)
        self.mesh1d.set_values('nnodey', nodey)
        self.mesh1d.description1d['network_node_ids'].extend([f'{xy[0]:.6f}_{xy[1]:.6f}' for xy in nodes])
        self.mesh1d.description1d["network_node_long_names"].extend([f'x={xy[0]:.6f}_y={xy[1]:.6f}' for xy in nodes])

        # Add edge node data to mesh
        self.mesh1d.set_values('nedge_nodes', np.ravel(network_edge_nodes))
        self.mesh1d.meshgeomdim.numedge = len(mesh1d_edge_nodes)
        self.mesh1d.set_values('edge_nodes', np.ravel(mesh1d_edge_nodes))
        
        self.mesh1d.edge_branchidx  = mesh1d_edge_branchidx
        self.mesh1d.edge_branchoffset  = mesh1d_edge_branchoffset
        self.mesh1d.edge_x = mesh1d_edge_x
        self.mesh1d.edge_y = mesh1d_edge_y
                
        # Add mesh branchidx and offset to mesh
        dimensions.numnode = len(mesh1d_node_branchidx)
        self.mesh1d.set_values('branchidx', mesh1d_node_branchidx)
        self.mesh1d.set_values('branchoffsets', mesh1d_node_branchoffset)

        # Process the 1d network (determine x and y locations) and determine schematised branches
        schematised, _ = self.mesh1d.process_1d_network()
        for idx, geometry in schematised.items():
            self.schematised.at[idx, 'geometry'] = geometry
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _generate_1d_spacing(self, anchor_pts:list, one_d_mesh_distance:float) -> np.array:
        """
        Generates 1d distances, called by function generate offsets

        Parameters:
        anchor_pts : list
            list of points to offset nodes from (structures)
        one_d_mesh_distance : float
            distance between 1d nodes
        
        Result:
            offsets : np.array
        """
        offsets = []
        for i in range(len(anchor_pts) - 1):
            section_length = anchor_pts[i+1] - anchor_pts[i]
            if section_length <= 0.0:
                raise ValueError('Section length must be larger than 0.0')
            nnodes = max(2, int(round(section_length / one_d_mesh_distance) + 1))
            offsets.extend(np.linspace(anchor_pts[i], anchor_pts[i+1], nnodes - 1, endpoint=False).tolist())
        offsets.append(anchor_pts[-1])

        return np.asarray(offsets)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_offsets(self, one_d_mesh_distance:float, structures:pd.DataFrame=None, max_dist_to_struc:float=None, urban_branches:list=None):
        """
        Method to generate 1d network grid point locations. The distances are generated
        based on the 1d mesh distance and anchor points. The anchor points can for
        example be structures; every structure should be seperated by a gridpoint.

        Parameters
        one_d_mesh_distance : float
            distance between 1d nodes
        structures:
            dataframe with structures (anchor points)
        max_dist_to_struc : float
            maximum distance to structure
        urban_branches : list
            list of urban (closed) branches
        """
        # For each branch
        for branch in self.branches.itertuples():
            # Distribute points along network [1d mesh]
            if urban_branches is not None:
                if branch.Index in urban_branches:
                    offsets = self._generate_1d_spacing([0.0, branch.geometry.length], branch.geometry.length)
                else:
                    offsets = self._generate_1d_spacing([0.0, branch.geometry.length], one_d_mesh_distance)
            else:
                offsets = self._generate_1d_spacing([0.0, branch.geometry.length], one_d_mesh_distance)
            self.offsets[branch.Index] = offsets
        
        if structures is not None:

            # Get structure data from dfs
            ids_offsets = structures[['branchid', 'chainage']]
            idx = (structures['branchid'] != '')
            if idx.any():
                logger.warning('Some structures are not linked to a branch.')
            ids_offsets = ids_offsets.loc[idx, :]

            # For each branch
            for branch_id, group in ids_offsets.groupby('branchid'):

                # Check if structures are located at the same offset
                u, c = np.unique(group['chainage'], return_counts=True)
                if any(c > 1):
                    logger.warning('Structures {} have the same location.'.format(
                        ', '.join(group.loc[np.isin(group['chainage'], u[c>1])].index.tolist())))
                
                branch = self.branches.at[branch_id, 'geometry']
                # Limits are the lengths at which the structures are located
                limits = sorted(group['chainage'].unique())
                
                anchor_pts = [0.0, branch.length]
                offsets = self._generate_1d_spacing(anchor_pts, one_d_mesh_distance)

                # Merge limits with start and end of branch
                limits = [-1e-3] + limits + [branch.length + 1e-3]
                    
                # If any structures
                if len(limits) > 2:

                    # also check if the calculation point are close enough to the structures
                    if max_dist_to_struc is not None:          
                        additional = []

                        # Skip the first and the last, these are no structures
                        for i in range(1, len(limits)-1):
                            # if the distance between two limits is large than twice the max distance to structure,
                            # the mesh point will be too far away. Add a limit on the minimum of half the length and
                            # two times the max distance
                            dist_to_prev_limit = limits[i] - (max(additional[-1], limits[i-1]) if any(additional) else limits[i-1])
                            if dist_to_prev_limit > 2 * max_dist_to_struc:
                                additional.append(limits[i] - min(2 * max_dist_to_struc, dist_to_prev_limit / 2))

                            dist_to_next_limit = limits[i+1] - limits[i]
                            if dist_to_next_limit > 2 * max_dist_to_struc:
                                additional.append(limits[i] + min(2 * max_dist_to_struc, dist_to_next_limit / 2))

                        # Join the limits
                        limits = sorted(limits + additional)
                          
                    # Get upper and lower limits
                    upper_limits = limits[1:]
                    lower_limits = limits[:-1]
                    
                    # Determine the segments that are missing a grid point
                    in_range = [((offsets > lower) & (offsets < upper)).any() for lower, upper in zip(lower_limits, upper_limits)]

                    while not all(in_range):
                        # Get the index of the first segment without grid point
                        i = in_range.index(False)
                        
                        # Add it to the anchor pts
                        anchor_pts.append((lower_limits[i] + upper_limits[i]) / 2.)
                        anchor_pts = sorted(anchor_pts)
                        
                        # Generate new offsets
                        offsets = self._generate_1d_spacing(anchor_pts, one_d_mesh_distance)
                
                        # Determine the segments that are missing a grid point
                        in_range = [((offsets > lower) & (offsets < upper)).any() for lower, upper in zip(lower_limits, upper_limits)]
                    
                    if len(anchor_pts) > 2:
                        logger.info(f'Added 1d mesh nodes on branch {branch_id} at: {anchor_pts}, due to the structures at {limits}.')

                # Set offsets for branch id
                self.offsets[branch_id] = offsets
        
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_node_idx_offset(self, branch_id:str, pt:shapely.geometry.Point, nnodes:int=1) -> tuple:
        """
        Get the index and offset of a node on a 1d branch.
        The nearest node is looked for.
        """

        # Project the point on the branch
        dist = self.schematised[branch_id].project(pt)

        # Get the branch data from the networkdata
        branchidx = self.mesh1d.description1d['network_branch_ids'].index(self.str2chars(branch_id, self.idstrlength)) + 1
        pt_branch_id = self.mesh1d.get_values('branchidx', as_array=True)
        idx = np.where(pt_branch_id == branchidx)
        
        # Find nearest offset
        offsets = self.mesh1d.get_values('branchoffset', as_array=True)[idx]
        isorted = np.argsort(np.absolute(offsets - dist))
        isorted = isorted[:min(nnodes, len(isorted))]

        # Get the offset
        offset = [offsets[imin] for imin in isorted]
        # Get the id of the node
        node_id = [idx[0][imin] + 1 for imin in isorted]

        return node_id, offset



class CrossSections:

    def __init__(self, hydamo):
        self.hydamo = hydamo
        self.crosssections = []
        self.default_definition = ""
        self.default_definition_shift = 0.0
        self.default_location = ""

        self.crosssection_loc = {}
        self.crosssection_def = {}

        self.get_roughnessname = self.get_roughness_description        
    
        self.convert = fmconverter.CrossSectionsIO(self)       

    def get_roughness_description(self, roughnesstype, value):

        if np.isnan(float(value)):
            raise ValueError('Roughness value should not be NaN.')
        
        # Convert integer to string
        #if isinstance(roughnesstype, int):
        #    roughnesstype = hydamo_to_dflowfm.roughness_gml[roughnesstype]
        
        # Get name
        name = f'{roughnesstype}_{float(value)}'

        # Check if the description is already known
        if name.lower() in map(str.lower, self.hydamo.roughness_definitions.keys()):
            return name
        
        # Convert roughness type string to integer for dflowfm
        delft3dfmtype = roughnesstype

        if roughnesstype.lower() == 'stricklerks':
            raise ValueError()

        # Add to dict
        self.hydamo.roughness_definitions[name] = {
            'frictionid': name,
            'frictiontype': delft3dfmtype,
            'frictionvalue': value
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

    def add_yz_definition(self, yz=None, thalweg=None, roughnesstype=None, roughnessvalue=None, name=None):
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
            name = f'yz_{yz}:08d'
        
        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary        
        self.crosssection_def[name] = {
            'id' : name,
            'type': 'yz',
            'thalweg': np.round(thalweg,decimals=3),
            'yzcount': len(z),
            'ycoordinates': self.hydamo.list_to_str(length),            
            'zcoordinates': self.hydamo.list_to_str(z),
            'sectioncount': 1,
            'frictionids': roughnessname,
            'frictionpositions': self.hydamo.list_to_str([length[0], length[-1]])
        }

        return name

    def add_circle_definition(self, diameter, roughnesstype, roughnessvalue, name=None):
        """
        Add circle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """        
        # Get name if not given
        if name is None:
            name = f'circ_d{diameter:.3f}'
        
        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            'id' : name,
            'type': 'circle',
            'thalweg': 0.0,
            'diameter': diameter,
            'frictionid': roughnessname
        }

        return name        

    def add_rectangle_definition(self, height, width, closed, roughnesstype, roughnessvalue, name=None):
        """
        Add rectangle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """        
        # Get name if not given
        if name is None:
            name = f'rect_h{height:.3f}_w{width:.3f}'

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            'id' : name,
            'type': 'rectangle',
            'thalweg': 0.0,
            'height': height,
            'width': width,
            'closed': int(closed),
            'frictionid': roughnessname
        }

        return name

    def add_trapezium_definition(self, slope, maximumflowwidth, bottomwidth, closed, roughnesstype, roughnessvalue, bottomlevel=None, name=None):
        """
        Add rectangle cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """        
        # Get name if not given
        if name is None:
            name = f'trapz_s{slope:.1f}_bw{bottomwidth:.1f}_bw{maximumflowwidth:.1f}'
        
        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)
        
        if bottomlevel is None:
            bottomlevel = 0.0
            
        if not closed:
            levels = f'{bottomlevel} 100'
            flowwidths = f'{bottomwidth:.2f} {bottomwidth + 2.*((100.0-bottomlevel)*slope):.2f}'
        else:
            levels = f'0 {((maximumflowwidth - bottomwidth)/2.0) / slope:.2f}'
            flowwidths = f'{bottomwidth:.2f} {maximumflowwidth:.2f}'

        # Add to dictionary
        self.crosssection_def[name] = {
            'id' : name,
            'type': 'zw',
            'thalweg': 0.0,
            'numlevels': 2,
            'levels': levels,
            'flowwidths': flowwidths,
            'totalwidths': flowwidths,
            'frictionid': roughnessname
        }

        return name

    def add_zw_definition(self, numLevels, levels, flowWidths, totalWidths,roughnesstype, roughnessvalue,
                                 name=None):
        """
        Add zw cross section. The cross section name is derived from the shape and roughness,
        so similar cross sections will result in a single definition.
        """
        # Get name if not given
        if name is None:
            name = f'zw_h{levels.replace(" ","_"):.1f}_w{flowWidths.replace(" ","_"):.1f}'

        # Get roughnessname
        roughnessname = self.get_roughnessname(roughnesstype, roughnessvalue)

        # Add to dictionary
        self.crosssection_def[name] = {
            'id': name,
            'type': 'zw',
            'thalweg': 0.0,
            'numlevels': int(numLevels),
            'levels': levels,
            'flowwidths': flowWidths,
            'totalwidths': totalWidths,
            'frictionid': roughnessname
        }

        return name

    def add_crosssection_location(self, branchid, chainage, definition, minz=np.nan, shift=0.0):

        descr = f'{branchid}_{chainage:.1f}'
        # Add cross section location
        self.crosssection_loc[descr] = {
            'id': descr,
            'branchid': branchid,
            'chainage': chainage,
            'shift': shift,
            'definitionId': definition,
        }

    def get_branches_without_crosssection(self):
        # First find all branches that match a cross section
        branch_ids = {dct['branchid'] for _, dct in self.crosssection_loc.items()}
        # Select the branch-ids that do nog have a matching cross section
        branches = self.hydamo.branches
        no_crosssection = branches.index[~np.isin(branches.index, list(branch_ids))]

        return no_crosssection.tolist()

    def get_structures_without_crosssection(self):
        struc_ids =  [dct['id'] for _, dct in self.crosssection_def.items()]
        bridge_ids = [dct['csDefId'] for _, dct in self.hydamo.structures.bridges.items()] 
        no_cross_bridge = np.asarray(bridge_ids)[~np.isin(bridge_ids , struc_ids)].tolist() 
        no_crosssection = no_cross_bridge              
        return no_crosssection

    def get_bottom_levels(self):
        """Method to determine bottom levels from cross sections"""

        # Initialize lists
        data = []
        geometry = []
        
        for key, css in self.crosssection_loc.items():
            # Get location
            geometry.append(self.dflowfmmodel.network.schematised.at[css['branchid'], 'geometry'].interpolate(css['chainage']))
            shift = css['shift']

            # Get depth from definition if yz and shift
            definition = self.crosssection_def[css['definitionId']]
            minz = shift
            if definition['type'] == 'yz':
                minz += min(float(z) for z in definition['zCoordinates'].split())
            
            data.append([css['branchid'], css['chainage'], minz])

        # Add to geodataframe
        gdf = gpd.GeoDataFrame(
            data=data,
            columns=['branchid', 'chainage', 'minz'],
            geometry=geometry
        )
        return gdf
     
    @validate_arguments(config=dict(arbitrary_types_allowed=True))  
    def crosssection_to_yzprofiles(self, crosssections:ExtendedGeoDataFrame, 
                                   roughness:ExtendedDataFrame, 
                                   branches:ExtendedGeoDataFrame, 
                                   roughness_variant:RoughnessVariant=None) -> dict:
    
        """
        Function to convert hydamo cross sections 'dwarsprofiel' to
        dflowfm input.
        d
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
            length = np.r_[0, np.cumsum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1])))]
            yz = np.c_[length, xyz[:, -1]]
            # the GUI cannot cope with identical y-coordinates. Add 1 cm to a 2nd duplicate.
            yz[:,0] = np.round(yz[:,0],3)
            for i in range(1,yz.shape[0]):
                if yz[i,0]==yz[i-1,0]:
                    yz[i,0] +=0.01
                    
            # determine thalweg
            if branches is not None:                              
                branche_geom = branches[branches.code==css.branch_id].geometry.values

                if css.geometry.intersection(branche_geom[0]).geom_type=='MultiPoint':
                    thalweg_xyz = css.geometry.intersection(branche_geom[0])[0].coords[:][0]                            
                else:
                    thalweg_xyz = css.geometry.intersection(branche_geom[0]).coords[:][0]                
                # and the Y-coordinate of the thalweg
                thalweg = np.hypot( thalweg_xyz[0]-xyz[0,0], thalweg_xyz[1]-xyz[0,1])
            else: 
                thalweg = 0.0
            
            if roughness_variant == RoughnessVariant.HIGH:
                ruwheid=roughness[roughness['profielpuntid']==css.globalid].ruwheidhoog
            if roughness_variant == RoughnessVariant.LOW:
                ruwheid=roughness[roughness['profielpuntid']==css.globalid].ruwheidlaag
                
            # Add to dictionary
            cssdct[css.code] = {
                'branchid': css.branch_id,
                'chainage': css.branch_offset,
                'yz': yz,
                'thalweg':thalweg,
                'typeruwheid': roughness[roughness['profielpuntid']==css.globalid].typeruwheid.values[0],
                'ruwheid': float(ruwheid)
            }
        
        return cssdct

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def parametrised_to_profiles(self, 
                                 parametrised:ExtendedDataFrame, 
                                 parametrised_values:ExtendedDataFrame,  
                                 branches:list,
                                 roughness_variant:RoughnessVariant=None) -> dict:
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
            branch = [branch for branch in branches if branch.globalid==param.hydroobjectid]
        
            values = parametrised_values[parametrised_values.normgeparamprofielid==param.normgeparamprofielid]
        
            #Drop profiles for which not enough data is available to write (as rectangle)
            # nulls = pd.isnull(parambranches[['bodembreedte', 'bodemhoogtebenedenstrooms', 'bodemhoogtebovenstrooms']]).any(axis=1).values
            # parambranches = parambranches.drop(ExtendedGeoDataFrame(geotype=LineString), parambranches.index[nulls], index_col='code',axis=0)
            # parambranches.drop(parambranches.index[nulls], inplace=True)
            
            if pd.isnull(values[values.soortparameter=='bodemhoogte benedenstrooms'].waarde).values[0]:
                print('bodemhoogte benedenstrooms not available for profile {}.'.format(param.globalid))
            if pd.isnull(values[values.soortparameter=='bodembreedte'].waarde).values[0]:
                print('bodembreedte not available for profile {}.'.format(param.globalid))
            if pd.isnull(values[values.soortparameter=='bodemhoogte bovenstrooms'].waarde).values[0]:
                print('bodemhoogte bovenstrooms not available for profile {}.'.format(param.globalid))
            
            # Determine characteristics
            botlev = (values[values.soortparameter=='bodemhoogte benedenstrooms'].waarde.values[0] + values[values.soortparameter=='bodemhoogte benedenstrooms'].waarde.values[0]) / 2.0       
            
            
            if pd.isnull(values[values.soortparameter=='taludhelling linkerzijde'].waarde).values[0]:
                css_type=='rectangle'
            else:
                css_type = 'trapezium'
                dh1 = values[values.soortparameter=='hoogte insteek linkerzijde'].waarde.values[0] - botlev
                dh2 = values[values.soortparameter=='hoogte insteek rechterzijde'].waarde.values[0] - botlev
                height = (dh1 + dh2) / 2.0
                # Determine maximum flow width and slope (both needed for output)
                maxflowwidth = values[values.soortparameter=='bodembreedte'].waarde.values[0] + values[values.soortparameter=='taludhelling linkerzijde'].waarde.values[0] * dh1 + values[values.soortparameter=='taludhelling rechterzijde'].waarde.values[0] * dh2
                slope = (values[values.soortparameter=='taludhelling linkerzijde'].waarde.values[0] + values[values.soortparameter=='taludhelling rechterzijde'].waarde.values[0]) / 2.0
            
            if roughness_variant==RoughnessVariant.LOW:
                roughness =  values.ruwheidlaag.values[0]
            elif roughness_variant==RoughnessVariant.HIGH:
                roughness = values.ruwheidhoog.values[0]
            else:
                ValueError('Invalid value for roughness_variant; should be "High" or "Low".')
            # Determine name for cross section
            if css_type == 'trapezium':
                cssdct[branch[0].Index] = {
                    'type': css_type,
                    'slope': round(slope, 2),
                    'maximumflowwidth': round(maxflowwidth, 1),
                    'bottomwidth': round(values[values.soortparameter=='bodembreedte'].waarde.values[0], 3),
                    'closed': 0,
                    'thalweg': 0.0,
                    'typeruwheid': values.typeruwheid.values[0],
                    'ruwheid': roughness,
                    'bottomlevel': botlev
                }
            elif css_type == 'rectangle':
                cssdct[branch[0].Index] = {
                    'type': css_type,
                    'height': 5.0,
                    'width': round(values[values.soortparameter=='bodembreedte'].waarde.values[0], 3),
                    'closed': 0,
                    'thalweg': 0.0,
                    'typeruwheid': values.typeruwheid.values[0],
                    'ruwheid': roughness,        
                    'bottomlevel': botlev
                }

        return cssdct
   
class ExternalForcings:

    def __init__(self, hydamo):
        self.hydamo = hydamo

        self.initial_waterlevel_polygons = gpd.GeoDataFrame(columns=['waterlevel', 'geometry','locationtype'])
        self.initial_waterdepth_polygons = gpd.GeoDataFrame(columns=['waterdepth', 'geometry','locationtype'])
        self.missing = None

        self.boundary_nodes = {}
        self.lateral_nodes = {}
        self.pattern =  "^[{]?[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}[}]?$"
   
        self.convert = fmconverter.ExternalForcingsIO(self)

    def set_initial_waterlevel(self, level, polygon=None, name=None, locationtype='1d'):
        """
        Method to set initial water level. A polygon can be given to
        limit the initial water level to a certain extent. 

        """
        # Get name is not given as input
        if name is None:
            name = 'wlevpoly{:04d}'.format(len(self.initial_waterlevel_polygons) + 1)

        # Add to geodataframe
        if polygon==None:
            new_df = pd.DataFrame({'waterlevel': level, 'geometry': polygon, 'locationtype':locationtype}, index=[name])
            self.initial_waterlevel_polygons =  new_df
        else:            
            self.initial_waterlevel_polygons.loc[name] = {'waterlevel': level, 'geometry': polygon, 'locationtype':locationtype}

    def set_missing_waterlevel(self, missing):
        """
        Method to set the missing value for the water level.
        this overwrites the water level at missing value in the mdu file.

        Parameters
        ----------
        missing : float
            Water depth
        """
        self.mdu_parameters['WaterLevIni'] = missing
    
    def set_initial_waterdepth(self, depth, polygon=None, name=None, locationtype='1d'):
        """
        Method to set the initial water depth in the 1d model. The water depth is
        set by determining the water level at the locations of the cross sections.
        
        Parameters
        ----------
        depth : float
            Water depth
        """
         # Get name is not given as input
        if name is None:
            name = 'wlevpoly{:04d}'.format(len(self.initial_waterdepth_polygons) + 1)
        # Add to geodataframe
        if polygon==None:
            
            new_df = pd.DataFrame({'waterdepth': depth, 'geometry': polygon, 'locationtype':locationtype}, index=[name])
            
            self.initial_waterdepth_polygons =  new_df
        else:
            self.initial_waterdepth_polygons.loc[name] = {'waterdepth': depth, 'geometry': polygon, 'locationtype':locationtype}
        
    def add_rainfall_2D(self, fName, bctype='rainfall'):
        """
        Parameters
        ----------
        fName : str
            Location of netcdf file containing rainfall rasters
        bctype : str
            Type of boundary condition. Currently only rainfall is supported
        """        
        assert bctype in ['rainfall']
        
        # Add boundary condition
        self.boundaries['rainfall_2D'] = {
            'file_name': fName,
            'bctype': bctype+'bnd',
        }
        


    @validate_arguments
    def add_boundary_condition(self, name:str, pt, quantity:str, series, mesh1d=None) -> None:
        """
        Add boundary conditions to model:
        - The boundary condition can be discharge or waterlevel
        - Is specified by a geographical location (pt) and a branchid
        - If no branchid is given, the nearest is searched
        - The boundary condition is added to the end of the given or nearest branch.
        - To specify a time dependendend boundary: a timeseries with values should be given
        - To specify a constant boundary a float should be given
        
        Parameters
        ----------
        name : str
            ID of the boundary condition
        pt : tuple or shapely.geometry.Point
            Location of the boundary condition
        bctype : str
            Type of boundary condition. Currently only discharge and waterlevel are supported
        series : pd.Series or float
            If a float, a constant in time boundary condition is used. If a pandas series,
            the values per time step are used. Index should be in datetime format
        branchid : str, optional
            ID of the branch. If None, the branch nearest to the given location (pt) is
            searched, by default None
        """

        assert quantity in ['dischargebnd', 'waterlevelbnd']

        unit = 'm3/s' if quantity=='dischargebnd' else 'm'
        if name in self.boundary_nodes.keys():
            raise KeyError(f'A boundary condition with name "{name}" is already present.')
        
        if isinstance(pt, tuple):
            pt = Point(*pt)

        # Find the nearest node   
        if len(mesh1d._mesh1d.mesh1d_node_id) == 0:
            raise KeyError('To find the closest node a 1d mesh should be created first.')                      
        nodes1d = np.asarray([n for n in zip(mesh1d._mesh1d.mesh1d_node_x,mesh1d._mesh1d.mesh1d_node_y,mesh1d._mesh1d.mesh1d_node_id)])
        get_nearest = KDTree(nodes1d[:,0:2])
        distance, idx_nearest = get_nearest.query(pt)
        nodeid = nodes1d[idx_nearest,2]

        # Convert time to minutes
        if isinstance(series, pd.Series):
            times = ((series.index - series.index[0]).total_seconds() / 60.).tolist()
            values = series.values.tolist()
            startdate = pd.datetime.strftime(series.index[0],'%Y-%m-%d %H:%M:%S')
        else:
            times = None
            values = series
            startdate = '0000-00-00 00:00:00'
        
        # Add boundary condition
        self.boundary_nodes[name] = {
            'id': name,
            'quantity': quantity,
            'value': values,
            'time': times,                       
            'time_unit': f'minutes since {startdate}',
            'value_unit' : unit,            
            'value' : values,
            'nodeid': nodeid,            
        }
        
        # Check if a 1d2d link should be removed
        #self.dflowfmmodel.network.links1d2d.check_boundary_link(self.boundaries[name])


    @validate_arguments
    def add_rain_series(self, name:str, values:list, times:list) -> None:
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
            'code' : name,
            'bctype' : 'rainfall',
            'filetype' : 1,
            'method' : 1,
            'operand' : 'O',
            'value': values,
            'time': times,
            'geometry': None,
            'branchid': None
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_lateral(self, id:str, branchid:str, chainage:str, discharge:Union[pd.Series, str]) -> None:
        """Add a lateral to an FM model

        Args:
            id (str): Id of th lateral node
            name (str): name of the node
            branchid (str): branchid it is snapped to
            chainage (str): chainage on the branch
            discharge (str, or pd.Series): discharge type: REALTIME when linked to RR, or float (constant value) or a pd.Series with time index
        """
         # Convert time to minutes
        if isinstance(discharge, pd.Series):
            times = ((discharge.index - discharge.index[0]).total_seconds() / 60.).tolist()
            values = discharge.values.tolist()
            startdate = pd.datetime.strftime(discharge.index[0],'%Y-%m-%d %H:%M:%S')
        else:
            times = None
            values = discharge
            startdate = '0000-00-00 00:00:00'
            if discharge != 'realtime':     
                discharge = None

        self.lateral_nodes[id] = {
            'id' : id,
            'name' : id,
            'type' : 'discharge',
            'locationtype' : '1d',
            'branchid': branchid,
            'chainage': chainage,
            'time': times,                       
            'time_unit': f'minutes since {startdate}',
            'value_unit' : 'm3/s',            
            'value' : values,
            'discharge': discharge        
        }                  

class Structures:

    def __init__(self, hydamo):
        self.hydamo = hydamo
        self.generalstructures_df = []
        self.rweirs_df = pd.DataFrame()
        self.orifices_df =pd.DataFrame()
        self.uweirs_df = pd.DataFrame()
        self.culverts_df = pd.DataFrame()
        self.bridges_df = pd.DataFrame()
        self.pumps_df = pd.DataFrame()
        self.compounds_df = pd.DataFrame()
     
        self.convert = fmconverter.StructuresIO(self)

    @validate_arguments
    def add_rweir(self,     id: str=None, 
                            name:str=None, 
                            branchid:str=None, 
                            chainage:float=None, 
                            crestlevel:float=None, 
                            crestwidth:float=None, 
                            corrcoeff:float=None, 
                            usevelocityheight:str=None, 
                            allowedflowdir:str=None) ->None:
        """
        Function to add a regular weir. Arguments correspond to the required input of DFlowFM.
        """
        if allowedflowdir is None: allowedflowdir = 'both'
        if usevelocityheight is None: usevelocityheight='true'
        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'crestlevel':crestlevel,'crestwidth':crestwidth,
                            'corrcoeff':corrcoeff,'usevelocityheight':usevelocityheight,'allowedflowdir':allowedflowdir}, index=[id])        
        self.rweirs_df = pd.concat([self.rweirs_df, dct], ignore_index=True)   
        
    @validate_arguments
    def add_orifice(self,   id: str=None, 
                            name:str=None, 
                            branchid:str=None, 
                            chainage:float=None, 
                            crestlevel:float=None, 
                            crestwidth:float=None, 
                            corrcoeff:float=None, 
                            usevelocityheight:str=None, 
                            allowedflowdir:str=None, 
                            gateloweredgelevel:float=None,                             
                            uselimitflowpos:bool=None,
                            limitflowpos:str=None,
                            uselimitflowneg:bool=None,
                            limitflowneg:str=None) ->None:
        """
        Function to add a orifice. Arguments correspond to the required input of DFlowFM.
        """
        if allowedflowdir is None: allowedflowdir = 'both'
        if usevelocityheight is None: usevelocityheight='true'       
        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'crestlevel':crestlevel,'crestwidth':crestwidth,
                            'corrcoeff':corrcoeff,'usevelocityheight':usevelocityheight,'allowedflowdir':allowedflowdir,
                            'gateloweredgelevel':gateloweredgelevel,'uselimitflowpos':uselimitflowpos,
                            'limitflowpos':limitflowpos,'uselimitflowneg':uselimitflowneg,'limitflowneg':limitflowneg}, index=[id])        
        self.orifices_df = pd.concat([self.orifices_df, dct], ignore_index=True)   
        
        
    @validate_arguments
    def add_uweir(self,     id: str=None, 
                            name:str=None, 
                            branchid:str=None, 
                            chainage:float=None, 
                            crestlevel:float=None, 
                            crestwidth:float=None, 
                            dischargecoeff:float=None, 
                            usevelocityheight:str=None, 
                            allowedflowdir:str=None, 
                            numlevels:float=None,                             
                            yvalues:str=None,
                            zvalues:str=None) ->None:
        """
        Function to add a universalweir. Arguments correspond to the required input of DFlowFM.
        """
        if allowedflowdir is None: allowedflowdir = 'both'
        if usevelocityheight is None: usevelocityheight='true'
        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'crestlevel':crestlevel,'crestwidth':crestwidth,
                            'dischargecoeff':dischargecoeff,'usevelocityheight':usevelocityheight,'numlevels':numlevels, 'allowedflowdir':allowedflowdir,
                            'yvalues':yvalues,'zvalues':zvalues}, index=[id])        
        self.uweirs_df = pd.concat([self.uweirs_df, dct], ignore_index=True)            
            
    @validate_arguments
    def add_bridge(self,   id: str=None, 
                            name:str=None, 
                            branchid:str=None, 
                            chainage:float=None, 
                            length:float=None, 
                            inletlosscoeff:float=None, 
                            outletlosscoeff:float=None, 
                            csdefid:str=None, 
                            shift:float=None, 
                            allowedflowdir:str=None, 
                            frictiontype:str=None,
                            friction:float=None) ->None:

        if allowedflowdir is None: allowedflowdir = 'both'
        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'length':length,'inletlosscoeff':inletlosscoeff,
                            'outletlosscoeff':outletlosscoeff,'csdefid':csdefid,'shift':shift, 'allowedflowdir':allowedflowdir,
                            'frictiontype':frictiontype,'friction':friction}, index=[id])        
        self.bridges_df = pd.concat([self.bridges_df, dct], ignore_index=True)   
    
                   
    @validate_arguments
    def add_culvert(self,   id: str=None, 
                            name:str=None, 
                            branchid:str=None, 
                            chainage:float=None, 
                            leftlevel:float=None, 
                            rightlevel:float=None, 
                            length:float=None, 
                            inletlosscoeff:float=None, 
                            outletlosscoeff:float=None, 
                            crosssection:dict=None, 
                            allowedflowdir:str=None, 
                            valveonoff:int=None, 
                            numlosscoeff:int=None, 
                            valveopeningheight:float=None, 
                            relopening:list=None, 
                            losscoeff:list=None,
                            frictiontype:str=None,
                            frictionvalue:float=None) ->None:

        if allowedflowdir is None: allowedflowdir = 'both'
        if valveonoff is None: valveonoff = 0        
        if valveopeningheight is None: valveopeningheight = 0 
        
        roughnessname = self.hydamo.crosssections.get_roughness_description(frictiontype, frictionvalue)

        if crosssection['shape'] == 'circle':
            definition = self.hydamo.crosssections.add_circle_definition(crosssection['diameter'], frictiontype, frictionvalue, name=id)
        elif crosssection['shape'] == 'rectangle':
            definition = self.hydamo.crosssections.add_rectangle_definition(crosssection['height'], crosssection['width'], crosssection['closed'], frictiontype, frictionvalue, name=id)
        else:
                NotImplementedError(f'Cross section with shape \"{crosssection["shape"]}\" not implemented.')

        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'rightlevel':rightlevel,'leftlevel':leftlevel,
                            'length':length, 'inletlosscoeff':inletlosscoeff,'outletlosscoeff':outletlosscoeff,'csdefid':definition,
                            'allowedflowdir':allowedflowdir,'valveonoff':valveonoff,'numlosscoeff':numlosscoeff, 'valveopeningheight':valveopeningheight,
                            'relopening':[relopening],'losscoeff':[losscoeff] }, index=[id])       
        
        self.culverts_df = pd.concat([self.culverts_df, dct], ignore_index=True)
    

   
    @validate_arguments
    def add_pump(self, id: str=None, 
                       name:str=None, 
                       branchid:str=None, 
                       chainage:float=None, 
                       orientation:str=None, 
                       numstages:int=None, 
                       controlside:str=None, 
                       capacity:float=None, 
                       startlevelsuctionside:list=None, 
                       stoplevelsuctionside:list=None,
                       startleveldeliveryside:list=None,
                       stopleveldeliveryside:list=None) ->None:

        if numstages is None: numstages = 1 
        if orientation is None: orientation = 'positive'
        if controlside is None: controlside = 'suctionSide'        
        dct = pd.DataFrame({'id':id, 'name':name,'branchid':branchid,'chainage':chainage,'orientation':orientation,'numstages':numstages,'controlside':controlside, 
        'capacity':capacity,'startlevelsuctionside':[startlevelsuctionside],'stoplevelsuctionside':[stoplevelsuctionside] ,'startleveldeliveryside':[startleveldeliveryside],'stopleveldeliveryside':[stopleveldeliveryside]}, index=[id])        
        self.pumps_df = pd.concat([self.pumps_df, dct], ignore_index=True)
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def as_dataframe(self,
                     generalstructures:bool=False,
                     pumps:bool=False,
                     rweirs:bool=False,
                     bridges:bool=False,
                     culverts:bool=False,
                     uweirs:bool=False,
                     orifices:bool=False,
                     compounds:bool=False) -> pd.DataFrame:
        """
        Returns a dataframe with the structures. Specify with the keyword arguments what structure types need to be returned.
        """
        dfs = []
        for df, descr, add in zip([self.generalstructures, self.culverts, self.rweirs, self.bridges, self.pumps, self.uweirs, self.orifices, self.compounds],
                                  ['generalstructure', 'culvert', 'weir','bridge', 'pump', 'uweir','orifice','compound'],
                                  [generalstructures, culverts, rweirs, bridges, pumps, uweirs, orifices, compounds]):
            if any(df) and add:
                # df = pd.DataFrame.from_dict(df, orient='index')
                df.insert(loc=0, column='structype', value=descr, allow_duplicates=True)
                dfs.append(df)

        if len(dfs) > 0:
            return pd.concat(dfs, sort=False)

class ObservationPoints:

    def __init__(self, hydamo):
        self.hydamo = hydamo        
        self.observation_points = pd.DataFrame()  

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_points(self, crds:list, names, locationTypes=None, snap_distance:float=5.) -> None:
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

        if isinstance(names,str):
            names = [names]
            crds = [crds]

        if locationTypes is not None:
            if isinstance(names,str):
                locationTypes = [locationTypes]
            
            # split 1d and 2d points, as the first ones need to be snapped to branches
            obs2d = gpd.GeoDataFrame()
            obs2d['name'] = [n for nn,n in enumerate(names) if locationTypes[nn]=='2d']
            obs2d['locationtype'] = '2d'
            obs2d['geometry'] = [Point(*pt) if not isinstance(pt, Point) else pt for ipt,pt in enumerate(crds) if (locationTypes[ipt]=='2d')]
            obs2d['x'] = [pt.coords[0][0] for pt in obs2d['geometry']]
            obs2d['y'] = [pt.coords[0][1] for pt in obs2d['geometry']]
            names1d = [n for n_i,n in enumerate(names) if locationTypes[n_i]=='1d']
            crds1d = [c for c_i,c in enumerate(crds) if locationTypes[c_i]=='1d']
        else:
            names1d = names
            crds1d = crds

        obs1d = gpd.GeoDataFrame()
        obs1d['name'] = names1d
        obs1d['geometry'] = [Point(*pt) if not isinstance(pt, Point) else pt for pt in crds1d]        
        obs1d['locationtype'] = '1d'        
        find_nearest_branch(self.hydamo.branches, obs1d, method='overal', maxdist=snap_distance)      
        obs1d.rename(columns={'branch_id':'branchid', 'branch_offset': 'chainage'}, inplace=True)
        obs = obs1d.append(obs2d, sort=True) if locationTypes is not None else obs1d
                    
        # Add to dataframe        
        self.observation_points = obs
                    
class StorageNodes:

    def __init__(self, hydamo):
        self.storagenodes = {}

        self.hydamo = hydamo

    def add_storagenode(self, id, nodeid, usestreetstorage='true', nodetype='unspecified',
                        name=np.nan, usetable='false',
                        bedlevel=np.nan, area=np.nan, streetlevel=np.nan, streetstoragearea=np.nan, storagetype='reservoir',
                        levels=np.nan, storagearea=np.nan, interpolate='linear'):
        base = {"type": "storageNode",
                'id': id,
                'name': id if np.isnan(name) else name,
                'useStreetStorage': usestreetstorage,
                'nodeType': nodetype,
                'nodeId': nodeid,
                'useTable': usetable}
        if usetable == 'false':
            out = {**base,
                   'bedLevel': bedlevel,
                   'area': area,
                   'streetLevel': streetlevel,
                   'streetStorageArea': streetstoragearea,
                   'storageType': storagetype}
        elif usetable == 'true':
            assert len(levels.split()) == len(storagearea.split()), 'Number of levels does not equal number of storagearea'
            out = {**base,
                   'numLevels': len(levels.split()),
                   'levels': area,
                   'storageArea': streetlevel,
                   'interpolate': interpolate}
        else:
            raise ValueError("Value of key 'usetable' is not supported. Either use 'true' or 'false")
        self.storagenodes[id] = remove_nan_values(out)


def remove_nan_values(base):
    base_copy = base.copy()
    for k, v in base.items():
        if isinstance(v, float):
            if np.isnan(v):
                base_copy.pop(k)
    return base_copy



#     def compound_structures(self, idlist, structurelist):
#         """
#         Method to add compound structures to the model.
        
#         """
#         geconverteerd = hydamo_to_dflowfm.generate_compounds(idlist, structurelist, self.structures)
        
#          # Add to dict
#         for compound in geconverteerd.itertuples():
#             self.structures.add_compound(
#                 id=compound.code,
#                 numstructures=compound.numstructures,
#     	        structurelist=compound.structurelist    	        
#             )
        
  #     def __init__(self, crosssections):
#         self.crosssections = crosssections

#     def from_datamodel(self, crsdefs=None, crslocs=None):
#         """"
#         From parsed data models of crsdefs and crs locs
#         """

#         if crslocs is not None:
#             for crsloc_idx, crsloc in crslocs.iterrows():
#                 # add location
#                 self.crosssections.add_crosssection_location(branchid=crsloc['branch_id'],
#                                                              chainage=crsloc['branch_offset'],
#                                                              shift=crsloc['shift'],
#                                                              definition=crsloc['crosssectiondefinitionid'])

#         if crsdefs is not None:
#             crsdefs = crsdefs.drop_duplicates(subset=['crosssectiondefinitionid'])
#             for crsdef_idx, crsdef in crsdefs.iterrows():
#                 # Set roughness value on default if cross-section has non defined (e.g. culverts)
#                 roughtype = crsdef['frictionid'].split('_')[0] if isinstance(crsdef['frictionid'], str) else 'Chezy'
#                 roughval = float(crsdef['frictionid'].split('_')[-1]) if isinstance(crsdef['frictionid'], str) else 45
#                 # add definition
#                 if crsdef['type'] == 'circle':
#                     self.crosssections.add_circle_definition(diameter=crsdef['diameter'],
#                                                              roughnesstype=roughtype,
#                                                              roughnessvalue=roughval,
#                                                              name=crsdef['crosssectiondefinitionid'])
#                 elif crsdef['type'] == 'rectangle':
#                     self.crosssections.add_rectangle_definition(height=crsdef['height'],
#                                                                 width=crsdef['width'],
#                                                                 closed=crsdef['closed'],
#                                                                 roughnesstype=roughtype,
#                                                                 roughnessvalue=roughval,
#                                                                 name=crsdef['crosssectiondefinitionid'])

#                 elif crsdef['type'] == 'trapezium':
#                     self.crosssections.add_trapezium_definition(slope=(crsdef['t_width'] - crsdef['width'])/2/crsdef['height'],
#                                                                 maximumflowwidth=crsdef['t_width'],
#                                                                 bottomwidth=crsdef['width'],
#                                                                 closed=crsdef['closed'],
#                                                                 roughnesstype=roughtype,
#                                                                 roughnessvalue=roughval,
#                                                                 name=crsdef['crosssectiondefinitionid'])
                    
#                 elif crsdef['type'] == 'zw':
#                     self.crosssections.add_zw_definition(numLevels=crsdef["numlevels"],
#                                                          levels=crsdef["levels"],
#                                                          flowWidths=crsdef["flowwidths"],
#                                                          totalWidths=crsdef["totalwidths"],
#                                                          roughnesstype=roughtype,
#                                                          roughnessvalue=roughval,
#                                                          name=crsdef['crosssectiondefinitionid'])

#                 elif crsdef['type'] == 'yz':
#                     # TODO BMA: add yz
#                     raise NotImplementedError

#                 else:
#                     raise NotImplementedError


                
            
# class ExternalForcingsIO:

#     def __init__(self, external_forcings):
#         self.external_forcings = external_forcings

#     def from_hydamo(self, boundary_conditions):

#         # Read from Hydamo
#         bcdct = hydamo_to_dflowfm.generate_boundary_conditions(boundary_conditions, self.external_forcings.dflowfmmodel.network.schematised)

#         # Add all items
#         for key, item in bcdct.items():
#             self.external_forcings.add_boundary_condition(key, item['geometry'], item['bctype'], item['value'], branchid=item['branchid'])
#             # # Add to dataframe         
#             # self.external_forcings.boundaries.loc[key] = item
#             # Check if a 1d2d link should be removed
#             #self.external_forcings.dflowfmmodel.network.links1d2d.check_boundary_link(self.external_forcings.boundaries.loc[key])

#     def read_laterals(self, locations, lateral_discharges=None, rr_boundaries=None):
#         """
#         Process laterals

#         Parameters
#         ----------
#         locations: gpd.GeoDataFrame
#             GeoDataFrame with at least 'geometry' (Point) and the column 'code'
#         lateral_discharges: pd.DataFrame
#             DataFrame with lateral discharges. The index should be a time object (datetime or similar).
#         rr_boundaries: pd.DataFrame
#             DataFrame with RR-catchments that are coupled 
#         """

#         if rr_boundaries is None: rr_boundaries = []
#         # Check argument
#         checks.check_argument(locations, 'locations', gpd.GeoDataFrame, columns=['geometry'])
#         if lateral_discharges is not None:
#             checks.check_argument(lateral_discharges, 'lateral_discharges', pd.DataFrame)

#         # Check if network has been loaded
#         network1d = self.external_forcings.dflowfmmodel.network.mesh1d
#         if not network1d.meshgeomdim.numnode:
#             raise ValueError('1d network has not been generated or loaded. Do this before adding laterals.')

#         # in case of 3d points, remove the 3rd dimension
#         locations['geometry2'] = [Point([point.geometry.x, point.geometry.y]) for _,point in locations.iterrows()]    
#         locations.drop('geometry', inplace=True, axis=1)
#         locations.rename(columns={'geometry2':'geometry'}, inplace=True)
        
#         # Find nearest 1d node per location and find the nodeid
#         #lateral_crds = np.vstack([loc.geometry.coords[0] for loc in locations.itertuples()])             
#         #nodes1d = network1d.get_nodes()
#         #get_nearest = KDTree(nodes1d)
#         #_, nearest_idx = get_nearest.query(lateral_crds[:,0:2])
                
#         # Get time series and add to dictionary
#         #for nidx, lateral in zip(nearest_idx, locations.itertuples()):
#         for lateral in locations.itertuples():
#             # crd = nodes1d[nearest_idx]
#             #nid = f'{nodes1d[nidx][0]:g}_{nodes1d[nidx][1]:g}'
            
#             # Check if a time is provided for the lateral
#             if lateral.code in rr_boundaries:                
#                 # Add to dictionary
#                 self.external_forcings.laterals[lateral.code] = {
#                     'branchid': lateral.branch_id,
#                     'branch_offset': str(lateral.branch_offset)                    
#                 }
#             else:
#                 if lateral_discharges is None:
#                     logger.warning(f'No lateral_discharges provied. {lateral.code} expects them. Skipping.')
#                     continue
#                 else:
#                     if lateral.code not in lateral_discharges.columns:
#                         logger.warning(f'No data found for {lateral.code}. Skipping.')
#                         continue
                    
#                 # Get timeseries
#                 series = lateral_discharges.loc[:, lateral.code]
                
#                 # Add to dictionary
#                 self.external_forcings.laterals[lateral.code] = { 
#                     'branchid': lateral.branch_id,
#                     'branch_offset': str(lateral.branch_offset), 
#                     'timeseries': series            
#                 }


# class StorageNodesIO:

#     def __init__(self, storagenodes):
#         self.storagenodes = storagenodes

#     def storagenodes_from_datamodel(self, storagenodes):
#         """"From parsed data model of storage nodes"""
#         for storagenode_idx, storagenode in storagenodes.iterrows():
#             self.storagenodes.add_storagenode(
#                 id=storagenode.id,
#                 name=storagenode.name if 'name' in storagenode.code else np.nan,
#                 usestreetstorage=storagenode.usestreetstorage,
#                 nodetype='unspecified',
#                 nodeid=storagenode.nodeid,
#                 usetable='false',
#                 bedlevel=storagenode.bedlevel,
#                 area=storagenode.area,
#                 streetlevel=storagenode.streetlevel,
#                 streetstoragearea=storagenode.streetstoragearea,
#                 storagetype=storagenode.storagetype
#             )

