import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon
import logging
import meshkernel as mk
import geopandas as gpd
from tqdm.auto import tqdm
from hydrolib.dhydamo.geometry.gridgeom import geometry

logger = logging.getLogger(__name__)

class Links1d2d:
    def __init__(self, network, mesh2d, hydamo):
        
        self.mesh1d  = {}
        self.mesh1d['nodes1d'] = np.c_[network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y]
        self.mesh1d['edges1d'] = network._mesh1d.mesh1d_edge_nodes + 1
        
        self.mesh2d = mesh2d
        self.hydamo = hydamo

        self.network = network

        # List for 1d 2d links
        self.nodes1d = []
        self.faces2d = []
        #setattr(self.network._link1d2d, 'link1d2d', np.array([]))

    def generate_1d_to_2d(self, max_distance=np.inf, branchid=None):
        """
        Generate 1d2d links from 1d nodes. Each 1d node is connected to
        the nearest 2d cell. A maximum distance can be specified to remove links
        that are too long. Also the branchid can be specified, if you only want
        to generate links from certain 1d branches.

        Parameters
        ----------
        max_distance : int, float
            The maximum length of a link. All longer links are removed.
        branchid : str or list
            ID's of branches for which the connection from 1d to 2d is made.
        """
        logger.info("Generating links from 1d to 2d based on distance.")

        # Create KDTree for faces
        faces2d = np.c_[
            self.mesh2d.get_values("facex"), self.mesh2d.get_values("facey")
        ]
        get_nearest = KDTree(faces2d)

        # Get network geometry
        # nodes1d = self.mesh1d.get_nodes()
        # idx = self.mesh1d.get_nodes_for_branch(branchid)
        nodes1d = self.mesh1d['nodes1d']
        idx = np.full(nodes1d.shape[0], dtype=bool, fill_value=True)
        # Get nearest 2d nodes
        distance, idx_nearest = get_nearest.query(nodes1d[idx])
        close = distance < max_distance

        # Add link data
        nodes1didx = np.arange(len(nodes1d))[idx]
        self.nodes1d.extend(nodes1didx[close] + 1)
        self.faces2d.extend(idx_nearest[close] + 1)

        # # Remove conflicting 1d2d links
        for _,bc in self.hydamo.boundary_conditions.iterrows():        
            self.check_boundary_link(bc)

    def generate_2d_to_1d(
        self,
        max_distance=np.inf,
        intersecting=True,
        branchid=None,
        shift_to_centroid=False,
    ):
        """
        Generate 1d2d links from 2d cells. A maximum distance can be specified
        to remove links that are too long. Also a branchid can be specified to only
        generate links to certain branches.

        In case of a 1D and 2D grid that is on top of each other the user might want
        to generate links only for intersecting cells, where in case of non-overlapping meshes
        the use might want to use the shortest distance. This behaviour can be specified with
        the option intersecting:
        1. intersecting = True: each 2d cell crossing a 1d branch segment is connected to
            the nearest 1d cell.
        2. intersecting = False: each 2d cell is connected to the nearest 1d cell,
            If the link crosses another cell it is removed.
        In case of option 2. setting a max distance will speed up the the process a bit.

        Parameters
        ----------
        max_distance : int, float
            Maximum allowed length for a link
        intersecting : bool
            Make connections for intersecting 1d and 2d cells or based on
            nearest neighbours
        branchid : str or list of str
            Generate only to specified 1d branches
        """
        logger.info(
            f'Generating links from 2d to 1d based on {"intersection" if intersecting else "distance"}.'
        )

        # Collect polygons for cells
        centers2d = self.mesh2d.get_faces(geometry="center")
        idx = np.arange(len(centers2d), dtype="int")
        # Create KDTree for 1d cells
        nodes1d = self.mesh1d['nodes1d']#self.mesh1d.get_nodes()
        nodes1didx = np.full(nodes1d.shape[0], dtype=bool, fill_value=True)
        get_nearest = KDTree(nodes1d[nodes1didx])

        # Make a pre-selection
        if max_distance < np.inf:
            # Determine distance from 2d to nearest 1d
            distance, _ = get_nearest.query(centers2d)
            idx = idx[distance < max_distance]

        # Create GeoDataFrame
        logger.info(f"Creating GeoDataFrame of ({len(idx)}) 2D cells.")
        cells = gpd.GeoDataFrame(
            data=centers2d[idx],
            columns=["x", "y"],
            index=idx + 1,
            geometry=[
                Polygon(cell)
                for i, cell in enumerate(self.mesh2d.get_faces())
                if i in idx
            ],
        )

        # Find intersecting cells with branches
        logger.info("Determine intersecting or nearest branches.")
        if branchid is None:
            branches = self.hydamo.branches
        elif isinstance(branchid, str):
            branches = self.hydamo.branches.loc[[branchid]]
        else:
            branches = self.hydamo.branches.loc[branchid]

        if intersecting:
            geometry.find_nearest_branch(branches, cells, method="intersecting")
        else:
            geometry.find_nearest_branch(
                branches, cells, method="overal", maxdist=max_distance
            )

        # Drop the cells without intersection
        cells.dropna(subset=["branch_offset"], inplace=True)
        faces2d = np.c_[cells.x, cells.y]

        # Get nearest 1d nodes
        distance, idx_nearest = get_nearest.query(faces2d)
        close = distance < max_distance

        # Add link data
        nodes1didx = np.where(nodes1didx)[0][idx_nearest]
        self.nodes1d.extend(nodes1didx[close] + 1)
        self.faces2d.extend(cells.index.values[close])

        if not intersecting:
            logger.info("Remove links that cross another 2D cell.")
            # Make sure only the nearest cells are accounted by removing all links that also cross another cell
            links = self.get_1d2dlinks(as_gdf=True)
            todrop = []

            # Remove links that intersect multiple cells
            cellbounds = cells.bounds.values.T
            for link in tqdm(
                links.itertuples(),
                total=len(links),
                desc="Removing links crossing mult. cells",
            ):
                selectie = cells.loc[
                    geometry.possibly_intersecting(cellbounds, link.geometry)
                ].copy()
                if selectie.intersects(link.geometry).sum() > 1:
                    todrop.append(link.Index)
            links.drop(todrop, inplace=True)

            # Re-assign
            del self.nodes1d[:]
            del self.faces2d[:]

            self.nodes1d.extend(links["node1did"].values.tolist())
            self.faces2d.extend(links["face2did"].values.tolist())

        # Shift centers of 2d faces to centroid if they are part of a 1d-2d link
        if shift_to_centroid:
            # Get current centers
            cx, cy = self.mesh2d.get_faces(geometry="center").T
            # Calculate centroids for cells with link
            idx = np.array(self.faces2d) - 1
            centroids = np.vstack(
                [cell.mean(axis=0) for cell in np.array(self.mesh2d.get_faces())[idx]]
            ).T
            cx[idx] = centroids[0]
            cy[idx] = centroids[1]
            # Set values back to geometry
            self.mesh2d.set_values("facex", cx)
            self.mesh2d.set_values("facey", cy)

        # Remove conflicting 1d2d links
        for _,bc in self.hydamo.boundary_conditions.iterrows():        
            self.check_boundary_link(bc)

    def check_boundary_link(self, bc):
        """
        Since a boundary conditions is not picked up when there is a bifurcation
        in the first branch segment, potential 1d2d links should be removed.

        This function should be called whenever a boundary conditions is added,
        or the 1d2d links are generated.
        """

        # Can only be done after links have been generated
        if not self.nodes1d or not self.faces2d:
            return None

        # Find the nearest node with the KDTree
        nodes1d = self.mesh1d['nodes1d'] #mesh1d.get_nodes()
        get_nearest = KDTree(nodes1d)
        _, idx_nearest = get_nearest.query(
            [float(pt) for pt in np.array(bc["geometry"].coords[:][0])][0:2]
        )
        node_id = idx_nearest + 1

        # Check 1. Determine if the nearest node itself is not a bifurcation
        edge_nodes = self.mesh1d["edges1d"]
        counts = {u: c for u, c in zip(*np.unique(edge_nodes, return_counts=True))}
        if counts[node_id] > 1:
            logger.warning(
                f"The boundary condition at {node_id} is not a branch end. Check if it is picked up by dflowfm."
            )

        # Check 2. Check if any 1d2d links are connected to the node or next node. If so, remove.
        # Find the node(s) connect to 'node_id'
        to_remove = np.unique(edge_nodes[(edge_nodes == node_id).any(axis=1)])
        for item in to_remove:
            while item in self.nodes1d:
                loc = self.nodes1d.index(item)
                self.nodes1d.pop(loc)
                self.faces2d.pop(loc)
                nx, ny = nodes1d[item - 1]                
                logger.info(
                    f"Removed link(s) from 1d node: ({nx:.2f}, {ny:.2f}) because it is too close to boundary condition at node {node_id:.0f}."
                )

    def get_1d2dlinks(self, as_gdf=False):
        """
        Method to get 1d2d links as array with coordinates or geodataframe.

        Parameters
        ----------
        as_gdf : bool
            Whether to export as geodataframe (True) or numpy array (False)
        """

        if not any(self.nodes1d):
            return None

        # Get 1d nodes and 2d faces
        nodes1d = self.mesh1d['nodes1d']
        faces2d = self.mesh2d.get_faces(geometry="center")

        # Get links
        links = np.dstack(
            [nodes1d[np.array(self.nodes1d) - 1], faces2d[np.array(self.faces2d) - 1]]
        )

        if not as_gdf:
            return np.array([line.T for line in links])
        else:
            return gpd.GeoDataFrame(
                data=np.c_[self.nodes1d, self.faces2d],
                columns=["node1did", "face2did"],
                geometry=[LineString(line.T) for line in links],
            )

    def remove_1d2d_from_numlimdt(self, file, threshold, node="2d"):
        """
        Remove 1d2d links based on numlimdt file
        """
        if node == "1d":
            links = self.get_1d2dlinks(as_gdf=True)

        with open(file) as f:
            for line in f.readlines():
                x, y, n = line.split()
                if int(n) >= threshold:
                    if node == "2d":
                        self.remove_1d2d_link(
                            float(x), float(y), mesh=node, max_distance=2.0
                        )
                    else:
                        # Find the 1d node connected to the link
                        idx = links.distance(Point(float(x), float(y))).idxmin()
                        x, y = links.at[idx, "geometry"].coords[0]
                        self.remove_1d2d_link(x, y, mesh=node, max_distance=2.0)

    def remove_1d2d_link(self, x, y, mesh, max_distance=None):
        """
        Remove 1d 2d link based on x y coordinate.
        Mesh can specified, 1d or 2d.
        """
        if mesh == "1d":
            pts = self.mesh1d['nodes1d']            
        elif mesh == "2d":
            pts = np.c_[self.mesh2d.get_faces(geometry="center")]            
        else:
            raise ValueError('Mesh should be "1d" or "2d".')

        if max_distance is None:
            max_distance=10.

        # Find nearest link
        dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        if dists.min() > max_distance:
            print('No links within the maximum distance. Doing nothing.')            
            return None
        imin = np.argmin(dists)

        # Determine what rows to remove (if any)
        linkdim = self.nodes1d if mesh == "1d" else self.faces2d
        to_remove = [link for link in (linkdim) if link == (imin + 1)]
        for item in to_remove:
            while item in linkdim:
                loc = linkdim.index(item)
                self.nodes1d.pop(loc)
                self.faces2d.pop(loc)

    def remove_1d_endpoints(self):
        """Method to remove 1d2d links from end points of the 1d mesh. The GUI
        will interpret every endpoint as a boundary conditions, which does not
        allow a 1d 2d link at the same node. To avoid problems with this, use
        this method.
        """
        # Can only be done after links have been generated
        if not self.nodes1d or not self.faces2d:
            return None

        nodes1d = self.mesh1d['nodes1d'] 
        edge_nodes = self.mesh1d['edges1d']

        # Select 1d nodes that are only present in a single edge
        edgeid, counts = np.unique(edge_nodes, return_counts=True)
        to_remove = edgeid[counts == 1]

        for item in to_remove:
            while item in self.nodes1d:
                loc = self.nodes1d.index(item)
                self.nodes1d.pop(loc)
                self.faces2d.pop(loc)
                nx, ny = nodes1d[item - 1]
                print(
                    f"Removed link(s) from 1d node: ({nx:.2f}, {ny:.2f}) because it is connected to an end-point."
                )

    def remove_links1d2d_within_polygon(self, polygon) -> None:
        """Remove 1d2d links within a given polygon or multipolygon

        Args:
            network (Network): The network from which the links are removed
            polygon (Union[Polygon, MultiPolygon]): The polygon that indicates which to remove
        """

        # Create an array with 2d facecenters and 1d nodes, that form the links
        nodes1d = self.mesh1d['nodes1d'][np.array(self.nodes1d)-1]
        faces2d = np.c_[
            self.mesh2d.get_values("facex"), self.mesh2d.get_values("facey")
        ][np.array(self.faces2d) - 1]
        

        # Check which links intersect the provided area
        index = np.zeros(len(nodes1d), dtype=bool)
        for part in geometry.as_polygon_list(polygon):
            index |= geometry.points_in_polygon(nodes1d, part)
            index |= geometry.points_in_polygon(faces2d, part)

        # Remove these links
        for i in reversed(np.where(index)[0]):
            self.nodes1d.pop(i)
            self.faces2d.pop(i)
            
    def convert_to_hydrolib(self):
        """
        convert_to_hydrolib Convert gruidgeom objects back to hydrolib-core

        _extended_summary_

        Returns:
            fm.geometry.network -object, filled with 2d and 1d2d objects
        """
        nodex = self.mesh2d.get_values('nodex', as_array=True)
        nodey = self.mesh2d.get_values('nodey', as_array=True)
        facez = self.mesh2d.get_values('facez', as_array=True)
        edge_nodes = self.mesh2d.get_values('edge_nodes', as_array=True) - 1        
        self.network._mesh2d._set_mesh2d(nodex, nodey, edge_nodes)
        self.network._mesh2d.mesh2d_face_z = facez

        # Add the 1d2d 
        nodes1d = np.array(self.nodes1d) - 1

        # Faces do not have to be the same between meshgeom and hydrolib. Find
        # the exact matches.
        gr_faces = np.c_[self.mesh2d.get_values("facex"), self.mesh2d.get_values("facey")]
        gr_faces = gr_faces[np.array(self.faces2d) - 1]
        mk_faces = np.c_[self.network._mesh2d.mesh2d_face_x, self.network._mesh2d.mesh2d_face_y]
        distances, faces2d = KDTree(mk_faces).query(gr_faces)
        print(f'Max distance between faces: {distances.max()}') # error out if we do not find an exact match
        
        contacts = self.network._link1d2d.meshkernel.contacts_get()
        contacts.mesh1d_indices = contacts.mesh1d_indices#[keep]
        contacts.mesh2d_indices = contacts.mesh2d_indices#[keep]        
       
        contacts =  mk.Contacts(nodes1d, faces2d)
        self.network._link1d2d.meshkernel.contacts_set(contacts)
    	
        # npresent = len(self.network._link1d2d.link1d2d)
        # new_links = np.c_[nodes1d, faces2d]
        # self.network._link1d2d.link1d2d = np.append(
        #     self.network._link1d2d.link1d2d,
        #     new_links,
        #     axis=0,
        # )
        # self.network._link1d2d.link1d2d_contact_type = np.append(
        #     self.network._link1d2d.link1d2d_contact_type, 
        #     np.full(nodes1d.size, 3)
        # )
        # self.network._link1d2d.link1d2d_id = np.append(
        #    self.network._link1d2d.link1d2d_id,
        #    np.array([f"{n1d:d}_{f2d:d}" for n1d, f2d in np.c_[nodes1d, faces2d]])
        # )
        # self.network._link1d2d.link1d2d_long_name = np.append(
        #     self.network._link1d2d.link1d2d_long_name,
        #     np.array([f"{n1d:d}_{f2d:d}" for n1d, f2d in np.c_[nodes1d, faces2d]])
        # )


        

