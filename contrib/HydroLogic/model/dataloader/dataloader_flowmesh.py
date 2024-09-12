from typing import List, Tuple
import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio._err import CPLE_AppDefinedError
from rasterio.transform import from_bounds
from scipy.interpolate import griddata, interp1d
from scipy.spatial import KDTree
import time


def load_mesh(input_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the mesh nodes faces' x and y-coordinates from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file

    Returns:
        node_x (np.ndarray): x-coordinates of the mesh nodes faces
        node_y (np.ndarray): y-coordinates of the mesh nodes faces
   
    try:
        with UGrid(
            input_file_path,
            "r",
        ) as ug:

            node_x = ug.variable_get_data_double(r"Mesh2d_face_x")
            node_y = ug.variable_get_data_double(r"Mesh2d_face_y")
    except OSError:
        # Linux support not yet implemented in UGridpy
        with Dataset(input_file_path) as nc:
            node_x = np.asarray(nc.variables[r"Mesh2d_face_x"][:]).flatten()
            node_y = np.asarray(nc.variables[r"Mesh2d_face_y"][:]).flatten()
     """
    with Dataset(input_file_path) as nc:
        node_x = np.asarray(nc.variables[r"Mesh2d_face_x"][:]).flatten()
        node_y = np.asarray(nc.variables[r"Mesh2d_face_y"][:]).flatten()
    return node_x, node_y


def load_meta_data(input_file_path) -> List:
    """
    Retrieves the svariables from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file

    Returns:
        List: list of variables that can be read.
    """
    variables = []

    with Dataset(input_file_path) as nc:
        for variable in nc.variables:
            if not hasattr(nc.variables[variable], "coordinates"):
                continue

            if ("Mesh2d_face_x Mesh2d_face_y" in nc.variables[variable].coordinates) or ("mesh2d_face_x mesh2d_face_y" in nc.variables[variable].coordinates):
                variables.append(variable)

    return variables


def load_data(input_file_path: str, variable: str) -> np.ndarray:
    """
    Retrieves the selected mesh data (i.e. variable) from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file
        variable (str): variable to return

    Returns:
        data (np.ndarray): data at the mesh nodes
    """

    with Dataset(input_file_path) as nc:
        data = np.asarray(nc.variables[variable][:]).flatten()

    return data


def load_fou_data(input_file_path: str, variable: str) -> np.ndarray:
    """
    Retrieves the selected mesh data (i.e. variable) from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file
        variable (str): variable to return

    Returns:
        data (np.ndarray): data at the mesh nodes
    """

    # Fou files don't have a time axis, so we can simply load them. This function isn't very useful :)
    return load_data(input_file_path, variable)


def load_map_data(input_file_path: str, variable: str) -> np.ndarray:
    """
    Retrieves the selected mesh data (i.e. variable) from the netCDF file at input_file_path.
    Also restores original dimensions

    Args:
        input_file_path (str): path to the input file
        variable (str): variable to return

    Returns:
        map_data (np.ndarray): data at the mesh nodes (time, spatial)
    """
    # Load data from file
    # try:
        # with UGrid(
            # input_file_path,
            # "r",
        # ) as ug:
# 
            # map_data = ug.variable_get_data_double(variable)
    # except OSError:
        # Linux support not yet implemented in UGridpy
    with Dataset(input_file_path) as nc:
        map_data = np.asarray(nc.variables[variable][:]).flatten()

    # reshape data using dimensions from netCDF input file (time, spatial)
    nc = Dataset(input_file_path)
    map_dim = nc.variables[variable].get_dims()
    map_dims = (map_dim[0].size, map_dim[1].size)

    map_data = np.reshape(map_data, map_dims)
    nc.close()

    return map_data


def load_classmap_data(
    input_file_path: str, variable: str, method: str = "average", ret_map_data=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the selected mesh data (i.e. variable) from the .clm netCDF file at input_file_path
    Returns (1) the raw data (clm_data) at mesh locations and (2) the filled data (map_data), using class definitions
    Optional: provide method for replacing class numbers with bin values (lower bound, upper bound, average). Default is average

    Args:
        input_file_path (str): path to the input file
        variable (str): variable to return
        method (str): method to use when replacing classes with values. "lower" replaces class numbers with lower bound;
                      "upper" replaces class numbers with upper bound; "average" replaces class numbers with average of upper and lower bounds.
                      Regardless, the lass class will take the lower bound as it goes to infinity

    Returns:
        clm_data (np.ndarray): classmap data at the mesh nodes (time, spatial)
        map_data (np.ndarray): filled classmap data at the mesh nodes, using class definition (time, spatial)
    """

    # load data for processing
    clm_data = load_data(input_file_path, variable)
    nc = Dataset(input_file_path)

    # Load class bounds
    class_bounds_name = nc.variables[variable].getncattr("flag_bounds")
    class_bounds_dim = nc.variables[class_bounds_name].get_dims()
    class_bound_dims = (class_bounds_dim[0].size, class_bounds_dim[1].size)

    # reshape
    class_bounds = np.reshape(
        load_data(input_file_path, class_bounds_name),
        class_bound_dims,
    )

    # Replace class values with values from class definition using interp1d (no actual interpolation is done as mapping is on the points provided)
    # Allow different class interpertation methods
    if method == "lower":
        y = class_bounds[:, 0]
    elif method == "upper":
        y = class_bounds[:, 1]
    elif method == "average":
        y = np.average(class_bounds, axis=1)

    x = np.arange(0, class_bounds.shape[0]) + 1  # classes are 1 indexed
    y[-1] = class_bounds[-1, 0]  # use lower bound for last class, as it goes to infinity

    if ret_map_data:
        # 'interpolate' -> we only querry at the points provided so no real interpolation is taking place. Done for convenience
        f = interp1d(x, y, bounds_error=False, fill_value=(np.nan, np.nan))
        map_data = f(clm_data)

    # Reshape data using dimensions from netCDF input file (time, spatial)
    map_dim = nc.variables[variable].get_dims()
    map_dims = (map_dim[0].size, map_dim[1].size)

    if ret_map_data:
        map_data = np.reshape(map_data, map_dims)

    clm_data = np.reshape(clm_data, map_dims)
    nc.close()  # close file
    return clm_data, map_data


def create_raster(
    node_x: np.ndarray, node_y: np.ndarray, resolution: float, margin: float = 1.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a raster that spans from min(node_x) to max(node_x) and from min(node_y) to max(node_y)
    with user defined resolution

    Args:
        node_x (np.ndarray): x-coordinates of the mesh nodes
        node_y (np.ndarray): y-coordinates of the mesh nodes
        resolution (float): resolution of the raster
        margin (float): margin around the node face center that the raster covers, in percentage.
                        A value of 1 corresponds to 100%, the default value of 1.05 corresponds
                        to 105% of the extend of the nodes face coordinates

    Returns:
        grid_x (np.ndarray): x-coordinates of the raster points
        grid_y (np.ndarray): y-coordinates of the raster points
        bounds (np.ndarray): outermost coordinates of the raster (west, east, south, north)
    """

    bounds, n_cells = get_bounds(
        node_x=node_x, node_y=node_y, resolution=resolution, margin=margin
    )
    xrange = np.linspace(start=bounds[0], stop=bounds[1], num=n_cells[0], endpoint=True)
    yrange = np.linspace(start=bounds[2], stop=bounds[3], num=n_cells[1], endpoint=True)
    grid_x, grid_y = np.meshgrid(xrange, yrange)

    return grid_x, grid_y, bounds


def get_bounds(
    node_x: np.ndarray, node_y: np.ndarray, resolution: float, margin: float
) -> np.ndarray:
    """
    Determines the raster bounds based on given resolution and margin.
    Guarantees that the resolution is maintained and extends span if the span in a direction is not an exact multiple of resolution.

    Args:
        node_x (np.ndarray): x-coordinates of the mesh nodes
        node_y (np.ndarray): y-coordinates of the mesh nodes
        resolution (float): resolution of the raster
        margin (float): margin around the node face center that the raster covers, in percentage.
                        A value of 1 corresponds to 100%, the default value of 1.05 corresponds
                        to 105% of the extend of the nodes face coordinates

    Returns:
        bounds (np.ndarray): outermost coordinates of the raster (west, east, south, north)
        n_cells(np.ndarray): number of cells in x and y direction (x, y)
    """

    # Determine span of nodes
    span_x = max(node_x) - min(node_x)
    span_y = max(node_y) - min(node_y)

    # Compute how many cells cover the span*margin
    n_cells_x = np.ceil(span_x * margin / resolution).astype(int)
    n_cells_y = np.ceil(span_y * margin / resolution).astype(int)

    # Compute the new span of n_cells * resolution
    fitted_span_x = n_cells_x * resolution
    fitted_span_y = n_cells_y * resolution

    # Compute difference w.r.t. original span
    diff_span_x = fitted_span_x - span_x
    diff_span_y = fitted_span_y - span_y

    # Determine bounds
    min_x = min(node_x) - diff_span_x / 2
    max_x = max(node_x) + diff_span_x / 2

    min_y = min(node_y) - diff_span_y / 2
    max_y = max(node_y) + diff_span_y / 2

    # Return results
    bounds = np.array([min_x, max_x, min_y, max_y])
    n_cells = np.array([n_cells_x, n_cells_y])

    return bounds, n_cells


def mesh_data_to_raster(
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_data: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    interpolation: str = "nearest",
    distance_tol: float = -999,
) -> np.ndarray:
    """
    1. Maps the data at the mesh nodes to the raster points.
       see e.g. https://hatarilabs.com/ih-en/how-to-create-a-geospatial-raster-from-xy-data-with-python-pandas-and-rasterio-tutorial
    2. Sets raster points to NaN if further away than distance_tol from a mesh node.
       This allows 'holes' in grids and other funky shapes to exist in the raster.
       Otherwise the convex hull of the mesh is interpolated, resulting in potentially unwanted results.

    Args:
        node_x (np.ndarray): x-coordinates of the mesh nodes
        node_y (np.ndarray): y-coordinates of the mesh nodes
        node_data (np.ndarray): data at the mesh nodes
        grid_x (np.ndarray): x-coordinates of the raster points
        grid_y (np.ndarray): y-coordinates of the raster points
        interpolation (str): interpolation method for scipy.spatial.griddata. (1. "linear", 2. "nearest", 3. "cubic")
        distance_tol (float): maximum distance for raster points from mesh nodes. If distance is larger, raster point will be set to NaN

    Returns:
        new_grid_data/grid_data (np.ndarray): data at raster points. Is NaN at points further from mesh node than distance_tol, if distance_tol > 0.
                                              Otherwise, raster points are interpolated according to scipy.spatial.griddata, stopping at the mesh's convex hull
    """

    # map data from mesh network to raster
    points = list(zip(node_x, node_y))

    grid_data = griddata(points, node_data, (grid_x, grid_y), method=interpolation)

    if distance_tol > 0:
        # set raster points that are too far away from mesh nodes to NaN
        # Store mesh points and raster points in KDTrees and compute distance if < distance_tol
        meshtree = KDTree(points)
        rastertree = KDTree(list(zip(grid_x.flatten(), grid_y.flatten())))
        sdm = rastertree.sparse_distance_matrix(meshtree, distance_tol, output_type="coo_matrix")

        # Compute number of entries per row (in getnnz()), if 0 then the raster pixel will be NaN as its too far from a mesh node
        skip_ix = sdm.getnnz(axis=1) == 0

        # Skip raster pixels that are too far from nodes (set them to nan)
        new_grid_data = grid_data.flatten()
        new_grid_data[skip_ix] = np.nan
        new_grid_data = np.reshape(new_grid_data, grid_data.shape)

        return new_grid_data

    else:
        return grid_data


def write_tiff(
    output_file_path: str, new_grid_data: np.ndarray, bounds: np.ndarray, epsg: int
) -> None:
    """
    Saves new_grid_data to a tiff file. new_grid_data should be a raster.

    Args:
        output_file_path (str): location of the output file
        new_grid_data (np.ndarray): data at grid points
        bounds (np.ndarray): outermost coordinates of the raster (west, east, south, north)
        epsg (int): coordinate reference system (CPS) that is stored in the tiff-file

    Returns:
        None
    """

    transform = from_bounds(
        west=bounds[0],
        east=bounds[1],
        south=bounds[3],
        north=bounds[2],
        width=new_grid_data.shape[0],
        height=new_grid_data.shape[1],
    )
    if epsg != 0:
        try:
            raster_crs = CRS.from_epsg(code=epsg)
        except:# (CRSError, CPLE_AppDefinedError):
            raster_crs = CRS.from_epsg(28992)
    else: raster_crs = CRS.from_epsg(28992)

    t_file = rasterio.open(
        output_file_path,
        "w",
        driver="GTiff",
        height=new_grid_data.shape[1],
        width=new_grid_data.shape[0],
        count=1,
        dtype=new_grid_data.dtype,
        crs=raster_crs,
        transform=transform,
    )
    t_file.write(new_grid_data, 1)
    t_file.close()


def mesh_to_tiff(
    data: np.ndarray,
    input_file_path: str,
    output_file_path: str,
    resolution: float,
    distance_tol: float,
    interpolation: str = "nearest",
) -> None:
    """
    Wrapper function that turns mesh data into a tiff.
    The mesh data should be a 1D array, with the same size as the coordinates in the netCDF input file.

    Args:
        data (np.ndarray): data at the mesh nodes.
        input_file_path (str): location of the input file
        output_file_path (str): location of the output file
        resolution (float): resolution of the raster
        distance_tol (float): maximum distance for raster points from mesh nodes. If distance is larger, raster point will be set to NaN
        interpolation (str): interpolation method for scipy.spatial.griddata. (1. "linear", 2. "nearest", 3. "cubic")

    Returns:
        grid_x (np.ndarray): grid with x-coordinates of raster
        grid_y (np.ndarray): grid with y-coordinates of raster
        new_grid_data (np.ndarray): new_grid_data
    """
    start_time = time.time()

    # create raster grid
    node_x, node_y = load_mesh(input_file_path)
    grid_x, grid_y, bounds = create_raster(node_x, node_y, resolution)
    
    # map mesh data to grid and set raster points outside of distance_tol form a mesh node to nan
    new_grid_data = mesh_data_to_raster(
        node_x,
        node_y,
        data,
        grid_x,
        grid_y,
        interpolation=interpolation,
        distance_tol=distance_tol,
    )
    
    # read crs from input file
    nc = Dataset(input_file_path)
    epsg = nc.variables["projected_coordinate_system"].getncattr("epsg")

    # write tif
    write_tiff(output_file_path, new_grid_data, bounds, epsg)

    print(f"Total time taken: {time.time() - start_time} seconds")

    return grid_x, grid_y, new_grid_data
