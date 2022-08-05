from typing import Tuple

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.interpolate import griddata, interp1d
from scipy.spatial import KDTree
from ugrid import UGrid


def load_mesh(input_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the mesh nodes x and y-coordinates from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file

    Returns:
        node_x (np.ndarray): x-coordinates of the mesh nodes
        node_y (np.ndarray): y-coordinates of the mesh nodes
    """
    with UGrid(
        input_file_path,
        "r",
    ) as ug:

        node_x = ug.variable_get_data_double(r"Mesh2d_face_x")
        node_y = ug.variable_get_data_double(r"Mesh2d_face_y")

    return node_x, node_y


def load_data(input_file_path: str, variable: str) -> np.ndarray:
    """
    Retrieves the selected mesh data (i.e. variable) from the netCDF file at input_file_path

    Args:
        input_file_path (str): path to the input file
        variable (str): variable to return

    Returns:
        data (np.ndarray): data at the mesh nodes
    """

    with UGrid(
        input_file_path,
        "r",
    ) as ug:

        data = ug.variable_get_data_double(variable)

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
    with UGrid(
        input_file_path,
        "r",
    ) as ug:
        map_data = ug.variable_get_data_double(variable)

    # reshape data using dimensions from netCDF input file (time, spatial)
    nc = Dataset(input_file_path)
    map_dim = nc.variables[variable].get_dims()
    map_dims = (map_dim[0].size, map_dim[1].size)

    map_data = np.reshape(map_data, map_dims)
    nc.close()

    return map_data


def load_classmap_data(
    input_file_path: str, variable: str, method: str = "average"
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

    # 'interpolate' -> we only querry at the points provided so no real interpolation is taking place. Still a convenient implementation
    f = interp1d(x, y, bounds_error=False, fill_value=(np.nan, np.nan))
    map_data = f(clm_data)

    # Reshape data using dimensions from netCDF input file (time, spatial)
    map_dim = nc.variables[variable].get_dims()
    map_dims = (map_dim[0].size, map_dim[1].size)

    clm_data = np.reshape(clm_data, map_dims)
    map_data = np.reshape(map_data, map_dims)

    nc.close()  # close file
    return clm_data, map_data


def create_raster(
    node_x: np.ndarray, node_y: np.ndarray, resolution: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a raster that spans from min(node_x) to max(node_x) and from min(node_y) to max(node_y)
    with user defined resolution

    Args:
        node_x (np.ndarray): x-coordinates of the mesh nodes
        node_y (np.ndarray): y-coordinates of the mesh nodes
        resolution (float): resolution of the raster

    Returns:
        grid_x (np.ndarray): x-coordinates of the raster points
        grid_y (np.ndarray): y-coordinates of the raster points
        bounds (np.ndarray): outermost coordinates of the raster (west, east, south, north)
    """

    bounds = np.array([min(node_x), max(node_x), min(node_y), max(node_y)])
    xrange = np.arange(bounds[0], bounds[1] + resolution, resolution)
    yrange = np.arange(bounds[2], bounds[3] + resolution, resolution)
    grid_x, grid_y = np.meshgrid(xrange, yrange)

    return grid_x, grid_y, bounds


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
        # set raster points that are too far away from mesh nodes to zero
        # Store mesh points and raster points in KDTrees and compute distance if < distance_tol
        meshtree = KDTree(points)
        rastertree = KDTree(list(zip(grid_x.flatten(), grid_y.flatten())))
        sdm = rastertree.sparse_distance_matrix(meshtree, distance_tol, output_type="coo_matrix")

        # Compute number of entries per row (in getnnz()), if none then the raster pixel will be skipped as its too far from a mesh node
        skip_ix = sdm.getnnz(axis=1) == 0

        # Skip raster pixels that are too far from nodes (set them to nan)
        new_grid_data = grid_data.flatten()
        new_grid_data[skip_ix] = np.nan
        new_grid_data = np.reshape(new_grid_data, grid_data.shape)

        return new_grid_data

    else:
        return grid_data


def write_tiff(
    output_file_path: str, new_grid_data: np.ndarray, bounds: np.ndarray, epsg: int = 28992
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
    raster_crs = CRS.from_epsg(epsg)

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

    return grid_x, grid_y, new_grid_data
