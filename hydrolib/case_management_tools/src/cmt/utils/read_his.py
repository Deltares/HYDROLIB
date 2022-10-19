import netCDF4 as nc
import numpy as np
import pandas as pd

LAYERS = ["bridge", "culvert", "lateral", "pump", "station", "weirgen"]
TIME_DIM = "time"


def get_ids(ds, layer="station"):
    """
    Get a list of ids of objects in layer

    Args:
        ds (netCDF4 Dataset): NetCDF4 his input Dataset.
        layer (str, optional): his layer. Defaults to "station".

    Returns:
        list: ids of objects in layer

    """

    id_var = f"{layer}_id"
    return ["".join(i).strip() for i in ds[id_var][:].data.astype(str)]


def get_idx(ds, ids, layer="station"):
    """
    Get the idices of a list of ids

    Args:
        ds (netCDF4 Dataset): NetCDF4 his input Dataset.
        ids (list): ids of objects in layer (see get_ids())
        layer (str, optional): his layer. Defaults to "station".

    Returns:
        list: indices of objects in layer

    """

    layer_ids = get_ids(ds, layer)
    return [layer_ids.index(i) for i in ids if i in layer_ids]


def get_var_by_name(ds, long_name, layer="station"):
    """
    Get a Dataset variable by specifying its long name and layer

    Args:
        ds (netCDF4 Dataset): NetCDF4 his input Dataset.
        long_name (str): long name of time dependent variable
        layer (str, optional): his layer. Defaults to "station".

    Returns:
        Dataset.Variable: NetCDF4 dataset variable.

    """

    id_var = f"{layer}_id"
    layer_dim = ds[id_var].dimensions[0]
    layer_vars = [
        i for i in ds.variables.values() if (i.dimensions == (TIME_DIM, layer_dim))
    ]
    return next((i for i in layer_vars if i.long_name == long_name), None)


def get_time(ds, time_dim=TIME_DIM):
    f"""
    Get time.variable as numpy datetime array

    Args:
        ds (netCDF4 Dataset): NetCDF4 his input Dataset.
        time_dim (str, optional): name of time-dimension Defaults to {TIME_DIM}.

    Returns:
        np.array: Numpy datetime array with timestamps

    """

    time_var = ds[time_dim]
    return np.array(
        nc.num2date(time_var[:].data, units=time_var.units), dtype="datetime64[s]"
    )


def get_timeseries(ds, long_name, ids=None, layer="station", statistic="max"):
    """
    Extract timeseries from netCDF4 in his-format in Pandas Dataframe

    Args:
        ds (netCDF4 Dataset): NetCDF4 his input Dataset.
        long_name (str): long name of time dependent variable
        ids (list): ids of objects in layer (see get_ids()). Defaults to None
        layer (str, optional): his layer. Defaults to "station".

    Returns:
        df (pd.DataFrame): Pandas DataFrame with timeseries for selected layer
        and ids.

    """

    var = get_var_by_name(ds, long_name, layer)
    if ids is not None:
        idx = get_idx(ds, ids, layer)
        result = pd.DataFrame(
            np.array([var[:, i].data for i in idx]).T, columns=ids, index=get_time(ds)
        )
    else:
        ids = get_ids(ds, layer)
        result = pd.DataFrame(var[:].data, columns=ids, index=get_time(ds))
    if statistic == "max":
        result = result.max()

    return result
