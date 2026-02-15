import numpy as np


def arrival_indices(
    map_data: np.ndarray,
    arrival_threshold: float = 0.02,
) -> np.ndarray:
    """
    Computes the arrival index. This is the index that water_height >= threshold
    Works with classmaps and maps, as long as arrival_threshold corresponds to waterdepth (maps) and class (classmap).

    Args:
        map_data (np.ndarray): map data (time, nodes)
        arrival_threshold: value >= arrival_threshold there is inundation

    Returns:
        ix_arrival (np.ndarray): arrival indices
    """

    # Determine where waterheight >= threshold
    arrived = map_data >= arrival_threshold

    # Find first time that this is the case
    indices = np.argmax(arrived, axis=0).astype(float)

    # if never inundated, then nan
    nevers = np.invert(np.amax(arrived, axis=0))

    indices[nevers] = np.nan

    return indices


def arrival_times(
    map_data: np.ndarray,
    time_step: np.timedelta64,
    time_unit: str = "s",
    arrival_threshold: float = 3,
    event_start: np.timedelta64 = None,
) -> np.ndarray:
    """
    Computes the arrival times. This is the time-index*time_step that water_height >= threshold
    Works with classmaps and maps, as long as arrival_threshold corresponds to waterdepth (maps) and class (classmap).

    Args:
        map_data (np.ndarray): map data (time, nodes)
        time_step (np.timedelta64): time step between two observations in classmap
        time_unit (str): desired output time format
        arrival_threshold: value >= arrival_threshold there is inundation

    Returns:
        t_arrival (np.ndarray): arrival times
    """

    # Determine where waterheight >= threshold
    indices = arrival_indices(map_data=map_data, arrival_threshold=arrival_threshold)

    # Turn indices into time
    t_arrival = np.multiply(indices, time_step / np.timedelta64(1, time_unit))

    if event_start is not None:
        t_arrival -= event_start / np.timedelta64(1, time_unit)

    return t_arrival


def rising_speeds(
    map_data: np.ndarray,
    time_step: np.timedelta64,
    time_unit: str = "s",
) -> np.ndarray:
    """
    Computes the rising speeds. This is defined as dh/dt between two subsequent datapoints

    Args:
        map_data (np.ndarray): filled classmap data (time, nodes)
        time_step (np.timedelta64): time step between two observations in classmap

    Returns:
        dh_dt (np.ndarray): rising_speeds
    """

    dh = map_data[1:, :] - map_data[:-1, :]
    dt = time_step / np.timedelta64(1, time_unit)
    dh_dt = dh / dt

    return dh_dt


def rising_speeds_ssm(
    map_data: np.ndarray,
    time_step: np.timedelta64,
    min_h: float = 0.02,
    max_h=1.5,
) -> np.ndarray:
    """
    Computes the rising speeds. This is defined as dh/dt between two subsequent datapoints

    Args:
        map_data (np.ndarray): filled classmap data (time, nodes)
        time_step (np.timedelta64): time step between two observations in classmap
        min_h (float): minimum depth (m) at which depths are considered
        max_h (float): maximum depth (m) at which depths are considered

    Returns:
        dh/dt (np.ndarray): rising_speeds
    """
    # TODO allow for more waterlevels than 1.5
    arrival_ixs = arrival_indices(map_data=map_data, arrival_threshold=min_h)
    max_h_ixs = arrival_indices(map_data=map_data, arrival_threshold=max_h)

    low_h = np.take_along_axis(map_data, arrival_ixs[np.newaxis, :], axis=0)
    upp_h = np.take_along_axis(map_data, max_h_ixs[np.newaxis, :], axis=0)
    dh = upp_h - low_h

    d_ix = max_h_ixs - arrival_ixs
    dt = d_ix * time_step

    return dh / dt


def height_of_mrs(map_data: np.ndarray, dh_dt: np.ndarray) -> np.ndarray:
    """
    Computes the water depth at maximum rising speed.


    Usage example of take_along_axis and argmax. The following prints zero:

    max_rs_ix = np.argmax(dh_dt, axis=0)
    max_rs = np.take_along_axis(dh_dt, max_rs_ix[np.newaxis, :], axis=0)
    print(np.sum(max_rs - np.amax(dh_dt, axis=0, keepdims=True)))

    Args:
        map_data (np.ndarray): filled classmap data (time, nodes)
        dh_dt (np.ndarray): rising_speeds

    Returns:
        h_mrs (np.ndarray): gevaarhoogte. water depth at maximum rising speed
    """

    # find maximum rising speed per point in mesh
    max_rs_ix = np.argmax(dh_dt, axis=0)

    # apply these indices to waterdepth mesh to obtain waterdepths at moment of maximum rising speed
    h_mrs = np.take_along_axis(map_data[1:, :], max_rs_ix[np.newaxis, :], axis=0)

    # profit
    return h_mrs[0, :]
