import numpy as np


def arrival_times(
    clm_data: np.ndarray,
    time_step: np.timedelta64,
    time_unit: str = "s",
    arrival_threshold: float = 3,
) -> np.ndarray:
    """
    Computes the arrival times. This is the time-index that water_height >= threshold

    Args:
        clm_data (np.ndarray): classmap data (time, nodes)
        time_step (np.timedelta64): time step between two observations in classmap
        time_unit (str): desired output time format
        arrival_threshold: class >= arrival_threshold there is inundation

    Returns:
        t_arrival (np.ndarray): arrival times
    """

    # Determine where waterheight >= threshold
    arrived = clm_data >= arrival_threshold

    # Find first time that this is the case
    indices = np.argmax(arrived, axis=0).astype(float)

    # Turn indices into time
    t_arrival = indices * time_step / np.timedelta64(1, time_unit)

    # if never inundated, then nan
    nevers = np.invert(np.amax(arrived, axis=0))
    t_arrival[nevers] = np.nan

    return t_arrival


def rising_speeds(
    map_data: np.ndarray, time_step: np.timedelta64, time_unit: str = "s"
) -> np.ndarray:
    """
    Computes the rising speeds. This is defined as dh/dt between two datapoints

    Args:
        map_data (np.ndarray): filled classmap data (time, nodes)
        time_step (np.timedelta64): time step between two observations in classmap
        time_unit (str): desired output time format

    Returns:
        dh_dt (np.ndarray): rising_speeds
    """

    dh = map_data[1:, :] - map_data[:-1, :]
    dt = time_step / np.timedelta64(1, time_unit)
    dh_dt = dh / dt

    return dh_dt


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
