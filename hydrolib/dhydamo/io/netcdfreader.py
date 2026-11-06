import logging
from datetime import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

logger = logging.getLogger(__name__)


def time_index(dataset: Dataset, time: datetime | pd.Timestamp | str, tolerance: float = 1.0) -> int:
    """Index of the time step in a D-Flow FM output file (map, his or fou) closest to a given time.

    Args:
        dataset (Dataset): opened netCDF file with a 'time' variable
        time (datetime | pd.Timestamp | str): requested time, e.g. "2016-06-02 12:00". A time without a
            time zone is taken in the time zone of the file (usually UTC).
        tolerance (float, optional): maximum difference in seconds that is accepted without a warning.
            Defaults to 1.0.

    Returns:
        int: index of the nearest time step
    """
    if "time" not in dataset.variables:
        raise ValueError(f"No time variable in {dataset.filepath()}.")
    variable = dataset["time"]
    times = pd.DatetimeIndex(
        num2date(variable[:], variable.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    )
    requested = pd.Timestamp(time)
    if requested.tzinfo is not None:
        requested = requested.tz_convert("UTC").tz_localize(None)

    if requested < times[0] or requested > times[-1]:
        raise ValueError(f"Time {requested} is outside the period of {dataset.filepath()}: {times[0]} - {times[-1]}.")
    index = int(np.argmin(np.abs((times - requested).total_seconds())))
    difference = abs((times[index] - requested).total_seconds())
    if difference > tolerance:
        logger.warning(f"No output at {requested}; the nearest time step {times[index]} is used.")
    else:
        logger.info(f"Time step {index} ({times[index]}) is used.")
    return index
