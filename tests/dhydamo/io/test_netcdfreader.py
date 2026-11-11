import logging

import numpy as np
import pytest
from netCDF4 import Dataset

from hydrolib.dhydamo.io.netcdfreader import time_index


@pytest.fixture
def his_like():
    """An in-memory netCDF file with hourly time steps on 1 June 2016 (0, 1, ..., 5 h)."""
    dataset = Dataset("in_memory.nc", "w", diskless=True)
    dataset.createDimension("time", None)
    time = dataset.createVariable("time", "f8", ("time",))
    time.units = "seconds since 2016-06-01 00:00:00 +00:00"
    time[:] = np.arange(6) * 3600.0
    yield dataset
    dataset.close()


def test_exact_time(his_like, caplog):
    with caplog.at_level(logging.WARNING):
        assert time_index(his_like, "2016-06-01 03:00") == 3
    assert caplog.messages == []


def test_nearest_time_with_warning(his_like, caplog):
    with caplog.at_level(logging.WARNING):
        assert time_index(his_like, "2016-06-01 03:20") == 3
        assert time_index(his_like, "2016-06-01 03:40") == 4
    assert len(caplog.messages) == 2
    assert "nearest time step" in caplog.messages[0]


def test_time_zone_is_converted_to_utc(his_like):
    # 04:00 in UTC+2 is 02:00 UTC
    assert time_index(his_like, "2016-06-01 04:00+02:00") == 2


def test_time_outside_the_period_raises(his_like):
    with pytest.raises(ValueError, match="outside the period"):
        time_index(his_like, "2016-06-02 00:00")


def test_file_without_time_raises():
    with Dataset("no_time.nc", "w", diskless=True) as dataset:
        with pytest.raises(ValueError, match="No time variable"):
            time_index(dataset, "2016-06-01")
