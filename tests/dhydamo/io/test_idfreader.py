import datetime
import re
from pathlib import Path

import numpy as np
import pytest
from hydrolib.dhydamo.io import idfreader

SEEPAGE = Path("hydrolib/sample_data/data/rasters/seepage").resolve()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20160601123045", datetime.datetime(2016, 6, 1, 12, 30, 45)),
        ("201606011230", datetime.datetime(2016, 6, 1, 12, 30)),
        # used to fail with "year 2016060112 is out of range"
        ("2016060112", datetime.datetime(2016, 6, 1, 12)),
        ("20160601", datetime.datetime(2016, 6, 1)),
        # used to give the month and day of today
        ("2016", datetime.datetime(2016, 1, 1)),
        ("2016-06-01", datetime.datetime(2016, 6, 1)),
    ],
)
def test_to_datetime(text: str, expected: datetime.datetime) -> None:
    assert idfreader.to_datetime(text) == expected


def test_decompose_with_a_compiled_pattern() -> None:
    # isinstance(pattern, pattern) used to raise a TypeError
    pattern = re.compile(r"(?P<name>[a-z]+)_(?P<time>\d+)_l(?P<layer>\d+)")

    d = idfreader.decompose("bdgflf_20160528_l1.idf", pattern=pattern)

    assert (d["name"], d["time"], d["layer"]) == (
        "bdgflf",
        datetime.datetime(2016, 5, 28),
        1,
    )


def test_open_and_affine() -> None:
    dataset = idfreader.open(SEEPAGE / "bdgflf_20160528_l1.idf").squeeze()

    assert dataset.shape == (460, 500)
    assert dataset["time"].values == np.datetime64("2016-05-28")
    affine = idfreader.affine_from_idf(dataset)
    assert (affine.a, affine.e) == (50.0, -50.0)
    # the centre of the first cell
    assert affine @ (0.5, 0.5) == (float(dataset.x[0]), float(dataset.y[0]))


def test_open_several_files_along_time() -> None:
    dataset = idfreader.open(SEEPAGE / "bdgflf_*_l1.idf")

    assert dataset.sizes["time"] == 3
    assert list(dataset["time"].dt.day.values) == [28, 14, 28]
