"""Reduce a D-Flow FM his-file to the variables used by the tests, for tests/reference_model.

Keeps the time, the names and coordinates of stations and structures, and the time series read by
test_run.py (via timeseries_from_other_model) and by the RTC tests (set_state_from_modelrun).

Usage:
    python tests/tools/reduce_his.py DFM_his.nc tests/reference_model/dflowfm/output/DFM_his.nc
"""

import re
import sys
from datetime import datetime

from netCDF4 import Dataset

TIME_SERIES = [
    "waterlevel",
    "weirgen_discharge",
    "weirgen_s1up",
    "weirgen_s1dn",
    "weirgen_crest_level",
    "pump_structure_discharge",
    "pump_capacity",
    "uniweir_discharge",
    "uniweir_s1up",
    "uniweir_crest_level",
    "bridge_discharge",
    "cmpstru_discharge",
    "culvert_discharge",
    "culvert_gate_lower_edge_level",
    "culvert_gate_opening_height",
    "orifice_crest_level",
    "orifice_gate_lower_edge_level",
]
# names and locations of stations and structures
STATIC = re.compile(r"^(time|(?!lateral).*_name|.*_input_geom_.*|station_[xy]_coordinate|projected_coordinate_system)$")


def reduce_his(src_path, dst_path):
    with Dataset(src_path) as src, Dataset(dst_path, "w", format="NETCDF4") as dst:
        dst.setncatts({name: src.getncattr(name) for name in src.ncattrs()})
        dst.setncattr("history", f"{datetime.now():%Y-%m-%d} reduced to the variables used by the tests (reduce_his.py)")
        names = [name for name in src.variables if STATIC.match(name) or name in TIME_SERIES]
        dimensions = {dim for name in names for dim in src[name].dimensions}
        for dim in dimensions:
            size = src.dimensions[dim]
            dst.createDimension(dim, None if size.isunlimited() else len(size))
        for name in names:
            var = src[name]
            fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            new = dst.createVariable(name, var.datatype, var.dimensions, fill_value=fill, zlib=True)
            new.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            new[:] = var[:]
        missing = sorted(set(TIME_SERIES) - set(names))
        if missing:
            print(f"Not in {src_path}: {missing}")


if __name__ == "__main__":
    reduce_his(sys.argv[1], sys.argv[2])
