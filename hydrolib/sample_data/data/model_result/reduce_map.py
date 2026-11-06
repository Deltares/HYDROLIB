"""Reduce a D-Flow FM map file to the 1D variables needed for initial conditions.

Keeps the 1D node coordinates, branch and chainage, and bed level, and the water level, water depth and velocity at the last time
step(s), so the file is small enough for the repository. The result can be read with
HyDAMO.external_forcings.read_values_from_modelrun(..., netcdf_index=-1).

Usage:
    python reduce_map.py DFM_map.nc DFM_map_last.nc [number of last time steps, default 1]
"""

import sys
from datetime import datetime

from netCDF4 import Dataset

STATIC_VARIABLES = [
    "mesh1d_node_x",
    "mesh1d_node_y",
    "mesh1d_node_branch",
    "mesh1d_node_offset",
    "mesh1d_edge_x",
    "mesh1d_edge_y",
    "mesh1d_edge_branch",
    "mesh1d_edge_offset",
    "mesh1d_edge_nodes",
    "network1d_branch_id",
    "mesh1d_flowelem_bl",
    "projected_coordinate_system",
]
TIME_VARIABLES = ["time", "mesh1d_s1", "mesh1d_waterdepth", "mesh1d_u1"]


def reduce_map(src_path, dst_path, n_last=1):
    with Dataset(src_path) as src, Dataset(dst_path, "w", format="NETCDF4") as dst:
        dst.setncatts({name: src.getncattr(name) for name in src.ncattrs()})
        dst.setncattr(
            "history",
            f"{src.getncattr('history') if 'history' in src.ncattrs() else ''}\n"
            f"{datetime.now():%Y-%m-%dT%H:%M:%S}: reduced to the last {n_last} time step(s) with reduce_map.py",
        )

        n_time = src.dimensions["time"].size
        start = max(n_time - n_last, 0)
        for name in STATIC_VARIABLES + TIME_VARIABLES:
            if name not in src.variables:
                continue
            var = src.variables[name]
            for dim in var.dimensions:
                if dim not in dst.dimensions:
                    size = None if dim == "time" else src.dimensions[dim].size
                    dst.createDimension(dim, size)
            fill_value = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
            out = dst.createVariable(name, var.dtype, var.dimensions, zlib=True, complevel=4, fill_value=fill_value)
            out.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs() if attr != "_FillValue"})
            if "time" in var.dimensions:
                out[:] = var[start:]
            else:
                out[:] = var[:]


if __name__ == "__main__":
    reduce_map(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
