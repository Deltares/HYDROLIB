# =========================================================================================
#
# License: LGPL
#
# Author: Stefan de Vries, Waterschap Drents Overijsselse Delta
#
# =========================================================================================

import os
from pathlib import Path

import pandas as pd
from read_dhydro import net_nc2gdf, read_locations


def export_model_gis(mdu_path, nc_path, output_path):
    """
     Uses other existing sub-functions to export a D-Hydro schematisation to GIS
    This function cleans up the dflowfm folder to make sure it can be loaded into hydrolib.
    


    Parameters:
         mdu_path : path
             Path to mdu file
         nc_path
             Path to the FlowFM_map.nc, containing the results of the simulation 
         output_path
             Path where the shapefiles are saved as output  
    ___________________________________________________________________________________________________________
    
     Returns:
         Shapefiles with d-hydro schematisation (e.g. branches, nodes, structures etc.)

    """

    network = net_nc2gdf(nc_path)
    network["1d_branches"]["branchnr"] = network["1d_branches"].index
    network["1d_branches"] = network["1d_branches"].rename(columns={"id": "branchid"})
    branche_ids = network["1d_branches"][["branchid", "branchnr"]].copy()
    network["1d_meshnodes"] = network["1d_meshnodes"].rename(
        columns={"id": "meshnodeid"}
    )
    network["1d_meshnodes"] = network["1d_meshnodes"].rename(
        columns={"branch": "branchnr"}
    )
    network["1d_meshnodes"] = pd.merge(
        network["1d_meshnodes"], branche_ids, on="branchnr"
    )
    for key in list(network.keys()):
        if len(network[key]):
            network[key].to_file(os.path.join(output_path, str(key + ".shp")))

    gdfs_results = read_locations(
        mdu_path, ["cross_sections_locations", "structures", "boundaries", "laterals"]
    )
    gdfs_results["cross_sections_locations"] = gdfs_results["cross_sections_locations"][
        ["id", "branchid", "chainage", "shift", "definitionid", "geometry"]
    ]
    gdfs_results["cross_sections_locations"] = gdfs_results[
        "cross_sections_locations"
    ].rename(columns={"definitionid": "def_id", "locationtype": "type"})
    gdfs_results["structures"] = gdfs_results["structures"][
        ["id", "name", "type", "branchid", "chainage", "geometry"]
    ]
    gdfs_results["laterals"] = gdfs_results["laterals"][
        ["id", "nodeid", "branchid", "chainage", "geometry"]
    ]
    for key in list(gdfs_results.keys()):
        if len(gdfs_results[key]):
            gdfs_results[key].to_file(os.path.join(output_path, str(key + ".shp")))


if __name__ == "__main__":
    mdu_path = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\Flow1D.mdu"
    )
    nc_path = Path(
        r"C:\Users\devop\Documents\Scripts\Hydrolib\HYDROLIB\contrib\Arcadis\scripts\exampledata\Dellen\Model_cleaned\dflowfm\dellen_net.nc"
    )
    output_path = Path(r"C:\Users\devop\Desktop\Dellen")
    export_model_gis(mdu_path, nc_path, output_path)
