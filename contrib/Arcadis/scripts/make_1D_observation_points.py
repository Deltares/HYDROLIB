# ===============================================================================
#
# Copyright (c) 2022 Arcadis Nederland B.V.
#
# Without prior written consent of Arcadis Nederland B.V., this code
# may not be published or duplicated.
#
# ===============================================================================
#! @script_name             Make observation points
#! @summary                 This script makes observation points for a selection of 1D channels within D-Hydro.
#! @author                  Robbert de Lange
#! @reviewer
#! @script_version          1.0
#! @date                    2022 - 03 - 18
#! @field                   Waterbeheer
#! @application             D-Hydro
#! @status                  Draft
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from read_dhydro import branch_gui2df, net_nc2gdf, read_locations

from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.obs.models import ObservationPointModel


def make_obs_points(mdu_path, output_path, prefix="rOut", fraction=0.95):
    """
    ___________________________________________________________________________________________________________

    Ontwikkelaar: Robbert de Lange
    Tester: -
    Status: Draft
    ___________________________________________________________________________________________________________

    Parameters
    ----------
    input_file : string
        pad naar input net_nc bestand in de dflowfm map van het model
    output_folder : string
        pad naar output folder
    observation_len: integer
        lengte tussen elk observatiepunt op de watergang
    ___________________________________________________________________________________________________________

    Resultaat
    -------
    Een observation.ini bestand in de gekozen output folder. Deze kan in de mdu en folder worden gezet van het D-Hydro model.
    De naam van het bestand is "1DObservationpoints.ini". Dit kan handmatig worden aangepast.

    """
    fm = FMModel(Path(mdu_path))
    net_nc = fm.geometry.netfile.filepath

    branches = net_nc2gdf(
        os.path.join(Path(mdu_path).parent, net_nc), results=["1d_branches"]
    )["1d_branches"]
    obs_branches = branches[branches.id.str.startswith(prefix)].copy()
    obs_branches.rename(columns={"id": "branchid"}, inplace=True)
    obs_branches.drop(columns=["name", "order"], inplace=True)
    obs_branches.insert(0, "name", "meas" + obs_branches.branchid)
    # obs_branches.insert(2,'locationtype',"1d")
    obs_branches.insert(2, "chainage", obs_branches.length * fraction)
    obs_branches.drop(
        columns=["length", "geometry", "branchType", "isLengthCustom"], inplace=True
    )

    towrite = ObservationPointModel(observationpoint=obs_branches.to_dict("records"))
    towrite.save(Path(output_path) / (str("1d_obspoints.ini")))


if __name__ == "__main__":
    mdu_path = r"C:\scripts\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\flowFM.mdu"
    output_path = r"C:\scripts\AHT_scriptjes\make_obs_points"
    # branches = branch_gui2df(os.path.join(Path(mdu_path).parent, fm.geometry.branchfile))

    print("dummy")
