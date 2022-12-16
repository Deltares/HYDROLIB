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

from read_dhydro import net_nc2gdf

from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.obs.models import ObservationPointModel


def make_obs_points(mdu_path, output_path, prefix="rOut", fraction=0.95):
    """
    ___________________________________________________________________________________________________________

    Ontwikkelaar: Robbert de Lange
    Tester: -
    Status: Draft
    ___________________________________________________________________________________________________________

    Creates new observation point on each branch. Point is placed at a location 'branch length * fraction'

    Parameters
    mdu_path : str
       Path to mdu file containing the D-hydro model structure
    output_path : str
        Path where the 1DObservationpoints.ini file is saved
    prefix : str
        Name of observation points is prefix + branchid
    fraction : float
        Fraction of the branch length where the observation point should be placed
        Between 0 and 1.0
    ___________________________________________________________________________________________________________

    Returns:
        A observation.ini file ("1DObservationpoints.ini") is saved in the output folder.
        This file can be moved to the folder of a D-Hydro model and be registered in
        the MDU file (keyword ObsFile)

    """

    if fraction < 0 or fraction > 1:
        raise Exception("Fraction should be a float number between 0 and 1.0")

    fm = FMModel(Path(mdu_path))
    net_nc = fm.geometry.netfile.filepath

    branches = net_nc2gdf(
        os.path.join(Path(mdu_path).parent, net_nc), results=["1d_branches"]
    )["1d_branches"]
    obs_branches = branches[branches.id.str.startswith(prefix)].copy()
    obs_branches.rename(columns={"id": "branchid"}, inplace=True)
    obs_branches.drop(columns=["name", "order"], inplace=True)
    obs_branches.insert(0, "name", "meas" + obs_branches.branchid)
    obs_branches.insert(2, "chainage", obs_branches.length * fraction)
    obs_branches.drop(
        columns=["length", "geometry", "branchType", "isLengthCustom"], inplace=True
    )

    towrite = ObservationPointModel(observationpoint=obs_branches.to_dict("records"))
    towrite.save(Path(output_path) / (str("1d_obspoints.ini")))


if __name__ == "__main__":
    mdu_path = r"C:\Users\devop\Documents\Scripts\Hydrolib - kopie\HYDROLIB\contrib\Arcadis\scripts\exampledata\Zwolle-Minimodel_clean\1D2D-DIMR\dflowfm\FlowFM.mdu"
    output_path = r"C:\Users\devop\Desktop\Dellen"
    # branches = branch_gui2df(os.path.join(Path(mdu_path).parent, fm.geometry.branchfile))
    make_obs_points(mdu_path, output_path, prefix="rOut", fraction=0.95)
    print("dummy")
