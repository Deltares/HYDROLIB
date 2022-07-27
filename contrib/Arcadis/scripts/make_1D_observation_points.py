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

import numpy as np
import pandas as pd
import xarray as xr

# ER MOET NOG INGEBOUWD WORDEN DAT JE DE KEUZE KAN MAKEN IN WELKE CHANNELS JE WILT


def aht_make_obs_points(input_file, output_path, observation_len):
    """
    ### Beschrijf hier wat je script precies doet. ###
    Dit script maakt observation points op de 1D watergangen op verschillende punten, wat gekozen kan worden door de gebruiker.
    Dit script maakt enkel nog een observatiepunt elke 100 meter per watergang. Het herkent niet of watergangen in elkaar overlopen.
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
    ###DIT MOET NOG AANGEPAST WORDEN, DIT IS SPECIFIEK VOOR HET D-HYDRO GEVAL VAN WRIJ
    df = get_channel_data(input_file)[:55]

    ################################

    obslen = 500
    output_file = os.path.join(output_path, "1DObservationpoints.ini")
    with open(output_file, "w") as file:
        file.write(
            "[General]\n    fileVersion           = 2.00\n    fileType              = obsPoints\n"
        )

        for index, row in df.iterrows():
            maxlen = row["lengte"]
            steps = np.arange(0, maxlen, obslen)
            for chainage in steps:
                file.write(
                    "\n[ObservationPoint]\n    name                  = "
                    + "obs_"
                    + str(index)
                    + "_"
                    + str(int(chainage))
                    + "\n    branchId              = "
                    + str(index)
                    + "\n    chainage              = "
                    + str(chainage)
                    + "\n "
                )


def aht_get_channels(file):
    df = get_channel_data(input_file)
    return [df.index]


def load_dataset(input_path):
    return xr.open_dataset(input_path)


def get_channel_data(input_path):
    ds = load_dataset(input_path)
    names = [x.decode("utf-8").strip() for x in ds["network_branch_id"].data]
    data = list(ds["network_edge_length"].data)
    df = pd.DataFrame(data=data, index=names, columns=["lengte"])
    # print (data)
    # par = choose_param(ds)
    # extensie = '.csv'  #input('Extentension choice (.csv, .jpg or .shp)')
    ds.close()
    return df


if __name__ == "__main__":
    input_file = r"\\chh6RD93\Arcadis\WRIJ_overstromingsberekeningen\DR48\1D2D_modelbouw\036_nieuw_uitstroom\dflowfm\dr48_net.nc"
    output_path = r"C:\scripts\AHT_scriptjes\make_obs_points"
    aht_make_obs_points(input_file, output_path, 1000)
