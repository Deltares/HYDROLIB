from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr

from hydrolib.core.dflowfm import FMModel, StructureModel

liwo_meta_fields = {}
liwo_meta_fields["Scenario"] = [
    "Scenario Identificatie",
    "Scenarionaam",
    "Scenariodatum",
    "Projectnaam",
    "Eigenaar overstromingsinformatie",
    "Beschrijving scenario",
    "Versie resultaat",
    "Motivatie rekenmethode",
    "Doel",
    "Berekeningsmethode",
    "Houdbaarheid scenario",
]
liwo_meta_fields["Locatie"] = [
    "y-coordinaten doorbraaklocatie",
    "x-coordinaten doorbraaklocatie",
    "Naam buitenwater",
    "Naam waterkering",
    "Naam doorbraaklocatie",
    "Gebiedsnaam",
]
liwo_meta_fields["Bres"] = [
    "overstromingskans - Norm (dijktraject)",
    "overstromingskans - DPV referentie (dijktraject)",
    "overstromingskans - DPV referentie (trajectdeel)",
    "overstromingskans - beheerdersoordeel (trajectdeel)",
    "Keringbeheerder",
    "Materiaal kering",
    "Lengte dijkringdeel",
    "Bresdiepte",
    "Duur bresgroei in verticale richting",
    "Initiele bresbreedte",
    "Methode bresgroei",
    "Startmoment bresgroei",
    "Maximale bresbreedte",
    "Kritieke stroomsnelheid (Uc)",
    "bresgroeifactor 1 (f1)",
    "bresgroeifactor 2 (f2)",
    "Afvoer coefficient (Ce)",
    "Initial Crest [m+NAP]",
    "Rivierknoop",
    "Boezemknoop",
    "Lowest crest",
    "Gridhoogte",
    "Maximaal bresdebiet",
    "Maximale instroom",
    "Wieldiepte",
    "Maximale waterstand",
]
liwo_meta_fields["Buitenwater"] = [
    "Buitenwatertype",
    "Maximale buitenwaterstand",
    "Stormvloedkering open",
    "Stormvloedkering gesloten",
    "Compartimentering van de boezem",
    "Gemiddeld meerpeil",
    "Eigenschappen getij",
    "Piekduur",
    "Stormduur",
    "Timeshift",
    "Debiet randvoorwaarden(rvw) locatie",
    "Waterstand randvoorwaarden(rvw) locatie",
    "Overschrijdingsfrequentie",
    "Maximale afvoer Rijn",
    "Maximale afvoer Maas",
]
liwo_meta_fields["Model"] = [
    "Datum modelschematisatie",
    "Variant beschrijving",
    "Bodemhoogte model",
    "Ruwheid model",
    "Start berekening",
    "Einde berekening",
    "Rekenduur",
    "Start simulatie",
    "Einde simulatie",
    "Duur",
    "Modelversie",
    "Modelleersoftware",
    "Modelresolutie",
]
liwo_meta_fields["Resterende"] = [
    "Regionale keringen of hoge lijnelementen standzeker",
    "Overige opmerkingen",
]
liwo_meta_fields["Bestanden"] = [
    "Maximale stroomsnelheid (asc of zip)",
    "Animatie waterdiepte (inc of zip)",
    "Maximale waterdiepte (asc of zip)",
    "Rapportage (pdf of word)",
    "3Di resultaat (nc)",
    "Bathymetrie (tif)",
]

modelrun_mdufile = None
modelrun_diafile = None
modelrun_mapfile = None
modelrun_clmfile = None
modelrun_hisfile = None


def _build_liwo_multiindex():
    cols = [
        (section, field)
        for section in liwo_meta_fields.keys()
        for field in liwo_meta_fields[section]
    ]
    df = pd.DataFrame(columns=cols)

    print(df)
    return df


def _fill_liwo_scenario(df: pd.DataFrame):
    global fmmodel
    hisdata = xr.open_dataset(modelrun_hisfile)
    df1 = pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
    df = df.append(df1, ignore_index=True)

    df.loc[0][("Scenario", "Scenariodatum")] = hisdata.attrs["date_created"]

    #    structures = StructureModel()
    strs = fmmodel.geometry.structurefile[0].structure

    df.loc[0][("Locatie", "y-coordinaten doorbraaklocatie")] = 1.23

    df.to_excel(r"d:\dam_ar\dflowfm_models\WsVV\liwo_import.xlsx")

    pass


def _set_modelrun_files(mdupath: Union[Path, str]):
    """Derive output file paths for a given D-Flow FM mdu file.

    Args:
        mdupath (Union[Path, str]): Path to the D-Flow FM MDU input file.
    """
    global modelrun_mdufile, modelrun_diafile, modelrun_mapfile, modelrun_clmfile, modelrun_hisfile, fmmodel

    if isinstance(mdupath, str):
        modelrun_mdufile = Path(mdupath)
    elif isinstance(mdupath, Path):
        modelrun_mdufile = mdupath
    else:
        raise ValueError("Argument mdupath must be either str or Path.")

    md_ident = modelrun_mdufile.stem
    workdir = modelrun_mdufile.parent

    fmmodel = FMModel(modelrun_mdufile, recurse=False)
    outputdir = Path(fmmodel.output.outputdir or "DFM_OUTPUT_" + md_ident)
    if not outputdir.is_absolute():
        outputdir = workdir / outputdir

    modelrun_diafile = outputdir / (md_ident + ".dia")
    modelrun_mapfile = outputdir / (
        fmmodel.output.mapfile.filepath or md_ident + "_map.nc"
    )
    modelrun_clmfile = outputdir / (
        fmmodel.output.classmapfile.filepath or md_ident + "_clm.nc"
    )
    modelrun_hisfile = outputdir / (
        fmmodel.output.hisfile.filepath or md_ident + "_his.nc"
    )

    print(f"Using model files:")
    print(f" - MDU file : {modelrun_mdufile}")
    print(f" - .dia file: {modelrun_diafile}")
    print(f" - map file : {modelrun_mapfile}")
    print(f" - clm file : {modelrun_clmfile}")
    print(f" - his file : {modelrun_hisfile}")


if __name__ == "__main__":
    _set_modelrun_files(r"d:\dam_ar\dflowfm_models\WsVV\testmodel\DFM.mdu")
    df = _build_liwo_multiindex()
    _fill_liwo_scenario(df)
