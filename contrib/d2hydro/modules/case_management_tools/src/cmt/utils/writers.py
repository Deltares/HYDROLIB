import json
from pathlib import Path
import sys
from typing import List
import itertools
import pandas as pd
from hydrolib.core.io.bc.models import ForcingModel, TimeSeries, QuantityUnitPair


def _boundary_to_timeseries(boundary):
    df = pd.DataFrame(boundary["time_series"])

    # parse datetime
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["time"])
    df["minutes"] = (
        df["datetime"] - df.iloc[0]["datetime"]
        ).dt.total_seconds() / 60

    # extract datablock
    datablock = df[["minutes", "value"]].values.tolist()

    # define time QuantityUnitPair
    time_pair = QuantityUnitPair(
        quantity="time",
        unit=f"minutes since {df.iloc[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # define value QuantityUnitPair
    value_pair = QuantityUnitPair(quantity=boundary["quantity"], unit='m')

    # fill TimeSeries object
    ts = TimeSeries(datablock=datablock,
                    name=boundary["objectid"],
                    function="timeseries",
                    quantityunitpair=[time_pair, value_pair],
                    timeinterpolation='linear',
                    offset=0.0,
                    factor=1.0
                    )
    return ts


def write_stowa_buien(meteo_path, events):
    import stowabui
    meteo_event = stowabui.MeteoEvent()
    for i in events:
        file_stem = i["id"]
        specs = i["stowa_bui"]
        if type(specs["duration"]) == int:
            specs["duration"] = pd.Timedelta(hours=specs["duration"])
        meteo_event.update(specs).write_meteo(meteo_path, file_stem)


def write_flow_boundaries(flow_path, events):
    for i in events:
        filepath = flow_path / f"{i['id']}.bc"
        fm = ForcingModel(
            forcing=[_boundary_to_timeseries(j) for j in i["boundaries"]]
            )
        fm.filepath = filepath
        fm.save()
