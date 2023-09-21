import itertools
import json
import sys
from pathlib import Path
from typing import List
from zipfile import ZipFile

import pandas as pd

from hydrolib.core.io.bc.models import ForcingModel, QuantityUnitPair, TimeSeries


def _get_timespecs(boundaries):
    def _get_datetime(i):
        start_datetime = pd.to_datetime(i[0]["date"]) + pd.to_timedelta(i[0]["time"])
        timestep = (
            pd.to_datetime(i[1]["date"]) + pd.to_timedelta(i[1]["time"])
        ) - start_datetime
        end_datetime = pd.to_datetime(i[-1]["date"]) + pd.to_timedelta(i[-1]["time"])
        time_delta = end_datetime - start_datetime
        time_delta += timestep
        return start_datetime, time_delta

    time_specs = [_get_datetime(i["time_series"]) for i in boundaries]
    start_datetime = max(i[0] for i in time_specs)
    end_datetime = min([i[0] + i[1] for i in time_specs])

    return start_datetime, end_datetime - start_datetime


def _boundary_to_timeseries(boundary, timedelta):
    df = pd.DataFrame(boundary["time_series"])
    total_minutes = timedelta.total_seconds() / 60
    # parse datetime
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["time"])
    df.sort_values("datetime", inplace=True)
    df["minutes"] = (df["datetime"] - df.iloc[0]["datetime"]).dt.total_seconds() / 60

    # extract datablock
    datablock = df[["minutes", "value"]].values.tolist()
    if df.iloc[-1]["minutes"] < total_minutes:
        datablock += [[total_minutes, datablock[-1][1]]]
    # define time QuantityUnitPair

    time_pair = QuantityUnitPair(
        quantity="time",
        unit=f"minutes since {df.iloc[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')}",
    )

    # define value QuantityUnitPair
    value_pair = QuantityUnitPair(quantity=boundary["quantity"], unit="m")

    # fill TimeSeries object
    ts = TimeSeries(
        datablock=datablock,
        name=boundary["objectid"],
        function="timeseries",
        quantityunitpair=[time_pair, value_pair],
        timeinterpolation="linear",
        offset=0.0,
        factor=1.0,
    )
    return ts


def write_stowa_buien(meteo_bc_path, events):
    import stowabui

    meteo_event = stowabui.MeteoEvent()
    boundary_conditions = []
    for i in events:
        file_stem = i["id"]
        specs = i["stowa_bui"]
        if type(specs["duration"]) == int:
            specs["duration"] = pd.Timedelta(hours=specs["duration"])
        meteo_event.update(specs).write_meteo(meteo_bc_path, file_stem)
        boundary_conditions += [
            dict(
                id=i["id"],
                name=i["name"],
                start_datetime=meteo_event.starts[meteo_event.season],
                timedelta=meteo_event.duration,
            )
        ]
    return boundary_conditions


def write_flow_boundaries(flow_bc_path, events):
    boundary_conditions = []
    for i in events:
        filepath = flow_bc_path / f"{i['id']}.bc"
        start_datetime, timedelta = _get_timespecs(i["boundaries"])
        fm = ForcingModel(
            forcing=[_boundary_to_timeseries(j, timedelta) for j in i["boundaries"]]
        )
        fm.filepath = filepath
        fm.save()
        boundary_conditions += [
            dict(
                id=i["id"],
                name=i["name"],
                start_datetime=start_datetime,
                timedelta=timedelta,
            )
        ]
    return boundary_conditions


def write_models(models_path, models, src_dir):
    for i in models:
        src_path = src_dir / i["path"]
        dst_path = models_path / rf"{i['id']}"
        dst_path.mkdir(parents=True, exist_ok=True)
        if src_path.suffix.lower() == ".zip":
            zipdata = ZipFile(src_path)
            zipinfos = zipdata.infolist()
            for zipinfo in zipinfos:
                filepath = Path(zipinfo.filename)
                if not zipinfo.is_dir():
                    relpath = filepath.relative_to(src_path.stem)
                    zipinfo.filename = str(relpath)
                    zipdata.extract(zipinfo, dst_path)


def write_rr_conditions(rr_ini_path, conditions, src_dir):
    for i in conditions:
        filename = f"{i['id']}.3b"
        src_path = src_dir / i["file"]
        if src_path.suffix.lower() == ".zip":
            zipdata = ZipFile(src_path)
            zipinfos = zipdata.infolist()
            zipinfo = next((i for i in zipinfos if i.filename.endswith(".3b")), None)
            if zipinfo is not None:
                zipinfo.filename = filename
                zipdata.extract(zipinfo, rr_ini_path)
