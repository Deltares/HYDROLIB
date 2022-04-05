
from cmt.project import Project
from pathlib import Path

cases_json = Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\populate_cases.json")
#project = Project(filepath=r"../data/test").from_stochastics(cases_json)


# %%
from cmt.utils.readers import read_stochastics


from hydrolib.core.io.bc.models import ForcingModel, TimeSeries, QuantityUnitPair

# %%
from cmt.utils.readers import read_stochastics
import pandas as pd

from cmt.utils.writers import _boundary_to_timeseries

stochast_dict = read_stochastics(cases_json)
events = stochast_dict["boundary_conditions"]["flow"]

i = events[0]
flow_path = Path(".")
filepath = flow_path / f"{i['id']}.bc"
fm = ForcingModel(
    forcing=[_boundary_to_timeseries(j) for j in i["boundaries"]]
    )

#%%



"""
time_series = bc["boundaries"][0]["time_series"]




df = pd.DataFrame(time_series)
df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
    df["time"]
)

df["minutes"] = (df["datetime"] - df.iloc[0]["datetime"]).dt.total_seconds() / 60

datablock = df[["minutes", "value"]].values.tolist()

time_pair = QuantityUnitPair(quantity="time", unit=f"minutes since {df.iloc[0]['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")

value_pair = QuantityUnitPair(quantity=bc["boundaries"][0]["quantity"], unit='m')

ts = TimeSeries(datablock=datablock,
                name=bc["boundaries"][0]["objectid"],
                function="timeseries",
                quantityunitpair=[time_pair, value_pair],
                timeinterpolation='linear',
                offset=0.0,
                factor=1.0
                )
fm = ForcingModel(forcing=[ts])
"""