"""Generator for STOWA meteo forcing"""

import csv
from pathlib import Path

import pandas as pd

data_path = Path(__file__).parent.joinpath("data").absolute().resolve()
patterns_xlsx = data_path / "patronen.xlsx"


bui_header = """*Name of this file: {file_path}
*Date and time of construction: {now}
*Comments are following an * (asterisk) and written above variables
1
*Aantal stations
1
*Station Name
'STOWA_BUI'
*Number_of_events seconds_per_timestamp
1 3600
*Start datetime and number of timestamps in the format: yyyy#m#d:#h#m#s:#d#h#m#s
*Observations per timestamp (row) and per station (column)
{time_specs}
"""

evp_header = """*Evaporationfile
*Meteo data: evaporation intensity in mm/day
*First record: start date, data in mm/day
*Datum (year month day), evaporation (mm/dag)
"""

starts = {
    "zomer": pd.Timestamp(year=2000, month=6, day=1),
    "winter": pd.Timestamp(year=2000, month=1, day=1),
}


class MeteoEvent(object):
    """A meteo event, including rainfall and evaporation."""

    def __init__(self, duration=pd.Timedelta(hours=24), starts=starts):
        self.pattern = "HOOG"
        self.season = "zomer"
        self.name_pattern = "{hours}H_{volume}MM_{pattern}_{season}"
        self.starts = starts
        self.volume = self._Volume()
        self.duration = duration

    class _Volume(object):
        evaporation = 0
        rainfall = 15

    def get_rainfall_series(self):
        """
        Get rainfall series for a given pattern.

        Returns
        -------
        pandas.Series: series with pd.Timestamp index and mm rainfall per timestamp.

        """
        hours = self.duration / pd.Timedelta(hours=1)

        if (hours).is_integer():
            sheet_name = f"{int(hours)}uur"
            df = pd.read_excel(patterns_xlsx, sheet_name=sheet_name, index_col=0)
            series = df[self.pattern] * self.volume.rainfall
            series.index = [
                self.starts[self.season] + pd.Timedelta(hours=i) for i in series.index
            ]
            series.index.name = "Timestamp"
            return series

    def get_evaporation_series(self):
        """
        Get evaporation series for a given pattern.

        Returns
        -------
        pandas.Series: series with pd.Timestamp index and mm evaporation per timestamp.

        """
        days = int(self.duration / pd.Timedelta(days=1) + 1)
        index = [self.starts[self.season] + pd.Timedelta(days=i) for i in range(days)]
        series = pd.Series(data=[self.volume.evaporation] * days, index=index)

        return series

    def update(self, specs):
        for k, v in specs.items():
            if k == "volume":
                self.volume.rainfall = v
            else:
                setattr(self, k, v)
        return self

    def write_meteo(self, meteo_dir: Path, file_stem: str = None):
        """
        Method to write meteofiles (DEFAULT.BUI and DEFAULT.EVP) based on values per catchment.

        Args:
            meteo_dir (str | Path): Directiory where meteo_file is to be written to
            file_stem (str, optional): Pattern to use for meteo (.bui, .evp, etc) file
                names. When None the pattern defaults to:

                {hours}H_{volume}MM_{pattern}_{season}

        Returns:
            file_stem (TYPE): DESCRIPTION.

        """
        # make dir if not exists
        meteo_path = Path(meteo_dir)
        meteo_path.mkdir(parents=True, exist_ok=True)

        # precipitation
        series = self.get_rainfall_series()

        hours = self.duration / pd.Timedelta(hours=1)
        if file_stem is None:
            file_stem = self.name_pattern.format(
                hours=int(hours),
                volume=int(self.volume.rainfall),
                pattern=self.pattern,
                season=self.season.upper(),
            )

        file_path = meteo_path / f"{file_stem}.BUI"

        now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S GMT")
        time_specs = "{start} {days} {hours} {minutes} {seconds}".format(
            start=self.starts[self.season]
            .strftime("%Y X%m X%d X%H X%M X%S")
            .replace("X0", ""),  # as compact + platform (windows/linux) independent
            days=self.duration.components.days,
            hours=self.duration.components.hours,
            minutes=self.duration.components.minutes,
            seconds=self.duration.components.seconds,
        )

        header = bui_header.format(file_path=file_path, now=now, time_specs=time_specs)

        file_path.write_text(header)

        series.to_csv(
            file_path, header=False, index=False, sep=" ", float_format="%.3f", mode="a"
        )

        # evaporation
        file_path = meteo_path / f"{file_stem}.EVP"
        file_path.write_text(evp_header)
        series = self.get_evaporation_series()

        series.to_csv(
            file_path,
            float_format="%.3f",
            date_format="%#Y  %#m  %#d ",
            quoting=csv.QUOTE_NONE,
            sep=" ",
            header=False,
            mode="a",
            escapechar=" ",
        )

        # temperature
        file_path = meteo_path / f"{file_stem}.TMP"
        file_path.write_text("")

        return file_stem
