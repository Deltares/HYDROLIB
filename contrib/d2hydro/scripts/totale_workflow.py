import sys

from cmt.project import Project
from pathlib import Path, PurePath
from zipfile import ZipFile
import shutil

from pandas import DataFrame

stochastics_json = Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\dellen\populate_cases.json")
project = Project(filepath=r"../data/stochast").from_stochastics(stochastics_json)

#%%
project = Project.from_manifest(Path(r"../data/stochast/manifest.json"))