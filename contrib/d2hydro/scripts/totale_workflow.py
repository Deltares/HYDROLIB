import shutil
import sys
from pathlib import Path, PurePath
from zipfile import ZipFile

from cmt.project import Project
from pandas import DataFrame

stochastics_json = Path(
    r"d:\repositories\HYDROLIB\contrib\d2hydro\data\dellen\populate_cases.json"
)
project = Project(filepath=r"../data/stochast").from_stochastics(stochastics_json)

#%%
project = Project.from_manifest(Path(r"../data/stochast/manifest.json"))
