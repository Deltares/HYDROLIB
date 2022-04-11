import sys

from cmt.project import Project
from pathlib import Path, PurePath
from zipfile import ZipFile
import shutil

stochastics_json = Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\dellen\populate_cases.json")
project = Project(filepath=r"../data/stochast").from_stochastics(stochastics_json)

# %%
from hydrolib.core.io.fnm.models import RainfallRunoffModel

fnm= RainfallRunoffModel(Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\stochast\models\D-Hydro-model\RR\Sobek_3b.fnm"))