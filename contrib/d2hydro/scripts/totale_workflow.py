from cmt.project import Project
from pathlib import Path


stochastics_json = Path(r"../data/dellen/populate_cases.json")
project = Project(filepath=r"../data/stochast").from_stochastics(stochastics_json)