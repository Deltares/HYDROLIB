from pathlib import Path

from cmt.project import Project

stochastics_json = Path(r"../data/dellen/populate_cases.json")
project = Project(filepath=r"../data/stochast").from_stochastics(stochastics_json)
