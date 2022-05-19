import json
from pathlib import Path

from hydrolib.core.io.fnm.models import RainfallRunoffModel

sobek_fnm_path = Path(
    r"d:\repositories\HYDROLIB\contrib\d2hydro\data\working_project\models\e112_f01_c14_multiple_catchments_ini_input\rr\Sobek_3b.fnm"
)
cases_json = Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\populate_cases.json")


rr_model = RainfallRunoffModel(filepath=sobek_fnm_path)

cases_dict = json.loads(cases_json.read_text())
