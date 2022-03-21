from hydrolib.core.io.fnm.models import RainfallRunoffModel
from pathlib import Path

sobek_fnm_path = Path(r"d:\repositories\HYDROLIB\contrib\d2hydro\data\working_project\models\e112_f01_c14_multiple_catchments_ini_input\rr\Sobek_3b.fnm")

rr_model = RainfallRunoffModel(filepath=sobek_fnm_path)