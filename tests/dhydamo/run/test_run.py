
import pathlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from tests.dhydamo.io import test_from_hydamo, test_to_hydrolibcore
from tests.dhydamo.rr import test_setup_rr
from tests.dhydamo.rtc import test_rtc


def _find_dimr(dhydro_path="C:/Program Files/Deltares", version=None):
    dimr_path = None

    # Find latest D-Hydro installation
    # @TODO only checks windows
    # @TODO assumes program files installation
    dhydro_path = pathlib.Path(dhydro_path)
    if dhydro_path.exists():
        installs = [p for p in dhydro_path.iterdir() if p.is_dir() and p.name.startswith("D-HYDRO Suite")]
        if len(installs) > 0:
            sort_keys = [p.name.replace("D-HYDRO Suite", "").strip().split(" ")[0] for p in installs]
            installs = [x for _, x in sorted(zip(sort_keys, installs))]
            if version is None:
                dimr_path = str(installs[-1].joinpath("plugins/DeltaShell.Dimr/kernels/x64/bin/run_dimr.bat"))
            else:
                installs = [i for i in installs if str(version) in str(i)]
                if installs == []:
                    raise ValueError(f"No D-Hydro installation found with version {version}")
                dimr_path = str(installs[0].joinpath("plugins/DeltaShell.Dimr/kernels/x64/bin/run_dimr.bat"))
    return dimr_path

@pytest.mark.slow
@pytest.mark.skipif(_find_dimr() is None, reason="D-Hydro not installed or run_dimr.bat not found")

# change version here to get test a different DHYDRO version
def test_run_model(version=None):
    # Read hydamo object only once    
    hydamo = test_from_hydamo._add_structures_manually()
    hydamo = test_from_hydamo._convert_structures(hydamo_obj=hydamo)

    # Add RR component
    drrmodel = test_setup_rr._setup_rr_model(hydamo=hydamo)
  
    # Setup model
    fm, output_path = test_to_hydrolibcore._write_model(drrmodel=drrmodel, hydamo_obj=hydamo, full_test=True)
    
    # write RR
    drrmodel.d3b_parameters["Timestepsize"] = "60" 
    drrmodel.d3b_parameters["StartTime"] = "'2016/06/01;00:00:00'"
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;00:00:00'"
    drrmodel.d3b_parameters["UnsaturatedZone"] = 1
    drrmodel.d3b_parameters["UnpavedPercolationLikeSobek213"] = -1
    drrmodel.d3b_parameters["VolumeCheckFactorToCF"] = 100000
    rr_writer = DRRWriter(drrmodel, output_dir=output_path, name="test", wwtp=(199000.0, 396000.0))
    rr_writer.write_all()
    
    # Add RTC component
    drtcmodel = test_rtc._setup_rtc_model(hydamo=hydamo, fm=fm, output_path=output_path)
    drtcmodel.write_xml_v1()

    # Find DIMR path
    dimr_path = _find_dimr(version=version)
    # dimr_path = dimr_path.replace('2026.01', '2025.01')

    # Write DIMR config
    dimr = DIMRWriter(output_path=output_path, dimr_path=dimr_path)
    dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
    dimr.add_crs()
    dimr.write_runbat(debuglevel=6, runlog='dimr.log')

    # Call DIMR .bat file
    result = subprocess.run(["cmd", "/c", "run.bat"], cwd=output_path, capture_output=True, text=True,  check=False)
    print(result.stderr)
    assert result.returncode == 0
    # p = subprocess.run("run.bat", cwd=output_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = p.communicate()
    # stdout = stdout.decode("ascii")
    # stderr = stderr.decode("ascii")

output_path = pathlib.Path(__file__).parent / ".." / ".." / "model"
output_path.mkdir(parents=False, exist_ok=True)

# path where the input data is located
@pytest.mark.slow
@pytest.mark.skipif(_find_dimr() is None, reason="D-Hydro not installed or run_dimr.bat not found")
@pytest.mark.parametrize(
    ("his_file", "location_type", "location_id", "variable"),
     [  (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'observation_point', 'ObsP_113GIS' , 'waterlevel'           ),           
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'weir'             , 'S_96550'     , 'discharge'            ),            
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'weir'             , 'S_96550'     , 'waterlevel_upstream'  ),  
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'weir'             , 'S_96550'     , 'waterlevel_downstream'),
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'pump'             , '113GIS'      , 'discharge'            ), 
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'uweir'            , 'uwtest'      , 'discharge'            ),
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'bridge'           , 'KBR_test'    , 'discharge'            ),
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'compound'         , 'cmp_S_97947' , 'discharge'            ),
        (output_path / 'dflowfm' / 'output' / 'DFM_his.nc', 'culvert'          , 'D_25561'     , 'discharge'            ),
    ],
)
def test_read_output(his_file, location_type, location_id, variable):
    if not (output_path / 'dflowfm').exists():
        test_run_model()    
    hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()
 
    his_reference = output_path   / '..' / 'reference_model' / 'dflowfm' / 'output'/ 'DFM_his.nc'
    assert his_reference.exists()

    his_test = output_path / 'dflowfm' / 'output' / 'DFM_his.nc'
    assert his_test.exists()
 
    geom_test, series_test = hydamo.external_forcings.convert.timeseries_from_other_model(his_test, location_type, location_id, variable)
    
    geom_ref, series_ref = hydamo.external_forcings.convert.timeseries_from_other_model(his_reference, location_type, location_id, variable)

    _, ax = plt.subplots()
    ax.plot(series_ref.index, series_ref.values, label='Reference', color='blue')
    ax.plot(series_test.index, series_test.values, label='New model', color='Red', linestyle='--')
    ax.set_ylabel(variable)
    ax.set_title(f'{location_type} {location_id}: {variable}')
    ax.legend()
    ax.grid()
    plt.savefig(output_path / '..' / 'figures' / f'test_run_{location_type}_{location_id}_{variable}.png')

    assert series_ref.shape[0] == series_test.shape[0]
    assert np.round(np.max(series_ref),1) == np.round(np.max(series_test),1)

    assert geom_test == geom_ref
