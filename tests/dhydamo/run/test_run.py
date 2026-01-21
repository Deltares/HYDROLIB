
import pathlib
import subprocess
import pytest

from tests.dhydamo.io import test_to_hydrolibcore, test_from_hydamo
from tests.dhydamo.rtc import test_rtc
from tests.dhydamo.rr import test_setup_rr
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter

def _find_dimr(dhydro_path="C:/Program Files/Deltares"):
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
            dimr_path = str(installs[-1].joinpath("plugins/DeltaShell.Dimr/kernels/x64/bin/run_dimr.bat"))

    return dimr_path

@pytest.mark.slow
@pytest.mark.skipif(_find_dimr() is None, reason="D-Hydro not installed or run_dimr.bat not found")
def test_run_model():
    # Read hydamo object only once
    hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()
    hydamo = test_from_hydamo._convert_structures(hydamo=hydamo)

    # Add RR component
    drrmodel = test_setup_rr._setup_rr_model(hydamo=hydamo)
  
    # Setup model
    fm, output_path = test_to_hydrolibcore._write_model(drrmodel=drrmodel, hydamo=hydamo, full_test=True)
    
    # write RR
    drrmodel.d3b_parameters["Timestepsize"] = "60" 
    drrmodel.d3b_parameters["StartTime"] = "'2016/06/01;00:00:00'"
    drrmodel.d3b_parameters["EndTime"] = "'2016/06/03;01:00:00'"
    drrmodel.d3b_parameters["UnsaturatedZone"] = 1
    drrmodel.d3b_parameters["UnpavedPercolationLikeSobek213"] = -1
    drrmodel.d3b_parameters["VolumeCheckFactorToCF"] = 100000
    rr_writer = DRRWriter(drrmodel, output_dir=output_path, name="test", wwtp=(199000.0, 396000.0))
    rr_writer.write_all()
    
    # Add RTC component
    drtcmodel = test_rtc._setup_rtc_model(hydamo=hydamo, fm=fm)
    drtcmodel.write_xml_v1()

    # Find DIMR path
    dimr_path = _find_dimr()

    # Write DIMR config
    dimr = DIMRWriter(output_path=output_path, dimr_path=dimr_path)
    dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
    dimr.write_runbat()

    # Call DIMR .bat file
    p = subprocess.Popen("run.bat", cwd=output_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode("ascii")
    stderr = stderr.decode("ascii")

