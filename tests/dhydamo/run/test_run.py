import pathlib
import subprocess
import pytest

from tests.dhydamo.io import test_to_hydrolibcore, test_from_hydamo
from tests.dhydamo.rtc import test_setup_rtc
from tests.dhydamo.rr import test_setup_rr
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter


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
            dimr_path = str(installs[-1].joinpath("plugins/DeltaShell.Dimr/kernels/x64/dimr/scripts/run_dimr.bat"))

    return dimr_path

@pytest.mark.slow
def test_run_model():
    # Read hydamo object only once
    hydamo = test_from_hydamo.test_hydamo_object_from_gpkg()
    hydamo = test_from_hydamo.test_convert_structures(hydamo=hydamo)

    hydamo.structures.add_rweir(
        id="rwtest",
        name="rwtest",
        branchid="W_1386_0",
        chainage=2.0,
        crestlevel=12.5,
        crestwidth=3.0,
        corrcoeff=1.0,
    )
    hydamo.structures.add_orifice(
        id="orifice_test",
        branchid="W_242213_0",
        chainage=43.0,
        crestlevel=18.00,
        gateloweredgelevel=18.5,
        crestwidth=7.5,
        corrcoeff=1.0,
    )
    hydamo.structures.add_uweir(
        id="uweir_test",
        branchid="W_242213_0",
        chainage=2.0,
        crestlevel=18.00,
        crestwidth=7.5,
        dischargecoeff=1.0,
        numlevels=4,
        yvalues="0.0 1.0 2.0 3.0",
        zvalues="19.0 18.0 18.2 19",
    )

    cmpnd_ids = ["cmpnd_1","cmpnd_2","cmpnd_3"]
    cmpnd_list = [["D_24521", "D_14808"],["D_21450", "D_19758"],["D_19757", "D_21451"]]
    hydamo.structures.convert.compound_structures(cmpnd_ids, cmpnd_list)

    # Add RR component
    drrmodel = test_setup_rr.test_setup_rr_model(hydamo=hydamo)

    # Setup model
    fm, output_path = test_to_hydrolibcore.test_write_model(drrmodel=drrmodel, hydamo=hydamo, full_test=True)

    # Add RTC component
    drtcmodel = test_setup_rtc.test_setup_rtc_model(hydamo=hydamo)
    drtcmodel.write_xml_v1()

    # Find DIMR path
    dimr_path = _find_dimr()
    assert dimr_path is not None

    # Write DIMR config
    dimr = DIMRWriter(output_path=output_path, dimr_path=dimr_path)
    dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
    dimr.write_runbat()

    # Call DIMR .bat file
    p = subprocess.Popen("run.bat", cwd=output_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode("ascii")
    stderr = stderr.decode("ascii")

    # @TODO write actual check if the model output is as expected