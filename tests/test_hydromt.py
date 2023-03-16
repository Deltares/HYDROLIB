"""Test for hydromt plugin model class DFlowFMModel"""

import pdb
from os.path import abspath, dirname, join

import numpy as np
import pytest
from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog

from hydrolib.hydromt_delft3dfm import DFlowFMModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples", "hydromt")

_models = {
    "piave": {
        "example": "dflowfm_piave",
        "ini": "dflowfm_build.ini",
        "model": DFlowFMModel,
        "data": "artifact_data",
        "region": {"bbox": [12.4331, 46.4661, 12.5212, 46.5369]},
        "snap_offset": 40,
    },
    "local": {
        "example": "dflowfm_local",
        "ini": "dflowfm_build_local.ini",
        "model": DFlowFMModel,
        "data": join(TESTDATADIR, "test_data.yaml"),
        "region": {"geom": f"{join(TESTDATADIR, 'local_data','1D_extent.geojson')}"},
        "snap_offset": 25,
    },
}


@pytest.mark.parametrize("model", list(_models.keys()))
def test_model_class(model):
    _model = _models[model]
    # read model in examples folder
    root = join(EXAMPLEDIR, _model["example"])
    mod = _model["model"](root=root, mode="r")
    mod.read()
    # run test_model_api() method
    non_compliant_list = mod._test_model_api()
    assert len(non_compliant_list) == 0


@pytest.mark.timeout(300)  # max 5 min
@pytest.mark.parametrize("model", list(_models.keys()))
def test_model_build(tmpdir, model):
    _model = _models[model]
    # test build method
    # compare results with model from examples folder
    root = str(tmpdir.join(model))
    logger = setuplog(__name__, join(root, "hydromt.log"), log_level=10)
    mod1 = _model["model"](
        root=root,
        mode="w",
        data_libs=[_model["data"]],
        network_snap_offset=_model["snap_offset"],
        openwater_computation_node_distance=40,
        logger=logger,
    )
    # Build method options
    region = _model["region"]
    config = join(TESTDATADIR, _model["ini"])
    opt = parse_config(config)
    # Build model
    mod1.build(region=region, opt=opt)
    # Check if model is api compliant
    non_compliant_list = mod1._test_model_api()
    assert len(non_compliant_list) == 0

    # Compare with model from examples folder
    # (need to read it again for proper geoms check)
    mod1 = _model["model"](root=root, mode="r", logger=logger)
    mod1.read()
    root = join(EXAMPLEDIR, _model["example"])
    mod0 = _model["model"](root=root, mode="r")
    mod0.read()
    # check if equal
    equal, errors = mod0._test_equal(mod1)
    # skip config.filepath (root is different)
    if "config.filepath" in errors:
        errors.pop("config.filepath", None)
        if len(errors) == 0:
            equal = True
    assert equal, errors
