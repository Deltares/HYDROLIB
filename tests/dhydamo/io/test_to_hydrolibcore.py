import sys

sys.path.insert(0, r"D:\3640.20\HYDROLIB-dhydamo")
import os

# and from hydrolib-core
from hydrolib.core.io.dimr.models import DIMR, FMComponent
from hydrolib.core.io.inifield.models import IniFieldModel
from hydrolib.core.io.onedfield.models import OneDFieldModel
from hydrolib.core.io.structure.models import *
from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.ext.models import ExtModel
from hydrolib.core.io.mdu.models import FMModel
from hydrolib.core.io.bc.models import ForcingModel
from hydrolib.core.io.friction.models import FrictionModel
from hydrolib.core.io.obs.models import ObservationPointModel

# Importing relevant classes from Hydrolib-dhydamo
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from hydrolib.dhydamo.geometry.viz import plot_network


from tests.dhydamo.io import test_from_hydamo


def setup_model():
    fm = FMModel()
    # Set start and stop time
    fm.time.refdate = 20160601
    fm.time.tstop = 2 * 3600 * 24

    os.chdir(r"D:\3640.20\HYDROLIB-dhydamo\hydrolib\notebooks")
    data_path = Path("../tests/data").resolve()
    assert data_path.exists()
    # path to write the models
    output_path = Path("../tests/model").resolve()
    assert output_path.exists()

    hydamo = test_from_hydamo.test_hydamo_object_from_gpkg()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=None,
    )
    hydamo.crosssections.convert.profiles(
        crosssections=hydamo.profile,
        crosssection_roughness=hydamo.profile_roughness,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
        branches=hydamo.branches,
        roughness_variant="High",
    )

    # Set a default cross section
    default = hydamo.crosssections.add_rectangle_definition(
        height=5.0,
        width=5.0,
        closed=False,
        roughnesstype="StricklerKs",
        roughnessvalue=30,
        name="default",
    )
    hydamo.crosssections.set_default_definition(definition=default, shift=10.0)

    hydamo.external_forcings.set_initial_waterdepth(1.5)

    # main filepath
    fm.filepath = Path(output_path) / "fm" / "test.mdu"
    # we first need to set the forcing model, because it is referred to in the ext model components
    return hydamo, fm


def test_convert_to_hydrolibmodel():

    hydamo, fm = setup_model()

    models = Df2HydrolibModel(hydamo)
    assert len(models.friction_defs) == 3
    assert len(models.crossdefs) == 353


def test_add_to_filestructure():
    hydamo, fm = setup_model()

    models = Df2HydrolibModel(hydamo)
    output_path = Path("../tests/model").resolve()
    assert output_path.exists()

    fm.geometry.structurefile = [StructureModel(structure=models.structures)]
    fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
    fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)

    fm.geometry.frictfile = []
    for i, fric_def in enumerate(models.friction_defs):
        fric_model = FrictionModel(global_=fric_def)
        fric_model.filepath = f"roughness_{i}.ini"
        fm.geometry.frictfile.append(fric_model)

    fm.output.obsfile = [ObservationPointModel(observationpoint=models.obspoints)]

    extmodel = ExtModel()
    extmodel.boundary = models.boundaries_ext
    extmodel.lateral = models.laterals_ext
    fm.external_forcing.extforcefilenew = extmodel

    fm.geometry.inifieldfile = IniFieldModel(initial=models.inifields)

    assert hasattr(fm, "geometry")
    assert hasattr(fm.geometry, "inifieldfile")
    # this does not work yet
    # assert hasattr(onedfieldmodel, "filepath")

    return fm


def test_write_model():

    fm = test_add_to_filestructure()

    output_path = Path("../tests/model").resolve()
    assert output_path.exists()

    dimr = DIMR()
    dimr.component.append(
        FMComponent(
            name="test",
            workingDir=Path(output_path) / "fm",
            model=fm,
            inputfile=fm.filepath,
        )
        #    RRComponent(name="test", workingDir="."", inputfile=fm.filepath, model=rr)
    )
    dimr.save(recurse=True)

    assert (output_path / "fm" / "test.mdu").exists()
    assert (output_path / "fm" / "crsdef.ini").exists()
    assert (output_path / "fm" / "network.nc").exists()
    assert (output_path / "fm" / "initialwaterdepth.ini").exists()
