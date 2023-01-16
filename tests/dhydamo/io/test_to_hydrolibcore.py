import sys

sys.path.insert(0, r".")
import os

# and from hydrolib-core
from hydrolib.core.dimr.models import DIMR, FMComponent
from hydrolib.core.dflowfm.inifield.models import IniFieldModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel
from hydrolib.core.dflowfm.structure.models import *
from hydrolib.core.dflowfm.crosssection.models import *
from hydrolib.core.dflowfm.ext.models import ExtModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.bc.models import ForcingModel
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.obs.models import ObservationPointModel

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

    data_path = Path("hydrolib/tests/data").resolve()
    assert data_path.exists()
    # path to write the models
    output_path = Path("hydrolib/tests/model").resolve()
    assert output_path.exists()

    hydamo = test_from_hydamo.test_hydamo_object_from_gpkg()

    hydamo.structures.convert.weirs(
        hydamo.weirs,
        hydamo.profile_group,
        hydamo.profile_line,
        hydamo.profile,
        hydamo.opening,
        hydamo.management_device,
    )

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
    from shapely.geometry import Point

    hydamo.observationpoints.add_points(
        [Point((200200, 395600)), (200200, 396200)],
        ["ObsPt1", "ObsPt2"],
        locationTypes=["1d", "1d"],
        snap_distance=10.0,
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
    output_path = Path("hydrolib/tests/model").resolve()
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

    output_path = Path("hydrolib/tests/model").resolve()
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
