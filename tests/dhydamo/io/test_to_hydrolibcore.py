import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from hydrolib.core.base.models import DiskOnlyFileModel
from hydrolib.core.dflowfm.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.net.models import NetworkModel
from hydrolib.core.dflowfm.obs.models import ObservationPointModel
from hydrolib.core.dflowfm.structure.models import StructureModel

# and from hydrolib-core
from hydrolib.core.dimr.models import DIMR, FMComponent
from hydrolib.dhydamo.converters.df2hydrolibmodel import (
    EXT_FILE_VERSION,
    Df2HydrolibModel,
)

# Importing relevant classes from Hydrolib-dhydamo
from hydrolib.dhydamo.converters.hydamo2df import RoughnessVariant
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.hydamo import HyDAMO
from hydrolib.dhydamo.geometry import mesh
from shapely.geometry import LineString, Point, Polygon

from tests.dhydamo.io import test_from_hydamo

# path where the input data is located
data_path = Path("hydrolib/sample_data/data").resolve()
assert data_path.exists()

# path to write the models
output_path = (Path(__file__).parent / ".." / ".." / "model").resolve()
output_path.mkdir(parents=False, exist_ok=True)


def setup_model(
    hydamo_obj: HyDAMO | None = None,
    full_test: bool = False,
    model_path: Path | str | None = None,
) -> tuple[HyDAMO, FMModel]:
    # model_path: folder to write the model to, defaults to the global output_path
    model_path = output_path if model_path is None else Path(model_path)
    fm = FMModel()
    assert isinstance(fm.geometry.netfile, NetworkModel)
    # Set start and stop time
    fm.time.refdate = 20160601
    fm.time.tstop = 48 * 3600
    hydamo = hydamo_obj

    if hydamo is None:
        hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()
        hydamo = test_from_hydamo._convert_structures(hydamo_obj=hydamo)

    if full_test:
        hydamo.observationpoints.add_points(
            [Point(199617, 394885), Point(199421, 393769), Point(199398, 393770)],
            ["Obs_BV152054", "ObsS_96684", "ObsO_test"],
            locationTypes=["1d", "1d", "1d"],
            snap_distance=10.0,
        )

        hydamo.observationpoints.add_points(
            [
                Point(200198, 396489),
                Point(201129, 396269),
                Point(199996, 393899),
                Point(199665, 395323),
            ],
            ["ObsS_96544", "ObsP_113GIS", "ObsS_96789", "Obs_ORIF"],
            locationTypes=["1d", "1d", "1d", "1d"],
            snap_distance=10.0,
        )

        hydamo.structures.add_orifice(
            id="orifice_test",
            branchid="W_242213_0",
            chainage=43.0,
            crestlevel=18.00,
            gateloweredgelevel=18.5,
            crestwidth=7.5,
            corrcoeff=1.0,
            uselimitflowpos=False,
            limitflowpos=0.0,
            uselimitflowneg=False,
            limitflowneg=0.0,
        )
    structures = hydamo.structures.as_dataframe(
        rweirs=True,
        bridges=True,
        uweirs=True,
        culverts=True,
        orifices=True,
        pumps=True,
    )
    assert structures is not None
    if full_test:
        objects = pd.concat(
            [structures, hydamo.observationpoints.observation_points], axis=0
        )
    else:
        objects = structures

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=20,
        max_dist_to_struc=None,
        structures=objects,
    )
    # crosssections
    hydamo.crosssections.convert.profiles(
        crosssections=hydamo.profile,
        crosssection_roughness=hydamo.profile_roughness,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
        branches=hydamo.branches,
        roughness_variant=RoughnessVariant.HIGH,
    )
    missing = hydamo.crosssections.get_branches_without_crosssection()
    missing_after_interpolation = mesh.mesh1d_order_numbers_from_attribute(
        hydamo.branches,
        missing,
        order_attribute="naam",
        network=fm.geometry.netfile.network,
        exceptions=["W_1386_0"],
    )

    # Set a default cross section
    profiel = np.array([[0, 21], [2, 19], [7, 19], [9, 21]])
    default = hydamo.crosssections.add_yz_definition(
        yz=profiel,
        thalweg=4.5,
        roughnesstype="StricklerKs",
        roughnessvalue=25.0,
        name="default",
    )

    hydamo.crosssections.set_default_definition(definition=default, shift=0.0)
    hydamo.crosssections.set_default_locations(missing_after_interpolation)

    if full_test:
        # storage node
        hydamo.storagenodes.add_storagenode(
            id="sto_test",
            xy=(141001, 395030),
            name="sto_test",
            usetable="true",
            levels=" ".join(np.arange(17.1, 19.6, 0.1).astype(str)),
            storagearea=" ".join(np.arange(100, 1000, 900 / 25.0).astype(str)),
            interpolate="linear",
            network=fm.geometry.netfile.network,
        )

        # boundary conditions
    hydamo.external_forcings.convert.boundaries(
        hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network
    )
    series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 1.0)
    series.index = [
        pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)
    ]
    hydamo.external_forcings.add_boundary_condition(
        "RVW_01",
        (197464.0, 392130.0),
        "dischargebnd",
        series,
        fm.geometry.netfile.network,
    )
    # initial condition
    hydamo.external_forcings.set_initial_waterdepth(1.5)

    # Add 2d network
    if full_test:
        extent = gpd.read_file(data_path.joinpath("2D_extent.shp")).geometry.iloc[0]
        assert isinstance(extent, Polygon)
        network = fm.geometry.netfile.network
        mesh.mesh2d_add_rectilinear(network, extent, dx=20, dy=20)
        mesh.mesh2d_clip(network, hydamo.branches.geometry["W_2646_0"].buffer(20.0))
        mesh.mesh2d_altitude_from_raster(
            network,
            data_path.joinpath("rasters/AHN_2m_clipped_filled.tif"),
            "face",
            "mean",
            fill_value=-999,
        )
        mesh.links1d2d_add_links_1d_to_2d(network)
        mesh.links1d2d_remove_1d_endpoints(network)

    # main filepath
    fm.filepath = model_path / "dflowfm" / "DFM.mdu"
    # we first need to set the forcing model, because it is referred to in the ext model components
    return hydamo, fm


def test_convert_to_hydrolibmodel() -> None:
    hydamo, _ = setup_model()

    models = Df2HydrolibModel(hydamo)
    assert len(models.friction_defs) == 4
    assert len(models.crossdefs) == 443


def _add_to_filestructure(
    drrmodel: DRRModel | None = None,
    hydamo_obj: HyDAMO | None = None,
    full_test: bool = False,
    model_path: Path | str | None = None,
) -> FMModel:
    model_path = output_path if model_path is None else Path(model_path)
    hydamo, fm = setup_model(
        hydamo_obj=hydamo_obj, full_test=full_test, model_path=model_path
    )

    if drrmodel is not None:
        hydamo.external_forcings.convert.laterals(
            hydamo.laterals, rr_boundaries=drrmodel.external_forcings.boundary_nodes
        )

    models = Df2HydrolibModel(hydamo, assign_default_profiles=True)

    fm.geometry.structurefile = [StructureModel(structure=models.structures)]
    fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
    fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)

    fm.geometry.frictfile = []
    for i, fric_def in enumerate(models.friction_defs):
        fric_model = FrictionModel(global_=fric_def)
        fric_model.filepath = Path(f"roughness_{i}.ini")
        fm.geometry.frictfile.append(fric_model)

    fm.output.obsfile = [ObservationPointModel(observationpoint=models.obspoints)]

    fm.external_forcing.extforcefilenew = models.ext_model()

    # initial fields: writes the data files and links them in the initial field file
    models.write_initial_fields(fm, model_path / "dflowfm")

    return fm


def test_add_to_filestructure() -> None:
    fm = _add_to_filestructure()
    assert hasattr(fm, "geometry")
    assert hasattr(fm.geometry, "inifieldfile")


def _write_model(
    drrmodel: DRRModel | None = None,
    hydamo_obj: HyDAMO | None = None,
    full_test: bool = False,
    model_path: Path | str | None = None,
) -> tuple[FMModel, Path]:
    model_path = output_path if model_path is None else Path(model_path)
    fm = _add_to_filestructure(
        drrmodel=drrmodel,
        hydamo_obj=hydamo_obj,
        full_test=full_test,
        model_path=model_path,
    )

    fm.geometry.uniformwidth1d = 1.0  # default  breedte
    fm.geometry.bedlevtype = 1  # 1: at cell center (tiles xz,yz,bl,bob=max(bl)), 2: at face (tiles xu,yu,blu,bob=blu), 3: at face (using mean node values), 4: at face
    fm.geometry.changestructuredimensions = False  # Change the structure dimensions in case these are inconsistent with the channel dimensions.

    fm.sediment.sedimentmodelnr = 0

    fm.numerics.cflmax = 0.7  # Maximum Courant nr.
    fm.numerics.advectype = (
        33  # Adv type, 0=no, 33=Perot q(uio-u) fast, 3=Perot q(uio-u).
    )

    fm.volumetables.increment = (
        0.2  # parameter setting advised by Deltares for better performance
    )
    fm.volumetables.usevolumetables = (
        True  # parameter setting advised by Deltares for better performance
    )

    fm.restart.restartfile = DiskOnlyFileModel(
        filepath=None
    )  # Restart file, only from netCDF-file, hence: either *_rst.nc or *_map.nc.
    fm.restart.restartdatetime = None  # Restart time [YYYYMMDDHHMMSS], only relevant in case of restart from *_map.nc.
    fm.sediment.sedimentmodelnr = 4
    fm.output.mapformat = (
        4  # parameter setting advised by Deltares for better performance
    )
    fm.output.ncformat = (
        4  # parameter setting advised by Deltares for better performance
    )
    fm.output.ncnoforcedflush = (
        True  # parameter setting advised by Deltares for better performance
    )
    fm.output.ncnounlimited = True
    fm.output.outputdir = Path("output")
    fm.output.statsinterval = [
        1
    ]  # Interval [s] between screen step outputs in seconds simulation time, if negative in seconds wall clock time.
    fm.output.timingsinterval = []  # Timings output interval TimingsInterval.

    if hasattr(fm, "sediment"):
        delattr(fm, "sediment")

    dimr = DIMR()
    dimr.component.append(
        FMComponent(
            name="DFM",
            workingDir=model_path / "dflowfm",
            model=fm,
            inputFile=fm.filepath,
        )
    )
    dimr.save(recurse=True)

    return fm, model_path


def test_write_model() -> None:
    _, output_path = _write_model()
    assert (output_path / "dflowfm" / "DFM.mdu").exists()
    assert (output_path / "dflowfm" / "crsdef.ini").exists()
    assert (output_path / "dflowfm" / "network.nc").exists()
    assert (output_path / "dflowfm" / "InitialWaterDepth.ini").exists()


def test_default_profiles_without_a_default_definition(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # without a default definition no locations are added; they used to get the definition None
    hydamo = HyDAMO()
    branches = gpd.GeoDataFrame(
        {"code": ["b1"]}, geometry=[LineString([(0, 0), (100, 0)])]
    )
    hydamo.branches.set_data(branches, index_col="code")
    hydamo.crosssections.set_default_locations(["b1"])

    with caplog.at_level(logging.WARNING):
        models = Df2HydrolibModel(hydamo, assign_default_profiles=True)

    assert models.crosslocs == []
    assert any(
        "Not all branches have a cross section" in message
        for message in caplog.messages
    )


def test_ext_model_file_version(tmp_path: Path) -> None:
    # D-HYDRO 2026.02 refuses the default version 3.00 of hydrolib-core 1.6 (issue #248)
    models = Df2HydrolibModel(HyDAMO())
    extmodel = models.ext_model()
    extmodel.save(filepath=tmp_path / "bnd.ext")

    lines = (tmp_path / "bnd.ext").read_text().splitlines()
    assert any(
        line.split("#")[0].split() == ["fileVersion", "=", EXT_FILE_VERSION]
        for line in lines
    )


def test_constant_and_qh_boundaries_write_time_interpolation(tmp_path: Path) -> None:
    # the D-HYDRO GUI (2026.02) requires timeInterpolation in every [Forcing] block, also for
    # constant values and QH tables, where hydrolib-core does not write it
    fm = FMModel()
    assert isinstance(fm.geometry.netfile, NetworkModel)
    network = fm.geometry.netfile.network
    mesh.mesh1d_add_branch_from_linestring(
        network, LineString([(0.0, 0.0), (100.0, 0.0)]), node_distance=100
    )

    hydamo = HyDAMO()
    forcings = hydamo.external_forcings
    forcings.add_boundary_condition(
        "constant", (0.0, 0.0), "waterlevelbnd", 1.5, network
    )
    forcings.add_boundary_condition(
        "qh", (100.0, 0.0), "qhbnd", {"Q": [0.0, 1.0], "H": [1.0, 2.0]}, network
    )
    models = Df2HydrolibModel(hydamo)
    models.forcingmodel.save(tmp_path / "boundaryconditions.bc")

    blocks = (tmp_path / "boundaryconditions.bc").read_text().split("[Forcing]")[1:]
    functions = [re.findall(r"function\s*=\s*(\w+)", block) for block in blocks]
    assert functions == [["constant"], ["qhtable"]]
    assert all(re.search(r"timeInterpolation\s*=\s*linear", block) for block in blocks)
