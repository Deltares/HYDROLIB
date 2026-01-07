import sys

sys.path.insert(0, r".")
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from hydrolib.core.dflowfm.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.dflowfm.ext.models import ExtModel
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.inifield.models import DiskOnlyFileModel, IniFieldModel
from hydrolib.core.dflowfm.mdu.models import FMModel
from hydrolib.core.dflowfm.obs.models import ObservationPointModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel
from hydrolib.core.dflowfm.structure.models import StructureModel

# and from hydrolib-core
from hydrolib.core.dimr.models import DIMR, FMComponent

# Importing relevant classes from Hydrolib-dhydamo
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.geometry import mesh
from tests.dhydamo.io import test_from_hydamo

# path where the input data is located
data_path = Path("hydrolib/tests/data").resolve()
assert data_path.exists()
# path to write the models
output_path = Path("hydrolib/tests/model").resolve()
output_path.mkdir(parents=True, exist_ok=True)
assert output_path.exists()


def setup_model(hydamo=None, full_test=False):
    fm = FMModel()
    # Set start and stop time
    fm.time.refdate = 20160601
    fm.time.tstop = 2 * 3600 * 24

    if hydamo is None:
        hydamo, _ = test_from_hydamo._hydamo_object_from_gpkg()
        hydamo = test_from_hydamo._convert_structures(hydamo=hydamo)

    structures = None
    if full_test:
        hydamo.observationpoints.add_points(
            [Point(199617,394885), Point(199421,393769), Point(199398,393770)],
            ["Obs_BV152054", "ObsS_96684","ObsO_test"],
            locationTypes=["1d", "1d", "1d"],
            snap_distance=10.0,
            )

        hydamo.observationpoints.add_points(
            [Point(200198,396489), Point(201129, 396269), Point(200264, 394761), Point(199665, 395323)],
            ["ObsS_96544", "ObsP_113GIS", 'Obs_UWR', 'Obs_ORIF'],
            locationTypes=["1d", "1d","1d", "1d"],
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
        )
        structures = hydamo.structures.as_dataframe(
            rweirs=True,
            bridges=True,
            uweirs=True,
            culverts=True,
            orifices=True,
            pumps=True,
        )
        objects = pd.concat([structures, hydamo.observationpoints.observation_points], axis=0)    
        
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
            roughness_variant="High",
        )
        missing = hydamo.crosssections.get_branches_without_crosssection()
        missing_after_interpolation = mesh.mesh1d_order_numbers_from_attribute(hydamo.branches, 
                missing, 
                order_attribute='naam', 
                network=fm.geometry.netfile.network,  
                exceptions=['W_1386_0'])

        # Set a default cross section
        profiel=np.array([[0,21],[2,19],[7,19],[9,21]])
        default = hydamo.crosssections.add_yz_definition(yz=profiel, 
                                                        thalweg = 4.5,
                                                        roughnesstype='StricklerKs',
                                                        roughnessvalue=25.0, 
                                                        name='default'
                                                        )

        hydamo.crosssections.set_default_definition(definition=default, shift=0.0)
        hydamo.crosssections.set_default_locations(missing_after_interpolation)   
    
        # storage node
        hydamo.storagenodes.add_storagenode(       
            id='sto_test',
            xy=(141001, 395030),                
            name='sto_test',
            usetable="true",
            levels=' '.join(np.arange(17.1, 19.6, 0.1).astype(str)),
            storagearea=' '.join(np.arange(100, 1000, 900/25.).astype(str)),
            interpolate="linear",
            network=fm.geometry.netfile.network
            )
        
        # boundary conditions
        hydamo.external_forcings.convert.boundaries(hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network)
        series = pd.Series(np.sin(np.linspace(2, 8, 120) * -1) + 1.0)
        series.index = [pd.Timestamp("2016-06-01 00:00:00") + pd.Timedelta(hours=i) for i in range(120)]
        hydamo.external_forcings.add_boundary_condition(
            "RVW_01", (197464.0, 392130.0), "dischargebnd", series, fm.geometry.netfile.network
        )
        # initial condition
        hydamo.external_forcings.set_initial_waterdepth(1.5)

        # Add 2d network
        extent = gpd.read_file(data_path.joinpath("2D_extent.shp")).at[0, "geometry"]
        network = fm.geometry.netfile.network
        mesh.mesh2d_add_rectilinear(network, extent, dx=20, dy=20)
        mesh.mesh2d_clip(network, hydamo.branches.loc['W_2646_0'].geometry.buffer(20.))
        mesh.mesh2d_altitude_from_raster(network, data_path.joinpath("rasters/AHN_2m_clipped_filled.tif"), "face", "mean", fill_value=-999)
        mesh.links1d2d_add_links_1d_to_2d(network)
        mesh.links1d2d_remove_1d_endpoints(network)

    # main filepath
    fm.filepath = Path(output_path) / "fm" / "test.mdu"
    # we first need to set the forcing model, because it is referred to in the ext model components
    return hydamo, fm


def test_convert_to_hydrolibmodel():
    hydamo, _ = setup_model()

    models = Df2HydrolibModel(hydamo)
    assert len(models.friction_defs) == 3
    assert len(models.crossdefs) == 353


def _add_to_filestructure(drrmodel=None, hydamo=None, full_test=False):
    hydamo, fm = setup_model(hydamo=hydamo, full_test=full_test)

    if drrmodel is not None:
        hydamo.external_forcings.convert.laterals(
            hydamo.laterals,                        
            rr_boundaries=drrmodel.external_forcings.boundary_nodes
        )


    if full_test:
        models = Df2HydrolibModel(hydamo, assign_default_profiles=True)
    else:
        models = Df2HydrolibModel(hydamo)

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

    for ifield, onedfield in enumerate(models.onedfieldmodels):
        # eventually this is the way, but it has not been implemented yet in Hydrolib core
        # fm.geometry.inifieldfile.initial[ifield].datafile = OneDFieldModel(global_=onedfield)

        # this is a workaround to do the same
        onedfield_filepath = output_path / "fm" / "initialwaterdepth.ini"
        onedfieldmodel = OneDFieldModel(global_=onedfield)
        onedfieldmodel.save(filepath=onedfield_filepath)
        fm.geometry.inifieldfile.initial[ifield].datafile = DiskOnlyFileModel(
            filepath=onedfield_filepath
        )

    return fm

def test_add_to_filestructure(drrmodel=None, hydamo=None, full_test=False):
    fm = _add_to_filestructure(drrmodel=drrmodel, hydamo=hydamo, full_test=full_test)
    assert hasattr(fm, "geometry")
    assert hasattr(fm.geometry, "inifieldfile")
    # this does not work yet
    # assert hasattr(onedfieldmodel, "filepath")


def _write_model(drrmodel=None, hydamo=None, full_test=False):
    fm = _add_to_filestructure(drrmodel=drrmodel, hydamo=hydamo, full_test=full_test)

    dimr = DIMR()
    dimr.component.append(
        FMComponent(
            name="DFM",
            workingDir=Path(output_path) / "fm",
            model=fm,
            inputfile=fm.filepath,
        )
        #    RRComponent(name="test", workingDir="."", inputfile=fm.filepath, model=rr)
    )
    dimr.save(recurse=True)

    return fm, output_path

def test_write_model(drrmodel=None, hydamo=None, full_test=False):
    _, output_path = _write_model(drrmodel=drrmodel, hydamo=hydamo, full_test=full_test)
    assert (output_path / "fm" / "test.mdu").exists()
    assert (output_path / "fm" / "crsdef.ini").exists()
    assert (output_path / "fm" / "network.nc").exists()
    assert (output_path / "fm" / "initialwaterdepth.ini").exists()
