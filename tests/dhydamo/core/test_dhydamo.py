from hydrolib.dhydamo.core.hydamo import HyDAMO
from pathlib import Path

hydamo_data_path = Path(__file__).parent / ".." / ".." / ".." / "hydrolib" / "tests" / "data"


def test_create_hydamo_object():

    # initialize a hydamo object
    extent_file = hydamo_data_path / "OLO_stroomgebied_incl.maas.shp"
    assert extent_file.exists()
    hydamo = HyDAMO(extent_file=extent_file)

    assert hydamo.clipgeo.area == 139373665.08206934

    # all data is contained in one geopackage called 'Example model'
    gpkg_file = hydamo_data_path / "Example_model.gpkg"
    assert gpkg_file.exists()

    # Read branches
    hydamo.branches.read_gpkg_layer(str(gpkg_file), layer_name="HydroObject", index_col="code")
    assert len(hydamo.branches) == 61
    assert hydamo.branches.length.sum() == 28371.461117125935

    # read profiles
    hydamo.profile.read_gpkg_layer(
        gpkg_file,
        layer_name="ProfielPunt",
        groupby_column="profiellijnid",
        order_column="codevolgnummer",
        id_col="code",
    )
    # TODO: Add tests

    # read roughness
    hydamo.profile_roughness.read_gpkg_layer(gpkg_file, layer_name="RuwheidProfiel")
    hydamo.profile.snap_to_branch(hydamo.branches, snap_method="intersecting")
    hydamo.profile.dropna(axis=0, inplace=True, subset=["branch_offset"])
    hydamo.profile_line.read_gpkg_layer(gpkg_file, layer_name="profiellijn")
    hydamo.profile_group.read_gpkg_layer(gpkg_file, layer_name="profielgroep")
    hydamo.profile.drop("code", axis=1, inplace=True)
    hydamo.profile["code"] = hydamo.profile["profiellijnid"]
    # TODO: Add tests

    # Read Weirs
    hydamo.weirs.read_gpkg_layer(gpkg_file, layer_name="Stuw")
    hydamo.weirs.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
    hydamo.weirs.dropna(axis=0, inplace=True, subset=["branch_offset"])
    hydamo.opening.read_gpkg_layer(gpkg_file, layer_name="Kunstwerkopening")
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
    # TODO: Add tests

    # Read culverts
    hydamo.culverts.read_gpkg_layer(gpkg_file, layer_name="DuikerSifonHevel", index_col="code")
    hydamo.culverts.snap_to_branch(hydamo.branches, snap_method="ends", maxdist=5)
    hydamo.culverts.dropna(axis=0, inplace=True, subset=["branch_offset"])
    assert len(hydamo.culverts) == 90
    assert hydamo.culverts.length.sum() == 2497.687230867272

    # Read management device
    hydamo.management_device.read_gpkg_layer(gpkg_file, layer_name="Regelmiddel")
    idx = hydamo.management_device.loc[hydamo.management_device["duikersifonhevelid"].notnull()].index
    for i in idx:
        globid = hydamo.culverts.loc[
            hydamo.culverts["code"].eq(hydamo.management_device.at[i, "duikersifonhevelid"]), "globalid"
        ].values[0]
        hydamo.management_device.at[i, "duikersifonhevelid"] = globid
    assert len(hydamo.management_device) == 27

    # Read pumpstations
    hydamo.pumpstations.read_gpkg_layer(gpkg_file, layer_name="Gemaal", index_col="code")
    hydamo.pumpstations.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
    hydamo.pumps.read_gpkg_layer(gpkg_file, layer_name="Pomp", index_col="code")
    hydamo.management.read_gpkg_layer(gpkg_file, layer_name="Sturing", index_col="code")
    # TODO: Add tests

    # Read bridges
    hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="Brug", index_col="code")
    hydamo.bridges.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
    hydamo.bridges.dropna(axis=0, inplace=True, subset=["branch_offset"])
    # TODO: Add tests

    # Read boundary conditions
    hydamo.boundary_conditions.read_gpkg_layer(
        gpkg_file, layer_name="hydrologischerandvoorwaarde", index_col="code"
    )
    hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=10)
    # TODO: Add tests

    # Read catchments
    hydamo.catchments.read_gpkg_layer(gpkg_file, layer_name="afvoergebiedaanvoergebied", index_col="code")
    # TODO: Add tests

    # Read laterals
    hydamo.laterals.read_gpkg_layer(gpkg_file, layer_name="lateraleknoop")
    for ind, cat in hydamo.catchments.iterrows():
        hydamo.catchments.loc[ind, "lateraleknoopcode"] = hydamo.laterals[
            hydamo.laterals.globalid == cat.lateraleknoopid
        ].code.values[0]
    hydamo.laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5000)
    # TODO: Add tests

    # Convert
    hydamo.structures.convert.weirs(
        hydamo.weirs,
        hydamo.profile_group,
        hydamo.profile_line,
        hydamo.profile,
        hydamo.opening,
        hydamo.management_device,
    )
    hydamo.structures.convert.culverts(hydamo.culverts, management_device=hydamo.management_device)
    hydamo.structures.convert.bridges(
        hydamo.bridges,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
    )
    hydamo.structures.convert.pumps(hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management)
    # TODO: Add tests

    # Add structures manually
    hydamo.structures.add_rweir(
        id="rwtest",
        name="rwtest",
        branchid="W_1386_0",
        chainage=2.0,
        crestlevel=18.0,
        crestwidth=3.0,
        corrcoeff=1.0,
    )
    hydamo.structures.add_orifice(
        id="otest",
        name="otest",
        branchid="W_1386_0",
        chainage=5.0,
        crestlevel=18.0,
        crestwidth=3.0,
        corrcoeff=1.0,
        gateloweredgelevel=17.0,
        uselimitflowpos=False,
        limitflowpos=0.0,
        uselimitflowneg=False,
        limitflowneg=0.0,
    )
    hydamo.structures.add_uweir(
        id="uwtest",
        name="uwtest",
        branchid="W_1386_0",
        chainage=6.0,
        dischargecoeff=0.9,
        crestlevel=18.0,
        numlevels=3,
        yvalues="0 0.5 1.0",
        zvalues="19.0 18.0 19.0",
    )
    hydamo.structures.add_bridge(
        id="btest",
        name="btest",
        branchid="W_1386_0",
        chainage=7.0,
        length=8.0,
        csdefid="rect_3.60",
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        shift=0.0,
        frictiontype="Manning",
        friction=0.06,
    )
    hydamo.structures.add_culvert(
        id="ctest",
        name="ctest",
        branchid="W_1386_0",
        chainage=8.0,
        leftlevel=18.0,
        rightlevel=17.0,
        length=30.0,
        crosssection={"shape": "circle", "diameter": 0.40},
        inletlosscoeff=0.9,
        outletlosscoeff=0.9,
        frictiontype="Manning",
        frictionvalue=0.06,
    )
    hydamo.structures.add_pump(
        id="ptest",
        name="ptest",
        branchid="W_1386_0",
        chainage=9.0,
        capacity=1.0,
        startlevelsuctionside=[14.0],
        stoplevelsuctionside=[13.8],
    )

    return hydamo
