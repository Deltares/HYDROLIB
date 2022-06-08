import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import pyproj
from shapely.geometry import Polygon


def analyse_bres_run(
    filepath, outpath, bresdebiet_trial, bresdebiet_actual, shpfile, dijkringnummer
):
    """
    PLEASE NOTE: This script works if there is a run used as input with a dambreak.
    This script can be used to analyse a dambreak run in D-Hydro. The script analyses the fou, his, map en dia file.

    Parameters
    ----------
    filepath : string
        Path to dflowfm folder.
    outpath : string
        Output path.
    bresdebiet_trial : TYPE
        DESCRIPTION.
    bresdebiet_actual : TYPE
        DESCRIPTION.
    shpfile : string
        Path to shapefile.
    dijkringnummer : integer
        Number of the dijkring.

    Returns
    -------
    All outputs can be found in output folder.

    """
    mapnc, hisnc, founc, diag, bres_coord_x, bres_coord_y = read_data(filepath)

    plot_his(hisnc, outpath, bresdebiet_trial, bresdebiet_actual)
    read_dia(diag, outpath)
    plot_overstroming(founc, mapnc, shpfile, dijkringnummer, bres_coord_x, bres_coord_y)


def read_data(filepath):
    for root, dirs, files in os.walk(os.path.join(filepath, "output")):
        for file in files:
            if file.endswith("map.nc"):
                mapnc = xr.open_dataset(os.path.join(filepath, "output", file))
                print(file)
            elif file.endswith("his.nc"):
                hisnc = xr.open_dataset(os.path.join(filepath, "output", file))
                print(file)
            elif file.endswith("fou.nc"):
                founc = xr.open_dataset(os.path.join(filepath, "output", file))
                print(file)
            elif file.endswith(".dia"):
                diafile = file

    strucfile = os.path.join(filepath, "structures.ini")

    with open(os.path.join(filepath, "output", diafile)) as d:
        diag = d.readlines()  # regels uit diagfile in lijst van strings

    inProj = pyproj.Proj(init="epsg:28992")
    outProj = pyproj.Proj(init="epsg:3857")

    if bresdebiet_actual or bresdebiet_trial:
        s = open(os.path.join(filepath, strucfile))
        struc = s.readlines()
        s.close()

        for i in range(len(struc)):
            if " " in struc[i]:
                if struc[i].split()[-1] == "dambreak":
                    dam = i
        for i in range(dam - 5, len(struc)):
            if " " in struc[i]:
                if struc[i].split()[0] == "StartLocationX":
                    bres_coord_x1 = struc[i].split()[-1]
                if struc[i].split()[0] == "StartLocationY":
                    bres_coord_y1 = struc[i].split()[-1]
        bres_coord_x, bres_coord_y = pyproj.transform(
            inProj, outProj, bres_coord_x1, bres_coord_y1
        )

    return mapnc, hisnc, founc, diag, bres_coord_x, bres_coord_y


def create_plot_grid(mapnc):
    xn = mapnc.Mesh2d_node_x
    yn = mapnc.Mesh2d_node_y

    link = mapnc.Mesh2d_face_nodes
    # fill all nan values with the last available (for triangles/squares/etc.)
    for i in range(len(link)):
        if np.any(np.isnan(link.data[i])):
            ind = np.where(np.isnan(link[i]))[0]
            last_value_i = ind[0] - 1
            for j in ind:
                link.data[i][j] = link.data[i][last_value_i]
    link_int = link.astype(int)

    x = xn.data[link_int - 1]  # x coordinates of cells/polygons
    y = yn.data[link_int - 1]  # y coordinates of cells/polygons

    # create list of shapely polygons (which can be used in plotting with geopandas)
    poly = np.empty((len(x), len(x[0])), dtype=tuple)
    polygons = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            poly[i][j] = (x[i][j], y[i][j])
        shapelypolygon = Polygon(list(poly[i]))
        polygons.append(shapelypolygon)

    return polygons


def read_dia(diag, outpath):
    # begin en eind van de lijst met info over rekenstappen vinden
    found = False
    for i in range(len(diag)):
        if diag[i] == "** INFO   : Done writing initial output to file(s).\n":
            start = i + 1
            found = True
        elif (diag[i] == "** INFO   : \n") & found:
            end = i
            break
    # de kleinste rekentijdstap uit de laatste kolom van rekenstappen halen
    if not found:
        print("ERROR: diag output for minimal time step not found")
        end = 0  # om de andere informatie wel zonder fout uit te kunnen lezen
    else:
        timesteplist = np.zeros(end - start)
        for i in range(start, end):
            line = diag[i]
            timestep = line.split()[-1]
            if timestep.replace(".", "", 1).isdigit():
                if (
                    float(timestep) < 100
                ):  # om andere regels die ertussen staan weg te filteren
                    timesteplist[i - start] = timestep
        timesteplist2 = timesteplist[timesteplist > 0]
        timestepmin = min(timesteplist2)

    # andere informatie uit de diag file halen (staat allemaal onderaan)
    for i in range(end, len(diag)):
        line = diag[i]
        if "average timestep       (s)" in line:
            timestepav = float(line.split()[-1])
        elif "time modelinit         (s)" in line:
            t_init = float(line.split()[-1])
        elif "total time in timeloop (h)" in line:
            t_total_h = float(line.split()[-1])

    print(
        f"total time [h] = {t_total_h}\n"
        f"average time step [s] = {timestepav}\n"
        f"minimal time step [s] = {timestepmin}\n"
        f"initialisation time [s] = {t_init}\n\n"
        "afgeronde waardes onder elkaar (zie ook output file: diagnostics_summary.txt):\n"
        f"{t_total_h:.2f}\n"
        f"{timestepav:.2f}\n"
        f"{timestepmin:.2f}\n"
        f"{t_init:.2f}"
    )  # nog 2 decimalen en alles onder elkaar printen

    with open(os.path.join(outpath, "diagnostics_summary.txt"), "w") as dia_out:
        dia_out.write(
            f"Total time [h]: {t_total_h:.2f}\nAverage time step [s]: {timestepav:.2f}\nMinimal time step [s]: {timestepmin:.2f}\nInitialisation time [s] {t_init:.2f}"
        )


def plot_his(hisnc, outpath, bresdebiet_trial, bresdebiet_actual):
    if bresdebiet_trial & bresdebiet_actual:
        print(
            "Beide bresdebiet types zijn geselecteerd, kies er eentje en probeer opnieuw, anders wordt de afvoer door de bres niet berekend"
        )
    elif bresdebiet_actual:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.dambreak_discharge)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("debiet [m3/s]")
        ax.set_title("Bresdebiet", fontsize=14)
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        plt.rc("font", size=14)  # fontsize global
        ax.tick_params(axis="x", rotation=30, labelright=True)
        # save
        plt.savefig(os.path.join(outpath, "bresdebiet.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.dambreak_cumulative_discharge)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("cumulatief debiet [m3]")
        ax.set_title("Cumulatief bresdebiet", fontsize=14)
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        plt.rc("font", size=14)  # fontsize global
        ax.tick_params(axis="x", rotation=30, labelright=True)
        # save
        plt.savefig(os.path.join(outpath, "bres_cdis.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            hisnc.time, hisnc.dambreak_crest_level, color="black", label="breshoogte"
        )
        ax.plot(
            hisnc.time,
            hisnc.dambreak_s1dn,
            color="blue",
            label="water level benedenstrooms",
        )
        ax.plot(
            hisnc.time,
            hisnc.dambreak_s1up,
            color="cyan",
            label="water level bovenstrooms",
        )
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("hoogte [m + NAP]")
        ax.set_title("Water level", fontsize=14)
        # ax.set_xlim([14929, 14944])
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        ax.tick_params(axis="x", rotation=30, labelright=True)
        ax.legend(loc="upper right")
        plt.rc("font", size=14)  # fontsize global
        # save
        plt.savefig(os.path.join(outpath, "hoogte.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.dambreak_breach_width_time_derivative)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("bresgroei [m/s]")
        ax.set_title("Bresgroei", fontsize=14)
        # ax.set_xlim([14929, 14944])
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        ax.tick_params(axis="x", rotation=30, labelright=True)
        plt.rc("font", size=14)  # fontsize global
        #    ax.set_ylim([0,max(hisnc.dambreak_breach_width_time_derivative)*1.05])
        # save
        plt.savefig(os.path.join(outpath, "bresgroei.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.dambreak_structure_head)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("verval [m]")
        ax.set_title("Verval", fontsize=14)
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        ax.tick_params(axis="x", rotation=30, labelright=True)
        plt.rc("font", size=14)  # fontsize global
        # save
        plt.savefig(os.path.join(outpath, "bres_verval.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.dambreak_crest_width)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("bresbreedte [m]")
        ax.set_title("Bresbreedte", fontsize=14)
        # ax.set_xlim([14929, 14944])
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 2))
        plt.rc("font", size=14)  # fontsize global
        ax.tick_params(axis="x", rotation=30, labelright=True)
        # save
        plt.savefig(os.path.join(outpath, "bres_breedte.png"), dpi=200)
    elif bresdebiet_trial:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(hisnc.time, hisnc.cross_section_discharge)
        ax.grid()
        ax.set_xlabel("tijd")
        ax.set_ylabel("debiet [m3/s]")
        ax.set_title("Bresdebiet", fontsize=14)
        # ax.set_xlim([14929, 14944])
        # als er meer of minder x-ticks moeten komen, zoekt nu eerste en laatste standaard x_tick en stapgrootte = 1 (dag)
        # ax.set_xticks(np.arange(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 1))
        ax.tick_params(axis="x", rotation=30, labelright=True)
        ax.set_ylim([0, max(hisnc.cross_section_discharge) * 1.05])
        plt.rc("font", size=14)  # fontsize global
        # save
        plt.savefig(os.path.join(outpath, "bresdebiet.png"))
    else:
        print("Geen bresdebiet opgegeven")

    plt.close("all")


def plot_overstroming(
    founc, mapnc, shpfile, dijkringnummer, bres_coord_x, bres_coord_y
):
    max_wd = founc.Mesh2d_fourier001_max_depth

    dr = gpd.read_file(shpfile, crs=28992)
    dr = dr.to_crs(epsg=3857)

    # geom2 = gpd.points_from_xy(max_wd.Mesh2d_face_x, max_wd.Mesh2d_face_y)
    polygons = create_plot_grid(mapnc)
    geom2 = polygons
    max_wd = max_wd.where(max_wd > 0.1)
    dr = dr.where(dr.nummer == dijkringnummer)

    gdf_max_wd = gpd.GeoDataFrame({"waterdiepte": max_wd}, geometry=geom2, crs=28992)
    gdf_max_wd = gdf_max_wd.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = gdf_max_wd.plot(
        column="waterdiepte",
        cmap="Blues",
        markersize=2,
        vmin=0,
        vmax=6,
        legend=True,
        legend_kwds={"label": "Waterdiepte [m]"},
    )
    dr.boundary.plot(ax=ax, color="black")
    if bresdebiet_actual or bresdebiet_trial:
        plt.plot(bres_coord_x, bres_coord_y, "ro", ms=5, label="bres")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Maximale waterdiepte", fontsize=10)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.rc("font", size=7)  # fontsize global
    plt.savefig(os.path.join(outpath, "waterdiepte_max.png"), dpi=300)
    plt.close("all")


if __name__ == "__main__":
    filepath = r"\\c1j6rd93\Arcadis\WRIJ_overstromingsinformatie\DR47_50_51_oplevering\20220511_finalversies\dr50_update\DR50_Eefdensebrug-zuid_T10000\dflowfm"
    outpath = r"C:\scripts\AHT_scriptjes\WRIJ"
    shpfile = r"C:\Users\\delanger3781\ARCADIS\WRIJ - D-HYDRO modellen & scenarioberekeningen - Documents\WRIJ - Gedeelde projectmap\06 Work in Progress\GIS\dijkringen\dijkringen_wrij.shp"
    dijkringnummer = 50
    
    bresdebiet_trial = False
    bresdebiet_actual = True

    analyse_bres_run(
        filepath, outpath, bresdebiet_trial, bresdebiet_actual, shpfile, dijkringnummer
    )
