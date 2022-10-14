import geopandas as gpd
import pandas as pd
import pkg_resources
from geovoronoi import points_to_coords, voronoi_regions_from_coords
from shapely.geometry import Point


def voronoi_extra(gdf_points, gdf_areas, group_id=""):

    """
      Create voronois based on areas and points based on geovoronoi.
      Circumvents errors of geovoronoi and also preserves data.

      Parameters:
           gdf_points: GeoDataFrame
              Input points
           gdf_areas: GeoDataFrame
              Input areas
           group_id: str
               Column id for witch the points need to be dissolved
    ___________________________________________________________________________________________________________
      Warning:
           Major changes in geovoronoi since 0.2.0
           Areas without point will not return an area
    """

    # for later on, dissolve areas into 1 are, we can use this later on if the voronoi script fails
    gdf_areas["dummy"] = 1
    df_totarea = gdf_areas.dissolve(by="dummy")
    tot_geom = df_totarea.iloc[0].geometry

    # create a unique index number
    gdf_points = gdf_points[
        ~gdf_points.index.duplicated(keep="first")
    ]  # verwijder dubbele punten in Sobek
    gdf_points["indx"] = gdf_points.index
    areas_missing = 0
    areas_error = 0
    points_used = pd.DataFrame()

    # loop over each area and make voronois
    voronois = []
    for index, row in gdf_areas.iterrows():

        # setting up of data
        area_geom = row.loc["geometry"]
        if area_geom.is_valid == False:  # correct for invalid shapes
            area_geom = area_geom.buffer(0)

        # select the points that are within the area
        gdf_points_clip = gpd.sjoin(
            gdf_points,
            gpd.GeoDataFrame(geometry=[area_geom], crs=gdf_areas.crs),
            predicate="within",
        )

        # creating voronoi, based on number of points (minimal 4 points needed)
        if len(gdf_points_clip) == 0:  # no points, skip
            areas_missing = areas_missing + 1
            continue
        elif len(gdf_points_clip) == 1:  # one point, voronoi equals area
            voronoi = gdf_points_clip
            voronoi.geometry = [row.geometry]
            points_used = points_used.append(gdf_points_clip)
        else:
            if len(gdf_points_clip) < 4:  # not enough points
                # copy points with small offset so we have at least 3 points
                gdf_points_dummy = gdf_points_clip.copy()
                gdf_points_dummy["geometry"] = gdf_points_dummy.translate(
                    xoff=0.001, yoff=0.001
                )
                gdf_points_extra_clip = gpd.GeoDataFrame(
                    pd.concat([gdf_points_clip, gdf_points_dummy], ignore_index=True),
                    crs=gdf_points.crs,
                )
                gdf_points_extra_clip = gpd.clip(gdf_points_extra_clip, area_geom)
            else:
                gdf_points_extra_clip = gdf_points_clip.copy()
            points_used = points_used.append(gdf_points_extra_clip)

            if pkg_resources.get_distribution("geovoronoi").version < "0.4.0":
                # add extra points at outer regions, this prevens certain errors
                xmin, ymin, xmax, ymax = area_geom.buffer(
                    area_geom.area ** 0.5 * 10
                ).envelope.bounds
                point1 = gdf_points_extra_clip.iloc[[0]].copy()
                point1["geometry"] = Point(xmax, ymax)
                point2 = gdf_points_extra_clip.iloc[[0]].copy()
                point2["geometry"] = Point(xmin, ymax)
                point3 = gdf_points_extra_clip.iloc[[0]].copy()
                point3["geometry"] = Point(xmin, ymin)
                point4 = gdf_points_extra_clip.iloc[[0]].copy()
                point4["geometry"] = Point(xmax, ymin)
                gdf_points_extra_clip = gdf_points_extra_clip.append(
                    [point1, point2, point3, point4]
                )

            coords = points_to_coords(gdf_points_extra_clip.geometry)
            try:
                if pkg_resources.get_distribution("geovoronoi").version >= "0.4.0":
                    vr_area, vr_points = voronoi_regions_from_coords(coords, area_geom)
                else:
                    vr_area, vr_points, vr_links = voronoi_regions_from_coords(
                        coords, area_geom
                    )
            except:
                # if voronoi fails, try by making bigger voronois and clip to area
                try:
                    if pkg_resources.get_distribution("geovoronoi").version >= "0.4.0":
                        vr_area, vr_points = voronoi_regions_from_coords(
                            coords, tot_geom
                        )
                    else:
                        vr_area, vr_points, vr_links = voronoi_regions_from_coords(
                            coords, tot_geom
                        )
                except:
                    print("error at area with index ", str(index))
                    areas_error = areas_error + 1
                    continue

            if pkg_resources.get_distribution("geovoronoi").version >= "0.4.0":
                # voronoi = gpd.GeoDataFrame(pd.concat([gdf_points_extra_clip.iloc[value] for key, value in vr_points.items()]))
                # voronoi.geometry = [value for key, value in vr_area.items()]
                voronoi = gpd.GeoDataFrame(pd.concat(
                    [
                        gdf_points_extra_clip.iloc[value]
                        for key, value in vr_points.items()
                    ]),
                    geometry=[value for key, value in vr_area.items()],
                    crs=gdf_points.crs,
                )
            else:
                voronoi = gpd.GeoDataFrame(
                    [gdf_points_extra_clip.iloc[id[0]] for id in vr_links],
                    geometry=vr_area,
                    crs=gdf_points.crs,
                )

            voronoi = gpd.clip(voronoi, area_geom)

        # adding the voronoi to the previous ones
        voronois.append(voronoi)

    # postprocessing of results
    gdf_result = gpd.GeoDataFrame(pd.concat(voronois), crs=gdf_points.crs)
    gdf_result = gdf_result.dissolve(
        by="indx", aggfunc="first"
    )  # dissolve to prevent dupplicates due to not enough points
    if group_id in gdf_result.columns:
        gdf_result.dissolve(by=group_id, aggfunc="first")
    if "index_right" in gdf_result.columns:
        gdf_result = gdf_result.drop(["index_right"], axis=1)
    if "indx" in gdf_result.columns:
        gdf_result = gdf_result.drop(["indx"], axis=1)

    # print number of failed areas
    print(str(len(gdf_areas)) + " gebieden verwerkt")
    print(str(areas_missing) + " gebieden zonder punten.")
    print(str(areas_error) + " met errors.")

    return gdf_result
