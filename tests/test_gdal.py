from osgeo import gdal


def test_gdal_zonal_stats_is_available():
    assert gdal.VersionInfo("VERSION_NUM") >= "3012000"
    assert gdal.Algorithm("raster", "zonal-stats").GetName() == "zonal-stats"
