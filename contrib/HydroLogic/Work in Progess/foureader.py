# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 00:59:40 2021

@author: Krijn Prins Hydrologic
"""
"""============================================================================
Provides
    1. Creates an object to convert variables from the fou netcdf output files
    from the D-Hydro model (...fou.nc) into .tif file format
    Use: foureader(inputdir, outputdir)
        Parameters
        ----------
        inputdir : absolute windows directory path to the ...fou.nc D-Hydro output file
        outputdir  : absolute windows directory path to the location newly
        created tiff files
        ncvar: required variable to get the max value in the fou file from, usually 'Mesh2d_fourierxxx_max_depth'
        coords:
            3 options to find the lower left boundary used for the tif files:
                'uselowerleft' : (standard) uses the list [x,y] values as defined in output_settings
                'innetcdf' : uses the given netcdf to search the lower left corner available
                'windowspath to Shapefile': needs a windowspath to a
                    shapefile of the area to determine the lower left boundary

        Returns
        -------
        foureader object with supported functions to extract variables:
            Tif making functions:
            get_fou_Tiff(): creates a .tif from the extracted
                values of the .fou file in the outputdir directory

==========================================================================="""
# /usr/bin/env python
import numpy as np
from glob import glob
import os, sys
import ugfile as uf
import tiffwriter as tw
import output_settings as c
import pathlib


class foureader(object):
    """Convert FOU file to TIFF file"""

    def __init__(self, ncinput, ncoutput, ncvar, coords="uselowerleft"):
        self.__ncinput = ncinput
        self.__ncoutput = ncoutput
        self.__ncvar = ncvar
        self.__ncband = c.nband
        self.__coords = coords

    def get_fou_data(self, **kwargs):
        fnin = list(pathlib.Path(self.__ncinput).glob("*fou.nc"))
        # ListOfNCfiles = self.__ncinput + os.sep + "*fou.nc"
        # inputfiles = glob(ListOfNCfiles)
        file = fnin[0]
        # for file in inputfiles:
        # filename = os.path.splitext(os.path.basename(file))[0]
        self.ds = uf.DatasetUG(file, "r")

    def get_fou_Tiff(self, **kwargs):
        ListOfNCfiles = self.__ncinput + os.sep + "*fou.nc"
        inputfiles = glob(ListOfNCfiles)

        for file in inputfiles:
            filename = os.path.splitext(os.path.basename(file))[0]
            ds = uf.DatasetUG(file, "r")
            bbox, polygons, coords = ds.get_polygons(self.__ncvar)

            # assign lower left boundary
            lower_left = []
            if self.__coords == "innetcdf":
                if "mesh2d_node_x" in ds.variables:
                    lower_left = [
                        min(ds.variables["mesh2d_node_x"][:].data),
                        min(ds.variables["mesh2d_node_y"][:].data),
                    ]
                    # print ('fouused')
                elif "Mesh2d_node_x" in ds.variables:
                    lower_left = [
                        min(ds.variables["Mesh2d_node_x"][:].data),
                        min(ds.variables["Mesh2d_node_y"][:].data),
                    ]
                    # print ('coordinates in fou used for boundary box')
                else:
                    raise Exception(
                        "lower left not found in fou, use shppath or lower left in output settings"
                    )
                # c.lower_left = foutoLol(shpfile)
            elif self.__coords == "uselowerleft":
                lower_left = c.lower_left
                # print('usinglowerleft')
            elif self.__coords != "innetcdf":
                lower_left = tw.shptoLol(shpfile=self.__coords)
                # print ('usingshapefile')

                # make tif
            bbox[0] = (
                np.floor((bbox[0] - lower_left[0]) / c.xres) * c.xres
                + lower_left[0]
            )
            bbox[1] = (
                np.floor((bbox[1] - lower_left[1]) / c.yres) * c.yres
                + lower_left[1]
            )

            if not self.__ncvar in ds.variables.keys():
                sys.stderr.write("Variable " + self.__ncvar + " not found!\n")
                sys.exit(1)
            my_varobj = ds.variables[self.__ncvar]
            tiffname = filename + "-" + self.__ncvar + ".tif"
            tiffname2 = self.__ncoutput + os.sep + tiffname
            sys.stderr.write("Opening new TIFF: %s\n" % tiffname2)
            my_vardata = ds.variables[self.__ncvar][:]
            my_tiff = tw.tiffwriter(
                tiffname2,
                bbox,
                [c.xres, c.yres],
                self.__ncband,
                c.nptype,
                c.epsg,
                flipped=True,
            )
            pixels = my_tiff.from_polygons(polygons, my_vardata, c.nodata)
            my_tiff.fillband(1, pixels, c.nodata)
            my_tiff.close()
            ds.close()
