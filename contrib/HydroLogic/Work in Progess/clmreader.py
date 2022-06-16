# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:41:13 2021

@author: Krijn Prins
"""
"""============================================================================
Provides
    1. Creates an object to convert variables from clm netcdf output files
    from the D-Hydro model (...clm.nc) into .tif file format
    Use: clmreader(inputdir, outputdir, ncvar)
        Parameters: required
        ----------
            inputdir : absolute windows directory path to the ...clm.nc D-Hydro output file
            outputdir  : absolute windows directory path to the location newly 
            created .tif files
            ncvar: the variable in the clm.nc file a class tif is made of
            (usually 'mesh2d_waterdepth' or 'Mesh2d_waterdepth')
        
        Parameters: optional
        ----------
            coords: 
                3 options to find the lower left boundary used for the tif files:
                    'uselowerleft' : (standard) uses the list [x,y] values as defined in output_settings
                    'innetcdf' : uses the given netcdf to search the lower left corner available
                    'windowspath to Shapefile': needs a windowspath to a 
                    shapefile of the area to determine the lower left boundary 
            lower: lower boundary to calculate arrivaltimes and risingspeeds (m) 
            determines whenever a cell is considered wet (standard 0.02m)
            upper: upper boundary to calculate risingspeeds (m) determines together 
            with the lower boundary the height the range where the risiningspeed is calculated
            (standard 1.5m)
            modelname: modelname to add to the name of the produced tiff file
        
        Parameters: external in the output_settings.py script
            lower_left : list of coördinates [x,y] is only used when coords = 'uselowerleft'
            xres : x resolution pixel size (m)
            yres : y resolution pixel size (m)
            epsg : coördinate system 
            nptype : numpy floats type
            nodata : set the no-data value for the tiff
            timeunit : used timeunit in the tiff
            time_format : used timeformat '%Y-%m-%d %H:%M:%S'
            nband : band of the tif file

        Returns
        -------
        clmreader object with supported functions to extract variables:
            Calculating functions:
                getArrivalTimes() : Calculate the arrivaltimes for cells considered wet
                defined by a waterdepth higher than the 'lower' parameter
                returns a list of the arrival times
                getRisingSpeeds() : Calculate the rising speeds for cells using 3 methods:
                    method = 'fixed_boundaries':
                        calculates the rising speeds by 
                        deviding the difference between the defined upper and
                        lower boundaries by the time it takes to get from this lower to this upper boundary
                        returns 
                    method = 'lower_class' :
                        calculates the rising speeds by deviding
                        the difference of the lower class and the lower class + 1class
                        by the time it takes to get from this lower to this upper class
                    method = 'all_classes':
                        calculates the rising speeds by 
                        calculating the rising speeds for each class seperatly 
                        then taking the max values for each cell
                    returns a list containing the max rising and the height of the class having that rising speed
                   

            Tif making functions:
                note that the tif making functions also calculate first
                makearrivaltif(): runs getArrivaltimes() and produces
                    a tif file in the outputdir directory
                makerisingtif(method='method') : runs getRisingSpeeds() with the specified method
                    produces a tif in the outputdir directory
                makegevaarhoogtetif(): runs getRisingSpeeds('all_classes') and 
                    produces a tif file in the outputdir directory

==========================================================================="""
#import netCDF4 as nc
import numpy as np
import output_settings as c
import re
import pathlib
import sys
import ugfile as uf
import tiffwriter as tw


timefactor = {}
timefactor['seconds']  = 1.0
timefactor['minutes'] = 60.0
timefactor['hours'] = 3600.0
timefactor['days'] = 86400.0

class clmreader(object):
    """Construct the triplet of grid files from an incremental file.""" 

    def __init__(self, ncinput, ncoutput, ncvar, lower = 0.02, upper = 1.5, modelname = False, coords='uselowerleft'): 
        self.__ncinput = ncinput
        self.__ncoutput = ncoutput
        self.__ncvarname = ncvar
        self.__classes = {} 
        self.__lower_threshold = None 
        self.__upper_threshold = None
        self.__lower_definition = lower 
        self.__upper_definition = upper
        self.__modelname = modelname
        self.__configsettings = c
        self.__coords = coords
        self.__read_ncvar()
        self.__parse_classes()

    def __read_ncvar(self):
        fnin = list(pathlib.Path(self.__ncinput).glob('*clm.nc'))
        self.__ds = uf.DatasetUG(fnin[0],'r')
        self.__ncvar = self.__ds.variables[self.__ncvarname]
        self.__nctim = self.__ds.variables["time"]
        self.__lower_left = self.__fill_lower_left()
        
    def __fill_lower_left(self):
        self.__lower_left = []
        if self.__coords == 'innetcdf':
            if 'mesh2d_node_x' in self.__ds.variables:
                self.__lower_left = [min(self.__ds.variables['mesh2d_node_x'][:].data),
                min(self.__ds.variables['mesh2d_node_y'][:].data)]
            elif 'Mesh2d_node_x' in self.__ds.variables:
                self.__lower_left = [min(self.__ds.variables['Mesh2d_node_x'][:].data),
                min(self.__ds.variables['Mesh2d_node_y'][:].data)]
            else:
                raise Exception ('lower left not found in fou, use shppath or lower left in output settings')                   
        elif self.__coords == 'uselowerleft':
            self.__lower_left = c.lower_left
        elif self.__coords != 'innetcdf':
            self.__lower_left = tw.shptoLol(shpfile = self.__coords)
                 
    def __fill_classes(self):
          
        units = self.__ncvar.getncattr('units')
        regexp_midcls   = '([-\d\.]+)'+units+'_to_([-\d\.]+)'+units  # intermediate class search regular expression
        regexp_topcls   = 'above_([-\d\.]+)'+units                   # top class search regular expression
        regexp_btmcls   = 'below_([-\d\.]+)'+units                   # bottom class search regular expression
        
        clsvalues = self.__ncvar.getncattr('flag_values')
        
        clsdefstr = self.__ncvar.getncattr('flag_meanings')
        clsdefs   = clsdefstr.split()
        
        ncls = len(clsvalues)
        # Create translation table for the classes, stores the LOWER bound of the class !!
        for icls in range(ncls):
            m = re.search(regexp_btmcls,clsdefs[icls])
            if m:
                cls = np.nan
            m = re.search(regexp_topcls,clsdefs[icls])
            if m:
                cls = float(m.group(1))
            m = re.search(regexp_midcls,clsdefs[icls])
            if m:
                cls = float(m.group(1))
            self.__classes[int(clsvalues[icls])] = cls

    def __parse_classes(self):
        # Find the number of the first class >= 0.02 
        self.__fill_classes()
        for class_number in sorted(self.__classes.keys()):
            if self.__classes[class_number] >= self.__lower_definition: 
                self.__lower_threshold = class_number 
                break 
        if self.__lower_threshold is None: 
            raise RuntimeError(f'No class exceeds {self.__lower_definition} m.')
        # Find the number of the first class >= 1.5 
        for class_number in sorted(self.__classes.keys()): 
            if self.__classes[class_number] >= self.__upper_definition: 
                self.__upper_threshold = class_number 
                break 
        if self.__upper_threshold is None: 
            raise RuntimeError(f'No class exceeds {self.__upper_definition} m.')
    
    def getArrivalTimes(self,**kwargs): 
        # Inspect the data to construct the rise rate grid.
        global timefactor
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        npoly = self.__ncvar.shape[1]
        arrival_times = np.ma.zeros(npoly)
        arrival_times.mask = True
        times  = self.__nctim[:]
        timeunit = self.__nctim.getncattr('units').split()[0]
        tfact = timefactor[timeunit]/timefactor[c.timeunit] 

        for ipoly in range(npoly):
            tseries= self.__ncvar[:,ipoly]
            lts = None
            for time, class_ in list(zip(times,tseries)):
                if class_ >= self.__lower_threshold:
                    if lts is None:
                        lts = (time, class_)
                        arrival_times[ipoly] = time * tfact
                        break

            if (verbose):
                sys.stderr.write("Calculating arrival time Node : %8.8d, %6.1d%%\r"%(ipoly,round(ipoly*100.0/npoly)))
        return arrival_times
    
    
    def getRisingSpeeds(self, method = 'fixed_boundaries', **kwargs): 
        # Inspect the data to construct the rise rate grid.
        global timefactor
        verbose = False
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        npoly = self.__ncvar.shape[1]
        rise_speeds = np.ma.zeros(npoly)
        rise_speeds.mask = True
        times  = self.__nctim[:]
        timeunit = self.__nctim.getncattr('units').split()[0]
        tfact = timefactor[timeunit]/timefactor[c.timeunit] 
        #calculate the rising speeds by using the fixed boundaries method (SSM)
        if method == 'fixed_boundaries':
            for ipoly in range(npoly):
                tseries= self.__ncvar[:,ipoly]
                lts = None
                prev_class = -1
                for time, class_ in list(zip(times,tseries)):
                    if class_ >= self.__lower_threshold:
                        if lts is None:
                            lts = (time, class_)
                        elif class_ >= self.__upper_threshold:
                            dh = self.__classes[class_] - self.__lower_definition
                            dt = (time - lts[0]) * tfact
                            if dh > 0. and dt > 0.:
                                rise_speeds[ipoly] = dh/dt
                                break
            prev_class = class_
            if (verbose):
                sys.stderr.write("Calculating rising speed Node : %8.8d, %6.1d%%\r"%(ipoly,round(ipoly*100.0/npoly)))
        #calculate the risingspeeds by using the lower class method
        if method == 'lower_class':
            for ipoly in range(npoly):
                tseries= self.__ncvar[:,ipoly]
                lts = None
                prev_class = -1
                for time, class_ in list(zip(times,tseries)):
                    if class_ >= self.__lower_threshold:
                        if lts is None:
                            lts = (time, class_)
                        elif class_ >= (self.__lower_threshold + 1):
                            dh = self.__classes[class_] - self.__lower_definition
                            dt = (time - lts[0]) * tfact
                            if dh > 0. and dt > 0.:
                                rise_speeds[ipoly] = dh/dt
                                break
            prev_class = class_
            if (verbose):
                sys.stderr.write("Calculating rising speed Node : %8.8d, %6.1d%%\r"%(ipoly,round(ipoly*100.0/npoly)))
        if method == 'all_classes':
            risespeedsmax = np.ma.zeros(npoly)
            risespeedsmax.mask = True
            height = np.ma.zeros(npoly)
            height.mask = True
            for ipoly in range(npoly):
                tseries= self.__ncvar[:,ipoly]
                prev_class = 1
                prev_time = 0
                risespeedsipoly = [0]
                risemaxhoogte = [0]
                for time, class_ in list(zip(times,tseries)):
                    if class_ >= (self.__lower_threshold):
                        
                        if class_ >= (prev_class + 1):
                            dh = (self.__classes[class_] - self.__classes[prev_class])
                            dt = (time - prev_time) * tfact
                            if dh > 0. and dt > 0.:
                                risespeedsipoly.append((dh/dt))
                                risemaxhoogte.append(self.__classes[class_])
                                prev_time = time
                                prev_class = class_
                max_value = max(risespeedsipoly)
                index = risespeedsipoly.index(max_value)
                risespeedsmax[ipoly] = risespeedsipoly[index]
                height[ipoly] = risemaxhoogte[index]
                rise_speeds = [risespeedsmax, height]
        return rise_speeds  

    def makerisingtif(self, method = 'fixed_boundaries'):
        if method == 'fixed_boundaries':
            rise_speeds = self.getRisingSpeeds(verbose=True)
        if method == 'lower_class':
            rise_speeds = self.getRisingSpeeds(method = 'lower_class', verbose=True)
        if method == 'all_classes':
            rise_speeds = self.getRisingSpeeds(method = 'all_classes', verbose=True)
            rise_speeds = rise_speeds[0]
        bbox, polygons, coords = self.__ds.get_polygons(self.__ncvarname) 
        bbox[0] = np.floor((bbox[0]-self.__configsettings.lower_left[0])/self.__configsettings.xres)* self.__configsettings.xres + self.__configsettings.lower_left[0]
        bbox[1] = np.floor((bbox[1]-self.__configsettings.lower_left[1])/self.__configsettings.yres)*self.__configsettings.yres + self.__configsettings.lower_left[1]
        
        # Write calculated rising speeds to tiff
        my_vardata = rise_speeds    
        
        if self.__modelname is not False:
            tiffname_rising = self.__ncoutput + f'\\rising_speeds{self.__modelname}.tif' 
        else:
            tiffname_rising   = self.__ncoutput + '\\rising_speeds.tif'
        
    
        nband = c.nband
        
        sys.stderr.write("\n")
        sys.stderr.write("Opening new TIFF:\n")
        my_tiff = tw.tiffwriter(tiffname_rising, bbox, [c.xres, c.yres], nband, c.nptype, c.epsg, flipped=True)
        
        sys.stderr.write("Filling pixels:\n")
        pixels = my_tiff.from_polygons(polygons, my_vardata, c.nodata)
        
        sys.stderr.write("Writing Band 1 .... \n")
        my_tiff.fillband(1, pixels, c.nodata)
        
        my_tiff.close()
        #self.ds.close()
        
    def makearrivaltif(self):        
        arrival_times = self.getArrivalTimes(verbose=True)
        bbox, polygons, coords = self.__ds.get_polygons(self.__ncvarname) 
        bbox[0] = np.floor((bbox[0]-self.__configsettings.lower_left[0])/self.__configsettings.xres)* self.__configsettings.xres + self.__configsettings.lower_left[0]
        bbox[1] = np.floor((bbox[1]-self.__configsettings.lower_left[1])/self.__configsettings.yres)*self.__configsettings.yres + self.__configsettings.lower_left[1]
        
        if self.__modelname is not False:
            tiffname_arrival   = self.__ncoutput + f'\\arrivaltimes{self.__modelname}.tif' 
        else:
            tiffname_arrival   = self.__ncoutput + '\\arrivaltimes.tif'
        
        nband = c.nband

        # Write calculated arrival times to TIFF file
        sys.stderr.write("\n")
        sys.stderr.write("Opening new TIFF:\n")
        my_tiff = tw.tiffwriter(tiffname_arrival, bbox, [c.xres, c.yres], nband, c.nptype, c.epsg, flipped=True)
        
        sys.stderr.write("Filling pixels:\n")
        pixels = my_tiff.from_polygons(polygons, arrival_times, c.nodata)
        
        sys.stderr.write("Writing Band 1 .... \n")
        my_tiff.fillband(1, pixels, c.nodata)
        
        my_tiff.close()
        #ds.close()
            
        
    def makegevaarhoogtetif(self):    
        risespeeds = self.getRisingSpeeds(method= 'all_classes', verbose=True)
        height = risespeeds[1]        
        bbox, polygons, coords = self.__ds.get_polygons(self.__ncvarname) 
        bbox[0] = np.floor((bbox[0]-self.__configsettings.lower_left[0])/self.__configsettings.xres)* self.__configsettings.xres + self.__configsettings.lower_left[0]
        bbox[1] = np.floor((bbox[1]-self.__configsettings.lower_left[1])/self.__configsettings.yres)*self.__configsettings.yres + self.__configsettings.lower_left[1]
        
        if self.__modelname is not False:
            tiffname_arrival   = self.__ncoutput + f'\\gevaarhoogtes{self.__modelname}.tif' 
        else:
            tiffname_arrival   = self.__ncoutput + '\\gevaarhoogtes.tif'
        
        nband = c.nband

        # Write calculated arrival times to TIFF file
        sys.stderr.write("\n")
        sys.stderr.write("Opening new TIFF:\n")
        my_tiff = tw.tiffwriter(tiffname_arrival, bbox, [c.xres, c.yres], nband, c.nptype, c.epsg, flipped=True)
        
        sys.stderr.write("Filling pixels:\n")
        pixels = my_tiff.from_polygons(polygons, height, c.nodata)
        
        sys.stderr.write("Writing Band 1 .... \n")
        my_tiff.fillband(1, pixels, c.nodata)
        
        my_tiff.close()
        





