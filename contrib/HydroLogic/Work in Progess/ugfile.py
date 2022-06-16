# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 00:59:21 2021

@author: Student8
"""

import netCDF4 as nc
class DatasetUG(nc.Dataset):
    def store_atts(self,variable):
        global meshes
        new_record = {}
        for attrib_name in variable.ncattrs():
            new_record[attrib_name] = variable.getncattr(attrib_name)
        return new_record
    
    def get_polygons(self, vname): 
        meshes = {}
        my_variables = {}
        for varname, variable in self.variables.items():
            try:
                cfrole = variable.getncattr('cf_role') 
            except:
                cfrole = ''
            # store meshes and coordinate systems
            if (cfrole == 'mesh_topology'):
                meshes[varname] = self.store_atts(variable)
        my_var     = self.variables[vname]
        my_atts    = self.store_atts(my_var)
        my_mesh    = meshes[my_atts['mesh']]
        my_mapping = my_atts['grid_mapping']
        if my_mapping == '':
            my_mapping = 'projected_coordinate_system'

        if my_mapping in self.variables:
            my_coords  = self.store_atts(self.variables[my_mapping])
        else:
            raise ("\nProjection variable "+my_mapping+" not found!\n")
        
        face_node_connectivity = self.variables[my_mesh['face_node_connectivity']][:]
        node_coordinates_names = my_mesh['node_coordinates'].split()
        node_coordinates_x = self.variables[node_coordinates_names[0]][:]
        node_coordinates_y = self.variables[node_coordinates_names[1]][:]
            
        my_polygons = []
        for pol in face_node_connectivity:
            nvtx = pol.count() # to stop before masked values
            pgon = {}
            pgon['x'] = node_coordinates_x[pol[range(nvtx)]-1]
            pgon['y'] = node_coordinates_y[pol[range(nvtx)]-1]
            if my_polygons:
                xmin = min(xmin,min(pgon['x']))
                xmax = max(xmax,max(pgon['x']))
                ymin = min(ymin,min(pgon['y']))
                ymax = max(ymax,max(pgon['y']))
            else:
                xmin = min(pgon['x'])
                xmax = max(pgon['x'])
                ymin = min(pgon['y'])
                ymax = max(pgon['y'])
            my_polygons.append(pgon)
        my_bbox = [xmin, ymin, xmax, ymax]
        return (my_bbox, my_polygons, my_coords)
