from ast import Or
from calendar import c
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import sys

sys.path.append(r'D:\3640.20\HYDROLIB-core')
from hydrolib.core.io.structure.models import *
from hydrolib.core.io.crosssection.models import *
from hydrolib.core.io.ext.models import *
from hydrolib.core.io.net.models import *


from shapely.geometry import LineString, MultiPolygon, Point, Polygon

logger = logging.getLogger(__name__)


class Structures:

    def __init__(self):
        self.rweirs = []
        self.orifices = []
        self.uweirs = []
        self.culverts = []
        self.bridges = []
        self.pumps = []

        self.df_rweirs = []
        self.df_orifices = []
        self.df_uweirs = []
        self.df_culverts = []
        self.df_bridges = []
        self.df_pumps = []
            
        
    def weirs_from_hydamo(self, weirs, profile_groups=None, profile_lines=None, profiles=None,  opening=None, management_device=None, management=None, usevelocityheight='true'):
        """
        Method to convert dflowfm weirs from hydamo weirs.
        
        Split the HyDAMO weirs object into a subset with 'regular weirs' and 'universal weirs'. 
       
        For now the assumption is that weirs for which a profile is defined, are universal, other weirs are regular.
               
        """
        
        index = np.zeros((len(weirs.code)))
        if profile_groups is not None:
            if 'stuwid' in profile_groups: 
                index[np.isin(weirs.globalid , np.asarray(profile_groups.stuwid))]=1              

        rweirs = weirs[index==0]

        rw_and_or_df = pd.DataFrame(columns=['id','name','branchid','chainage','crestlevel',
                                         'crestwidth','corrcoeff','usevelocityheight','gateloweredgelevel',
                                         'allowedflowdir','uselimitflowpos', 'limitflowpos','uselimitflowneg','limitflowneg'], index=rweirs.code)

        for weir in rweirs.itertuples():
                        
            weir_opening = opening[opening.stuwid == weir.globalid]
            weir_mandev = management_device[management_device.kunstwerkopeningid == weir_opening.globalid.to_string(index=False)]
            
            # check if a separate name field is present
            if 'naam' in weirs:
                name = weir.naam
            else:
                name = weir.code

            if weir_mandev.overlaatonderlaat.to_string(index=False) == 'Overlaat':
                rw_and_or_df.at[weir.code, 'id'] = weir.code
                rw_and_or_df.at[weir.code, 'name'] = name                
                rw_and_or_df.at[weir.code, 'branchid'] = weir.branch_id
                rw_and_or_df.at[weir.code, 'chainage'] = weir.branch_offset
                rw_and_or_df.at[weir.code, 'crestlevel'] = weir_opening.laagstedoorstroomhoogte.values[0]
                rw_and_or_df.at[weir.code, 'crestwidth'] = weir_opening.laagstedoorstroombreedte.values[0]
                rw_and_or_df.at[weir.code, 'corrcoeff'] = weir.afvoercoefficient
                rw_and_or_df.at[weir.code, 'allowedflowdir'] = 'both'
                rw_and_or_df.at[weir.code, 'usevelocityheight'] = usevelocityheight          
                
            elif weir_mandev.overlaatonderlaat.to_string(index=False) == 'Onderlaat':
                if 'maximaaldebiet' not in weir_mandev:  
                    limitflow = 'false'
                    maxq =  0.0
                else:
                    limitflow = 'true'
                rw_and_or_df.at[weir.code, 'gateloweredgelevel'] = weir_mandev.hoogteopening.values[0]
                rw_and_or_df.at[weir.code,'uselimitflowpos'] = limitflow
                rw_and_or_df.at[weir.code,'limitflowpos'] = maxq
                rw_and_or_df.at[weir.code,'uselimitflowneg'] = limitflow
                rw_and_or_df.at[weir.code,'limitflowneg'] = maxq

        self.rweirs_df = rw_and_or_df[rw_and_or_df['uselimitflowpos'].isnull()]
        self.orifices_df = rw_and_or_df[~rw_and_or_df['uselimitflowpos'].isnull()]
        self.rweirs_df.drop(['gateloweredgelevel','uselimitflowpos','limitflowpos','uselimitflowneg','limitflowneg'], axis=1, inplace=True)

        for weir in self.rweirs_df.itertuples():
            struc = Weir(
                            id=weir.id,
                            name=weir.name,                                        
                            branchid=weir.branchid,
                            chainage=weir.chainage,
                            crestlevel=weir.crestlevel,
                            crestwidth=weir.crestwidth,
                            corrcoeff=weir.corrcoeff,
                            allowedflowdir=weir.allowedflowdir,
                            usevelocityheight=weir.usevelocityheight
                        )  
            # don't understand why this does not work. It does in debugging mode
            #[exec("struc.comments."+field[0]+"=''") for field in struc.comments]    
            
            
            
            self.rweirs.append(struc)
        for orifice in self.orifices_df.itertuples():            
            struc = Orifice(
                            id = orifice.id,
                            name = orifice,
                            branchid=orifice.branch_id,
                            chainage=orifice.branch_offset,
                            crestlevel=orifice.crestlevel,
                            crestwidth=orifice.crestwidth,                                                                        
                            corrcoeff=orifice.afvoercoefficient,
                            allowedflowdir=orifice.allowedflowdir,
                            usevelocityheight=orifice.usevelocityheight,
                            gateloweredgelevel=orifice.gateloweredgelevel,
                            uselimitflowpos=orifice.uselimitflowpos,
                            limitflowpos=orifice.limitflowpos,
                            uselimitflowneg=orifice.uselimitflowneg,
                            limitflowneg=orifice.limitflowneg
                            )
            [exec("struc.comments."+field[0]+"=''") for field in struc.comments]           
            self.orifices.append(struc)

            

                                                    

        uweirs = weirs[index==1]   

        self.uweirs_df = pd.DataFrame(columns=['id','name','branchid','chainage','crestlevel',
                                         'crestwidth','dischargecoeff','usevelocityheight','allowedflowdir',
                                         'numlevels','yvalues', 'zvalues'], index=uweirs.code)
       
        for uweir in uweirs.itertuples():                
            
            # check if a separate name field is present
            if 'naam' in uweirs:
                name = uweir.naam
            else:
                name = uweir.code

            prof=np.empty(0)
            if profiles is not None:
                    if 'stuwid' in profile_groups:
                        group = profile_groups[profile_groups['stuwid']==uweir.globalid]
                        line = profile_lines[profile_lines['profielgroepid']==group['globalid'].values[0]]
                        prof = profiles[profiles['globalid']==line['globalid'].values[0]]   
                        if not prof.empty:                    
                            counts = len(prof.geometry.iloc[0].coords[:])
                            xyz = np.vstack(prof.geometry.iloc[0].coords[:])
                            length = np.r_[0, np.cumsum(np.hypot(np.diff(xyz[:, 0]), np.diff(xyz[:, 1])))]
                            yzvalues = np.c_[length, xyz[:, -1]-np.min(xyz[:,-1])]                            
                            
            if len(prof)==0:
                # return an error it is still not found
                raise ValueError(f'{uweir.code} is not found in any cross-section.')           

            self.uweirs_df.at[uweir.code, 'id'] = uweir.code
            self.uweirs_df.at[uweir.code, 'name'] = name                
            self.uweirs_df.at[uweir.code, 'branchid'] = uweir.branch_id
            self.uweirs_df.at[uweir.code, 'chainage'] = uweir.branch_offset
            self.uweirs_df.at[uweir.code, 'crestlevel'] = uweir.laagstedoorstroomhoogte
            self.uweirs_df.at[uweir.code, 'crestwidth'] = uweir.kruinbreedte
            self.uweirs_df.at[uweir.code, 'discchargecoeff'] = uweir.afvoercoefficient
            self.uweirs_df.at[uweir.code, 'allowedflowdir'] = 'both'
            self.uweirs_df.at[uweir.code, 'usevelocityheight'] = usevelocityheight
            self.uweirs_df.at[uweir.code, 'numlevels'] = counts
            self.uweirs_df.at[uweir.code, 'yvalues'] = ' '.join([f'{yz[0]:7.3f}' for yz in yzvalues])
            self.uweirs_df.at[uweir.code, 'zvalues'] = ' '.join([f'{yz[1]:7.3f}' for yz in yzvalues])
           
        for uweir in self.uweirs_df.itertuples():
            struc = UniversalWeir(
                        id=uweir.id,
                        name=uweir.name,                                        
                        branchid=uweir.branchid,
                        chainage=uweir.chainage,
                        crestlevel=uweir.crestlevel,
                        crestwidth=uweir.crestwidth,
                        dischargecoeff=uweir.dischargecoeff,
                        allowedflowdir=uweir.allowedflowdir,
                        usevelocityheight=uweir.usevelocityheight,
                        numlevels = uweir.numlevels,
                        yvalues = uweir.yvalues,
                        zvalues = uweir.zvalues
                    )  
        self.uweirs.append(struc)
           
        #    [exec("struc.comments."+field[0]+"=''") for field in struc.comments]                           
           

    def bridges_from_hydamo(self, bridges, profile_groups=None, profile_lines=None, profiles=None):
        #         """
        #         Method to convert dflowfm bridges from hydamo bridges.
        #         """
        #         # Convert to dflowfm input
        #         geconverteerd = hydamo_to_dflowfm.generate_bridges(bridges, profile_groups=profile_groups, profile_lines=profile_lines,profiles=profiles)
        
        self.bridges_df = pd.DataFrame(columns=['id','name','branchid','chainage','csdefid','shift',
                                                'inletlosscoeff','outletlosscoeff','allowedflowdir',
                                                'length','frictiontype','friction'], index=bridges.code)

        for bridge in bridges.itertuples():        
            # first search in yz-profiles
            group = profile_groups[profile_groups['brugid']==bridge.globalid]
            line = profile_lines[profile_lines['profielgroepid']==group['globalid'].values[0]]
            prof = profiles[profiles['globalid']==line['globalid'].values[0]]   
            
            if len(prof) > 0:
                #bedlevel = np.min([c[2] for c in prof.geometry[0].coords[:]])  
                profile_id=prof.code.values[0]
            else:
                # return an error it is still not found
                raise ValueError(f'{bridge.code} is not found in any cross-section.')

            if 'naam' in bridges:
                name = bridge.naam
            else:
                name = bridge.code
            profile_id=prof.code.values[0]

            self.bridges_df.at[bridge.code, 'id'] = bridge.code
            self.bridges_df.at[bridge.code, 'name'] = name
            self.bridges_df.at[bridge.code, 'branchid'] = bridge.branch_id
            self.bridges_df.at[bridge.code, 'chainage'] = bridge.branch_offset
            self.bridges_df.at[bridge.code, 'csdefid'] = profile_id
            self.bridges_df.at[bridge.code, 'shift'] = '0.0'
            self.bridges_df.at[bridge.code, 'allowedflowdir'] = 'both'
            self.bridges_df.at[bridge.code, 'inletlosscoeff'] = bridge.intreeverlies
            self.bridges_df.at[bridge.code, 'outletlosscoeff'] = bridge.uittreeverlies
            self.bridges_df.at[bridge.code, 'length'] = bridge.lengte
            self.bridges_df.at[bridge.code, 'frictiontype'] = bridge.typeruwheid
            self.bridges_df.at[bridge.code, 'friction'] = bridge.ruwheid

        for bridge in self.bridges_df.itertuples():
            struc = Bridge(
                            id      = bridge.id,
                            name    = bridge.name,
                            branchid= bridge.branchid,
                            chainage= bridge.chainage,
                            csdefid = bridge.csdefid,
                            allowedflowdir=bridge.allowedflowdir,
                            shift = bridge.shift,
                            inletlosscoeff =bridge.inletlosscoeff,
                            outletlosscoeff = bridge.outletlosscoeff,
                            length = bridge.length,
                            frictiontype=bridge.frictiontype,
                            friction = bridge.friction
                           )
                        
            #[exec("struc.comments."+field[0]+"=''") for field in struc.comments]                                                     
            self.bridges.append(struc)

    def culverts_from_hydamo(self, culverts, management_device=None):
        """
        Method to convert dflowfm weirs from hydamo weirs.
        """
        self.culverts_df = pd.DataFrame(columns=['id','name','branchid','chainage','csdefid','leftlevel','rightlevel',
                                                'inletlosscoeff','outletlosscoeff','allowedflowdir',
                                                'length','frictiontype','friction','numlosscoeff',
                                                'valveonoff','losscoeff','valveopeningheight','relopening'], index=culverts.code)
        for culvert in culverts.itertuples():

            # Generate cross section definition name
            if culvert.vormkoker == 'Rond' or culvert.vormkoker == 'Ellipsvormig':
                crosssection = {'shape': 'circle', 'diameter': culvert.hoogteopening}
                definition = f'circ_d{culvert.hoogteopening:.2f}'
            elif culvert.vormkoker == 'Rechthoekig' or culvert.vormkoker == 'Onbekend' or culvert.vormkoker == 'Eivormig' or culvert.vormkoker=='Muilprofiel' or culvert.vormkoker=='Heulprofiel':
                crosssection = {'shape': 'rectangle', 'height': culvert.hoogteopening, 'width': culvert.breedteopening, 'closed': 1}
                definition = f'circ_h{culvert.hoogteopening:.2f}_w{culvert.breedteopening:.2f}'
            else:
                crosssection = {'shape': 'circle', 'diameter': 0.40}
                print(f'Culvert {culvert.code} has an unknown shape: {culvert.vormkoker}. Applying a default profile (round - 40cm)')
            
            
            # check whether an afsluitmiddel is present and take action dependent on its settings
            mandev = management_device[management_device.duikersifonhevelid==culvert.globalid]
            if mandev.empty:        
                allowedflowdir='both'
                valveonoff= 0
                numlosscoeff=None
                valveopeningheight=0
                relopening=None
                losscoeff=None
            else:      
                for _,i in mandev.iterrows():
                    if i['soortregelmiddel']=='terugslagklep':
                        allowedflowdir = 'positive'
                    elif i['soortregelmiddel']=='schuif':
                        valveonoff = 1
                        valveopeningheight = float(i['hoogteopening'])
                        numlosscoeff = 1
                        relopening = [float(i['hoogteopening'])/culvert.hoogteopening]
                        losscoeff = [float(i['afvoercoefficient'])]
                    else:
                        print(f'Type of closing device for culvert {culvert.code} is not implemented; only "schuif" and "terugslagklep" are allowed.')
            
            # check if a separate name field is present
            if 'naam' in culverts:
                name = culvert.naam
            else:    
                name = culvert.code
            self.culverts_df.at[culvert.code, 'id'] = culvert.code
            self.culverts_df.at[culvert.code, 'name'] = name
            self.culverts_df.at[culvert.code, 'branchid'] = culvert.branch_id
            self.culverts_df.at[culvert.code, 'chainage'] = culvert.branch_offset
            self.culverts_df.at[culvert.code, 'leftlevel'] = culvert.hoogtebinnenonderkantbov
            self.culverts_df.at[culvert.code, 'rightlevel'] = culvert.hoogtebinnenonderkantbene
            self.culverts_df.at[culvert.code, 'length'] = culvert.lengte
            self.culverts_df.at[culvert.code, 'inletlosscoeff'] = culvert.intreeverlies
            self.culverts_df.at[culvert.code, 'outletlosscoeff'] = culvert.uittreeverlies
            self.culverts_df.at[culvert.code, 'csdefid'] = definition
            self.culverts_df.at[culvert.code, 'allowedflowdir'] = 'both'
            self.culverts_df.at[culvert.code, 'valveonoff'] = valveonoff
            self.culverts_df.at[culvert.code, 'numlosscoeff'] = numlosscoeff
            self.culverts_df.at[culvert.code, 'valveopeningheight'] = valveopeningheight
            self.culverts_df.at[culvert.code, 'relopening'] = relopening
            self.culverts_df.at[culvert.code, 'losscoeff'] = losscoeff
            
        for culvert in self.culverts_df.itertuples():
            struc = Culvert(
                            id=culvert.id,
                            name=culvert.name,                                        
                            branchid=culvert.branchid,
                            chainage=culvert.chainage,
                            leftlevel=culvert.leftlevel,
                            rightlevel=culvert.rightlevel,                     
                            length=culvert.length,
                            inletlosscoeff=culvert.inletlosscoeff,
                            outletlosscoeff=culvert.outletlosscoeff,
                            csdefid=culvert.csdefid,
                            allowedflowdir=culvert.allowedflowdir,
                            valveonoff= culvert.valveonoff,
                            numlosscoeff=culvert.numlosscoeff,
                            valveopeningheight=culvert.valveopeningheight,
                            relopening=culvert.relopening,
                            losscoeff=culvert.losscoeff
                        )       
            #[exec("struc.comments."+field[0]+"=''") for field in struc.comments]                                                     
            self.culverts.append(struc)
 
    
    def pumps_from_hydamo(self, pumpstations, pumps=None, management=None):
    
        # DAMO contains m3/min, while D-Hydro needs m3/s
        pumps['maximalecapaciteit'] /= 60


        self.pumps_df = pd.DataFrame(columns=['id','name','branchid','chainage',
                                               'orientation','numstages','controlside',
                                                'capacity','startlevelsuctionside','stoplevelsuctionside'], index=pumps.code)
       
  
        # Add sturing to pumps
        for pump in pumps.itertuples():

            # Find sturing for pump
            sturingidx = (management.pompid == pump.globalid).values
            
            # find gemaal for pump
            gemaalidx = (pumpstations.globalid == pump.gemaalid).values
            
            # so first check if there are multiple pumps with one 'sturing'
            if sum(sturingidx) != 1:            
                raise IndexError(f'Multiple or no management rules found in hydamo.management for pump {pump.code}.')
                
            # If there als multiple pumping stations connected to one pump, raise an error
            if sum(gemaalidx) != 1:
                raise IndexError(f'Multiple or no pump stations (gemalen) found for pump {pump.code}.')

            # Find the idx if the pumping station connected to the pump
                # gemaalidx = gemalen.iloc[np.where(gemaalidx)[0][0]]['code']
                # Find the control for the pumping station (and thus for the pump)
                #@sturingidx = (sturing.codegerelateerdobject == gemaalidx).values

                #assert sum(sturingidx) == 1

            branch_id = pumpstations.iloc[np.where(gemaalidx)[0][0]]['branch_id']
            branch_offset = pumpstations.iloc[np.where(gemaalidx)[0][0]]['branch_offset']
            # Get the control by index
            pump_control = management.iloc[np.where(sturingidx)[0][0]]

            if pump_control.doelvariabele != 1 and pump_control.doelvariabele != 'waterstand':
                raise NotImplementedError('Sturing not implemented for anything else than water level (1).')

            # Add levels for suction side
            startlevelsuctionside = [pump_control['bovengrens']]
            stoplevelsuctionside = [pump_control['ondergrens']] 

            if 'naam' in pumps:
                name = pump.name
            else:
                name = pump.code

            self.pumps_df.at[pump.code,'id'] = pump.code                    
            self.pumps_df.at[pump.code,'name'] =name
            self.pumps_df.at[pump.code,'branchid'] =branch_id
            self.pumps_df.at[pump.code,'chainage'] = branch_offset
            self.pumps_df.at[pump.code,'orientation'] ='positive'
            self.pumps_df.at[pump.code,'numstages'] = 1
            self.pumps_df.at[pump.code,'controlside'] ='suctionSide'
            self.pumps_df.at[pump.code,'capacity'] = pump.maximalecapaciteit
            self.pumps_df.at[pump.code,'startlevelsuctionside'] = startlevelsuctionside
            self.pumps_df.at[pump.code,'stoplevelsuctionside'] = stoplevelsuctionside
        
        for pump in self.pumps_df.itertuples():
            struc = Pump(
                        id=pump.id,
                        name=pump.name,
                        branchid=pump.branchid,
                        chainage=pump.chainage,
                        orientation=pump.orientation,
                        numstages=pump.numstages,
                        controlside=pump.controlside,
                        capacity=pump.capacity,
                        startlevelsuctionside=pump.startlevelsuctionside,
                        stoplevelsuctionside=pump.stoplevelsuctionside
                        )
            self.pumps.append(struc)                     
                    
#     def compound_structures(self, idlist, structurelist):
#         """
#         Method to add compound structures to the model.
        
#         """
#         geconverteerd = hydamo_to_dflowfm.generate_compounds(idlist, structurelist, self.structures)
        
#          # Add to dict
#         for compound in geconverteerd.itertuples():
#             self.structures.add_compound(
#                 id=compound.code,
#                 numstructures=compound.numstructures,
#     	        structurelist=compound.structurelist    	        
#             )
        
                    

class CrossSections:
    pass

#     def __init__(self, crosssections):
#         self.crosssections = crosssections

#     def from_datamodel(self, crsdefs=None, crslocs=None):
#         """"
#         From parsed data models of crsdefs and crs locs
#         """

#         if crslocs is not None:
#             for crsloc_idx, crsloc in crslocs.iterrows():
#                 # add location
#                 self.crosssections.add_crosssection_location(branchid=crsloc['branch_id'],
#                                                              chainage=crsloc['branch_offset'],
#                                                              shift=crsloc['shift'],
#                                                              definition=crsloc['crosssectiondefinitionid'])

#         if crsdefs is not None:
#             crsdefs = crsdefs.drop_duplicates(subset=['crosssectiondefinitionid'])
#             for crsdef_idx, crsdef in crsdefs.iterrows():
#                 # Set roughness value on default if cross-section has non defined (e.g. culverts)
#                 roughtype = crsdef['frictionid'].split('_')[0] if isinstance(crsdef['frictionid'], str) else 'Chezy'
#                 roughval = float(crsdef['frictionid'].split('_')[-1]) if isinstance(crsdef['frictionid'], str) else 45
#                 # add definition
#                 if crsdef['type'] == 'circle':
#                     self.crosssections.add_circle_definition(diameter=crsdef['diameter'],
#                                                              roughnesstype=roughtype,
#                                                              roughnessvalue=roughval,
#                                                              name=crsdef['crosssectiondefinitionid'])
#                 elif crsdef['type'] == 'rectangle':
#                     self.crosssections.add_rectangle_definition(height=crsdef['height'],
#                                                                 width=crsdef['width'],
#                                                                 closed=crsdef['closed'],
#                                                                 roughnesstype=roughtype,
#                                                                 roughnessvalue=roughval,
#                                                                 name=crsdef['crosssectiondefinitionid'])

#                 elif crsdef['type'] == 'trapezium':
#                     self.crosssections.add_trapezium_definition(slope=(crsdef['t_width'] - crsdef['width'])/2/crsdef['height'],
#                                                                 maximumflowwidth=crsdef['t_width'],
#                                                                 bottomwidth=crsdef['width'],
#                                                                 closed=crsdef['closed'],
#                                                                 roughnesstype=roughtype,
#                                                                 roughnessvalue=roughval,
#                                                                 name=crsdef['crosssectiondefinitionid'])
                    
#                 elif crsdef['type'] == 'zw':
#                     self.crosssections.add_zw_definition(numLevels=crsdef["numlevels"],
#                                                          levels=crsdef["levels"],
#                                                          flowWidths=crsdef["flowwidths"],
#                                                          totalWidths=crsdef["totalwidths"],
#                                                          roughnesstype=roughtype,
#                                                          roughnessvalue=roughval,
#                                                          name=crsdef['crosssectiondefinitionid'])

#                 elif crsdef['type'] == 'yz':
#                     # TODO BMA: add yz
#                     raise NotImplementedError

#                 else:
#                     raise NotImplementedError

#     def from_hydamo(self, crosssections=None, crossection_roughness=None, param_profile=None, param_profile_values=None, branches=None, roughness_variant=None):
#         """
#         Method to add cross section from hydamo files. Two files
#         can be handed to the function, the cross section file (dwarsprofiel) and the
#         parametrised file (normgeparametriseerd). The
#         hierarchical order is 1. dwarsprofiel, 2. normgeparametriseerd.
#         Each branch will be assigned a profile following this order. If parametrised
#         and standard are not given, branches can be without cross section. In that case
#         a standard profile should be assigned
#         """

#         # first, make a selection as to use only the dwarsprofielen/parametrised that are related to branches, not structures
#         if crosssections is not None and not crosssections.empty:
#             if 'stuwid' not in crosssections:
#                 crosssections['stuwid'] = str(-999.)
#             if 'brugid' not in crosssections:                
#                 crosssections['brugid'] = str(-999.)
#             dp_branches = ExtendedGeoDataFrame(geotype=LineString, columns = crosssections.required_columns)
#             dp_branches.set_data(gpd.GeoDataFrame([i for i in crosssections.itertuples() if (len(i.brugid)<10)&(len(i.stuwid)<10)]))            
#         else:     
#             dp_branches = ExtendedGeoDataFrame(geotype=LineString)
            
       
#        # Assign cross-sections to branches
#         nnocross = len(self.crosssections.get_branches_without_crosssection())
#         logger.info(f'Before adding the number of branches without cross section is: {nnocross}.')
             
#         if not dp_branches is None:
#             # 1. Collect cross sections from 'dwarsprofielen'
#             yz_profiles = hydamo_to_dflowfm.dwarsprofiel_to_yzprofiles(dp_branches, crossection_roughness, branches, roughness_variant=roughness_variant)               
        
#             for name, css in yz_profiles.items():
#                 # Add definition
#                 self.crosssections.add_yz_definition(yz=css['yz'], thalweg=css['thalweg'], name=name, roughnesstype=css['typeruwheid'], roughnessvalue=css['ruwheid'])
#                 # Add location
#                 self.crosssections.add_crosssection_location(branchid=css['branchid'], chainage=css['chainage'], definition=name)
        
#         # Check the number of branches with cross sections
#         no_crosssection_id = self.crosssections.get_branches_without_crosssection()
#         no_crosssection = [b for b in branches.itertuples() if b.code in no_crosssection_id]
        
#         nnocross = len(no_crosssection)
#         logger.info(f'After adding \'dwarsprofielen\' the number of branches without cross section is: {nnocross}.')
#         if (nnocross == 0):
#             print('No further branches without a profile.')
#         elif param_profile is None:
#             print('No parametrised crossections available for branches.')
#         else: 
#             # Derive norm cross sections for norm parametrised
#             param_profiles_converted = hydamo_to_dflowfm.parametrised_to_profiles(param_profile, param_profile_values, no_crosssection, roughness_variant=roughness_variant)
#             # Get branch information
#             branchdata = self.crosssections.dflowfmmodel.network.branches.loc[list(param_profiles_converted.keys())]
#             branchdata['chainage'] = branchdata.length / 2.
        
#             # Add cross sections
#             for branchid, css in param_profiles_converted.items():
#                 chainage = branchdata.at[branchid, 'chainage']
                    
#                 if css['type'] == 'rectangle':
#                     name = self.crosssections.add_rectangle_definition(
#                         height=css['height'],
#                         width=css['width'],
#                         closed=css['closed'],
#                         roughnesstype=css['typeruwheid'],
#                         roughnessvalue=css['ruwheid']
#                     )
    
#                 if css['type'] == 'trapezium':
#                     name = self.crosssections.add_trapezium_definition(
#                         slope=css['slope'],
#                         maximumflowwidth=css['maximumflowwidth'],
#                         bottomwidth=css['bottomwidth'],
#                         closed=css['closed'],
#                         roughnesstype=css['typeruwheid'],
#                         roughnessvalue=css['ruwheid']                        
#                     ) 
    
#                 # Add location
#                 self.crosssections.add_crosssection_location(branchid=branchid, chainage=chainage, definition=name, shift=css['bottomlevel'])

#             nnocross = len(self.crosssections.get_branches_without_crosssection())
#             logger.info(f'After adding \'normgeparametriseerd\' the number of branches without cross section is: {nnocross}.')
                
            
# class ExternalForcingsIO:

#     def __init__(self, external_forcings):
#         self.external_forcings = external_forcings

#     def from_hydamo(self, boundary_conditions):

#         # Read from Hydamo
#         bcdct = hydamo_to_dflowfm.generate_boundary_conditions(boundary_conditions, self.external_forcings.dflowfmmodel.network.schematised)

#         # Add all items
#         for key, item in bcdct.items():
#             self.external_forcings.add_boundary_condition(key, item['geometry'], item['bctype'], item['value'], branchid=item['branchid'])
#             # # Add to dataframe         
#             # self.external_forcings.boundaries.loc[key] = item
#             # Check if a 1d2d link should be removed
#             #self.external_forcings.dflowfmmodel.network.links1d2d.check_boundary_link(self.external_forcings.boundaries.loc[key])

#     def read_laterals(self, locations, lateral_discharges=None, rr_boundaries=None):
#         """
#         Process laterals

#         Parameters
#         ----------
#         locations: gpd.GeoDataFrame
#             GeoDataFrame with at least 'geometry' (Point) and the column 'code'
#         lateral_discharges: pd.DataFrame
#             DataFrame with lateral discharges. The index should be a time object (datetime or similar).
#         rr_boundaries: pd.DataFrame
#             DataFrame with RR-catchments that are coupled 
#         """

#         if rr_boundaries is None: rr_boundaries = []
#         # Check argument
#         checks.check_argument(locations, 'locations', gpd.GeoDataFrame, columns=['geometry'])
#         if lateral_discharges is not None:
#             checks.check_argument(lateral_discharges, 'lateral_discharges', pd.DataFrame)

#         # Check if network has been loaded
#         network1d = self.external_forcings.dflowfmmodel.network.mesh1d
#         if not network1d.meshgeomdim.numnode:
#             raise ValueError('1d network has not been generated or loaded. Do this before adding laterals.')

#         # in case of 3d points, remove the 3rd dimension
#         locations['geometry2'] = [Point([point.geometry.x, point.geometry.y]) for _,point in locations.iterrows()]    
#         locations.drop('geometry', inplace=True, axis=1)
#         locations.rename(columns={'geometry2':'geometry'}, inplace=True)
        
#         # Find nearest 1d node per location and find the nodeid
#         #lateral_crds = np.vstack([loc.geometry.coords[0] for loc in locations.itertuples()])             
#         #nodes1d = network1d.get_nodes()
#         #get_nearest = KDTree(nodes1d)
#         #_, nearest_idx = get_nearest.query(lateral_crds[:,0:2])
                
#         # Get time series and add to dictionary
#         #for nidx, lateral in zip(nearest_idx, locations.itertuples()):
#         for lateral in locations.itertuples():
#             # crd = nodes1d[nearest_idx]
#             #nid = f'{nodes1d[nidx][0]:g}_{nodes1d[nidx][1]:g}'
            
#             # Check if a time is provided for the lateral
#             if lateral.code in rr_boundaries:                
#                 # Add to dictionary
#                 self.external_forcings.laterals[lateral.code] = {
#                     'branchid': lateral.branch_id,
#                     'branch_offset': str(lateral.branch_offset)                    
#                 }
#             else:
#                 if lateral_discharges is None:
#                     logger.warning(f'No lateral_discharges provied. {lateral.code} expects them. Skipping.')
#                     continue
#                 else:
#                     if lateral.code not in lateral_discharges.columns:
#                         logger.warning(f'No data found for {lateral.code}. Skipping.')
#                         continue
                    
#                 # Get timeseries
#                 series = lateral_discharges.loc[:, lateral.code]
                
#                 # Add to dictionary
#                 self.external_forcings.laterals[lateral.code] = { 
#                     'branchid': lateral.branch_id,
#                     'branch_offset': str(lateral.branch_offset), 
#                     'timeseries': series            
#                 }


# class StorageNodesIO:

#     def __init__(self, storagenodes):
#         self.storagenodes = storagenodes

#     def storagenodes_from_datamodel(self, storagenodes):
#         """"From parsed data model of storage nodes"""
#         for storagenode_idx, storagenode in storagenodes.iterrows():
#             self.storagenodes.add_storagenode(
#                 id=storagenode.id,
#                 name=storagenode.name if 'name' in storagenode.code else np.nan,
#                 usestreetstorage=storagenode.usestreetstorage,
#                 nodetype='unspecified',
#                 nodeid=storagenode.nodeid,
#                 usetable='false',
#                 bedlevel=storagenode.bedlevel,
#                 area=storagenode.area,
#                 streetlevel=storagenode.streetlevel,
#                 streetstoragearea=storagenode.streetstoragearea,
#                 storagetype=storagenode.storagetype
#             )

