# coding: latin-1
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
from pydantic.v1 import validate_arguments
import netCDF4 as nc
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.core.dflowfm.mdu.models import FMModel


class DIMRWriter:
    """Temporary files for DIMR-configuration - to be superseded by Hydrolib-core functionality."""

    @validate_arguments
    def __init__(
        self, dimr_path: str = None, output_path: Union[str, Path] = None
    ) -> None:
        """Class initialization for the DIMR writer. This is a temporary fix to include writers for RR and RTC, that are not yet fully included in Hydrolib-core, or have bugs. Eventually, this file should be redundant as this functionality is fully included in Hydrolib-core

        Args:
            dimr_path (Union[str, Path], optional): _description_. Defaults to None.
            output_path (Union[str, Path], optional): _description_. Defaults to None.
        """
        if dimr_path is None:
            self.run_dimr = r"C:\Program Files\Deltares\D-HYDRO Suite 2022.04 1D2D\plugins\DeltaShell.Dimr\kernels\x64\dimr\scripts\run_dimr.bat"
        else:
            self.run_dimr = dimr_path

        if output_path is None:
            self.output_path = Path(os.path.abspath("."))
        else:
            self.output_path = output_path

        self.template_dir = Path(os.path.abspath("."))

    @validate_arguments
    def write_runbat(self, debuglevel=6, runlog=None) -> None:
        """Generates a run.bat to run the model in DIMR. The path to the executable is provided by the user in the class initialization, or set to the default D-Hydro installation folder."""

        with open(os.path.join(self.output_path, "run.bat"), "w") as f:
            f.write("@ echo off\n")
            f.write("set OMP_NUM_THREADS=2\n")
            if runlog is None:
                f.write('call "' + str(self.run_dimr) + '" -d ' + str(debuglevel) + '\n')
            else:
                f.write('call "' + str(self.run_dimr) + '" -d ' + str(debuglevel) + ' > ' + str(runlog) + '\n')

    @validate_arguments
    def add_crs(self, netcdf_path: Union[str, Path] = None) -> None:
        """Reads the Netcdf file and addes the required attributes for a valid CRS."""        
        if netcdf_path is None:
            netfile = list((self.output_path / 'dflowfm').glob('*.nc'))[0]
            if not netfile.exists():
                raise FileNotFoundError(f"Netcdf file not found in {self.output_path / 'dflowfm'}. Provide the correct path via the netcdf_path argument.")
        else:
            if netcdf_path.is_dir():
                netfile = list(netcdf_path.glob('*.nc'))[0]
            elif netcdf_path.is_file():
                netfile = netcdf_path  
            else:        
                raise FileNotFoundError(f"Netcdf file not found at {netcdf_path}.")   
        netf = nc.Dataset(netfile, 'r+')
        netf.Conventions =  'CF-1.8 UGRID-1.0 Deltares-0.10'
        proj = netf.createVariable('projected_coordinate_system','i4')
        proj.epsg = 28992
        proj.grid_mapping_name = "Amersfoort / RD New" #"Rijksdriehoeksstelsel"
        proj.longitude_of_prime_meridian = 0.0#; // double
        proj.semi_major_axis = 6377397.155#; // double
        proj.semi_minor_axis = 6356078.962818189#; // double
        proj.inverse_flattening = 299.1528128#; // double
        proj.proj4_params = "+prdoj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs"
        proj.EPSG_code = "EPSG:28992"
        proj.value = "value is equal to EPSG code"
        proj.wkt = "PROJCS[\"Amersfoort / RD New\",\n    GEOGCS[\"Amersfoort\",\n        DATUM[\"Amersfoort\",\n            SPHEROID[\"Bessel 1841\",6377397.155,299.1528128,\n"             
        netf.close()   

    @staticmethod
    def _add_unique_coupler_item(
        coupler: ET.Element,
        source_name: str,
        target_name: str,
        seen_pairs: set[tuple[str, str]],
        gn_brackets: str,
    ) -> None:
        if source_name is None or target_name is None:
            return

        pair = (source_name, target_name)
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)

        item = ET.SubElement(coupler, gn_brackets + "item")
        item.text = ""
        item.tail = "\n"

        source = ET.SubElement(item, gn_brackets + "sourceName")
        source.text = source_name
        source.tail = "\n"

        target = ET.SubElement(item, gn_brackets + "targetName")
        target.text = target_name
        target.tail = "\n"

    #@validate_arguments(config=ConfigDict(arbitrary_types_allowed=True))
    def write_dimrconfig(
        self, fm: FMModel, rr_model: DRRModel = None, rtc_model: DRTCModel = None
    ) -> None:
        """Construct the contents of a DIMR-config file, using elements from FM, RTC and RR models.

        Args:
            fm (instance of class FM-model): Hydrolib-core model object. When only fm is set, only a 1d2d model is generated.
            rr_model (instance of a DRRModel object, optional): Class containig all information from the RR-model. Defaults to None.
            rtc_model (instance of a DRTCmodel, optional): Class containig all information from the RTC-model. Defaults to None.
        """
        # initiate flags for RR and RTC blocks
        RR = False
        RTC = False
        if rr_model is not None:
            RR = True
        if rtc_model is not None:
            RTC = True

        generalname = "http://schemas.deltares.nl/dimr"  # Config
        xsi_name = "http://www.w3.org/2001/XMLSchema-instance"
        gn_brackets = "{" + generalname + "}"

        # registering namespaces
        ET.register_namespace("", generalname)
        ET.register_namespace("xsi", xsi_name)

        # Parsing xml file
        configfile = ET.parse(self.template_dir / "dimr_config.xml")

        myroot = configfile.getroot()
        myroot[1][1].text = fm.filepath.parts[-2]
        myroot[1][2].text = fm.filepath.name

        # control blocks
        control = ET.Element(gn_brackets + "control")
        control.text = ""
        control.tail = "\n"
        myroot.insert(1, control)

        if RR or RTC:
            parallel = ET.SubElement(control, gn_brackets + "parallel")
            parallel.tail = "\n"
            parallel.text = ""

        if RTC:
            startgrouprtc = ET.SubElement(parallel, gn_brackets + "startGroup")
            startgrouprtc.text = ""
            startgrouprtc.tail = "\n"

            timertc = ET.SubElement(startgrouprtc, gn_brackets + "time")
            timertc.text = f"{0} {rtc_model.time_settings['step']} {fm.time.tstop}"
            timertc.tail = "\n"
            if bool(rtc_model.pid_controllers) or bool(rtc_model.interval_controllers):
                couplerrtcrr = ET.SubElement(startgrouprtc, gn_brackets + "coupler")
                couplerrtcrr.attrib = {"name": "flow_to_rtc"}
                couplerrtcrr.tail = "\n"

            startrtc = ET.SubElement(startgrouprtc, gn_brackets + "start")
            startrtc.attrib = {"name": "Real_Time_Control"}
            startrtc.tail = "\n"
            couplerrtc = ET.SubElement(startgrouprtc, gn_brackets + "coupler")
            couplerrtc.attrib = {"name": "rtc_to_flow"}
            couplerrtc.tail = "\n"

        if RR:
            startgrouprr = ET.SubElement(parallel, gn_brackets + "startGroup")
            startgrouprr.text = ""
            startgrouprr.tail = "\n"

            timerr = ET.SubElement(startgrouprr, gn_brackets + "time")
            timerr.text = (
                f"{0} {rr_model.d3b_parameters['Timestepsize']} {fm.time.tstop}"
            )
            timerr.tail = "\n"
            couplerrrf = ET.SubElement(startgrouprr, gn_brackets + "coupler")
            couplerrrf.attrib = {"name": "flow_to_rr"}
            couplerrrf.tail = "\n"
            startrr = ET.SubElement(startgrouprr, gn_brackets + "start")
            startrr.attrib = {"name": "RR"}
            startrr.tail = "\n"
            couplerfrr = ET.SubElement(startgrouprr, gn_brackets + "coupler")
            couplerfrr.attrib = {"name": "rr_to_flow"}
            couplerfrr.tail = "\n"
        if RR or RTC:
            startfm = ET.SubElement(parallel, gn_brackets + "start")
        else:
            startfm = ET.SubElement(control, gn_brackets + "start")
        startfm.attrib = {"name": "DFM"}
        startfm.tail = "\n"

        # component blocks
        if RTC:
            comprtc = ET.Element(gn_brackets + "component")
            comprtc.attrib = {"name": "Real_Time_Control"}
            comprtc.tail = "\n"
            librtc = ET.SubElement(comprtc, gn_brackets + "library")
            librtc.text = "FBCTools_BMI"
            librtc.tail = "\n"
            wdrtc = ET.SubElement(comprtc, gn_brackets + "workingDir")
            wdrtc.text = "rtc"
            wdrtc.tail = "\n"
            frtc = ET.SubElement(comprtc, gn_brackets + "inputFile")
            frtc.text = "."
            frtc.tail = "\n"
            myroot.insert(2, comprtc)

        if RR:
            comprr = ET.Element(gn_brackets + "component")
            comprr.attrib = {"name": "RR"}
            comprr.tail = "\n"
            librr = ET.SubElement(comprr, gn_brackets + "library")
            librr.text = "rr_dll"
            librr.tail = "\n"
            wdrr = ET.SubElement(comprr, gn_brackets + "workingDir")
            wdrr.text = "rr"
            wdrr.tail = "\n"
            frr = ET.SubElement(comprr, gn_brackets + "inputFile")
            frr.text = "Sobek_3b.fnm"
            frr.tail = "\n"
            myroot.insert(4, comprr)

            # FM to RR coupler
            couplerfmrr = ET.Element(gn_brackets + "coupler")
            couplerfmrr.attrib = {"name": "flow_to_rr"}
            couplerfmrr.tail = "\n"

            sourcefmrr = ET.SubElement(couplerfmrr, gn_brackets + "sourceComponent")
            sourcefmrr.text = "DFM"
            sourcefmrr.tail = "\n"

            targetfmrr = ET.SubElement(couplerfmrr, gn_brackets + "targetComponent")
            targetfmrr.text = "RR"
            targetfmrr.tail = "\n"

            for i in rr_model.external_forcings.boundary_nodes.keys():
                item = ET.SubElement(couplerfmrr, gn_brackets + "item")
                item.text = ""
                item.tail = "\n"

                source = ET.SubElement(item, gn_brackets + "sourceName")
                source.text = f"laterals/{i}/water_level"
                source.tail = "\n"

                target = ET.SubElement(item, gn_brackets + "targetName")
                target.text = f"catchments/{i}/water_level"
                target.tail = "\n"

            logfmrr = ET.SubElement(couplerfmrr, gn_brackets + "logger")
            logfmrr.text = ""
            logfmrr.tail = "\n"

            wdlogfmrr = ET.SubElement(logfmrr, gn_brackets + "workingDir")
            wdlogfmrr.text = "."
            wdlogfmrr.tail = "\n"

            flogfmrr = ET.SubElement(logfmrr, gn_brackets + "outputFile")
            flogfmrr.text = "dflowfm_to_rr.nc"
            flogfmrr.tail = "\n"
            myroot.append(couplerfmrr)

        if RTC:
            # RTC to FM coupler
            couplerrtcfm = ET.Element(gn_brackets + "coupler")
            couplerrtcfm.attrib = {"name": "rtc_to_flow"}
            couplerrtcfm.tail = "\n"

            sourcertcfm = ET.SubElement(couplerrtcfm, gn_brackets + "sourceComponent")
            sourcertcfm.text = "Real_Time_Control"
            sourcertcfm.tail = "\n"

            targetrtcfm = ET.SubElement(couplerrtcfm, gn_brackets + "targetComponent")
            targetrtcfm.text = "DFM"
            targetrtcfm.tail = "\n"

            rtcdict = {
                      "Crest level (s)": ["weirs","crestLevel"],
                      "Gate lower edge level (s)": ["orifices","gateLowerEdgeLevel"],
                      "Capacity (p)": ["pumps","capacity"]
                      }

            rtc_to_flow_seen_pairs = set()

            for i in rtc_model.all_controllers.keys():
                svar=rtc_model.all_controllers[i]['steering_variable']
                self._add_unique_coupler_item(
                    couplerrtcfm,
                    f"[Output]{i}/{svar}",
                    f"{rtcdict[svar][0]}/{i}/{rtcdict[svar][1]}",
                    rtc_to_flow_seen_pairs,
                    gn_brackets,
                )

            # check if there are user-specified controller that should be included
            if rtc_model.complex_controllers is not None:
                complex_config = rtc_model.complex_controllers["dimr_config"][0]
                for block in complex_config:
                    block_tag = block.tag.split("}", 1)[-1] if block.tag.startswith("{") else block.tag
                    if block_tag == "coupler":
                        if block.attrib["name"].lower().startswith("rtc_to_flow"):
                            for iblock in block:
                                iblock_tag = iblock.tag.split("}", 1)[-1] if iblock.tag.startswith("{") else iblock.tag
                                if iblock_tag == "item":
                                    source = iblock.find(".//{*}sourceName")
                                    target = iblock.find(".//{*}targetName")
                                    self._add_unique_coupler_item(
                                        couplerrtcfm,
                                        source.text if source is not None else None,
                                        target.text if target is not None else None,
                                        rtc_to_flow_seen_pairs,
                                        gn_brackets,
                                    )

            logrtcfm = ET.SubElement(couplerrtcfm, gn_brackets + "logger")
            logrtcfm.text = ""
            logrtcfm.tail = "\n"

            wdlogrtcfm = ET.SubElement(logrtcfm, gn_brackets + "workingDir")
            wdlogrtcfm.text = "."
            wdlogrtcfm.tail = "\n"

            flogrtcfm = ET.SubElement(logrtcfm, gn_brackets + "outputFile")
            flogrtcfm.text = "rtc_to_dflowfm.nc"
            flogrtcfm.tail = "\n"

            myroot.append(couplerrtcfm)

            # the Fm to RTC coupler is not always needed. It is for PID controllers.
            coupler_exists = False
            flow_to_rtc_seen_pairs = set()
            if bool(rtc_model.pid_controllers) or bool(rtc_model.interval_controllers):
                couplerfmrtc = ET.Element(gn_brackets + "coupler")
                couplerfmrtc.attrib = {"name": "flow_to_rtc"}
                couplerfmrtc.tail = "\n"

                sourcefmrtc = ET.SubElement(
                    couplerfmrtc, gn_brackets + "sourceComponent"
                )
                sourcefmrtc.text = "DFM"
                sourcefmrtc.tail = "\n"

                targetfmrtc = ET.SubElement(
                    couplerfmrtc, gn_brackets + "targetComponent"
                )
                targetfmrtc.text = "Real_Time_Control"
                targetfmrtc.tail = "\n"

                for i in rtc_model.pid_controllers.keys():
                    if rtc_model.pid_controllers[i]['target_variable'] == 'Discharge (op)':
                        source_name = f"observations/{rtc_model.pid_controllers[i]['observation_point']}/discharge"
                    elif rtc_model.pid_controllers[i]['target_variable'] == 'Water level (op)':
                        source_name = f"observations/{rtc_model.pid_controllers[i]['observation_point']}/water_level"
                    else:
                        raise ValueError('Invalid target variable in controller: should bo discharge or water level.')
                    target_name = f"[Input]{rtc_model.pid_controllers[i]['observation_point']}/{rtc_model.pid_controllers[i]['target_variable']}"
                    self._add_unique_coupler_item(
                        couplerfmrtc,
                        source_name,
                        target_name,
                        flow_to_rtc_seen_pairs,
                        gn_brackets,
                    )

                # Loop through all interval controllers
                for i in rtc_model.interval_controllers.keys():
                    if rtc_model.interval_controllers[i]["target_variable"] == "Discharge (op)":
                        source_name = f"observations/{rtc_model.interval_controllers[i]['observation_point']}/discharge"
                    elif rtc_model.interval_controllers[i]["target_variable"] == "Water level (op)":
                        source_name = f"observations/{rtc_model.interval_controllers[i]['observation_point']}/water_level"
                    else:
                        raise ValueError("Invalid target variable in controller: should be discharge or water level.")
                    target_name = f"[Input]{rtc_model.interval_controllers[i]['observation_point']}/{rtc_model.interval_controllers[i]['target_variable']}"
                    self._add_unique_coupler_item(
                        couplerfmrtc,
                        source_name,
                        target_name,
                        flow_to_rtc_seen_pairs,
                        gn_brackets,
                    )

                coupler_exists = True
            # it could be that are no PID controllers, but there are complex controllers
            if rtc_model.complex_controllers is not None:
                complex_config = rtc_model.complex_controllers["dimr_config"][0]
                for block in complex_config:
                    block_tag = block.tag.split("}", 1)[-1] if block.tag.startswith("{") else block.tag
                    if block_tag == "coupler":
                        if block.attrib["name"].lower().startswith("flow_to_rtc"):
                            if not coupler_exists:
                                couplerfmrtc = ET.Element(gn_brackets + "coupler")
                                couplerfmrtc.attrib = {"name": "flow_to_rtc"}
                                couplerfmrtc.tail = "\n"

                                sourcefmrtc = ET.SubElement(
                                    couplerfmrtc, gn_brackets + "sourceComponent"
                                )
                                sourcefmrtc.text = "DFM"
                                sourcefmrtc.tail = "\n"

                                targetfmrtc = ET.SubElement(
                                    couplerfmrtc, gn_brackets + "targetComponent"
                                )
                                targetfmrtc.text = "Real_Time_Control"
                                targetfmrtc.tail = "\n"

                                coupler_exists = True
                            for iblock in block:
                                iblock_tag = iblock.tag.split("}", 1)[-1] if iblock.tag.startswith("{") else iblock.tag
                                if iblock_tag == "item":
                                    source = iblock.find(".//{*}sourceName")
                                    target = iblock.find(".//{*}targetName")
                                    self._add_unique_coupler_item(
                                        couplerfmrtc,
                                        source.text if source is not None else None,
                                        target.text if target is not None else None,
                                        flow_to_rtc_seen_pairs,
                                        gn_brackets,
                                    )
            # in any of those two cases, add the controllers and the logger
            if coupler_exists:
                logrtc = ET.SubElement(couplerfmrtc, gn_brackets + "logger")
                logrtc.text = ""
                logrtc.tail = "\n"

                wdlogrtc = ET.SubElement(logrtc, gn_brackets + "workingDir")
                wdlogrtc.text = "."
                wdlogrtc.tail = "\n"

                flogrtc = ET.SubElement(logrtc, gn_brackets + "outputFile")
                flogrtc.text = "dflowfm_to_rtc.nc"
                flogrtc.tail = "\n"

                myroot.append(couplerfmrtc)

        if RR:
            # finally the RR to FM coupler
            couplerrrfm = ET.Element(gn_brackets + "coupler")
            couplerrrfm.attrib = {"name": "rr_to_flow"}
            couplerrrfm.tail = "\n"

            sourcerrfm = ET.SubElement(couplerrrfm, gn_brackets + "sourceComponent")
            sourcerrfm.text = "RR"
            sourcerrfm.tail = "\n"

            targetrrfm = ET.SubElement(couplerrrfm, gn_brackets + "targetComponent")
            targetrrfm.text = "DFM"
            targetrrfm.tail = "\n"

            for i in rr_model.external_forcings.boundary_nodes.keys():
                item = ET.SubElement(couplerrrfm, gn_brackets + "item")
                item.text = ""
                item.tail = "\n"

                source = ET.SubElement(item, gn_brackets + "sourceName")
                source.text = f"catchments/{i}/water_discharge"
                source.tail = "\n"

                target = ET.SubElement(item, gn_brackets + "targetName")
                target.text = f"laterals/{i}/water_discharge"
                target.tail = "\n"

            logrrfm = ET.SubElement(couplerrrfm, gn_brackets + "logger")
            logrrfm.text = ""
            logrrfm.tail = "\n"

            wdlogrrfm = ET.SubElement(logrrfm, gn_brackets + "workingDir")
            wdlogrrfm.text = "."
            wdlogrrfm.tail = "\n"

            flogrrfm = ET.SubElement(logrrfm, gn_brackets + "outputFile")
            flogrrfm.text = "rr_to_dflowfm.nc"
            flogrrfm.tail = "\n"

            myroot.append(couplerrrfm)
        # use the function from DRTC to finish the file with the correct namespace
        DRTCModel.finish_file(myroot, configfile, self.output_path / "dimr_config.xml")
