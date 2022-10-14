# coding: latin-1
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
from pydantic import validate_arguments

from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.core.io.mdu.models import FMModel


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

        if output_path is None:
            self.output_path = Path(os.path.abspath("."))
        else:
            self.output_path = output_path

        self.template_dir = Path(os.path.abspath("."))

    @validate_arguments
    def write_runbat(self) -> None:
        """Generates a run.bat to run the model in DIMR. The path to the executable is provided by the user in the class initialization, or set to the default D-Hydro installation folder."""

        with open(os.path.join(self.output_path, "run.bat"), "w") as f:
            f.write("@ echo off\n")
            f.write("set OMP_NUM_THREADS=2\n")
            f.write('call "' + str(self.run_dimr) + '"\n')

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
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
        myroot[1][1].text = "fm"
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
            if hasattr(rtc_model, "pid_controllers"):
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

            for i in rtc_model.all_controllers.keys():
                item = ET.SubElement(couplerrtcfm, gn_brackets + "item")
                item.text = ""
                item.tail = "\n"

                source = ET.SubElement(item, gn_brackets + "sourceName")
                source.text = f"[Output]{i}/Crest level (s)"
                source.tail = "\n"

                target = ET.SubElement(item, gn_brackets + "targetName")
                target.text = f"weirs/{i}/crestLevel"
                target.tail = "\n"

            # check if there are user-specified controller that should be included
            if rtc_model.complex_controllers is not None:
                complex_config = rtc_model.complex_controllers["dimr_config"][0]
                for block in complex_config:
                    if ET.tostring(block).decode().startswith("<coupler"):
                        if block.attrib["name"].lower().startswith("rtc_to_flow"):
                            for iblock in block:
                                if ET.tostring(iblock).decode().startswith("<item"):
                                    item = ET.SubElement(
                                        couplerrtcfm, gn_brackets + "item"
                                    )
                                    item.text = ""
                                    item.tail = "\n"

                                    source = ET.SubElement(
                                        item, gn_brackets + "sourceName"
                                    )
                                    source.text = iblock[0].text
                                    source.tail = "\n"

                                    target = ET.SubElement(
                                        item, gn_brackets + "targetName"
                                    )
                                    target.text = iblock[1].text
                                    target.tail = "\n"

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
            if hasattr(rtc_model, "pid_controllers"):
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
                    item = ET.SubElement(couplerfmrtc, gn_brackets + "item")
                    item.text = ""
                    item.tail = "\n"

                    source = ET.SubElement(item, gn_brackets + "sourceName")
                    source.text = f"observations/{rtc_model.pid_controllers[i]['observation_point']}/water_level"
                    source.tail = "\n"

                    target = ET.SubElement(item, gn_brackets + "targetName")
                    target.text = f"[Input]{rtc_model.pid_controllers[i]['observation_point']}/{rtc_model.pid_controllers[i]['target_variable']}"
                    target.tail = "\n"

                coupler_exists = True
            # it could be that are no PID controllers, but there are complex controllers
            if rtc_model.complex_controllers is not None:

                complex_config = rtc_model.complex_controllers["dimr_config"][0]
                for block in complex_config:
                    if ET.tostring(block).decode().startswith("<coupler"):
                        if block.attrib["name"].lower().startswith("flow_to_rtc"):
                            if not coupler_exists:
                                couplerfmrtc = ET.Element(gn_brackets + "coupler")
                                couplerfmrtc.attrib = {"name": "flow_to_rtc"}
                                couplerfmrtc.tail = "\n"
                                coupler_exists = True
                            for iblock in block:
                                if ET.tostring(iblock).decode().startswith("<item"):
                                    item = ET.SubElement(
                                        couplerfmrtc, gn_brackets + "item"
                                    )
                                    item.text = ""
                                    item.tail = "\n"

                                    source = ET.SubElement(
                                        item, gn_brackets + "sourceName"
                                    )
                                    source.text = iblock[0].text
                                    source.tail = "\n"

                                    target = ET.SubElement(
                                        item, gn_brackets + "targetName"
                                    )
                                    target.text = iblock[1].text
                                    target.tail = "\n"
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
                source.text = f"catchments/{i}/water_level"
                source.tail = "\n"

                target = ET.SubElement(item, gn_brackets + "targetName")
                target.text = f"laterals/{i}/water_level"
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
