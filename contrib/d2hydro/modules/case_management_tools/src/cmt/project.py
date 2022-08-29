import json
import logging
import shutil
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Union

import netCDF4 as nc
import pandas as pd
from cmt.run import dhydro
from cmt.utils.modifyers import prefix_to_paths
from cmt.utils.read_his import get_timeseries
from cmt.utils.readers import read_stochastics, read_text
from cmt.utils.writers import (
    write_flow_boundaries,
    write_models,
    write_rr_conditions,
    write_stowa_buien,
)
from pydantic import BaseModel

from hydrolib.core.io.rr.models import RainfallRunoffModel
from hydrolib.core.io.mdu.models import FMModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BoundaryCondition(BaseModel):
    id: str
    name: str
    start_datetime: datetime
    timedelta: timedelta


class InitialCondition(BaseModel):
    id: str
    name: str


class Run(BaseModel):
    completed: bool = False
    success: Optional[bool]
    start_datetime: Optional[datetime]
    run_duration: Optional[timedelta]
    results_id: Optional[str]
    output_deleted: Optional[bool]


class Case(BaseModel):
    id: str
    name: str
    meteo_bc_id: str
    flow_bc_id: str
    model_id: str
    fm_output_dir: Path
    start_datetime: datetime
    simulation_period: timedelta
    run: Run = Run()


class Model(BaseModel):
    id: str
    name: str
    mdu: str
    fnm: str


class Source(BaseModel):
    nc_file: str = "his"
    id: str
    layer: str
    variable: str
    statistic: str = "max"


class Sample(BaseModel):
    id: str
    name: str
    source: Source


class Result(BaseModel):
    id: str
    statistic: str = "max"
    samples: List[Sample]
    results: Optional[dict] = {}

    def __get_sample_ids(self, nc_file, layer, variable):
        return [
            i.id
            for i in self.samples
            if (i.source.layer == layer)
            & (i.source.nc_file == nc_file)
            & (i.source.variable == variable)
        ]

    def get_results(self, output_dir, case_id, model_name):
        def __get_result(self, nc_file, layer, variable):
            ids = self.__get_sample_ids(nc_file, layer, variable)
            with nc.Dataset(
                    output_dir.joinpath(f"{model_name}_{nc_file}.nc")
                    ) as ds:
                series = get_timeseries(
                    ds,
                    long_name=variable,
                    ids=ids,
                    layer=layer,
                    statistic=self.statistic,
                )
            return series

        sets = set(
            [
                (i.source.nc_file, i.source.layer, i.source.variable)
                for i in self.samples
            ]
        )
        result = pd.concat([__get_result(self, *i) for i in sets])
        self.results[case_id] = result.to_dict()
        return result

    def write_results(self, project_dir):
        results_dir = project_dir.joinpath("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir.joinpath(f"{self.id}.json")
        results_path.write_text(json.dumps(self.results, indent=2))


class BoundaryConditions(BaseModel):
    meteo: List[BoundaryCondition] = []
    flow: List[BoundaryCondition] = []


class InitialConditions(BaseModel):
    rr: List[InitialCondition] = []
    flow: List[BoundaryCondition] = []


class Project(BaseModel):
    filepath: Optional[Path] = None
    boundary_conditions: BoundaryConditions = BoundaryConditions()
    initial_conditions: InitialConditions = InitialConditions()
    models: List[Model] = []
    cases: List[Case] = []
    results: Optional[List[Result]]

    def _init_filepath(self):
        if self.filepath.exists():
            shutil.rmtree(self.filepath)
        self.filepath = self.filepath.absolute().resolve()
        self.filepath.mkdir(parents=True, exist_ok=True)

    def _create_subdir(self, subdir):
        dir_path = Path(self.filepath) / subdir
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def get_case(self, case_id):
        return next((i for i in self.cases if i.id == case_id), None)

    def get_results(self, results_id):
        return next((i for i in self.results if i.id == results_id), None)

    def get_fm_output_dir(self, case_id):
        case = self.get_case(case_id)
        model = self.get_model(case.model_id)
        fm_dir = Path(model.mdu).parent

        return self.filepath.joinpath("cases", case_id, fm_dir, case.fm_output_dir)

    def sample_output(self, case_id):
        case = self.get_case(case_id)
        model = self.get_model(case.model_id)
        model_name = fm_dir = Path(model.mdu).parent
        output_dir = self.get_fm_output_dir(case_id)
        his_nc_path = output_dir.joinpath(f"{model_name}_his.nc")

    def delete_output(self, case_id):
        case = self.get_case(case_id)
        output_dir = self.get_fm_output_dir(case_id)
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
                case.run.output_deleted = True
            except:
                logger.warning(f"Failed to remove: {output_dir}")

    def get_cases(self):
        return [i.id for i in self.cases]

    def get_model(self, model_id):
        return next((i for i in self.models if i.id == model_id))

    def run_post(self, case_id, delete_output=False):
        case = self.get_case(case_id)
        if case.run.results_id is not None:
            results = self.get_results(case.run.results_id)
            model = self.get_model(case.model_id)
            model_name = Path(model.mdu).stem
            output_dir = self.get_fm_output_dir(case_id)
            results.get_results(output_dir, case_id, model_name)
            results.write_results(self.filepath)

            if delete_output:
                self.delete_output(case_id)

    def run_cases(self, case_ids, workers=None, delete_output=False):
        logger.info(f"Start running batch: #cases: {len(case_ids)}, #workers {workers}")
        tp = ThreadPool(workers)
        for idx, case_id in enumerate(case_ids):
            progress = f" ({idx+1}/{len(case_ids)})"
            tp.apply_async(
                self.run_case, (case_id, progress, False, True, delete_output)
            )
        tp.close()
        tp.join()
        self.write_manifest()
        logger.info("Finished running batch")

    def run_case(
        self,
        case_id,
        progress="",
        stream_output=False,
        returncode=True,
        delete_output=False,
    ):
        start_datetime = datetime.now()
        case = self.get_case(case_id)
        logger.info(f"Start running case{progress}: '{case_id}'")
        if case is not None:
            case.run.start_datetime = start_datetime
            cwd = self.filepath.joinpath(f"cases/{case_id}").absolute().resolve()
            try:
                rc = dhydro.run(cwd, stream_output=stream_output, returncode=returncode)
            except:
                rc = 1

            if rc == 0:
                case.run.completed = True
                case.run.success = True
                run_duration = datetime.now() - start_datetime
                case.run.run_duration = run_duration
                logger.info(
                    f"finished running case: '{case_id}' in {run_duration.total_seconds()} secs"
                )
                self.run_post(case_id, delete_output=delete_output)
            else:
                case.run.completed = False
                case.run.success = False
                logger.error(f"Case: '{case_id}' crashed")
            self.write_manifest()

    def get_flow_boundary(self, bc_id):
        return next((i for i in self.boundary_conditions.flow if i.id == bc_id))

    def get_meteo_boundary(self, bc_id):
        return next((i for i in self.boundary_conditions.meteo if i.id == bc_id))

    def write_manifest(self):
        index_json = self.filepath / "manifest.json"

        index_json.write_text(
            self.json(
                exclude={"filepath"},
                indent=1,
            )
        )

    @classmethod
    def from_manifest(cls, manifest_json: Union[str, Path]):
        cls = cls.parse_raw(read_text(manifest_json))
        cls.filepath = Path(manifest_json).parent.absolute().resolve()
        return cls

    def from_stochastics(self, stochastics_json: Path):
        def __convert_to_samples(results):
            def __strip_file_name(filename):
                if filename.endswith("his.nc"):
                    return "his"
                elif filename.endswith("map.nc"):
                    return "map"

            def __get_source(sample):
                return Source(
                    id=sample["id"],
                    nc_file=__strip_file_name(sample["filename"]),
                    layer="station",
                    variable=sample["parameter"],
                )

            def __get_sample(sample):
                source = __get_source(sample)
                return Sample(id=sample["id"], name=sample["name"], source=source)

            return [__get_sample(i) for i in results]

        self._init_filepath()
        stochastics_dict = read_stochastics(stochastics_json)
        src_dir = stochastics_json.parent
        # do boundaries
        if "boundary_conditions" in stochastics_dict.keys():
            boundaries = stochastics_dict["boundary_conditions"]

            # do meteo boundaries
            if "meteo" in boundaries.keys():
                events = boundaries["meteo"]
                meteo_bc_path = self._create_subdir(r"boundary_conditions/meteo")
                events = [i for i in events if "stowa_bui" in i.keys()]
                bcs = write_stowa_buien(meteo_bc_path, events)
                self.boundary_conditions.meteo = [BoundaryCondition(**i) for i in bcs]

            # do flow boundaries
            if "flow" in boundaries.keys():
                flow_bc_path = self._create_subdir(r"boundary_conditions/flow")
                events = boundaries["flow"]
                bcs = write_flow_boundaries(flow_bc_path, events)
                self.boundary_conditions.flow = [BoundaryCondition(**i) for i in bcs]

        # do initial conditions
        if "initial_conditions" in stochastics_dict.keys():
            initial_conditions = stochastics_dict["initial_conditions"]
            if "rr" in initial_conditions.keys():
                rr_ini_path = self._create_subdir(r"initial_conditions/rr")
                conditions = initial_conditions["rr"]
                write_rr_conditions(rr_ini_path, conditions, src_dir)
                self.initial_conditions.rr = [
                    InitialCondition(id=i["id"], name=i["name"]) for i in conditions
                ]

        # do models
        if "models" in stochastics_dict.keys():
            models_path = self._create_subdir(r"models")
            models = stochastics_dict["models"]
            write_models(models_path, models, src_dir)
            self.models = [
                Model(id=i["id"], name=i["name"], mdu=i["mdu_file"], fnm=i["fnm_file"])
                for i in models
            ]
            self.results = [
                Result(
                    id=i["id"],
                    statistic=i["results"][0]["filter"].lower(),
                    samples=__convert_to_samples(i["results"]),
                )
                for i in models
            ]

        if "cases" in stochastics_dict.keys():
            cases_path = self._create_subdir(r"cases")
            cases = stochastics_dict["cases"]
            for model_id in set([i["model_id"] for i in cases]):
                model = self.get_model(model_id)
                cases_subset = [i for i in cases if i["model_id"] == model_id]
                mdu = FMModel(filepath=self.filepath / "models" / model.id / model.mdu)
                mdu_parent = mdu.filepath.parent
                mdu.general.pathsrelativetoparent = True
                mdu.time.tunit = "S"
                dimr_path = self.filepath / "models" / model.id / "dimr_config.xml"
                rtc_dir = self.filepath / "models" / model.id / "rtc"
                for section in ["geometry", "restart", "output"]:
                    mdu_section = getattr(mdu, section)
                    for i in mdu_section.__fields__.keys():
                        if not i in ["outputdir"]:
                            if hasattr(mdu_section, i):
                                setattr(
                                    mdu_section,
                                    i,
                                    prefix_to_paths(
                                        getattr(mdu_section, i),
                                        prefix=f"../../../models/{model_id}/{mdu_parent.name}",
                                    ),
                                )
                ext = mdu.external_forcing.extforcefilenew
                ext_path = str(ext.filepath)
                fnm = RainfallRunoffModel(
                    filepath=self.filepath / "models" / model.id / model.fnm
                )
                fnm_parent = fnm.filepath.parent
                fnm_prefix = f"../../../models/{model_id}/{fnm_parent.name}"
                exclude_suffices = [
                    ".log",
                    ".out",
                    ".dbg",
                    ".his",
                    ".abr",
                    ".msg",
                    ".fnm",
                ]
                exclude_names = ["sobek3b_progress.txt", "RSRR_OUT", "RR-ready"]
                for i in fnm.__fields__.keys():
                    setattr(
                        fnm,
                        i,
                        prefix_to_paths(
                            getattr(fnm, i),
                            prefix=fnm_prefix,
                            exclude_suffices=exclude_suffices,
                            exclude_names=exclude_names,
                        ),
                    )
                for i in cases_subset:

                    # read start_datetime
                    # set paths
                    flow_bc = self.get_flow_boundary(i["flow_bc_id"])
                    flow_bc_path = Path(
                        f"../../../boundary_conditions/flow/{flow_bc.id}.bc"
                    )
                    dst_path = cases_path / rf"{i['id']}"
                    for bc in ext.boundary:
                        bc.forcingfile.filepath = flow_bc_path

                    meteo_bc = self.get_meteo_boundary(i["meteo_bc_id"])
                    bui_path = Path(
                        f"../../../boundary_conditions/meteo/{meteo_bc.id}.bui"
                    )
                    evp_path = Path(
                        f"../../../boundary_conditions/meteo/{meteo_bc.id}.evp"
                    )

                    start_datetime = max(
                        flow_bc.start_datetime, meteo_bc.start_datetime
                    )
                    end_datetime = min(
                        flow_bc.start_datetime + flow_bc.timedelta,
                        meteo_bc.start_datetime + meteo_bc.timedelta,
                    )
                    time_delta = end_datetime - start_datetime

                    # write flow model
                    mdu_path = dst_path / model.mdu
                    ext.save(mdu_path.parent / ext_path)
                    ext.filepath.write_text(
                        ext.filepath.read_text().replace("locationFile =\n", "")
                    )
                    ext.filepath = ext_path
                    mdu.time.refdate = int(start_datetime.strftime("%Y%m%d"))
                    time = start_datetime.time()
                    mdu.time.tstart = timedelta(
                        hours=time.hour, minutes=time.minute, seconds=time.second
                    ).total_seconds()
                    mdu.time.tstop = time_delta.total_seconds()
                    mdu.save(dst_path / model.mdu)
                    fm_output_dir = mdu.output.outputdir

                    # copy rtc model
                    shutil.copytree(rtc_dir, dst_path / "rtc")

                    # write fnm model
                    fnm.bui_file.filepath = bui_path
                    fnm.verdampings_file = evp_path
                    fnm.save(dst_path / model.fnm)

                    # write dimr
                    shutil.copy(dimr_path, dst_path / "dimr_config.xml")
                    self.cases += [
                        Case(
                            id=i["id"],
                            name=i["name"],
                            meteo_bc_id=i["meteo_bc_id"],
                            flow_bc_id=i["flow_bc_id"],
                            model_id=i["model_id"],
                            fm_output_dir=fm_output_dir,
                            start_datetime=start_datetime,
                            simulation_period=time_delta,
                        )
                    ]
                    self.cases[-1].run.results_id = model_id

                    # write run.bat
                    bat_path = self.filepath / "models" / model.id / "run.bat"
                    if bat_path.exists():
                        shutil.copy(bat_path, dst_path / "run.bat")

        self.write_manifest()

        return self
