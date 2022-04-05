from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import sys
from cmt.utils.readers import read_stochastics
from cmt.utils.writers import write_stowa_buien, write_flow_boundaries


class BoundaryCondition(BaseModel):
    id: str
    name: str


class Case(BaseModel):
    id: str
    name: str
    meteo_bc_id: str
    flow_bc_id: str
    model_id: str
    filepath: Path


class Model(BaseModel):
    id: str
    name: str
    filepath: Path


class Models(BaseModel):
    List[Model]


class Cases(BaseModel):
    List[Case]


class BoundaryConditions(BaseModel):
    meteo: List[BoundaryCondition] = []
    flow: List[BoundaryCondition] = []


class Project(BaseModel):
    filepath: Optional[Path] = None
    boundary_conditions: BoundaryConditions = BoundaryConditions()
    models: Models = Models()
    Cases: Cases = Cases()

    def _create_subdir(self, subdir):
        dir_path = Path(self.filepath) / subdir
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def from_stochastics(self, stochastics_json: Path):
        stochastics_dict = read_stochastics(stochastics_json)
        if "boundary_conditions" in stochastics_dict.keys():
            boundaries = stochastics_dict["boundary_conditions"]

            # do meteo boundaries
            if "meteo" in boundaries.keys():
                events = boundaries["meteo"]
                meteo_path = self._create_subdir(r"boundary_conditions/meteo")
                events = [
                    i for i in events if "stowa_bui" in i.keys()
                    ]
                write_stowa_buien(meteo_path, events)
                self.boundary_conditions.meteo = [
                    BoundaryCondition(id=i["id"], name=i["name"]) for i in events
                    ]

            # do flow boundaries
            if "flow" in boundaries.keys():
                flow_path = self._create_subdir(r"boundary_conditions/flow")
                events = boundaries["flow"]
                write_flow_boundaries(flow_path, events)
                self.boundary_conditions.flow = [
                    BoundaryCondition(id=i["id"], name=i["name"]) for i in events
                    ]
        return self
