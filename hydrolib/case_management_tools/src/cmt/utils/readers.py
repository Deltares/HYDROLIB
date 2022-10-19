import json
from pathlib import Path
from typing import Union


def read_text(file_path: Union[str, Path]) -> str:
    if type(file_path) == str:
        file_path = Path(file_path)
    return file_path.read_text()


def read_stochastics(stochastics_json: Union[str, Path]) -> dict:
    return json.loads(read_text(stochastics_json))
