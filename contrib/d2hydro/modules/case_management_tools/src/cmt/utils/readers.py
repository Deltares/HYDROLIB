import json
from pathlib import Path

def read_stochastics(stochastics_json: Path):
    return json.loads(stochastics_json.read_text())
