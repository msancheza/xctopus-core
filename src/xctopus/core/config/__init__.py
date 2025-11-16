import yaml
from pathlib import Path

def load_config(path: str):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
