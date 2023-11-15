import numpy as np
from dataclasses import dataclass
import json
import glob


@dataclass
class SinglePoint:
    smiles: str
    positions: np.ndarray
    charge: float
    elements: list
    energies: dict

def load_sp_data():
    import sch_benchmark
    path = sch_benchmark.__path__
    sp_data = []

    for filename in glob.glob(f"{path}/data/*/*.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        for item in data:
            if "positions" not in item:
                continue
            sp = SinglePoint(item["smiles"], np.array(item["positions"]), item["charge"], item["elements"], item["energies"])
            sp_data.append(sp)
    return sp_data