import numpy as np
from dataclasses import dataclass
import json
import glob
import sch_benchmark

path = sch_benchmark.__path__[0]


@dataclass
class SinglePoint:
    title: str
    smiles: str
    positions: np.ndarray
    charge: float
    elements: list
    energies: dict


def load_hutchison_task():
    sp_data = []
    for filename in glob.glob(f"{path}/data/hutchison_conformers/*.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        for item in data:
            sp = SinglePoint(
                "hutchison_all",
                item["smiles"],
                np.array(item["positions"]),
                item["charge"],
                item["elements"],
                item["energies"],
            )
            sp_data.append(sp)
    return sp_data


def load_ionic_conformers_task():
    conf_data = []
    for filename in glob.glob(f"{path}/data/ionic_conformers/*.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        for item in data:
            conformer_grps = []
            smiles = item["smiles"]
            charge = item["charge"]
            elements = item["elements"]
            for method in [
                "QeqNN",
                "QeqNN-TB",
                "QRNN",
                "QRNN-TB",
                "SANI",
                "GFN2-xTB",
                "PM7",
                "wB97X-D/6-31G*",
            ]:
                positions = item[method]["positions"]
                energies = {method: item[method]["energy"]}
                conformer_grps.append(
                    SinglePoint(
                        method, smiles, np.array(positions), charge, elements, energies
                    )
                )
            conf_data.append(conformer_grps)
    return conf_data


def load_rotamer_task():
    rotamer_data = []
    for filename in glob.glob(f"{path}/data/rotamer_data/*.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        rotamer_grps = []
        for item in data:
            rotamer_grps.append(
                SinglePoint(
                    "rotamer_all",
                    item["smiles"],
                    np.array(item["positions"]),
                    item["charge"],
                    item["elements"],
                    item["energies"],
                )
            )
        rotamer_data.append(rotamer_grps)
    return rotamer_data

def load_tautobase_task():
    tautomers_data = []
    for filename in glob.glob(f"{path}/data/tautobase/*.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        
        tautomer_grps = []
        for item in data:
            tautomer_grps.append(SinglePoint("tautobase_all", item["smiles"], np.array(item["positions"]), item["charge"], item["elements"], item["energies"]))
        tautomers_data.append(tautomer_grps)
    return tautomers_data