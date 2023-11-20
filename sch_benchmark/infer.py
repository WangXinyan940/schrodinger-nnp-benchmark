from .io import (
    load_hutchison_task,
    load_ionic_conformers_task,
    load_rotamer_task,
    load_tautobase_task,
)
from .tools import calc_sp, calc_opt, calc_r2, group_by_smiles, group_by_elements, analyse_by_group, calc_rmsd
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.calculator import Calculator
from tqdm import tqdm, trange
from ase.units import Hartree, eV, kcal, mol
import pickle

######################################
#
# Load ASE calculator, benchmark data
#
######################################
EV_TO_KCAL_MOL = eV / (kcal / mol)
HARTREE_TO_KCAL_MOL = Hartree / (kcal / mol)


def compute_hutchison(calculator: Calculator, name: str = "new_method", test_dataset=-1):
    methods = [
        "QRNN-TB",
        "SANI",
        "GFN2-xTB",
        "PM7",
        "wB97X-D/6-31G*",
        "DLPNO-CCSD(T)/cc-pVTZ"
    ]

    data = load_hutchison_task(test_dataset=test_dataset)
    ener = []
    eref = {}
    elems = []
    smiles = []
    for method in methods:
        eref[method] = []
    for item in tqdm(data):
        ener.append(calc_sp(item, calculator))
        elems.append(item.elements)
        smiles.append(item.smiles)
        for method in methods:
            eref[method].append(item.energies[method])
    ener = np.array(ener)
    with open(f"{name}_hutchison.pkl", "wb") as f:
        pickle.dump([data, ener], f)
        
def load_hutchison(name: str = "new_method"):
    with open(f"{name}_hutchison.pkl", "rb") as f:
        data, ener = pickle.load(f)
    return data, ener


def compute_ionic_conformers(calculator: Calculator, name: str = "new_method", test_dataset=-1):
    data = load_ionic_conformers_task(test_dataset=test_dataset)
    result = []
    for conformers in tqdm(data):
        ref = conformers["wB97X-D/6-31G*"]
        opt_result = calc_opt(ref, calculator, name=name)
        result.append(opt_result)
    with open(f"{name}_ionic_conformers.pkl", "wb") as f:
        pickle.dump([data, result], f)

def load_ionic_conformers(name: str = "new_method"):
    with open(f"{name}_ionic_conformers.pkl", "rb") as f:
        data, result = pickle.load(f)
    return data, result
    

def compute_rotamer(calculator: Calculator, name: str = "new_method", test_dataset=-1):
    data = load_rotamer_task(test_dataset=test_dataset)
    result = []
    for rotamer in tqdm(data):
        rot_scan = []
        for conf in rotamer:
            rot_scan.append(calc_sp(conf, calculator))
        result.append(rot_scan)
    with open(f"{name}_rotamer.pkl", "wb") as f:
        pickle.dump([data, result], f)

def load_rotamer(name: str = "new_method"):
    with open(f"{name}_rotamer.pkl", "rb") as f:
        data, result = pickle.load(f)
    return data, result
    

def compute_tautobase(calculator: Calculator, name: str = "new_method", test_dataset=-1):

    data = load_tautobase_task(test_dataset=test_dataset)
    result = []
    for t1, t2 in tqdm(data):
        e1, e2 = calc_sp(t1, calculator), calc_sp(t2, calculator)
        result.append([e1, e2])
    
    with open(f"{name}_tautobase.pkl", "wb") as f:
        pickle.dump([data, result], f)

def load_tautobase(name: str = "new_method"):
    with open(f"{name}_tautobase.pkl", "rb") as f:
        data, result = pickle.load(f)
    return data, result

def compute_all(calculator: Calculator, name: str = "new_method", test_dataset=-1):
    compute_hutchison(calculator, name=name, test_dataset=test_dataset)
    compute_ionic_conformers(calculator, name=name, test_dataset=test_dataset)
    compute_rotamer(calculator, name=name, test_dataset=test_dataset)
    compute_tautobase(calculator, name=name, test_dataset=test_dataset)