from .io import (
    load_hutchison_task,
    load_ionic_conformers_task,
    load_rotamer_task,
    load_tautobase_task,
)
from .tools import calc_sp, calc_opt
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.calculator import Calculator
from tqdm import tqdm, trange

######################################
#
# Load ASE calculator, benchmark data
#
######################################


def compute_hutchison(calculator: Calculator):
    data = load_hutchison_task()
    ener = []
    for item in tqdm(data):
        ener.append(calc_sp(item, calculator))
    return ener

def compute_ionic_conformers(calculator: Calculator):
    data = load_ionic_conformers_task()
    result = []
    for conformers in tqdm(data):
        ref = conformers[-1]
        opt_result = calc_opt(ref, calculator)
        result.append(opt_result)
    return result

def compute_rotamer(calculator: Calculator):
    data = load_rotamer_task()
    result = []
    for rotamer in tqdm(data):
        rot_scan = []
        for conf in rotamer:
            rot_scan.append(calc_sp(conf, calculator))
        result.append(rot_scan)
    return result

def compute_tautobase(calculator: Calculator):
    data = load_tautobase_task()
    result = []
    for t1, t2 in tqdm(data):
        e1, e2 = calc_sp(t1, calculator), calc_sp(t2, calculator)
        result.append([e1, e2])
    return result