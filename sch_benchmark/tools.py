from .io import SinglePoint
from typing import Tuple, List
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS
import numpy as np
from ase.units import Hartree, eV, kcal, mol


def sp_to_atoms(sp: SinglePoint) -> Atoms:
    charges = np.zeros(len(sp.elements))
    charges[0] = sp.charge
    return Atoms(
        symbols=sp.elements,
        positions=sp.positions,
        cell=np.identity(3) * 100,
        pbc=[False, False, False],
        charges=charges,
    )

def group_by_smiles(smiles):
    grp = {}
    ngrp = []
    for smi in smiles:
        if smi not in grp:
            grp[smi] = len(grp)
        ngrp.append(grp[smi])
    return ngrp

def elem_to_str(elem):
    el_sorted = sorted(elem)
    tmp = {}
    for e in el_sorted:
        if e not in tmp:
            tmp[e] = 0
        tmp[e] += 1
    ret = ""
    for key in tmp:
        ret += f"{key}{tmp[key]}"
    return ret

def group_by_elements(elem: list) -> list:
    grp = {}
    ngrp = []
    for el in elem:
        el_str = elem_to_str(el)
        if el_str not in grp:
            grp[el_str] = len(grp)
        ngrp.append(grp[el_str])
    return ngrp

def analyse_by_group(val: np.ndarray, ref: np.ndarray, groups: List[int]):
    diff_grps = list(set(groups))
    new_val = np.zeros_like(val)
    new_ref = np.zeros_like(ref)
    groups_ = np.array(groups)
    r2_grp, mae_grp = [], []
    for grp in diff_grps:
        idx = np.where(groups_ == grp)[0]
        if len(idx) < 3:
            print(f"WARNING: group size is {len(idx)}")
            continue
        val_part = val[idx] - val[idx].mean()
        ref_part = ref[idx] - ref[idx].mean()
        new_ref[idx] = ref_part
        new_val[idx] = val_part
        r2_part = calc_r2(val_part, ref_part)
        mae_part = np.abs(val_part - ref_part).mean()
        r2_grp.append(r2_part)
        mae_grp.append(mae_part)
    return np.array(mae_part), np.array(r2_part), new_val, new_ref


def calc_opt(sp: SinglePoint, calculator: Calculator, name="calculator") -> SinglePoint:
    atoms = sp_to_atoms(sp)
    atoms.calc = calculator
    # do some opt work
    dyn = LBFGS(atoms)
    dyn.run(fmax=0.05)
    e = atoms.get_potential_energy() * eV / Hartree
    pos = atoms.get_positions()
    return SinglePoint(
        sp.title, sp.smiles, pos, sp.charge, sp.elements, {name: e}
    )


def calc_sp(sp: SinglePoint, calculator: Calculator) -> float:
    atoms = sp_to_atoms(sp)
    atoms.calc = calculator
    e = atoms.get_potential_energy() * eV / Hartree
    return e

def calc_r2(y1: np.ndarray, y2: np.ndarray) -> float:
    return np.corrcoef(y1, y2)[0, 1] ** 2

def compute_aligned_mobile(mobile, target):

    mu1 = mobile.mean(0)
    mu2 = target.mean(0)

    mobile = mobile - mu1
    target = target - mu2

    correlation_matrix = np.dot(np.transpose(mobile), target)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(V) * np.linalg.det(W_tr)) < 0.0
    if is_reflection:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)

    translation = mu2 - mu1.dot(rotation)

    return mobile.dot(rotation) + translation

def calc_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
    # align
    pos1 -= pos1.mean(axis=0)
    pos2 -= pos2.mean(axis=0)
    pos1 = compute_aligned_mobile(pos1, pos2)
    # calc rmsd
    return np.sqrt(np.power(pos1 - pos2, 2).sum(axis=1).mean())