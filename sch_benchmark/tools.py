from .io import SinglePoint
from .base import BaseDataSet
from typing import Tuple, List
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.bfgslinesearch import BFGSLineSearch
import numpy as np
from ase.units import Hartree, eV, kcal, mol
from rdkit import Chem
import tempfile
from pathlib import Path
try:
    import geometric
    import geometric.molecule
    from geometric.internal import Dihedral, PrimitiveInternalCoordinates
except ImportError as e:
    import warnings
    warnings.warn(f"GEOMETRIC is not installed, use ASE instead. The optimization accuracy may be affected and the accuracy would be overestimated.")

EV_TO_KCAL_MOL = eV / (kcal / mol)
HARTREE_TO_KCAL_MOL = Hartree / (kcal / mol)


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
        if len(idx) < 2:
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


def sp2mol(sp: SinglePoint):
    molecule = geometric.molecule.Molecule()
    molecule.elem = sp.elements
    molecule.xyzs = sp.positions.reshape((1, -1, 3))  # In Angstrom
    return molecule


def calc_opt(sp: SinglePoint, calculator: Calculator, name="calculator", opt_method = BaseDataSet.OptMethod.ASE) -> SinglePoint:
    if opt_method == BaseDataSet.OptMethod.GEOMETRY:
        mol = sp2mol(sp)
        engine = geometric.ase_engine.EngineASE(mol, calculator)
        initial_charges = np.zeros(len(engine.ase_atoms))
        initial_charges[0] = sp.charge
        engine.ase_atoms.set_initial_charges(initial_charges)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            prefix = tmpdir / "opt"
            m = geometric.optimize.run_optimizer(
                customengine=engine,
                check=1,
                prefix=str(prefix.absolute()),
                convergence_set="GAU_LOOSE",
                hessian="never",
                maxiter=128
            )
            e = m.qm_energies[-1] * eV / Hartree
            pos = m.xyzs[-1]

    elif opt_method == BaseDataSet.OptMethod.ASE:
        atoms = sp_to_atoms(sp)
        atoms.calc = calculator
        # do some opt work
        dyn = BFGSLineSearch(atoms)
        dyn.run(fmax=0.001)
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


class XTBCmdCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, method='GFN2-xTB', **kwargs):
        self.method = method
        super().__init__(**kwargs)

    def calculate(self, atoms, properties=['energy', 'forces'], system_changes = None):
        import subprocess
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xyz_path = tmp_path / 'tmp.xyz'
            # write atoms to xyz file
            ase.io.write(xyz_path, atoms)
            charge = int(atoms.get_initial_charges().sum())
            if self.method == 'GFN2-xTB':
                gfn = "2"
            else:
                gfn = "1"
            cmd = f'xtb tmp.xyz --sp --chrg {charge} --gfn {gfn} --grad'
            subprocess.run(cmd, shell=True, cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # read output
            with open(tmp_path / "tmp.engrad", "r") as f:
                text = f.readlines()
            lines = [line.strip() for line in text if line[0] != "#"]
            natoms = int(lines[0])
            energy = float(lines[1]) # energy in har
            grads = np.array([float(i) for i in lines[2:2+3*natoms]]).reshape((natoms, 3)) # force in Hartree/Bohr
            forces = - grads
            e_eV = energy * 27.2114
            forces_eV_A = forces * 51.422067
            self.results['energy'] = e_eV
            self.results['forces'] = forces_eV_A