from .io import SinglePoint
from typing import Tuple
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS
import numpy as np
from ase.units import Hartree, eV


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


def calc_opt(sp: SinglePoint, calculator: Calculator) -> SinglePoint:
    atoms = sp_to_atoms(sp)
    atoms.calc = calculator
    # do some opt work
    dyn = LBFGS(atoms)
    dyn.run(fmax=0.05)
    e = atoms.get_potential_energy() * eV / Hartree
    pos = atoms.get_positions()
    return SinglePoint(
        sp.title, sp.smiles, pos, sp.charge, sp.elements, {"calculator": e}
    )


def calc_sp(sp: SinglePoint, calculator: Calculator) -> float:
    atoms = sp_to_atoms(sp)
    atoms.calc = calculator
    e = atoms.get_potential_energy() * eV / Hartree
    return e
