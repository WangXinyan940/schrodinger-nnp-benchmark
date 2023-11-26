from .base import SinglePoint, BaseDataSet
from .io import load_rotamer_task
from .tools import calc_sp, calc_opt, calc_r2, analyse_by_group, group_by_smiles, HARTREE_TO_KCAL_MOL, EV_TO_KCAL_MOL
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def _anal_rotamer(data, ecalc, calc_method, ref_method):
    ediff = []
    for nrot in range(len(data)):
        rot_calc, rot_ref = [], []
        for nframe, frame in enumerate(data[nrot]):
            rot_calc.append(ecalc[nrot][nframe])
            rot_ref.append(frame.energies[ref_method])
        rot_calc = np.array(rot_calc)
        rot_ref = np.array(rot_ref)
        rot_calc = rot_calc - rot_calc.mean()
        rot_ref = rot_ref - rot_ref.mean()
        delta = rot_calc - rot_ref
        for item in delta:
            ediff.append(item)
    return np.array(ediff)


class Rotamer(BaseDataSet):

    def initialize(self):
        self.tasks = load_rotamer_task()
        self.method_ref = "wB97X-D/6-31G*"
        self.name = "rotamer"

    def inference_task(self, name, calculator, task):
        new_task = []
        for item in task:
            item.energies[name] = calc_sp(item, calculator)
            new_task.append(item)
        return new_task

    def analyse_method(self, method: str, tasks: List):
        ecalc = [[j.energies[method] for j in i] for i in tasks]
        ediff = _anal_rotamer(tasks, ecalc, method, self.method_ref)
        mae = np.abs(ediff).mean() * HARTREE_TO_KCAL_MOL
        title = f"{method}\nenergy mean absolute error: {mae:.4f}"

        plt.hist(ediff * HARTREE_TO_KCAL_MOL, bins=20)
        plt.xlabel("delta E (kcal/mol)")
        plt.title(title)
