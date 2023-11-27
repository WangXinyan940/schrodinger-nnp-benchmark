from .base import SinglePoint, BaseDataSet
from .io import load_tautobase_task
from .tools import calc_sp, calc_opt, calc_r2, analyse_by_group, group_by_smiles, HARTREE_TO_KCAL_MOL, EV_TO_KCAL_MOL
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def _anal_tautomer(data, ecalc, calc_method, ref_method):
    ref_delta, calc_delta = [], []
    for npair, (t1, t2) in enumerate(data):
        ref_delta.append(t1.energies[ref_method] - t2.energies[ref_method])
        calc_delta.append(ecalc[npair][0] - ecalc[npair][1])
    ref_delta = np.array(ref_delta)
    calc_delta = np.array(calc_delta)
    diff = calc_delta - ref_delta
    return diff


class Tautomer(BaseDataSet):

    def initialize(self):
        self.tasks = load_tautobase_task()
        self.method_ref = "wB97X-D/6-31G*"
        self.name = "tautomer"

    def inference_task(self, task, name, calculator):
        new_task = []
        for item in task:
            item.energies[name] = calc_sp(item, calculator)
            new_task.append(item)
        return new_task

    def analyse_method(self, method: str, tasks: List):
        ecalc = [[i[0].energies[method], i[1].energies[method]] for i in tasks]
        diff = _anal_tautomer(tasks, ecalc, method, self.method_ref)
        mae = np.abs(diff).mean() * HARTREE_TO_KCAL_MOL
        title = f"{method}\ndE mean absolute error: {mae:.4f}"

        plt.hist(diff * HARTREE_TO_KCAL_MOL, bins=20)
        plt.xlabel("delta E (kcal/mol)")
        plt.title(title)
