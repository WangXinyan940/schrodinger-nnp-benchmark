from .base import SinglePoint, BaseDataSet
from .io import load_hutchison_task
from .tools import (
    calc_sp,
    calc_r2,
    center_by_elements,
    analyse_by_group,
    group_by_elements,
    center_by_groups,
    group_by_smiles,
    HARTREE_TO_KCAL_MOL,
    EV_TO_KCAL_MOL,
)
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def draw_correlation_plot(val, ref, x_name, y_name, title):
    val = val * HARTREE_TO_KCAL_MOL
    ref = ref * HARTREE_TO_KCAL_MOL

    plt.scatter(ref, val, s=5)

    x_min = min(ref)
    x_max = max(ref)
    y_min = min(val)
    y_max = max(val)
    dx = x_max - x_min
    dy = y_max - y_min
    x_min -= dx * 0.1
    x_max += dx * 0.1
    y_min -= dy * 0.1
    y_max += dy * 0.1
    vmin = min((x_min, y_min))
    vmax = max((x_max, y_max))
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)

    plt.plot([vmin, vmax], [vmin, vmax], color="red", linewidth=1)

    plt.xlabel(f"{x_name} (kcal/mol)")
    plt.ylabel(f"{y_name} (kcal/mol)")
    r2_val = calc_r2(val, ref)
    plt.title(title)
    plt.tight_layout()
    return r2_val


class Hutchison(BaseDataSet):
    def initialize(self):
        self.tasks = load_hutchison_task()
        self.method_ref = "DLPNO-CCSD(T)/cc-pVTZ"
        self.name = "hutchison"

    def inference_task(self, task, name, calculator):
        task.energies[name] = calc_sp(task, calculator)
        return task

    def analyze_method(self, method: str, tasks: List):
        elements = [i.elements for i in tasks]
        smiles = [i.smiles for i in tasks]
        # groups = group_by_elements(elements)
        groups = group_by_smiles(smiles)
        eref = np.array([i.energies[self.method_ref] for i in tasks])
        eval = np.array([i.energies[method] for i in tasks])
        eref_centered = center_by_groups(eref, groups)
        eval_centered = center_by_groups(eval, groups)
        median_ae = np.mean(np.abs(eref_centered - eval_centered)) * HARTREE_TO_KCAL_MOL
        R2 = calc_r2(eval_centered * HARTREE_TO_KCAL_MOL, eref_centered * HARTREE_TO_KCAL_MOL)
        title = f"{method}\nMAE: {median_ae:.4f} (kcal/mol)  R$^2$: {R2:.4f}"

        draw_correlation_plot(eval_centered, eref_centered, self.method_ref, method, title)

    def show(self):
        keys = self.tasks[0].energies.keys()
        for key in keys:
            print(key)