from .base import SinglePoint, BaseDataSet
from .io import load_ionic_conformers_task
from .tools import (
    calc_sp,
    calc_opt,
    calc_rmsd,
    calc_r2,
    analyse_by_group,
    group_by_smiles,
    HARTREE_TO_KCAL_MOL,
    EV_TO_KCAL_MOL,
)
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def _anal_ionic_conf(sp_cal, sp_ref, cal_method, ref_method, group):
    grps = list(set(group))
    geom_rmsd = []
    rmsd_all, r2_all = [], []
    for grp in grps:
        idx = np.where(np.array(group) == grp)[0]
        ener_calc, ener_ref = [], []
        for i in idx:
            rmsd = calc_rmsd(sp_cal[i].positions, sp_ref[i].positions)
            geom_rmsd.append(rmsd)
            ener_calc.append(sp_cal[i].energies[cal_method])
            ener_ref.append(sp_ref[i].energies[ref_method])
        if len(ener_calc) < 3:
            print(f"WARNING: group size is {len(ener_calc)}")
            continue
        ener_calc = np.array(ener_calc)
        ener_ref = np.array(ener_ref)
        ener_calc = ener_calc - ener_calc.mean()
        ener_ref = ener_ref - ener_ref.mean()
        rmsd = np.sqrt(np.power(ener_calc - ener_ref, 2).mean())
        r2 = calc_r2(ener_calc, ener_ref)
        rmsd_all.append(rmsd)
        r2_all.append(r2)
    mean_rmsd = np.mean(rmsd_all) * HARTREE_TO_KCAL_MOL
    median_r2 = np.median(r2_all)
    return mean_rmsd, median_r2, geom_rmsd


class IonicConformer(BaseDataSet):
    def initialize(self):
        self.tasks = load_ionic_conformers_task()
        self.method_ref = "wB97X-D/6-31G*"
        self.name = "ionic_conformer"

    def inference_task(self, task, name, calculator):
        task[name] = calc_opt(
            task[self.method_ref], calculator, name, opt_method=self.opt_method
        )
        return task

    def analyse_method(self, method: str, tasks: List):
        sp_val = [i[method] for i in tasks]
        sp_ref = [i[self.method_ref] for i in tasks]

        smiles = [i[self.method_ref].smiles for i in tasks]
        smiles_grp = group_by_smiles(smiles)

        mean_rmsd, median_r2, geom_rmsd = _anal_ionic_conf(
            sp_val, sp_ref, method, self.method_ref, smiles_grp
        )
        title = f"{method}\ngeom mean RMSD: {np.mean(geom_rmsd):.4f}\nener mean RMSD: {mean_rmsd:.4f}  ener median R$^2$: {median_r2:.4f}"

        plt.hist(geom_rmsd, bins=20)
        plt.xlabel("geom RMSD (Å)")
        plt.title(title)
