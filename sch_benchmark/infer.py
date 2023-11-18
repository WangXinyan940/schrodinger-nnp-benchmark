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

######################################
#
# Load ASE calculator, benchmark data
#
######################################
EV_TO_KCAL_MOL = eV / (kcal / mol)
HARTREE_TO_KCAL_MOL = Hartree / (kcal / mol)

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
    return r2_val


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
    for method in methods:
        eref[method] = np.array(eref[method])
    group = group_by_smiles(smiles)

    plt.figure(figsize=(15, 10), dpi=100)
    for ii in range(3):
        for jj in range(2):
            if ii == 2 and jj == 1:
                continue
            plt.subplot(2, 3, ii * 2 + jj + 1)
            val = eref[methods[ii * 2 + jj]]
            ref = eref[methods[-1]]
            mae, r2, val, ref = analyse_by_group(val, ref, group)
            title = f"{methods[ii * 2 + jj]}\nmedian MAE: {np.median(mae)*HARTREE_TO_KCAL_MOL:.4f}  median R$^2$: {np.median(r2):.4f}"
            draw_correlation_plot(val, ref, methods[-1], methods[ii * 2 + jj], title)
            print(title)
    plt.subplot(2, 3, 6)
    val = ener
    ref = eref[methods[-1]]
    mae, r2, val, ref = analyse_by_group(val, ref, group)
    title = f"{name}\nmedian MAE: {np.median(mae)*HARTREE_TO_KCAL_MOL:.4f}  median R$^2$: {np.median(r2):.4f}"
    draw_correlation_plot(val, ref, methods[-1], name, title)
    print(title)
    plt.tight_layout()
    plt.savefig("hutchison.png")
    return ener, eref

def analyse_ionic_conf(sp_cal, sp_ref, cal_method, ref_method, group):
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

def compute_ionic_conformers(calculator: Calculator, name: str = "new_method", test_dataset=-1):
    methods = [
        "QRNN",
        "QRNN-TB",
        "SANI",
        "GFN2-xTB",
        "PM7",
        "wB97X-D/6-31G*",
    ]
    data = load_ionic_conformers_task(test_dataset=test_dataset)
    result = []
    for conformers in tqdm(data):
        ref = conformers[methods[-1]]
        opt_result = calc_opt(ref, calculator, name=name)
        result.append(opt_result)
    group = group_by_smiles([item.smiles for item in result])
    grps = list(set(group))
    plt.figure(figsize=(15, 10), dpi=100)
    for nmethod, method in enumerate(methods[:-1]):
        sp_cal = [conformers[method] for conformers in data]
        sp_ref = [conformers[methods[-1]] for conformers in data]
        mean_rmsd, median_r2, geom_rmsd = analyse_ionic_conf(sp_cal, sp_ref, method, methods[-1], group)
        title = f"{method}\ngeom mean RMSD: {np.mean(geom_rmsd):.4f}\nener mean RMSD: {mean_rmsd:.4f}  ener median R$^2$: {median_r2:.4f}"
        print(title)

        plt.subplot(2, 3, nmethod + 1)
        plt.hist(geom_rmsd, bins=20)
        plt.xlabel("geom RMSD (Å)")
        plt.title(title)

    # self
    plt.subplot(2, 3, 6)
    sp_cal = result
    sp_ref = [conformers[methods[-1]] for conformers in data]
    mean_rmsd, median_r2, geom_rmsd = analyse_ionic_conf(sp_cal, sp_ref, name, methods[-1], group)
    title = f"{name}\ngeom mean RMSD: {np.mean(geom_rmsd):.4f}\nener mean RMSD: {mean_rmsd:.4f}  ener median R$^2$: {median_r2:.4f}"
    print(title)
    plt.hist(geom_rmsd, bins=20)
    plt.xlabel("geom RMSD (Å)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("ionic_conformers.png")
    return result

def compute_rotamer(calculator: Calculator, test_dataset=-1):
    data = load_rotamer_task(test_dataset=test_dataset)
    result = []
    for rotamer in tqdm(data):
        rot_scan = []
        for conf in rotamer:
            rot_scan.append(calc_sp(conf, calculator))
        result.append(rot_scan)
    return result

def compute_tautobase(calculator: Calculator, test_dataset=-1):
    data = load_tautobase_task(test_dataset=test_dataset)
    result = []
    for t1, t2 in tqdm(data):
        e1, e2 = calc_sp(t1, calculator), calc_sp(t2, calculator)
        result.append([e1, e2])
    return result