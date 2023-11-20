from .infer import load_hutchison, load_ionic_conformers, load_rotamer, load_tautobase
from .tools import group_by_smiles, calc_r2, calc_rmsd, analyse_by_group
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Hartree, eV, kcal, mol
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


def analyse_hutchison(name: str = "new_method", methods: List = None, filter: Callable = None):
    if methods is None:
        methods = [
            "QRNN-TB",
            "SANI",
            "GFN2-xTB",
            "PM7",
            "wB97X-D/6-31G*"
        ]
    method_ref = "DLPNO-CCSD(T)/cc-pVTZ"

    data, ener = load_hutchison(name=name)

    if filter is not None:
        n_init = len(data)
        data_remain = []
        ener_remain = []
        for i in range(len(data)):
            if filter(data[i]):
                data_remain.append(data[i])
                ener_remain.append(ener[i])
        data = data_remain
        ener = np.array(ener_remain)
        n_final = len(data)
        print(f"{n_final}/{n_init} data remain after filtering")
    
    eref = np.array([i.energies[method_ref] for i in data])
    smiles = [sp.smiles for sp in data]
    smiles_grp = group_by_smiles(smiles)

    nfig = len(methods) + 1
    ncol = 2
    nrow = nfig // ncol
    if ncol * nrow < nfig:
        nrow += 1

    plt.figure(figsize=(nrow * 4, ncol * 6), dpi=100)
    for nmethod, method in enumerate(methods):
        eval = np.array([i.energies[method] for i in data])
        maes, r2s, new_eval, new_eref = analyse_by_group(eval, eref, smiles_grp)
        title = f"{method}\nmedian MAE: {np.median(maes)*HARTREE_TO_KCAL_MOL:.4f}  median R$^2$: {np.median(r2s):.4f}"

        plt.subplot(nrow, ncol, nmethod + 1)
        draw_correlation_plot(
            new_eval, new_eref, method_ref, method, title
        )
    
    plt.subplot(nrow, ncol, nmethod+2)
    maes, r2s, new_eval, new_eref = analyse_by_group(ener, eref, smiles_grp)
    title = f"{name}\nmedian MAE: {np.median(maes)*HARTREE_TO_KCAL_MOL:.4f}  median R$^2$: {np.median(r2s):.4f}"
    draw_correlation_plot(
        new_eval, new_eref, method_ref, name, title
    )
    plt.tight_layout()
    plt.savefig(f"{name}_hutchison.png")

    