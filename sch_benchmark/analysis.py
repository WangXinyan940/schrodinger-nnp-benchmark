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


def analyse_hutchison(
    name: str = "new_method",
    figname: str = None,
    methods: List = None,
    filter: Callable = None,
):
    if methods is None:
        methods = ["QRNN-TB", "SANI", "GFN2-xTB", "PM7", "wB97X-D/6-31G*"]
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
        draw_correlation_plot(new_eval, new_eref, method_ref, method, title)

    plt.subplot(nrow, ncol, nmethod + 2)
    maes, r2s, new_eval, new_eref = analyse_by_group(ener, eref, smiles_grp)
    title = f"{name}\nmedian MAE: {np.median(maes)*HARTREE_TO_KCAL_MOL:.4f}  median R$^2$: {np.median(r2s):.4f}"
    draw_correlation_plot(new_eval, new_eref, method_ref, name, title)
    plt.tight_layout()
    if figname is None:
        figname = f"{name}_hutchison.png"
    plt.savefig(figname)


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


def analyse_ionic_conformers(
    name: str = "new_method",
    figname: str = None,
    methods: List = None,
    filter: Callable = None,
):
    if methods is None:
        methods = ["QRNN-TB", "SANI", "GFN2-xTB", "PM7"]
    method_ref = "wB97X-D/6-31G*"
    data, result = load_ionic_conformers(name=name)
    if filter is not None:
        n_init = len(data)
        data_remain = []
        result_remain = []
        for i in range(len(result)):
            if filter(result[i]):
                data_remain.append(data[i])
                result_remain.append(result[i])
        data = data_remain
        result = result_remain
        n_final = len(data)
        print(f"{n_final}/{n_init} data remain after filtering")
    smiles = [sp[method_ref].smiles for sp in data]
    smiles_grp = group_by_smiles(smiles)

    nfig = len(methods) + 1
    ncol = 2
    nrow = nfig // ncol
    if ncol * nrow < nfig:
        nrow += 1

    plt.figure(figsize=(nrow * 4, ncol * 6), dpi=100)
    for nmethod, method in enumerate(methods):
        sp_val = [i[method] for i in data]
        sp_ref = [i[method_ref] for i in data]

        mean_rmsd, median_r2, geom_rmsd = _anal_ionic_conf(
            sp_val, sp_ref, method, method_ref, smiles_grp
        )
        title = f"{method}\ngeom mean RMSD: {np.mean(geom_rmsd):.4f}\nener mean RMSD: {mean_rmsd:.4f}  ener median R$^2$: {median_r2:.4f}"

        plt.subplot(nrow, ncol, nmethod + 1)
        plt.hist(geom_rmsd, bins=20)
        plt.xlabel("geom RMSD (Å)")
        plt.title(title)

    # self
    plt.subplot(nrow, ncol, nmethod + 2)
    sp_val = result
    sp_ref = [i[method_ref] for i in data]
    mean_rmsd, median_r2, geom_rmsd = _anal_ionic_conf(
        sp_val, sp_ref, name, method_ref, smiles_grp
    )
    title = f"{name}\ngeom mean RMSD: {np.mean(geom_rmsd):.4f}\nener mean RMSD: {mean_rmsd:.4f}  ener median R$^2$: {median_r2:.4f}"
    plt.hist(geom_rmsd, bins=20)
    plt.xlabel("geom RMSD (Å)")
    plt.title(title)
    plt.tight_layout()
    if figname is None:
        figname = f"{name}_ionic_conformers.png"
    plt.savefig(figname)


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


def analyse_rotamer(
    name: str = "new_method",
    figname: str = None,
    methods: List = None,
    filter: Callable = None,
):
    if methods is None:
        methods = ["QRNN-TB", "SANI", "GFN2-xTB", "PM7"]
    method_ref = "wB97X-D/6-31G*"
    data, result = load_rotamer(name=name)
    if filter is not None:
        n_init = len(data)
        data_remain = []
        result_remain = []
        for i in range(len(data)):
            if filter(data[i][0]):
                data_remain.append(data[i])
                result_remain.append(result[i])
        data = data_remain
        result = result_remain
        n_final = len(data)
        print(f"{n_final}/{n_init} data remain after filtering")

    nfig = len(methods) + 1
    ncol = 2
    nrow = nfig // ncol
    if ncol * nrow < nfig:
        nrow += 1

    plt.figure(figsize=(nrow * 4, ncol * 6), dpi=100)
    for nmethod, method in enumerate(methods):
        ecalc = [[j.energies[method] for j in i] for i in data]
        ediff = _anal_rotamer(data, ecalc, method, method_ref)
        mae = np.abs(ediff).mean() * HARTREE_TO_KCAL_MOL
        title = f"{method}\nenergy mean absolute error: {mae:.4f}"

        plt.subplot(nrow, ncol, nmethod + 1)
        plt.hist(ediff * HARTREE_TO_KCAL_MOL, bins=20)
        plt.xlabel("delta E (kcal/mol)")
        plt.title(title)

    # self
    plt.subplot(nrow, ncol, nmethod + 2)
    ecalc = result
    ediff = _anal_rotamer(data, ecalc, name, method_ref)
    mae = np.abs(ediff).mean() * HARTREE_TO_KCAL_MOL
    title = f"{name}\nenergy mean absolute error: {mae:.4f}"
    plt.hist(ediff * HARTREE_TO_KCAL_MOL, bins=20)
    plt.xlabel("delta E (kcal/mol)")
    plt.title(title)
    plt.tight_layout()
    if figname is None:
        figname = f"{name}_rotamer.png"
    plt.savefig(figname)


def _anal_tautomer(data, ecalc, calc_method, ref_method):
    ref_delta, calc_delta = [], []
    for npair, (t1, t2) in enumerate(data):
        ref_delta.append(t1.energies[ref_method] - t2.energies[ref_method])
        calc_delta.append(ecalc[npair][0] - ecalc[npair][1])
    ref_delta = np.array(ref_delta)
    calc_delta = np.array(calc_delta)
    diff = calc_delta - ref_delta
    return diff


def analyse_tatobase(
    name: str = "new_method",
    figname: str = None,
    methods: List = None,
    filter: Callable = None,
):
    if methods is None:
        methods = ["SANI", "QRNN", "QRNN-TB", "GFN2-xTB", "PM7"]
    method_ref = "wB97X-D/6-31G*"
    data, result = load_tautobase(name=name)
    if filter is not None:
        n_init = len(data)
        data_remain = []
        result_remain = []
        for i in range(len(data)):
            if filter(data[i][0]):
                data_remain.append(data[i])
                result_remain.append(result[i])
        data = data_remain
        result = result_remain
        n_final = len(data)
        print(f"{n_final}/{n_init} data remain after filtering")

    nfig = len(methods) + 1
    ncol = 2
    nrow = nfig // ncol
    if ncol * nrow < nfig:
        nrow += 1

    plt.figure(figsize=(nrow * 4, ncol * 6), dpi=100)
    for nmethod, method in enumerate(methods):
        ecalc = [[i[0].energies[method], i[1].energies[method]] for i in data]
        diff = _anal_tautomer(data, ecalc, method, method_ref)
        mae = np.abs(diff).mean() * HARTREE_TO_KCAL_MOL
        title = f"{method}\ndE mean absolute error: {mae:.4f}"

        plt.subplot(nrow, ncol, nmethod + 1)
        plt.hist(diff * HARTREE_TO_KCAL_MOL, bins=20)
        plt.xlabel("delta E (kcal/mol)")
        plt.title(title)

    # self
    diff = _anal_tautomer(data, result, name, method_ref)
    mae = np.abs(diff).mean() * HARTREE_TO_KCAL_MOL
    title = f"{name}\ndE mean absolute error: {mae:.4f}"
    plt.subplot(nrow, ncol, nmethod + 2)
    plt.hist(diff * HARTREE_TO_KCAL_MOL, bins=20)
    plt.xlabel("delta E (kcal/mol)")
    plt.title(title)

    plt.tight_layout()
    if figname is None:
        figname = f"{name}_tautobase.png"
    plt.savefig(figname)
