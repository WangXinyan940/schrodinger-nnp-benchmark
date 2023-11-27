import numpy as np
from dataclasses import dataclass
import json
import glob
import sch_benchmark
from typing import Callable, List
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import Pool
from functools import partial
from enum import Enum

path = sch_benchmark.__path__[0]

@dataclass
class SinglePoint:
    title: str
    smiles: str
    positions: np.ndarray
    charge: float
    elements: list
    energies: dict


class BaseDataSet:

    class OptMethod(Enum):
        ASE = 1
        GEOMETRY = 2

    def __init__(self):
        self.name = "Base"
        self.tasks = []
        self.method_ref = "wB97X-D/6-31G*"
        self.opt_method = self.OptMethod.ASE
        self.initialize()

    def inference(self, name, calculator, parallel: bool = False, nthreads: int = -1):
        len_tasks = len(self.tasks)
        if not parallel:
            for n in trange(len_tasks):
                i = self.tasks[n]
                self.tasks[n] = self.inference_task(i, name, calculator)
        else:
            if nthreads <= 0:
                n_proc = multiprocessing.cpu_count()
                nthreads = int(n_proc / 2)
            with Pool(processes = nthreads) as pool:
                results = list(tqdm(pool.imap(partial(self.inference_task, name = name, calculator = calculator), self.tasks), total=len_tasks))
            self.tasks = results


    def analyse(self, methods, figure: str = "{name}.png", filter: Callable = None):
        nfig = len(methods)
        ncol = 2
        nrow = nfig // ncol
        if ncol * nrow < nfig:
            nrow += 1

        plt.figure(figsize=(nrow * 4, ncol * 6), dpi=100)

        if filter is not None:
            n_init = len(self.tasks)
            tasks_remain = []
            for i in range(len(self.tasks)):
                if filter(self.tasks[i]):
                    tasks_remain.append(self.tasks[i])
            tasks = tasks_remain
            n_final = len(tasks)
            print(f"{n_final}/{n_init} data remain after filtering")
        else:
            tasks = self.tasks

        for nmethod, method in enumerate(methods):
            plt.subplot(nrow, ncol, nmethod + 1)
            self.analyse_method(method, tasks)
        plt.tight_layout()
        plt.savefig(figure.format(name=self.name))

    def initialize(self):
        raise NotImplementedError("Method [initialize] is not implemented")

    def inference_task(self, task, name, calculator):
        raise NotImplementedError("Method [inference_task] is not implemented")

    def analyse_method(self, method, tasks: List):
        raise NotImplementedError("Method [analyse_method] is not implemented")
