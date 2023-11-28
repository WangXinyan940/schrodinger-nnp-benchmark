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

    def inference(self, name, calculator):
        len_tasks = len(self.tasks)
        for n in trange(len_tasks):
            i = self.tasks[n]
            self.tasks[n] = self.inference_task(i, name, calculator)

    def analyze(self, methods, figure: str = "{name}.png", filter: Callable = None):
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
            self.analyze_method(method, tasks)
        plt.tight_layout()
        plt.savefig(figure.format(name=self.name))

    @classmethod
    def split(cls, dataset, fold: int = 5) -> List:
        from copy import deepcopy

        datasets = []
        for i in range(fold):
            datasets.append(deepcopy(dataset))
            datasets[i].tasks = []
        for i in range(len(dataset.tasks)):
            datasets[i % fold].tasks.append(dataset.tasks[i])
        return datasets

    @classmethod
    def merge(cls, datasets: List):
        from copy import deepcopy

        dataset = deepcopy(datasets[0])
        dataset.tasks = []
        for i in range(len(datasets)):
            dataset.tasks += datasets[i].tasks
        return dataset

    def initialize(self):
        raise NotImplementedError("Method [initialize] is not implemented")

    def inference_task(self, task, name, calculator):
        raise NotImplementedError("Method [inference_task] is not implemented")

    def analyse_method(self, method, tasks: List):
        raise NotImplementedError("Method [analyse_method] is not implemented")

    def save(self, filename: str):
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        import pickle

        with open(filename, "rb") as f:
            return pickle.load(f)
