import numpy as np
from dataclasses import dataclass
import json
import glob
import sch_benchmark
from typing import Callable, List
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.name = "Base"
        self.tasks = []
        self.method_ref = "wB97X-D/6-31G*"
        self.initialize()

    def setRefMethod(self, method):
        self.method_ref = method

    def getRefMethod(self):
        return self.method_ref

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def inference(self, name, calculator):
        for n, i in enumerate(self.tasks):
            self.tasks[n] = self.inference_task(name, calculator, i)

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

    def inference_task(self, name, calculator, task):
        raise NotImplementedError("Method [inference_task] is not implemented")

    def analyse_method(self, method, tasks: List):
        raise NotImplementedError("Method [analyse_method] is not implemented")