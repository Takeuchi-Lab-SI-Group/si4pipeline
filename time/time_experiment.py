#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm
from time import time

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import source.pipelineprocesser as plp


def option():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.soft_ipod(X, y, 0.02)
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 5)
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


def option_parallel():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.soft_ipod(X, y, 0.02)
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 20)
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 5)
    M2 = plp.lasso(X, y, 0.06)
    M = plp.union(M1, M2)

    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


def option_serial():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.soft_ipod(X, y, 0.02)
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 20)
    X = plp.extract_features(X, M)

    M = plp.stepwise_feature_selection(X, y, 10)
    X = plp.extract_features(X, M)

    M = plp.lasso(X, y, 0.04)
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


class PararellExperiment(metaclass=ABCMeta):
    def __init__(self, num_iter: int, num_results: int, num_worker: int):
        self.num_iter = num_iter
        self.num_results = num_results
        self.num_worker = num_worker

    @abstractmethod
    def iter_experiment(self, args) -> tuple:
        pass

    def experiment(self, dataset: list) -> list:
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, dataset), total=self.num_iter)
            )
        results = [result for result in results if result is not None]
        return results[: self.num_results]

    @abstractmethod
    def run_experiment(self):
        pass


class ExperimentPipeline(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        option: str,
        n: int,
        p: int,
        delta: float,
        oc: str,
        seed: int,
    ):
        super().__init__(
            num_iter=int(num_results * 1.12),
            num_results=num_results,
            num_worker=num_worker,
        )
        self.num_results = num_results
        self.option = option
        self.n = n
        self.p = p
        self.delta = delta
        self.oc = True if oc == "oc" else False
        self.seed = seed

    def iter_experiment(self, args) -> tuple:
        seed = args
        rng = np.random.default_rng(seed)

        for _ in range(1000):  # repeat while true feature is not selected
            X = rng.normal(size=(self.n, self.p))
            noise = rng.normal(size=self.n)

            beta = np.zeros(self.p)
            beta[:3] = self.delta
            y = X @ beta + noise
            num_missing = rng.binomial(self.n, 0.03)
            mask = rng.choice(self.n, num_missing, replace=False)
            y[mask] = np.nan

            pl = None
            if self.option == "default":
                pl = option()
            elif self.option == "parallel":
                pl = option_parallel()
            elif self.option == "serial":
                pl = option_serial()

            try:
                M, _ = pl(X, y)
            except Exception:
                continue

            if len(M) == 0:
                continue

            index = rng.choice(len(M))
            if self.delta == 0.0 or M[index] in range(3):
                try:
                    start = time()
                    _, result = pl.inference(
                        X, y, 1.0, index, is_result=True, over_conditioning=self.oc
                    )
                    end = time()
                    return result, end - start
                except Exception as e:
                    # print(e)
                    return None
        return None

    def run_experiment(self):
        seeds = [5000 * (self.seed + 1) + i for i in range(self.num_iter)]
        self.results = self.experiment(seeds)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=50)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument(
        "--option", type=str, default="default"
    )  # default, parallel, serial
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--p", type=int, default=80)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--oc", type=str, default="none")  # oc or none
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.n, args.p, args.option, args.seed)

    experiment = ExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        n=args.n,
        p=args.p,
        delta=args.delta,
        oc=args.oc,
        seed=args.seed,
    )

    experiment.run_experiment()

    if args.oc == "oc":
        result_path = f"time/results_{args.option}_oc"
    else:
        result_path = f"time/results_{args.option}"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = f"{args.n}_{args.p}_{args.delta}_{args.seed}.pkl"
    print(args.n, args.p, args.delta, args.seed)
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
