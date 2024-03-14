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
from scipy.stats import norm
from tqdm import tqdm

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import source.pipelineprocesser as plp


def option1():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.cook_distance(X, y, 3.0)
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 5)
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


def option2():
    X, y = plp.make_dataset()
    y = plp.definite_regression_imputation(X, y)

    M = plp.marginal_screening(X, y, 5)
    X = plp.extract_features(X, M)

    O = plp.dffits(X, y, 3.0)
    X, y = plp.remove_outliers(X, y, O)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.intersection(M1, M2)
    return plp.make_pipeline(output=M)


def option1_cv():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.cook_distance(X, y, 3.0, {2.0, 3.0})
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 5, {3, 5})
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3, {2, 3})
    M2 = plp.lasso(X, y, 0.08, {0.08, 0.12})
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


def option2_cv():
    X, y = plp.make_dataset()
    y = plp.definite_regression_imputation(X, y)

    M = plp.marginal_screening(X, y, 5, {3, 5})
    X = plp.extract_features(X, M)

    O = plp.dffits(X, y, 3.0, {2.0, 3.0})
    X, y = plp.remove_outliers(X, y, O)

    M1 = plp.stepwise_feature_selection(X, y, 3, {2, 3})
    M2 = plp.lasso(X, y, 0.08, {0.08, 0.12})
    M = plp.intersection(M1, M2)
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


class ExperimentDS(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        mode: str,
        n: int,
        p: int,
        delta: float,
        seed: int,
    ):
        super().__init__(
            num_iter=int(num_results * 1.02),
            num_results=num_results,
            num_worker=num_worker,
        )
        self.num_results = num_results
        self.mode = mode
        self.n = n
        self.p = p
        self.delta = delta
        self.seed = seed

    def iter_experiment(self, args) -> tuple:
        seed = args
        rng = np.random.default_rng(seed)

        for _ in range(1000):
            X = rng.normal(size=(self.n, self.p))
            noise = rng.normal(size=self.n)

            beta = np.zeros(self.p)
            beta[:3] = self.delta
            y = X @ beta + noise
            num_missing = rng.binomial(self.n, 0.03)
            mask = rng.choice(self.n, num_missing, replace=False)
            # y[mask] = np.nan

            pl = None
            if self.mode == "op1":
                pl = option1()
            elif self.mode == "op2":
                pl = option2()
            elif self.mode == "op1cv":
                pl = option1_cv()
            elif self.mode == "op2cv":
                pl = option2_cv()
            else:
                pass

            if pl is not None:
                if self.mode in ["op1cv", "op2cv"]:
                    pl.tune(
                        X[: self.n // 2, :],
                        y[: self.n // 2],
                        n_iter=16,
                        cv=5,
                        random_state=seed,
                    )
                M, _ = pl(X[: self.n // 2, :], y[: self.n // 2])
                if len(M) == 0:
                    continue
                index = rng.choice(len(M))
                if self.delta != 0.0 and M[index] not in range(3):
                    continue
            else:
                if self.mode != "op1and2":
                    raise ValueError("Invalid cv mode")
                mpls = plp.make_pipelines(option1_cv(), option2_cv())
                mpls.tune(
                    X[: self.n // 2, :],
                    y[: self.n // 2],
                    n_iters=16,
                    cv=5,
                    random_state=seed,
                )
                M, _ = mpls(X[: self.n // 2, :], y[: self.n // 2])
                if len(M) == 0:
                    continue
                index = rng.choice(len(M))
                if self.delta != 0.0 and M[index] not in range(3):
                    continue

            y = y[self.n // 2 :]
            X = X[self.n // 2 :, M]
            nan_mask = np.isnan(y)
            y = y[~nan_mask]
            X = X[~nan_mask, :]

            etas = np.linalg.inv(X.T @ X) @ X.T
            eta = etas[index]
            stat = eta @ y / np.sqrt(eta @ eta)
            return 2 * norm.cdf(-np.abs(stat))
        return None

    def run_experiment(self):
        seeds = [2000 * (self.seed + 1) + i for i in range(self.num_iter)]
        self.results = self.experiment(seeds)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=100)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--mode", type=str, default="op1and2")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.n, args.p, args.delta, args.seed)

    experiment = ExperimentDS(
        num_results=args.num_results,
        num_worker=args.num_worker,
        mode=args.mode,
        n=args.n,
        p=args.p,
        delta=args.delta,
        seed=args.seed,
    )

    experiment.run_experiment()

    result_path = f"results_ds"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = f"{args.mode}_{args.n}_{args.p}_{args.delta}_{args.seed}.pkl"
    print(args.n, args.p, args.delta, args.seed)
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
