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
from scipy.stats import norm, skewnorm, exponnorm, gennorm, t
from scipy.optimize import brentq
from scipy.integrate import quad
from tqdm import tqdm

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import source.pipelineprocesser as plp


def standardize(rv, param):
    if type(rv) == str:
        rv = get_rv_from_name(rv)
    mean = rv.mean(param)
    std = rv.std(param)
    return rv(param, loc=-mean / std, scale=1 / std)


def wasserstein_distance(rv):
    def func(x):
        return np.abs(rv.cdf(x) - norm.cdf(x))

    return quad(func, -np.inf, np.inf)[0]


def target_function(rv, target, metric="wasserstein"):
    if metric == "wasserstein":

        def func(param):
            return wasserstein_distance(standardize(rv, param)) - target

    else:
        raise ValueError("metric must be wasserstein")
    return func


def get_rv_from_name(rv_name):
    rv_dict = {
        "skewnorm": skewnorm,
        "exponnorm": exponnorm,
        "gennormsteep": gennorm,
        "gennormflat": gennorm,
        "t": t,
    }
    return rv_dict[rv_name]


def binary_search(rv_name, distance, metric):
    rv = get_rv_from_name(rv_name)
    func = target_function(rv, distance, metric=metric)
    if metric == "wasserstein":
        range_dict = {
            "skewnorm": (1e-4, 30.0),
            "exponnorm": (1e-4, 15.0),
            "gennormsteep": (1e-1, 2.0 - 1e-4),
            "gennormflat": (2.0 + 1e-4, 50.0),
            "t": (3.0, 200.0),
        }
    elif metric == "kl":
        range_dict = {
            "skewnorm": (1e-4, 30.0),
            "exponnorm": (1e-4, 15.0),
            "gennormsteep": (1e-1, 2.0 - 1e-4),
            "gennormflat": (2.0 + 1e-4, 50.0),
            "t": (3.0, 200.0),
        }
    return brentq(func, *range_dict[rv_name])


def standardized_rv_at_distance(distribution, distance, metric="wasserstein"):
    # if metric is kl, sensitive to adust range of binary search
    assert metric in ["wasserstein", "kl"], "metric must be wasserstein or kl"
    params_dict = {
        "skewnorm": {
            "0.03": 1.141679535895037,
            "0.06": 1.668027646656356,
            "0.09": 2.253555993158534,
            "0.12": 3.052977442461724,
            "0.15": 4.441693019739707,
        },
        "exponnorm": {
            "0.03": 0.5274333543184184,
            "0.06": 0.7361945074942922,
            "0.09": 0.9307079975424131,
            "0.12": 1.1365153042836023,
            "0.15": 1.372114598160624,
        },
        "gennormsteep": {
            "0.03": 1.685486347382175,
            "0.06": 1.446878209856004,
            "0.09": 1.2592111500311147,
            "0.12": 1.1075283854228473,
            "0.15": 0.9822742249929434,
        },
        "gennormflat": {
            "0.03": 2.4358709097539135,
            "0.06": 3.0868574329392504,
            "0.09": 4.188574703248306,
            "0.12": 6.60223527240027,
            "0.15": 23.021018170499307,
        },
        "t": {
            "0.03": 13.911718115376004,
            "0.06": 7.606345474941293,
            "0.09": 5.498186625845221,
            "0.12": 4.441398730633352,
            "0.15": 3.8067196925891835,
        },
    }
    if metric == "wasserstein" and distance in [0.03, 0.06, 0.09, 0.12, 0.15]:
        param = params_dict[distribution][str(distance)]
    else:
        param = binary_search(distribution, distance, metric=metric)

    rv = get_rv_from_name(distribution)
    return standardize(rv, param)


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
        seed: int,
        noise: str,
        distance: float,
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
        self.seed = seed
        self.noise = noise
        self.distance = distance

    def iter_experiment(self, args) -> tuple:
        X, y, seed = args
        rng = np.random.default_rng(seed)

        if self.option == "op1":
            pl = option1()
        elif self.option == "op2":
            pl = option2()
        else:
            raise ValueError("Invalid option")
        M, _ = pl(X, y)

        if len(M) == 0:
            return None
        index = rng.choice(len(M))

        try:
            _, result = pl.inference(X, y, 1.0, index, is_result=True)
        except Exception as e:
            return None
        return result

    def run_experiment(self):
        rv = standardized_rv_at_distance(self.noise, self.distance, "wasserstein")
        y_list = rv.rvs(size=(self.num_iter, self.n), random_state=self.seed)
        rng = np.random.default_rng(5000 * (self.seed + 1))
        seeds = [5000 * (self.seed + 1) + i for i in range(self.num_iter)]
        X_list = rng.normal(size=(self.num_iter, self.n, self.p))
        datasets = [(X, y, seed) for X, y, seed in zip(X_list, y_list, seeds)]
        self.results = self.experiment(datasets)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--option", type=str, default="none")  # op1 op2
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--noise", type=str, default="none")
    parser.add_argument("--distance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.noise, args.distance, args.seed)

    experiment = ExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        n=args.n,
        p=args.p,
        seed=args.seed,
        noise=args.noise,
        distance=args.distance,
    )

    experiment.run_experiment()

    result_path = f"robust/results_nongauss"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = f"{args.option}_{args.noise}_{args.distance}_{args.seed}.pkl"
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
