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

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import source.pipelineprocesser as plp


def option1():
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


def option1_cv():
    X, y = plp.make_dataset()
    y = plp.mean_value_imputation(X, y)

    O = plp.soft_ipod(X, y, 0.02, {0.02, 0.018})
    X, y = plp.remove_outliers(X, y, O)

    M = plp.marginal_screening(X, y, 5, {3, 5})
    X = plp.extract_features(X, M)

    M1 = plp.stepwise_feature_selection(X, y, 3, {2, 3})
    M2 = plp.lasso(X, y, 0.08, {0.08, 0.12})
    M = plp.union(M1, M2)
    return plp.make_pipeline(output=M)


def option2():
    X, y = plp.make_dataset()
    y = plp.definite_regression_imputation(X, y)

    M = plp.marginal_screening(X, y, 5)
    X = plp.extract_features(X, M)

    O = plp.cook_distance(X, y, 3.0)
    X, y = plp.remove_outliers(X, y, O)

    M1 = plp.stepwise_feature_selection(X, y, 3)
    M2 = plp.lasso(X, y, 0.08)
    M = plp.intersection(M1, M2)
    return plp.make_pipeline(output=M)


def option2_cv():
    X, y = plp.make_dataset()
    y = plp.definite_regression_imputation(X, y)

    M = plp.marginal_screening(X, y, 5, {3, 5})
    X = plp.extract_features(X, M)

    O = plp.cook_distance(X, y, 3.0, {2.0, 3.0})
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


class ExperimentPipeline(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        mode: str,
        option: str,
        n: int,
        seed: int,
    ):
        super().__init__(
            num_iter=int(num_results * 1.12),
            num_results=num_results,
            num_worker=num_worker,
        )
        self.num_results = num_results
        self.mode = mode
        self.option = option
        self.n = n
        self.seed = seed

    def iter_experiment(self, args) -> tuple:
        X, y, seed = args
        rng = np.random.default_rng(seed)

        pl = None
        if self.option == "op1":
            pl = option1()
        elif self.option == "op2":
            pl = option2()
        else:
            flag = True
            if self.option == "op1cv":
                pl = option1_cv()
            elif self.option == "op2cv":
                pl = option2_cv()
            else:
                flag = False
            if flag:
                pl.tune(X, y, n_iter=16, cv=5, random_state=seed)  # fix seed

        if pl is not None:
            M, _ = pl(X, y)
        else:
            if self.option != "op12cv":
                raise ValueError("Invalid option")
            pl = plp.make_pipelines(option1_cv(), option2_cv())
            pl.tune(
                X, y, n_iters=[16, 16], cv=5, random_state=seed
            )  # not n_iter but n_iters for MultiPipeline, fix seed
            M, _ = pl(X, y)

        if len(M) == 0:
            return (None, None)

        try:
            index = rng.choice(len(M))
            _, para_result = pl.inference(
                X, y, 1.0, index, is_result=True, over_conditioning=False
            )
            _, oc_result = pl.inference(
                X, y, 1.0, index, is_result=True, over_conditioning=True
            )
        except Exception as e:
            # print(e)
            return None
        return (para_result, oc_result)

    def run_experiment(self):
        with open("real/table_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
        X, y = dataset[self.mode]

        args = []
        rng = np.random.default_rng(self.seed)
        seeds = [2000 * (self.seed + 1) + i for i in range(self.num_iter)]
        for seed in seeds:
            index = rng.choice(X.shape[0], size=self.n, replace=False)
            args.append((X[index, :], y[index], seed))

        self.results = self.experiment(args)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--mode", type=str, default="none")
    parser.add_argument("--option", type=str, default="none")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.mode, args.option, args.n)

    experiment = ExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        mode=args.mode,
        option=args.option,
        n=args.n,
        seed=args.seed,
    )

    experiment.run_experiment()

    result_path = f"real/results"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name = f"{args.option}_{args.mode}_{args.n}_{args.seed}.pkl"
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
