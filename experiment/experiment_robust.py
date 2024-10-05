"""Module for robust experiments."""

import argparse
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time
from typing import Literal, cast

import numpy as np
from sicore import (  # type: ignore[import]
    SelectiveInferenceResult,
    generate_non_gaussian_rv,
)
from tqdm import tqdm  # type: ignore[import]

from experiment.utils import Results, option1, option1_multi, option2, option2_multi

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

warnings.simplefilter("ignore")


class RobustExperimentPipeline:
    """Experiment class for the data analysis pipeline."""

    def __init__(
        self,
        num_results: int,
        num_worker: int,
        option: Literal["op1", "op2", "all_cv"],
        n: int,
        d: int,
        robust_type: Literal[
            "estimated",
            "skewnorm",
            "exponnorm",
            "gennormsteep",
            "gennormflat",
            "t",
        ],
        distance: float,
        seed: int,
    ) -> None:
        """Initialize the experiment."""
        self.num_results = num_results
        self.num_iter = int(num_results * 1.1)
        self.num_worker = num_worker
        self.option = option
        self.n = n
        self.d = d
        self.robust_type = robust_type
        if robust_type in ["skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"]:
            self.rv = generate_non_gaussian_rv(robust_type, distance)
        self.seed = seed

    def experiment(
        self,
        seeds: list[int],
    ) -> list[tuple[SelectiveInferenceResult, float, float]]:
        """Conduct the experiment in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, seeds), total=self.num_iter),
            )
        results = [result for result in results if result is not None]
        return results[: self.num_results]

    def iter_experiment(
        self,
        seed: int,
    ) -> tuple[SelectiveInferenceResult, float, float] | None:
        """Iterate the experiment."""
        rng = np.random.default_rng(seed)

        for _ in range(1000):
            X = rng.normal(size=(self.n, self.d))
            match self.robust_type:
                case "estimated":
                    y = rng.normal(size=self.n)
                    sigma = None
                case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
                    seed_ = rng.integers(0, 2**32 - 1)
                    y = self.rv.rvs(size=self.n, random_state=seed_)
                    sigma = 1.0
            nan_mask = rng.choice(self.n, rng.binomial(self.n, 0.03), replace=False)
            y[nan_mask] = np.nan

            match self.option:
                case "op1":
                    manager = option1()
                case "op2":
                    manager = option2()
                case "all_cv":
                    manager = option1_multi() | option2_multi()
                    manager.tune(X, y, random_state=seed)

            M, _ = manager(X, y)
            if len(M) == 0:
                continue
            test_index = int(rng.choice(len(M)))

            try:
                start = time()
                _, result = manager.inference(
                    X,
                    y,
                    sigma,
                    test_index=test_index,
                    retain_result=True,
                )
                result = cast(SelectiveInferenceResult, result)
                elapsed = time() - start
                _, oc_p_value = manager.inference(
                    X,
                    y,
                    sigma,
                    test_index=test_index,
                    inference_mode="over_conditioning",
                )
                oc_p_value = cast(float, oc_p_value)
            except Exception as e:  # noqa: BLE001
                print(e)
                return None
            else:
                return result, oc_p_value, elapsed
        return None

    def run_experiment(self) -> None:
        """Conduct the experiments and save the results."""
        seeds = [5000 * (self.seed + 1) + i for i in range(self.num_iter)]
        full_results = self.experiment(seeds)
        self.results = Results(
            results=[result[0] for result in full_results],
            oc_p_values=[result[1] for result in full_results],
            times=[result[2] for result in full_results],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--option", type=str, default="all_cv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--robust_type", type=str, default="estimated")
    parser.add_argument("--distance", type=float, default=0.00)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.option)
    print(args.n, args.d, args.robust_type, args.distance, args.seed)

    experiment = RobustExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        n=args.n,
        d=args.d,
        robust_type=args.robust_type,
        distance=args.distance,
        seed=args.seed,
    )
    experiment.run_experiment()

    dir_path = Path("results_robust") / args.robust_type
    dir_path.mkdir(parents=True, exist_ok=True)

    match args.robust_type:
        case "estimated":
            results_file_path = (
                dir_path / f"{args.option}_{args.n}_{args.d}_{args.seed}.pkl"
            )
        case "skewnorm" | "exponnorm" | "gennormsteep" | "gennormflat" | "t":
            results_file_path = (
                dir_path / f"{args.option}_{args.distance}_{args.seed}.pkl"
            )

    with results_file_path.open("wb") as f:
        pickle.dump(experiment.results, f)
