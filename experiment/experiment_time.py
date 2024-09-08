"""Module for time experiments."""

import argparse
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time
from typing import Literal, cast

import numpy as np
from sicore import SelectiveInferenceResult  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from experiment.utils import (
    Results,
    option1,
    option1_parallel,
    option1_serial,
    option2,
)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

warnings.simplefilter("ignore")


class TimeExperimentPipeline:
    """Experiment class for the time experiments."""

    def __init__(
        self,
        num_results: int,
        num_worker: int,
        option: Literal["op1", "op2", "op1_parallel", "op1_serial"],
        n: int,
        d: int,
        seed: int,
    ) -> None:
        """Initialize the experiment."""
        self.num_results = num_results
        self.num_iter = int(num_results * 1.1)
        self.num_worker = num_worker
        self.option = option
        self.n = n
        self.d = d
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
        rng = np.random.default_rng([seed, self.n, self.d, self.seed])

        for _ in range(1000):
            X = rng.normal(size=(self.n, self.d))
            y = rng.normal(size=self.n)
            nan_mask = rng.choice(self.n, rng.binomial(self.n, 0.03), replace=False)
            y[nan_mask] = np.nan

            match self.option:
                case "op1":
                    manager = option1()
                case "op2":
                    manager = option2()
                case "op1_parallel":
                    manager = option1_parallel()
                case "op1_serial":
                    manager = option1_serial()

            M, _ = manager(X, y)
            if len(M) == 0:
                continue
            test_index = int(rng.choice(len(M)))

            try:
                start = time()
                _, result = manager.inference(
                    X,
                    y,
                    1.0,
                    test_index=test_index,
                    retain_result=True,
                )
                result = cast(SelectiveInferenceResult, result)
                elapsed = time() - start
                _, oc_p_value = manager.inference(
                    X,
                    y,
                    1.0,
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
        full_results = self.experiment(list(range(self.num_iter)))
        self.results = Results(
            results=[result[0] for result in full_results],
            oc_p_values=[result[1] for result in full_results],
            times=[result[2] for result in full_results],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--option", type=str, default="none")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--d", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.option)
    print(args.n, args.d, args.seed)

    experiment = TimeExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        n=args.n,
        d=args.d,
        seed=args.seed,
    )
    experiment.run_experiment()

    dir_path = Path("results_time")
    dir_path.mkdir(parents=True, exist_ok=True)

    results_file_path = dir_path / f"{args.option}_{args.n}_{args.d}_{args.seed}.pkl"
    with results_file_path.open("wb") as f:
        pickle.dump(experiment.results, f)
