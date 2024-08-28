"""Module for main experiments."""

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
from sicore.core.base import InfiniteLoopError  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from experiment.utils import Results, option1, option1_multi, option2, option2_multi

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

warnings.simplefilter("ignore")


class ExperimentPipeline:
    """Experiment class for the data analysis pipeline."""

    def __init__(
        self,
        num_results: int,
        num_worker: int,
        option: Literal["op1", "op2", "all_cv"],
        n: int,
        d: int,
        delta: float,
        seed: int,
    ) -> None:
        """Initialize the experiment."""
        self.num_results = num_results
        self.num_iter = int(num_results * 1.1)
        self.num_worker = num_worker
        self.option = option
        self.n = n
        self.d = d
        self.delta = delta
        self.seed = seed

    def experiment(
        self,
        seeds: list[int],
    ) -> list[tuple[float, float, float, float]]:
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
    ) -> tuple[float, float, float, float] | None:
        """Iterate the experiment."""
        rng = np.random.default_rng(seed)

        for _ in range(1000):
            X = rng.normal(size=(self.n, self.d))
            noise = rng.normal(size=self.n)
            beta = np.zeros(self.d)
            beta[:3] = self.delta
            y = X @ beta + noise
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
            if self.delta != 0.0 and M[test_index] not in range(3):
                continue

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
            except InfiniteLoopError:
                return None
            except Exception as e:  # noqa: BLE001
                print(e)  # noqa: T201
                return None
            else:
                return result.p_value, result.naive_p_value(), oc_p_value, elapsed
        return None

    def run_experiment(self) -> None:
        """Conduct the experiments and save the results."""
        seeds = [5000 * (self.seed + 1) + i for i in range(self.num_iter)]
        full_results = self.experiment(seeds)
        self.results = Results(
            p_values=[result[0] for result in full_results],
            naive_p_values=[result[1] for result in full_results],
            oc_p_values=[result[2] for result in full_results],
            times=[result[3] for result in full_results],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--option", type=str, default="op1")  # op1 op2 all_cv
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--inference_mode", type=str, default="parametric")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.option)  # noqa: T201
    print(args.n, args.d, args.delta, args.seed)  # noqa: T201

    experiment = ExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        n=args.n,
        d=args.d,
        delta=args.delta,
        seed=args.seed,
    )
    experiment.run_experiment()

    dir_path = Path(f"results_{args.option}")
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    results_file_path = dir_path / f"{args.n}_{args.d}_{args.delta}_{args.seed}.pkl"
    with results_file_path.open("wb") as f:
        pickle.dump(experiment.results, f)
