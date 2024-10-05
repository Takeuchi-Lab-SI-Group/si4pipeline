"""Module for real data experiments."""

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

from experiment.utils import Results, option1, option1_multi, option2, option2_multi

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".."))

warnings.simplefilter("ignore")


class RealExperimentPipeline:
    """Experiment class for the real data experiment."""

    def __init__(
        self,
        num_results: int,
        num_worker: int,
        option: Literal["op1", "op2", "all_cv"],
        n: int,
        key: str,
        seed: int,
    ) -> None:
        """Initialize the experiment."""
        self.num_results = num_results
        self.num_iter = int(num_results * 1.1)
        self.num_worker = num_worker
        self.option = option
        self.n = n
        self.key = key
        self.seed = seed

    def experiment(
        self,
        args: list[tuple[np.ndarray, np.ndarray, int]],
    ) -> list[tuple[SelectiveInferenceResult, float, float]]:
        """Conduct the experiment in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, args), total=self.num_iter),
            )
        results = [result for result in results if result is not None]
        return results[: self.num_results]

    def iter_experiment(
        self,
        args: tuple[np.ndarray, np.ndarray, int],
    ) -> tuple[SelectiveInferenceResult, float, float] | None:
        """Iterate the experiment."""
        X, y, seed = args
        rng = np.random.default_rng([seed, self.seed])

        match self.option:
            case "op1":
                manager = option1()
            case "op2":
                manager = option2()
            case "all_cv":
                manager = option1_multi() | option2_multi()
                manager.tune(X, y, random_state=rng.integers(2**32))

        M, _ = manager(X, y)
        if len(M) == 0:
            return None
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

    def run_experiment(self) -> None:
        """Conduct the experiments and save the results."""
        path = Path("experiment/dataset/table_dataset.pkl")
        with path.open("rb") as f:
            dataset = pickle.load(f)
        X, y = dataset[self.key]

        args = []
        rng = np.random.default_rng(self.seed)
        for seed in range(self.num_iter):
            index = rng.choice(X.shape[0], size=self.n, replace=False)
            args.append((X[index, :], y[index], seed))

        full_results = self.experiment(args)
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
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--key", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.option, args.key)
    print(args.n, args.seed)

    experiment = RealExperimentPipeline(
        num_results=args.num_results,
        num_worker=args.num_worker,
        option=args.option,
        key=args.key,
        n=args.n,
        seed=args.seed,
    )
    experiment.run_experiment()

    dir_path = Path("results_real")
    dir_path.mkdir(parents=True, exist_ok=True)

    results_file_path = dir_path / f"{args.option}_{args.key}_{args.n}_{args.seed}.pkl"
    with results_file_path.open("wb") as f:
        pickle.dump(experiment.results, f)
