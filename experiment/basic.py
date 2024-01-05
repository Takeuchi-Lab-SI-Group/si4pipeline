import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from source.model import option1, option2


class PararellExperiment:
    def __init__(self, seed, num_worker, num_result, option, n, p, beta):
        self.seed = seed
        self.num_worker = num_worker
        self.num_result = num_result
        self.num_iter = int(num_result * 1.03)
        self.option = option
        self.n = n
        self.p = p
        self.beta = beta

    def iter_experiment(self, args):
        X, y, test_index = args
        pipeline = self.option()  # to change
        try:
            _, result = pipeline.inference(X, y, 1.0, test_index, step=1e-6)
        except Exception as e:
            print(e)
            return None
        return result

    def experiment(self, dataset):
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, dataset), total=self.num_iter)
            )
        results = [result for result in results if result is not None]
        return results[: self.num_result]

    def run_experiment(self):
        # to change
        pipeline = self.option()
        n, p = self.n, self.p
        beta = np.zeros(p)
        beta[:3] = self.beta
        rng = np.random.default_rng(self.seed)

        dataset = []
        for _ in range(self.num_iter * 2):
            X = rng.normal(size=(n, p))
            y = X @ beta + rng.normal(size=(n,))
            # y[:5] += 4
            missing = rng.choice(list(range(n)), size=n // 10, replace=False)
            y[missing] = np.nan
            M, _ = pipeline(X, y)
            if len(M) > 0:
                test_index = rng.choice(len(M))
                if self.beta == 0.0 or M[test_index] in {0, 1, 2}:
                    dataset.append((X, y, test_index))
                    if len(dataset) == self.num_iter:
                        break

        results = self.experiment(dataset)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--num_result", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--option", type=int, default=1)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()

    if args.option == 1:
        option = option1
    elif args.option == 2:
        option = option2

    experiment = PararellExperiment(
        args.seed, args.num_worker, args.num_result, option, args.n, args.p, args.beta
    )
    results = experiment.run_experiment()

    dir_name = f"results_op{args.option}"
    file_name = f"seed{args.seed}_n{args.n}p{args.p}beta{args.beta:.1f}.pkl"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(f"{dir_name}/{file_name}", "wb") as f:
        pickle.dump(results, f)
