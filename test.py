import numpy as np
from source.pipeline import make_dataset, make_pipeline
from source.base_component import union, intersection, extract_features, remove_outliers
from source.missing_imputation import (
    mean_value_imputation,
    definite_regression_imputation,
)
from source.outlier_detection import cook_distance, dffits, soft_ipod
from source.feature_selection import (
    stepwise_feature_selection,
    lasso,
    marginal_screening,
)
from scipy.stats import norm
from tqdm import tqdm
from sicore import pvalues_qqplot
from scipy.stats import kstest
from concurrent.futures import ProcessPoolExecutor


def option1():
    X, y = make_dataset()
    y = mean_value_imputation(X, y)

    O = cook_distance(X, y, 3.0)
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = union(M1, M2)
    return make_pipeline(output=M)


def option2():
    X, y = make_dataset()
    y = definite_regression_imputation(X, y)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    O = dffits(X, y, 3.0)
    X, y = remove_outliers(X, y, O)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = intersection(M1, M2)
    return make_pipeline(output=M)


def option3():
    X, y = make_dataset()
    y = definite_regression_imputation(X, y)

    # O = cook_distance(X, y, 3.0)
    # X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = intersection(M1, M2)
    return make_pipeline(output=M)


class PararellExperiment:
    def __init__(self, num_worker=40, num_iter=2000):
        self.num_worker = num_worker
        self.num_iter = num_iter

    def iter_experiment(self, args):
        X, y, test_index = args
        pipeline = option1()  # to change
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
        return results

    def run_experiment(self):
        # to change
        pipeline = option1()
        n, p = 100, 10
        rng = np.random.default_rng()

        dataset = []
        for _ in range(self.num_iter * 2):
            X = rng.normal(size=(n, p))
            y = rng.normal(size=(n,))
            y[:5] += 4
            missing = rng.choice(list(range(n)), size=n // 10, replace=False)
            y[missing] = np.nan
            M, _ = pipeline(X, y)
            if len(M) > 0:
                index = rng.integers(0, len(M))
                dataset.append((X, y, index))
                if len(dataset) == self.num_iter:
                    break

        results = self.experiment(dataset)

        p_list = np.array([result.p_value for result in results])
        naive_p_list = np.array(
            [2 * norm.cdf(-np.abs(result.stat)) for result in results]
        )

        print(f"parametric: {np.mean(p_list < 0.05):.3f}")
        print(f"naive: {np.mean(naive_p_list < 0.05):.3f}")

        pvalues_qqplot(p_list, fname="qqplot.pdf")
        print(f'KS: {kstest(p_list, "uniform")[1]:.3f}')

        print(
            f"search count: {np.mean([result.search_count for result in results]):.2f}"
        )


if __name__ == "__main__":
    experiment = PararellExperiment()
    experiment.run_experiment()
