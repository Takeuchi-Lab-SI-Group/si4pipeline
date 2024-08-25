import numpy as np
from source.base_component import FeatureMatrix, ResponseVector


class MissingImputation:
    instance_counter = dict()

    def __init__(self, name: str):
        MissingImputation.instance_counter.setdefault(name, 0)
        self.name = f"{name}_{MissingImputation.instance_counter[name]}"
        MissingImputation.instance_counter[name] += 1

    def __call__(
        self, feature_matrix: FeatureMatrix, response_vector: ResponseVector
    ) -> ResponseVector:
        pl_structure = feature_matrix.pl_structure | response_vector.pl_structure
        pl_structure.update(self.name, self)
        return ResponseVector(pl_structure)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        imputer = self.compute_imputer(feature_matrix, response_vector)
        return imputer @ response_vector[~np.isnan(response_vector)]

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class MeanValueImputation(MissingImputation):
    def __init__(self, name="mean_value_imputation"):
        super().__init__(name)

    def compute_imputer(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        nan_mask = np.isnan(response_vector)
        num_missing = np.count_nonzero(nan_mask)
        n = len(response_vector)
        imputer = np.zeros((n, n - num_missing))  # (n, n - num_missing)
        imputer[nan_mask] = 1 / (n - num_missing)
        imputer[~nan_mask, :] = np.eye(n - num_missing)
        return imputer


class EuclideanImputation(MissingImputation):
    def __init__(self, name="euclidean_imputation"):
        super().__init__(name)

    def compute_imputer(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector
        nan_mask = np.isnan(y)
        num_missing = np.count_nonzero(nan_mask)
        n = len(y)
        imputer = np.zeros((n, n - num_missing))  # shape (n, n - num_missing)
        imputer[~nan_mask, :] = np.eye(n - num_missing)

        missing_index = np.where(nan_mask)[0]
        for index in missing_index:
            # euclidean distance
            X_euclidean = np.sqrt(
                np.sum((X[~nan_mask, :] - X[index]) ** 2, axis=1)
            )  # shape (n - num_missing, )
            idx = np.argmin(X_euclidean)
            imputer[index, idx] = 1.0
        return imputer


class ManhattanImputation(MissingImputation):
    def __init__(self, name="manhattan_imputation"):
        super().__init__(name)

    def compute_imputer(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector
        nan_mask = np.isnan(y)
        num_missing = np.count_nonzero(nan_mask)
        n = len(y)
        imputer = np.zeros((n, n - num_missing))  # shape (n, n - num_missing)
        imputer[~nan_mask, :] = np.eye(n - num_missing)

        missing_index = np.where(nan_mask)[0]
        for index in missing_index:
            # manhattan distance
            X_manhattan = np.sum(
                np.abs(X[~nan_mask] - X[index]), axis=1
            )  # shape (n - num_missing, )
            idx = np.argmin(X_manhattan)
            imputer[index, idx] = 1.0
        return imputer


class ChebyshevImputation(MissingImputation):
    def __init__(self, name="chebyshev_imputation"):
        super().__init__(name)

    def compute_imputer(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector
        nan_mask = np.isnan(y)
        num_missing = np.count_nonzero(nan_mask)
        n = len(y)
        imputer = np.zeros((n, n - num_missing))  # shape (n, n - num_missing)
        imputer[~nan_mask, :] = np.eye(n - num_missing)

        missing_index = np.where(nan_mask)[0]
        for index in missing_index:
            # manhattan distance
            X_chebyshev = np.max(
                np.abs(X[~nan_mask] - X[index]), axis=1
            )  # shape (n - num_missing, )
            idx = np.argmin(X_chebyshev)
            imputer[index, idx] = 1.0
        return imputer


class DefiniteRegressionImputation(MissingImputation):
    def __init__(self, name="definite_regression_imputation"):
        super().__init__(name)

    def compute_imputer(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector
        nan_mask = np.isnan(y)
        num_missing = np.count_nonzero(nan_mask)
        n = len(y)
        imputer = np.zeros((n, n - num_missing))
        imputer[~nan_mask, :] = np.eye(n - num_missing)  # shape (n, n - num_missing)

        beta_hat_front = (
            np.linalg.inv(X[~nan_mask, :].T @ X[~nan_mask, :]) @ X[~nan_mask, :].T
        )
        imputer[nan_mask, :] = X[nan_mask, :] @ beta_hat_front
        return imputer


def mean_value_imputation(feature_matrix, response_vector):
    return MeanValueImputation()(feature_matrix, response_vector)


def euclidean_imputation(feature_matrix, response_vector):
    return EuclideanImputation()(feature_matrix, response_vector)


def manhattan_imputation(feature_matrix, response_vector):
    return ManhattanImputation()(feature_matrix, response_vector)


def chebyshev_imputation(feature_matrix, response_vector):
    return ChebyshevImputation()(feature_matrix, response_vector)


def definite_regression_imputation(feature_matrix, response_vector):
    return DefiniteRegressionImputation()(feature_matrix, response_vector)
