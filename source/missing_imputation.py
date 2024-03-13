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
        raise NotImplementedError

    def compute_covariance(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class MeanValueImputation(MissingImputation):
    def __init__(self, name="mean_value_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        _, y = feature_matrix, response_vector.copy()

        # location of missing value
        missing_index = np.where(np.isnan(y))[0]

        # other than missing value and its averate
        y_delete = np.delete(y, missing_index)
        y_mean = np.mean(y_delete)

        # imputation
        y[missing_index] = y_mean
        return y

    def compute_covariance(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = sigma**2 * np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        # update covariance
        # (y_1 + y_2 + ... + y_(n-num_outliers)) / (n - num_outliers)
        each_var_cov_value = sigma**2 / (n - len(missing_index))

        cov[:, missing_index] = each_var_cov_value
        cov[missing_index, :] = each_var_cov_value
        return y_imputed, cov


class EuclideanImputation(MissingImputation):
    def __init__(self, name="euclidean_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector.copy()

        # location of missing value
        missing_index = np.where(np.isnan(y))[0]

        # imputation
        for index in missing_index:
            # euclidean distance
            X_euclidean = np.sqrt(np.sum((X - X[index]) ** 2, axis=1))

            # delete missing value
            X_euclidean_deleted = np.delete(X_euclidean, missing_index)
            idx_deleted = np.argmin(X_euclidean_deleted)

            # original index
            original_indices = np.arange(len(X_euclidean))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]

            y[index] = y[idx]
        return y

    def compute_covariance(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = sigma**2 * np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        # imputation
        idx_list = []
        for index in missing_index:
            # euclidean distance
            X_euclidean = np.sqrt(
                np.sum((feature_matrix - feature_matrix[index]) ** 2, axis=1)
            )

            # delete missing value
            X_euclidean_deleted = np.delete(X_euclidean, missing_index)
            idx_deleted = np.argmin(X_euclidean_deleted)

            # original index
            original_indices = np.arange(len(X_euclidean))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]
            idx_list.append(idx)

        for index, idx in zip(missing_index, idx_list):
            cov[index, idx] = sigma**2
            cov[idx, index] = sigma**2
        return y_imputed, cov


class ManhattanImputation(MissingImputation):
    def __init__(self, name="manhattan_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector.copy()

        # location of missing value
        missing_index = np.where(np.isnan(y))[0]

        # imputation
        for index in missing_index:
            # manhattan distance
            X_manhattan = np.sum(np.abs(X - X[index]), axis=1)

            # delete missing value
            X_manhattan_deleted = np.delete(X_manhattan, missing_index)
            idx_deleted = np.argmin(X_manhattan_deleted)

            # original index
            original_indices = np.arange(len(X_manhattan))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]

            y[index] = y[idx]
        return y

    def compute_covariance(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = sigma**2 * np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        # imputation
        idx_list = []
        for index in missing_index:
            # manhattan distance
            X_manhattan = np.sum(np.abs(feature_matrix - feature_matrix[index]), axis=1)

            # delete missing value
            X_manhattan_deleted = np.delete(X_manhattan, missing_index)
            idx_deleted = np.argmin(X_manhattan_deleted)

            # original index
            original_indices = np.arange(len(X_manhattan))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]
            idx_list.append(idx)

        for index, idx in zip(missing_index, idx_list):
            cov[index, idx] = sigma**2
            cov[idx, index] = sigma**2
        return y_imputed, cov


class ChebyshevImputation(MissingImputation):
    def __init__(self, name="chebyshev_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector.copy()

        # location of missing value
        missing_index = np.where(np.isnan(y))[0]

        # imputation
        for index in missing_index:
            # chebyshev distance
            X_chebyshev = np.max(np.abs(X - X[index]), axis=1)

            # delete missing value
            X_chebyshev_deleted = np.delete(X_chebyshev, missing_index)
            idx_deleted = np.argmin(X_chebyshev_deleted)

            # original index
            original_indices = np.arange(len(X_chebyshev))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]

            y[index] = y[idx]
        return y

    def compute_covariance(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = sigma**2 * np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        # imputation
        idx_list = []
        for index in missing_index:
            # chebyshev distance
            X_chebyshev = np.max(np.abs(feature_matrix - feature_matrix[index]), axis=1)

            # delete missing value
            X_chebyshev_deleted = np.delete(X_chebyshev, missing_index)
            idx_deleted = np.argmin(X_chebyshev_deleted)

            # original index
            original_indices = np.arange(len(X_chebyshev))
            valid_indices = np.delete(original_indices, missing_index)
            idx = valid_indices[idx_deleted]
            idx_list.append(idx)

        for index, idx in zip(missing_index, idx_list):
            cov[index, idx] = sigma**2
            cov[idx, index] = sigma**2
        return y_imputed, cov


class DefiniteRegressionImputation(MissingImputation):
    def __init__(self, name="definite_regression_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector.copy()

        # location of missing value
        missing_index = np.where(np.isnan(y))[0]

        X_delete = np.delete(X, missing_index, 0)
        y_delete = np.delete(y, missing_index).reshape(-1, 1)

        # imputation
        beta_hat = np.linalg.inv(X_delete.T @ X_delete) @ X_delete.T @ y_delete
        for index in missing_index:
            X_missing = X[index]
            y_new = X_missing @ beta_hat
            y[index] = y_new

        return y

    def compute_covariance(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = sigma**2 * np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        X = feature_matrix
        X_delete = np.delete(X, missing_index, 0)

        # update covariance
        factor = np.linalg.inv(X_delete.T @ X_delete)
        for index in missing_index:
            X_missing = X[index]
            for i in range(n):
                each_x = X[i]
                var_missing = sigma**2 * each_x @ factor @ X_missing.T
                cov[i, index] = var_missing
                cov[index, i] = var_missing

        return y_imputed, cov


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
