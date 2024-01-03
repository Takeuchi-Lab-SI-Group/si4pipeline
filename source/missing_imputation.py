import numpy as np
from source.base_component import MissingImputation


class MeanValueImputation(MissingImputation):
    def __init__(self, name="mean_value_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        _, y = feature_matrix, response_vector

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
    ) -> (np.ndarray, np.ndarray):
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = np.identity(n)
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


class ManhattanImputation(MissingImputation):
    def __init__(self, name="manhattan_imputation"):
        super().__init__(name)


class ChebyshevImputation(MissingImputation):
    def __init__(self, name="chebyshev_imputation"):
        super().__init__(name)


class DefiniteRegressionImputation(MissingImputation):
    def __init__(self, name="definite_regression_imputation"):
        super().__init__(name)

    def impute_missing(
        self, feature_matrix: np.ndarray, response_vector: np.ndarray
    ) -> np.ndarray:
        X, y = feature_matrix, response_vector

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
    ) -> (np.ndarray, np.ndarray):
        y_imputed = self.impute_missing(feature_matrix, response_vector)

        n = response_vector.shape[0]
        cov = np.identity(n)
        missing_index = np.where(np.isnan(response_vector))[0]

        # update covariance
        # (y_1 + y_2 + ... + y_(n-num_outliers)) / (n - num_outliers)
        each_var_cov_value = sigma**2 / (n - len(missing_index))

        cov[:, missing_index] = each_var_cov_value
        cov[missing_index, :] = each_var_cov_value
        return y_imputed, cov


class ProbabilisticRegressionImputation(MissingImputation):
    def __init__(self, name="probabilistic_regression_imputation"):
        super().__init__(name)


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


def probabilistic_regression_imputation(feature_matrix, response_vector):
    return ProbabilisticRegressionImputation()(feature_matrix, response_vector)
