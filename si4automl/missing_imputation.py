"""Module containing missing imputation methods."""

import numpy as np


class MissingImputation:
    """A class for missing imputation."""

    def __init__(self) -> None:
        """Initialize the MissingImputation object."""

    def impute_missing(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Perform the missing imputation."""
        imputer = self.compute_imputer(feature_matrix, response_vector)
        return imputer @ response_vector[~np.isnan(response_vector)]

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
        raise NotImplementedError


class MeanValueImputation(MissingImputation):
    """A class for the mean value imputation method."""

    def __init__(self) -> None:
        """Initialize the MeanValueImputation object."""
        super().__init__()

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
        _ = feature_matrix
        nan_mask = np.isnan(response_vector)
        num_missing = np.count_nonzero(nan_mask)
        n = len(response_vector)
        imputer = np.zeros((n, n - num_missing))  # (n, n - num_missing)
        imputer[nan_mask] = 1 / (n - num_missing)
        imputer[~nan_mask, :] = np.eye(n - num_missing)
        return imputer


class EuclideanImputation(MissingImputation):
    """A class for the euclidean imputation method."""

    def __init__(self) -> None:
        """Initialize the EuclideanImputation object."""
        super().__init__()

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
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
                np.sum((X[~nan_mask, :] - X[index]) ** 2, axis=1),
            )  # shape (n - num_missing, )
            idx = np.argmin(X_euclidean)
            imputer[index, idx] = 1.0
        return imputer


class ManhattanImputation(MissingImputation):
    """A class for the manhattan imputation method."""

    def __init__(self) -> None:
        """A class for the manhattan imputation method."""
        super().__init__()

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
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
                np.abs(X[~nan_mask] - X[index]),
                axis=1,
            )  # shape (n - num_missing, )
            idx = np.argmin(X_manhattan)
            imputer[index, idx] = 1.0
        return imputer


class ChebyshevImputation(MissingImputation):
    """A class for the chebyshev imputation method."""

    def __init__(self) -> None:
        """Initialize the ChebyshevImputation object."""
        super().__init__()

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
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
                np.abs(X[~nan_mask] - X[index]),
                axis=1,
            )  # shape (n - num_missing, )
            idx = np.argmin(X_chebyshev)
            imputer[index, idx] = 1.0
        return imputer


class DefiniteRegressionImputation(MissingImputation):
    """Definite regression imputation method."""

    def __init__(self) -> None:
        """Initialize the DefiniteRegressionImputation object."""
        super().__init__()

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix."""
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
