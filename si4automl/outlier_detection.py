"""Module containing outlier detection methods."""

import numpy as np
import sklearn.linear_model as lm  # type: ignore[import]
from sicore import polytope_below_zero  # type: ignore[import]


class OutlierDetection:
    """A class for outlier detection methods."""

    def __init__(self, parameter: float) -> None:
        """Initialize the OutlierDetection object."""
        self.parameter = parameter

    def detect_outliers(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the outlier detection."""
        raise NotImplementedError

    def reset_cache(self) -> None:
        """Reset the cache to execute the selective inference."""
        self.intervals: dict[
            int,
            dict[tuple[float, float], tuple[list[int], list[int]]],
        ] = {}

    def load_intervals(
        self,
        z: float,
        l: float,
        u: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float] | None:
        """Load the intervals to execute the selective inference."""
        for interval, indexes in self.intervals.setdefault(mask_id, {}).items():
            if interval[0] < z < interval[1]:
                M, O = indexes
                l = np.max([l, interval[0]]).item()
                u = np.min([u, interval[1]]).item()
                return M, O, l, u
        return None

    def save_intervals(
        self,
        l: float,
        u: float,
        M: list[int],
        O: list[int],
        mask_id: int = -1,
    ) -> None:
        """Save the intervals to execute the selective inference."""
        self.intervals[mask_id][(l, u)] = (M, O)

    def perform_si(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        feature_matrix: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
        l: float,
        u: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float]:
        """Perform the selective inference."""
        raise NotImplementedError


class CookDistance(OutlierDetection):
    """A class for cook distance method."""

    def __init__(self, parameter: float) -> None:
        """Initialize the CookDistance object."""
        super().__init__(parameter)

    def detect_outliers(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the outlier detection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        # cook's distance
        non_outlier = []
        outlier = []
        n, p = X.shape

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameter / n  # threshold value

        # outlier detection
        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]  # diagonal element of hat matrix
            Di_1 = (y.T @ (Px @ ej @ ej.T @ Px) @ y) / (
                y.T @ Px @ y
            )  # first term of Di
            Di_2 = ((n - p) * hi) / (p * (1 - hi) ** 2)  # second term of Di
            Di = Di_1 * Di_2

            if Di < threshold:
                non_outlier.append(i)
            else:
                outlier.append(i)

        O_ = [num_outlier_data[i] for i in outlier]
        return O + O_

    def perform_si(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        feature_matrix: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
        l: float,
        u: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float]:
        """Perform the selective inference."""
        results = self.load_intervals(z, l, u, mask_id)
        if results is not None:
            return results

        X, yz = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(yz, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        non_outlier = []
        outlier = []
        n, p = X.shape

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameter / n  # threshold value

        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]  # diagonal element of hat matrix
            Di_1 = (yz.T @ (Px @ ej @ ej.T @ Px) @ yz) / (
                yz.T @ Px @ yz
            )  # first term of Di
            Di_2 = ((n - p) * hi) / (p * (1 - hi) ** 2)  # second term of Di
            Di = Di_1 * Di_2

            if Di < threshold:
                non_outlier.append(i)
            else:
                outlier.append(i)

        l_list, u_list = [l], [u]
        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]
            H_1 = ((n - p) * hi) * Px @ ej @ ej.T @ Px
            H_2 = ((self.parameter * p * (1 - hi) ** 2) / n) * Px
            H = H_1 - H_2

            if i in outlier:
                H = -H

            intervals = polytope_below_zero(a, b, H, np.zeros(n), 0)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

        l = np.max(l_list).item()
        u = np.min(u_list).item()
        assert l < z < u, "l < z < u is not satisfied"

        O_ = [num_outlier_data[i] for i in outlier]
        O = O + O_

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u


class Dffits(OutlierDetection):
    """A class for dffits method."""

    def __init__(self, parameter: float) -> None:
        """Initialize the Dffits object."""
        super().__init__(parameter)

    def detect_outliers(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the outlier detection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        # dffits
        non_outlier = []
        outlier = []
        n, p = X.shape

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameter * p / (n - p)

        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]  # diagonal element of hat matrix
            dffits_i_1 = np.sqrt(hi * (n - p - 1)) / (1 - hi)  # one side of dffits
            dffits_i_2_denominator = y.T @ Px @ y - (
                (y.T @ Px @ ej @ ej.T @ Px @ y) / (1 - hi)
            )
            dffits_i_2 = (ej.T @ Px @ y) / np.sqrt(dffits_i_2_denominator)
            dffits_i = dffits_i_1 * dffits_i_2

            if dffits_i**2 < threshold:
                non_outlier.append(i)
            else:
                outlier.append(i)

        O_ = [num_outlier_data[i] for i in outlier]
        return O + O_

    def perform_si(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        feature_matrix: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
        l: float,
        u: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float]:
        """Perform the selective inference."""
        results = self.load_intervals(z, l, u, mask_id)
        if results is not None:
            return results

        X, yz = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(yz, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        non_outlier = []
        outlier = []
        n, p = X.shape

        # dffits
        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameter * p / (n - p)

        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]  # diagonal element of hat matrix
            dffits_i_1 = np.sqrt(hi * (n - p - 1)) / (1 - hi)  # one side of dffits
            dffits_i_2_denominator = yz.T @ Px @ yz - (
                (yz.T @ Px @ ej @ ej.T @ Px @ yz) / (1 - hi)
            )
            dffits_i_2 = (ej.T @ Px @ yz) / np.sqrt(dffits_i_2_denominator)
            dffits_i = dffits_i_1 * dffits_i_2

            if dffits_i**2 < threshold:
                non_outlier.append(i)
            else:
                outlier.append(i)

        l_list, u_list = [l], [u]
        for i in range(n):
            ej = np.zeros((n, 1))
            ej[i] = 1
            hi = hat_matrix[i][i]
            H_11 = ((hi * (n - p - 1)) / (1 - hi) ** 2) + (
                (self.parameter * p) / ((n - p) * (1 - hi))
            )
            H_1 = H_11 * Px @ ej @ ej.T @ Px
            H_2 = ((self.parameter * p) / (n - p)) * Px
            H = H_1 - H_2

            if i in outlier:
                H = -H

            intervals = polytope_below_zero(a, b, H, np.zeros(n), 0)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

        l = np.max(l_list).item()
        u = np.min(u_list).item()
        assert l < z < u, "l < z < u is not satisfied"

        O_ = [num_outlier_data[i] for i in outlier]
        O = O + O_

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u


class SoftIpod(OutlierDetection):
    """A class for soft ipod method."""

    def __init__(self, parameter: float) -> None:
        """Initialize the SoftIpod object."""
        super().__init__(parameter)

    def detect_outliers(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the outlier detection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        # soft-ipod
        # outlier: list[int] = []
        n = X.shape[0]

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        Pxy = Px @ y

        lasso = lm.Lasso(
            alpha=self.parameter,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-10,
        )
        lasso.fit(Px, Pxy)
        outlier = np.where(lasso.coef_ != 0)[0]

        O_ = [num_outlier_data[i] for i in outlier]
        return O + O_

    def perform_si(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        feature_matrix: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
        l: float,
        u: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float]:
        """Perform the selective inference."""
        results = self.load_intervals(z, l, u, mask_id)
        if results is not None:
            return results

        X, yz = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(yz, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        num_data, n = list(range(X.shape[0])), X.shape[0]
        num_outlier_data = [i for i in num_data if i not in O]

        # n = X.shape[0]

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        Pxy = Px @ yz

        lasso = lm.Lasso(
            alpha=self.parameter,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-10,
        )
        lasso.fit(Px, Pxy)
        outlier = np.where(lasso.coef_ != 0)[0].tolist()
        non_outlier = np.where(lasso.coef_ == 0)[0]
        signs = np.sign(lasso.coef_[outlier])

        X_caron_active = Px[:, non_outlier]
        X_caron_inactive = Px[:, outlier]

        Px_caron_inactive_perp = (
            np.identity(n)
            - X_caron_inactive
            @ np.linalg.inv(X_caron_inactive.T @ X_caron_inactive)
            @ X_caron_inactive.T
        )
        X_caron_inactive_plus = X_caron_inactive @ np.linalg.inv(
            X_caron_inactive.T @ X_caron_inactive,
        )

        A0_plus = X_caron_active.T @ Px_caron_inactive_perp @ Px / (self.parameter * n)
        A0_minus = -A0_plus

        b0_plus = (
            np.ones(len(non_outlier)) - X_caron_active.T @ X_caron_inactive_plus @ signs
        )
        b0_minus = (
            np.ones(len(non_outlier)) + X_caron_active.T @ X_caron_inactive_plus @ signs
        )

        A1 = (
            -np.diag(signs)
            @ np.linalg.inv(X_caron_inactive.T @ X_caron_inactive)
            @ X_caron_inactive.T
            @ Px
        )
        b1 = (
            -n
            * self.parameter
            * np.diag(signs)
            @ np.linalg.inv(X_caron_inactive.T @ X_caron_inactive)
            @ signs
        )

        soft_ipod_condition = [[A0_plus, b0_plus], [A0_minus, b0_minus], [A1, b1]]

        left_list = []
        right_list = []
        for Aj, bj in soft_ipod_condition:
            left = (Aj @ b).reshape(-1).tolist()
            right = (bj - Aj @ a).reshape(-1).tolist()
            left_list += left
            right_list += right

        l_list, u_list = [l], [u]
        for left, right in zip(left_list, right_list, strict=False):
            if np.around(left, 5) == 0:
                if right <= 0:
                    raise ValueError  # l must be less than u
                continue
            term = right / left
            if left > 0:
                u_list.append(term)
            else:
                l_list.append(term)

        l, u = np.max(l_list).item(), np.min(u_list).item()
        assert l < z < u, "l < z < u is not satisfied"

        # O_ = [num_outlier_data[i] for i in outlier]
        # O = O + O_
        O = O + [num_outlier_data[i] for i in outlier]

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u
