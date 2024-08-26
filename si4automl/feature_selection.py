"""Module containing feature selection methods."""

from typing import cast

import numpy as np
import sklearn.linear_model as lm  # type: ignore[import]
from sicore import polytope_below_zero  # type: ignore[import]


class FeatureSelection:
    """A class for feature selection methods."""

    def __init__(self, parameter: float) -> None:
        """Initialize the FeatureSelection object."""
        self.parameter = parameter

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the feature selection."""
        raise NotImplementedError

    def reset_intervals(self) -> None:
        """Reset the intervals to execute the selective inference."""
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


class StepwiseFeatureSelection(FeatureSelection):
    """A class for the stepwise feature selection method."""

    def __init__(self, parameter: int) -> None:
        """Initialize the StepwiseFeatureSelection object."""
        super().__init__(parameter)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the feature selection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # initialize
        min_mse = np.inf
        active_set: list[int] = []
        inactive_set = list(range(X.shape[1]))

        # stepwise feature selection
        self.parameter = cast(int, self.parameter)
        for _ in range(self.parameter):
            mse_list = []
            for inactive_feature in inactive_set:
                active_temp = [*active_set, inactive_feature]
                X_active_temp = X[:, active_temp]
                r = (
                    np.identity(X.shape[0])
                    - X_active_temp
                    @ np.linalg.pinv(X_active_temp.T @ X_active_temp)
                    @ X_active_temp.T
                ) @ y
                mse = r.T @ r
                mse_list.append(mse)

            min_mse_index = np.argmin(mse_list)
            min_mse_feature = mse_list[min_mse_index]

            if min_mse_feature < min_mse:
                min_mse = min_mse_feature
                active_set.append(inactive_set.pop(min_mse_index))
            else:
                break

        return [M[i] for i in active_set]

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

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        # initialize
        min_mse = np.inf
        active_set: list[int] = []
        inactive_set = list(range(X.shape[1]))

        # stepwise feature selection
        self.parameter = cast(int, self.parameter)
        for _ in range(self.parameter):
            mse_list = []
            for inactive_feature in inactive_set:
                active_temp = [*active_set, inactive_feature]
                X_active_temp = X[:, active_temp]
                r = (
                    np.identity(X.shape[0])
                    - X_active_temp
                    @ np.linalg.pinv(X_active_temp.T @ X_active_temp)
                    @ X_active_temp.T
                ) @ yz
                mse = r.T @ r
                mse_list.append(mse)

            min_mse_index = np.argmin(mse_list)
            min_mse_feature = mse_list[min_mse_index]

            if min_mse_feature < min_mse:
                min_mse = min_mse_feature
                active_set.append(inactive_set.pop(min_mse_index))
            else:
                break

        l_list, u_list = [l], [u]
        inactive_set = list(range(X.shape[1]))

        for i in range(self.parameter):
            X_active_k = X[:, active_set[0 : i + 1]]
            mse_active_k = (
                np.identity(X.shape[0])
                - X_active_k @ np.linalg.pinv(X_active_k.T @ X_active_k) @ X_active_k.T
            )
            inactive_set.remove(active_set[i])
            for inactive_feature in inactive_set:
                active_temp = active_set[0:i] + [inactive_feature]
                X_active_temp = X[:, active_temp]
                mse_active_temp = (
                    np.identity(X.shape[0])
                    - X_active_temp
                    @ np.linalg.pinv(X_active_temp.T @ X_active_temp)
                    @ X_active_temp.T
                )

                quad_A = mse_active_k - mse_active_temp

                intervals = polytope_below_zero(a, b, quad_A, np.zeros(X.shape[0]), 0)
                for left, right in intervals:
                    if left < z < right:
                        l_list.append(left)
                        u_list.append(right)
                        break

        l = np.max(l_list).item()
        u = np.min(u_list).item()
        assert l < z < u

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u


class MarginalScreening(FeatureSelection):
    """A class for the marginal screening method."""

    def __init__(self, parameter: int) -> None:
        """Initialize the MarginalScreening object."""
        super().__init__(parameter)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the feature selection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # marginal screening
        XTy_abs = np.abs(X.T @ y).flatten()
        sort_XTy_abs = np.argsort(XTy_abs)[::-1]

        self.parameter = cast(int, self.parameter)
        active_set = sort_XTy_abs[: self.parameter]
        return [M[i] for i in active_set]

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

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        XTyz_abs = np.abs(X.T @ yz).flatten()
        sort_XTyz_abs = np.argsort(XTyz_abs)[::-1]

        self.parameter = cast(int, self.parameter)
        active_set = sort_XTyz_abs[: self.parameter]
        inactive_set = sort_XTyz_abs[self.parameter :]

        left_list = []
        right_list = []
        for i in active_set:
            x_i = X[:, i]
            sign_i = np.sign(x_i.T @ yz)

            e1 = sign_i * x_i.T @ a
            e2 = sign_i * x_i.T @ b

            for j in inactive_set:
                x_j = X[:, j]

                e3 = x_j.T @ a
                e4 = x_j.T @ b

                e5 = -x_j.T @ a
                e6 = -x_j.T @ b

                left_list.append(e4 - e2)
                left_list.append(e6 - e2)
                right_list.append(e1 - e3)
                right_list.append(e1 - e5)

        l_list, u_list = [l], [u]
        for left, right in zip(left_list, right_list, strict=False):
            if np.around(left, 5) == 0:
                if right <= 0:
                    raise ValueError  # "l must be less than u"
                continue
            term = right / left
            if left > 0:
                u_list.append(term)
            else:
                l_list.append(term)

        l = np.max(l_list).item()
        u = np.min(u_list).item()
        assert l < z < u

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u


class Lasso(FeatureSelection):
    """A class for the lasso method."""

    def __init__(self, parameter: float) -> None:
        """Initialize the Lasso object."""
        super().__init__(parameter)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        """Perform the feature selection."""
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # lasso
        lasso = lm.Lasso(
            alpha=self.parameter,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-10,
        )
        lasso.fit(X, y)
        active_set = np.where(lasso.coef_ != 0)[0]
        return [M[i] for i in active_set]

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

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        lasso = lm.Lasso(
            alpha=self.parameter,
            fit_intercept=False,
            max_iter=5000,
            tol=1e-10,
        )
        lasso.fit(X, yz)
        active_set = np.where(lasso.coef_ != 0)[0].tolist()
        inactive_set = [i for i in range(X.shape[1]) if i not in active_set]
        signs = np.sign(lasso.coef_[active_set])

        X_active = X[:, active_set]
        X_inactive = X[:, inactive_set]

        X_active_plus = np.linalg.pinv(X_active.T @ X_active) @ X_active.T
        Pm = X_active @ X_active_plus

        A0_plus = (X_inactive.T @ (np.identity(X.shape[0]) - Pm)) / (
            self.parameter * X.shape[0]
        )
        A0_minus = (-X_inactive.T @ (np.identity(X.shape[0]) - Pm)) / (
            self.parameter * X.shape[0]
        )
        b0_plus = np.ones(X_inactive.shape[1]) - X_inactive.T @ X_active_plus.T @ signs
        b0_minus = np.ones(X_inactive.shape[1]) + X_inactive.T @ X_active_plus.T @ signs

        A1 = -np.diag(signs) @ X_active_plus
        b1 = (
            -self.parameter
            * X.shape[0]
            * np.diag(signs)
            @ np.linalg.inv(X_active.T @ X_active)
            @ signs
        )

        lasso_condition = [[A0_plus, b0_plus], [A0_minus, b0_minus], [A1, b1]]

        left_list = []
        right_list = []
        for Aj, bj in lasso_condition:
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

        l = np.max(l_list).item()
        u = np.min(u_list).item()
        assert l < z < u

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, mask_id)
        return M, O, l, u
