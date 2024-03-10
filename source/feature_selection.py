import numpy as np
import sklearn.linear_model as lm
from source.base_component import FeatureMatrix, ResponseVector, SelectedFeatures


class FeatureSelection:
    instance_counter = dict()

    def __init__(
        self,
        name: str,
        parameters,
        candidates,
    ):
        self.parameters = parameters
        self.candidates = candidates
        FeatureSelection.instance_counter.setdefault(name, 0)
        self.name = f"{name}_{FeatureSelection.instance_counter[name]}"
        FeatureSelection.instance_counter[name] += 1

    def __call__(
        self, feature_matrix: FeatureMatrix, response_vector: ResponseVector
    ) -> SelectedFeatures:
        pl_structure = feature_matrix.pl_structure | response_vector.pl_structure
        pl_structure.update(self.name, self)
        return SelectedFeatures(pl_structure)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> np.ndarray:
        raise NotImplementedError

    def reset_intervals(self):
        self.intervals = dict()

    def load_intervals(
        self,
        z: float,
        l: float,
        u: float,
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ):
        if candidate_id is None and mask_id is None:
            self.intervals.setdefault(None, dict())
            items = self.intervals[None].items()
        elif candidate_id is not None and mask_id is not None:
            self.intervals.setdefault(candidate_id, dict())
            self.intervals[candidate_id].setdefault(mask_id, dict())
            items = self.intervals[candidate_id][mask_id].items()
        else:
            raise ValueError("candidate_id and mask_id must be both None or not None")

        for interval, indexes in items:
            if interval[0] < z < interval[1]:
                M, O = indexes
                l = np.max([l, interval[0]])
                u = np.min([u, interval[1]])
                return M, O, l, u
        return None

    def save_intervals(
        self,
        l: float,
        u: float,
        M: list[int],
        O: list[int],
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ):
        if candidate_id is None and mask_id is None:
            self.intervals[None][(l, u)] = (M, O)
        elif candidate_id is not None and mask_id is not None:
            self.intervals[candidate_id][mask_id][(l, u)] = (M, O)
        else:
            raise ValueError("candidate_id and mask_id must be both None or not None")

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
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ) -> tuple[list[int], list[int], float, float]:
        raise NotImplementedError


class StepwiseFeatureSelection(FeatureSelection):
    def __init__(
        self, name="stepwise_feature_selection", parameters=None, candidates=None
    ):
        super().__init__(name, parameters, candidates)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # initialize
        active_set = []
        inactive_set = list(range(X.shape[1]))

        # stepwise feature selection
        for _ in range(self.parameters):
            X_active = X[:, active_set]
            r = y - X_active @ np.linalg.pinv(X_active.T @ X_active) @ X_active.T @ y
            correlation = X[:, inactive_set].T @ r

            ind = np.argmax(np.abs(correlation))
            active_set.append(inactive_set[ind])
            inactive_set.remove(inactive_set[ind])
        M = [M[i] for i in active_set]
        return M

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
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ) -> tuple[list[int], list[int], float, float]:  # type: ignore
        results = self.load_intervals(z, l, u, candidate_id, mask_id)
        if results is not None:
            return results

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        active_set = []
        inactive_set = list(range(X.shape[1]))
        signs = []

        for _ in range(self.parameters):
            X_active = X[:, active_set]
            r = yz - X_active @ np.linalg.pinv(X_active.T @ X_active) @ X_active.T @ yz
            correlation = X[:, inactive_set].T @ r

            ind = np.argmax(np.abs(correlation))
            active_set.append(inactive_set[ind])
            inactive_set.remove(inactive_set[ind])
            signs.append(np.sign(correlation[ind]))

        left_list = []
        right_list = []
        inactive_set = list(range(X.shape[1]))

        for i in range(self.parameters):
            X_active = X[:, active_set[:i]]
            x_jt = X[:, active_set[i]]
            sign_t = signs[i]
            F = (
                np.identity(X.shape[0])
                - X_active @ np.linalg.pinv(X_active.T @ X_active) @ X_active.T
            )
            inactive_set.remove(active_set[i])

            for j in inactive_set:
                x_j = X[:, j]

                left1 = (x_j - sign_t * x_jt).T @ F @ b
                left2 = (-x_j - sign_t * x_jt).T @ F @ b
                right1 = -(x_j - sign_t * x_jt).T @ F @ a
                right2 = -(-x_j - sign_t * x_jt).T @ F @ a

                left_list.append(left1)
                left_list.append(left2)
                right_list.append(right1)
                right_list.append(right2)

        l_list, u_list = [l], [u]
        for left, right in zip(left_list, right_list):
            if np.around(left, 5) == 0:
                if right <= 0:
                    raise ValueError("l must be less than u")
                continue
            term = right / left
            if left > 0:
                u_list.append(term)
            else:
                l_list.append(term)

        l = np.max(l_list)
        u = np.min(u_list)
        assert l < z < u, "l < z < u is not satisfied"

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, candidate_id, mask_id)
        return M, O, l, u


class MarginalScreening(FeatureSelection):
    def __init__(self, name="marginal_screening", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # marginal screening
        XTy_abs = np.abs(X.T @ y).flatten()
        sort_XTy_abs = np.argsort(XTy_abs)[::-1]

        active_set = sort_XTy_abs[: self.parameters]
        M = [M[i] for i in active_set]
        return M

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
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ) -> tuple[list[int], list[int], float, float]:
        results = self.load_intervals(z, l, u, candidate_id, mask_id)
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

        active_set = sort_XTyz_abs[: self.parameters]
        inactive_set = sort_XTyz_abs[self.parameters :]

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
        for left, right in zip(left_list, right_list):
            if np.around(left, 5) == 0:
                if right <= 0:
                    raise ValueError("l must be less than u")
                continue
            term = right / left
            if left > 0:
                u_list.append(term)
            else:
                l_list.append(term)

        l = np.max(l_list)
        u = np.min(u_list)
        assert l < z < u, "l < z < u is not satisfied"

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, candidate_id, mask_id)
        return M, O, l, u


class Lasso(FeatureSelection):
    def __init__(self, name="lasso", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)

    def select_features(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
    ) -> list[int]:
        X, y = feature_matrix, response_vector
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        y = np.delete(y, O).reshape(-1, 1)

        # lasso
        lasso = lm.Lasso(
            alpha=self.parameters, fit_intercept=False, max_iter=5000, tol=1e-10
        )
        lasso.fit(X, y)
        active_set = np.where(lasso.coef_ != 0)[0]
        M = [M[i] for i in active_set]
        return M

    def perform_si_prev(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        feature_matrix: np.ndarray,
        selected_features: list[int],
        detected_outliers: list[int],
        l: float,
        u: float,
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ) -> tuple[list[int], list[int], float, float]:
        results = self.load_intervals(z, l, u, candidate_id, mask_id)
        if results is not None:
            return results

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        lasso = lm.Lasso(
            alpha=self.parameters, fit_intercept=False, max_iter=5000, tol=1e-10
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
            self.parameters * X.shape[0]
        )
        A0_minus = (-X_inactive.T @ (np.identity(X.shape[0]) - Pm)) / (
            self.parameters * X.shape[0]
        )
        b0_plus = np.ones(X_inactive.shape[1]) - X_inactive.T @ X_active_plus.T @ signs
        b0_minus = np.ones(X_inactive.shape[1]) + X_inactive.T @ X_active_plus.T @ signs

        A1 = -np.diag(signs) @ X_active_plus
        b1 = (
            -self.parameters
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
        for left, right in zip(left_list, right_list):
            if np.around(left, 5) == 0:
                if right <= 0:
                    raise ValueError("l must be less than u")
                continue
            term = right / left
            if left > 0:
                u_list.append(term)
            else:
                l_list.append(term)

        l = np.max(l_list)
        u = np.min(u_list)
        assert l < z < u, "l < z < u is not satisfied"

        M = [M[i] for i in active_set]

        self.save_intervals(l, u, M, O, candidate_id, mask_id)
        return M, O, l, u

    def compute_quotient(self, numerator, denominator):
        if denominator == 0:
            return np.inf
        quotient = float(numerator / denominator)
        if quotient <= 0:
            return np.inf
        return quotient

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
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ) -> tuple[list[int], list[int], float, float]:
        results = self.load_intervals(z, l, u, candidate_id, mask_id)
        if results is not None:
            return results

        X = feature_matrix
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]

        a, b = np.delete(a, O), np.delete(b, O)

        z_eval = np.max([l, z - 0.1])
        while z_eval < z:
            yz = a + b * (z_eval + 1e-5)

            lasso = lm.Lasso(
                alpha=self.parameters, fit_intercept=False, max_iter=5000, tol=1e-10
            )
            lasso.fit(X, yz)
            active_set = np.where(lasso.coef_ != 0)[0].tolist()
            inactive_set = [i for i in range(X.shape[1]) if i not in active_set]

            beta_active = lasso.coef_[active_set]

            X_active = X[:, active_set]
            X_inactive = X[:, inactive_set]

            etaAz = np.array([])

            if X_active is not None:
                etaAz = (np.linalg.inv(X_active.T @ X_active) @ X_active.T) @ b

            gammaAz = np.array([])
            shAz = np.array([])

            if X_inactive is not None:
                if X_active is None:
                    e1 = yz
                else:
                    e1 = yz - (X_active @ beta_active)

                e2 = X_inactive.T @ e1
                shAz = e2 / (self.parameters * X.shape[0])
                if X_active is None:
                    gammaAz = (X_inactive.T @ b) / X.shape[0]
                else:
                    gammaAz = (
                        (X_inactive.T @ b) - ((X_inactive.T @ X_active) @ etaAz)
                    ) / X.shape[0]

            min1 = np.inf
            min2 = np.inf

            for i in range(len(etaAz)):
                numerator = -beta_active[i]
                denominator = etaAz[i]

                quotient = self.compute_quotient(numerator, denominator)

                if quotient < min1:
                    min1 = quotient

            for j in range(len(gammaAz)):
                numerator = (np.sign(gammaAz[j]) - shAz[j]) * self.parameters
                denominator = gammaAz[j]
                quotient = self.compute_quotient(numerator, denominator)

                if quotient < min2:
                    min2 = quotient

            tz = np.min([min1, min2])
            M_updated = [M[i] for i in active_set]

            z_next = np.min([z_eval + tz, u])
            self.save_intervals(z_eval, z_next, M_updated, O, candidate_id, mask_id)
            if z_eval < z < z_next:
                pass
                # print(z_eval, z, z_next)
            z_eval = z_next

        return self.load_intervals(z, l, u, candidate_id, mask_id)


class ElasticNet(FeatureSelection):
    def __init__(self, name="elastic_net", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)


class Lars(FeatureSelection):
    def __init__(self, name="lars", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)


def stepwise_feature_selection(
    feature_matrix, response_vector, parameters=10, candidates=None
):
    return StepwiseFeatureSelection(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def marginal_screening(feature_matrix, response_vector, parameters=10, candidates=None):
    return MarginalScreening(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def lasso(feature_matrix, response_vector, parameters=0.1, candidates=None):
    return Lasso(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def elastic_net(feature_matrix, response_vector, parameters=None, candidates=None):
    return ElasticNet(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def lars(feature_matrix, response_vector, parameters=None, candidates=None):
    return Lars(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )
