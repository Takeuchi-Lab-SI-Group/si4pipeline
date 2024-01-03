import numpy as np
from source.base_component import OutlierDetection
from sicore import polytope_to_interval


class CookDistance(OutlierDetection):
    def __init__(self, name="cook_distance", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)

    def detect_outliers(
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

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        # cook's distance
        non_outlier = []
        outlier = []
        n, p = X.shape

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameters / n  # threshold value

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
        O = O + O_
        return O

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
    ) -> (list[int], list[int], float, float):
        if any(self.intervals):
            for interval, indexes in self.intervals.items():
                if interval[0] < z < interval[1]:
                    M, O = indexes
                    l = np.max([l, interval[0]])
                    u = np.min([u, interval[1]])
                    return M, O, l, u

        X, y = feature_matrix, a + b * z
        M, O = selected_features, detected_outliers

        X = np.delete(X, O, 0)
        X = X[:, M]
        yz = np.delete(y, O).reshape(-1, 1)

        a, b = np.delete(a, O), np.delete(b, O)

        num_data = list(range(X.shape[0]))
        num_outlier_data = [i for i in num_data if i not in O]

        non_outlier = []
        outlier = []
        n, p = X.shape

        hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
        Px = np.identity(n) - hat_matrix
        threshold = self.parameters / n  # threshold value

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
            H_2 = ((self.parameters * p * (1 - hi) ** 2) / n) * Px
            H = H_1 - H_2

            if i in outlier:
                H = -H

            intervals = polytope_to_interval(a, b, H, np.zeros(n), 0)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

        l = np.max(l_list)
        u = np.min(u_list)
        assert l < z < u, "l < z < u is not satisfied"

        O_ = [num_outlier_data[i] for i in outlier]
        O = O + O_
        self.intervals[(l, u)] = (M, O)
        return M, O, l, u


class Dffits(OutlierDetection):
    def __init__(self, name="dffits", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)


class SoftIpod(OutlierDetection):
    def __init__(self, name="soft_ipod", parameters=None, candidates=None):
        super().__init__(name, parameters, candidates)


def cook_distance(feature_matrix, response_vector, parameters=3.0, candidates=None):
    return CookDistance(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def dffits(feature_matrix, response_vector, parameters=None, candidates=None):
    return Dffits(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )


def soft_ipod(feature_matrix, response_vector, parameters=None, candidates=None):
    return SoftIpod(parameters=parameters, candidates=candidates)(
        feature_matrix, response_vector
    )
