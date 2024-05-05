import numpy as np
from source.base_component import (
    FeatureMatrix,
    ResponseVector,
    IndexesOperator,
    Union,
    Intersection,
    RemoveOutliers,
    ExtractFeatures,
    SelectedFeatures,
)
from source.missing_imputation import MissingImputation
from source.outlier_detection import OutlierDetection
from source.feature_selection import FeatureSelection
from graphlib import TopologicalSorter
from itertools import product
from sicore import SelectiveInferenceNorm
from sicore.intervals import poly_lt_zero


class PipelineStructure:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.components = {"start": None}
        self.current_node = "start"
        self.tuned = False

    def update(self, node, component):
        self.add_node(node, component)
        self.add_edge(self.current_node, node)
        self.current_node = node

    def add_node(self, node, component):
        self.nodes.add(node)
        self.components[node] = component

    def add_edge(self, sender, reciever):
        self.edges.add((sender, reciever))

    def make_graph(self):
        self.graph = dict()
        for edge in self.edges:
            self.graph.setdefault(edge[1], set()).add(edge[0])
        ts = TopologicalSorter(self.graph)
        self.static_order = list(ts.static_order())

    def __call__(self, feature_matrix: np.ndarray, response_vector: np.ndarray):
        outputs = dict()
        for node in self.static_order:
            if node == "start":
                selected_features = list(range(feature_matrix.shape[1]))
                detected_outliers = []
                outputs[node] = (selected_features, detected_outliers)

            elif isinstance(
                self.components[node], (FeatureSelection, OutlierDetection)
            ):
                layer = self.components[node]
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers = outputs[parants[0]]

                if isinstance(layer, FeatureSelection):
                    selected_features = layer.select_features(
                        feature_matrix,
                        response_vector,
                        selected_features,
                        detected_outliers,
                    )
                elif isinstance(layer, OutlierDetection):
                    detected_outliers = layer.detect_outliers(
                        feature_matrix,
                        response_vector,
                        selected_features,
                        detected_outliers,
                    )
                else:
                    raise TypeError(
                        "Input must be FeatureSelection or OutlierDetection"
                    )
                outputs[node] = (selected_features, detected_outliers)

            elif isinstance(self.components[node], MissingImputation):
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers = outputs[parants[0]]
                response_vector = self.components[node].impute_missing(
                    feature_matrix, response_vector
                )
                outputs[node] = (selected_features, detected_outliers)

            elif isinstance(self.components[node], IndexesOperator):
                layer = self.components[node]
                parants = list(self.graph[node])
                selected_features_list = []
                detected_outliers_list = []
                for parant in parants:
                    selected_features, detected_outliers = outputs[parant]
                    selected_features_list.append(selected_features)
                    detected_outliers_list.append(detected_outliers)
                if isinstance(layer, Union):
                    process = layer.union
                elif isinstance(layer, Intersection):
                    process = layer.intersection
                else:
                    raise TypeError("Input must be Union or Intersection")
                if layer.mode == "selected_features":
                    selected_features = process(*selected_features_list)
                    detected_outliers = detected_outliers_list[0]
                elif layer.mode == "detected_outliers":
                    selected_features = selected_features_list[0]
                    detected_outliers = process(*detected_outliers_list)
                outputs[node] = (selected_features, detected_outliers)

            elif isinstance(self.components[node], (RemoveOutliers, ExtractFeatures)):
                parents = list(self.graph[node])
                assert len(parents) == 1
                selected_features, detected_outliers = outputs[parents[0]]
                outputs[node] = (selected_features, detected_outliers)

            elif node == "end":
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers = outputs[parants[0]]
                return selected_features, detected_outliers
        raise ValueError("There is no end node")

    def reset_intervals(self):
        for node in self.static_order:
            if isinstance(self.components[node], (FeatureSelection, OutlierDetection)):
                self.components[node].reset_intervals()
        self.cv_quadratic = dict()

    def inference(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma=None,
        test_index=None,  # int from 0 to |self.M|-1
        is_result=False,
        **kwargs,
    ):
        if "step" not in kwargs:
            kwargs["step"] = 1e-6

        self.X, self.y = feature_matrix, response_vector
        self.M, self.O = self(feature_matrix, response_vector)

        # shape of imputer is (n, n - num_missing)
        node = self.static_order[1]
        if isinstance(self.components[node], MissingImputation):
            self.imputer = self.components[node].compute_imputer(self.X, self.y)
        else:
            self.imputer = np.eye(self.y.shape[0])

        if sigma is None:
            residuals = (
                self.imputer @ self.y[~np.isnan(self.y)]
                - self.X
                @ np.linalg.inv(self.X.T @ self.X)
                @ self.X.T
                @ self.imputer
                @ self.y[~np.isnan(self.y)]
            )
            sigma = np.std(residuals, ddof=self.X.shape[1])

        n = self.y.shape[0]
        X = np.delete(self.X, self.O, 0)  # shape (n - |O|, p)
        X = X[:, self.M]  # shape (n - |O|, |M|)
        Im = np.delete(np.eye(n), self.O, 0)  # shape (n - |O|, n)

        etas = np.linalg.inv(X.T @ X) @ X.T @ Im  # shape (|M|, n)
        self.etas = etas @ self.imputer  # shape (|M|, n - num_missing)

        # to delete
        for i in range(len(self.M)):
            assert np.allclose(
                etas[i] @ self.imputer @ self.y[~np.isnan(self.y)],
                self.etas[i] @ self.y[~np.isnan(self.y)],
            ), "etas"

        if test_index is not None:
            self.etas = [self.etas[test_index]]

        self.calculators = []
        results = []
        for eta in self.etas:
            self.reset_intervals()
            max_tail = 20 * np.sqrt((sigma**2.0) * eta @ eta)

            calculator = SelectiveInferenceNorm(
                self.y[~np.isnan(self.y)], sigma**2.0, eta
            )
            result = calculator.inference(
                self.algorithm,
                self.model_selector,
                max_tail=max_tail,
                **kwargs,
            )
            results.append(result)
            self.calculators.append(calculator)

        if test_index is None:
            if is_result:
                return self.M, results
            else:
                return self.M, [result.p_value for result in results]
        else:
            if is_result:
                return self.M[test_index], results[0]
            else:
                return self.M[test_index], results[0].p_value

    def selection_event(
        self,
        X: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        candidate_id: int | None = None,
        mask_id: int | None = None,
    ):
        outputs = dict()
        feature_matrix = X

        for node in self.static_order:
            if node == "start":
                selected_features = list(range(feature_matrix.shape[1]))
                detected_outliers = []
                outputs[node] = (selected_features, detected_outliers, -np.inf, np.inf)

            elif isinstance(
                self.components[node], (FeatureSelection, OutlierDetection)
            ):
                layer = self.components[node]
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers, l, u = outputs[parants[0]]
                selected_features, detected_outliers, l, u = layer.perform_si(
                    a,
                    b,
                    z,
                    feature_matrix,
                    selected_features,
                    detected_outliers,
                    l,
                    u,
                    candidate_id,
                    mask_id,
                )
                outputs[node] = (selected_features, detected_outliers, l, u)

            elif isinstance(
                self.components[node],
                (MissingImputation, RemoveOutliers, ExtractFeatures),
            ):
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers, l, u = outputs[parants[0]]
                outputs[node] = (selected_features, detected_outliers, l, u)

            elif isinstance(self.components[node], IndexesOperator):
                layer = self.components[node]
                parants = list(self.graph[node])
                selected_features_list = []
                detected_outliers_list = []
                l_list, u_list = [], []
                for parant in parants:
                    selected_features, detected_outliers, l, u = outputs[parant]
                    selected_features_list.append(selected_features)
                    detected_outliers_list.append(detected_outliers)
                    l_list.append(l)
                    u_list.append(u)
                if isinstance(layer, Union):
                    process = layer.union
                elif isinstance(layer, Intersection):
                    process = layer.intersection
                else:
                    raise TypeError("Input must be Union or Intersection")
                if layer.mode == "selected_features":
                    selected_features = process(*selected_features_list)
                    detected_outliers = detected_outliers_list[0]
                elif layer.mode == "detected_outliers":
                    selected_features = selected_features_list[0]
                    detected_outliers = process(*detected_outliers_list)
                l, u = np.max(l_list), np.min(u_list)
                outputs[node] = (selected_features, detected_outliers, l, u)

            elif node == "end":
                parants = list(self.graph[node])
                assert len(parants) == 1
                selected_features, detected_outliers, l, u = outputs[parants[0]]

        if node != "end":
            raise ValueError("There is no end node")

        return selected_features, detected_outliers, l, u

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float):
        a, b = self.imputer @ a, self.imputer @ b
        selected_features, detected_outliers, l, u = self.selection_event(
            self.X, a, b, z, None, None
        )

        if not self.tuned:
            return (selected_features, detected_outliers), [l, u]
        else:
            (
                selected_candidate,
                selected_candidate_id,
                l,
                u,
                quadratic_at_each_candidate,
                _,
            ) = self.cross_validate_error(a, b, z, l, u)
            l_list, u_list = [l], [u]
            selected_quadratic = quadratic_at_each_candidate[selected_candidate_id]
            for candidate_id in range(self.n_iter):
                # check
                intervals = poly_lt_zero(
                    selected_quadratic - quadratic_at_each_candidate[candidate_id]
                )
                for interval in intervals:
                    if interval[0] < z < interval[1]:
                        l_list.append(interval[0])
                        u_list.append(interval[1])
            l, u = np.max(l_list), np.min(u_list)
            return (selected_features, detected_outliers, selected_candidate), [l, u]

    def cross_validate_error(self, a, b, z, l, u):
        feature_matrix = self.X
        old_mse = np.inf
        quadratic_at_each_candidate = dict()
        l_list, u_list = [l], [u]
        for candidate_id in range(self.n_iter):
            candidate = self.candidates[candidate_id]
            self.cv_quadratic.setdefault(candidate_id, dict())
            quadratic_list = []
            for mask_id in range(self.cv):
                mask = self.cv_masks[mask_id]
                self.cv_quadratic[candidate_id].setdefault(mask_id, dict())

                self.set_parameters(candidate)

                flag = False
                for interval in self.cv_quadratic[candidate_id][mask_id].keys():
                    if interval[0] < z < interval[1]:
                        l_list.append(interval[0])
                        u_list.append(interval[1])
                        quadratic_list.append(
                            self.cv_quadratic[candidate_id][mask_id][interval]
                        )
                        flag = True
                        break
                if flag:
                    continue  # if flag is True, go to next mask_id

                X_tr, a_tr, b_tr = feature_matrix[mask], a[mask], b[mask]
                (
                    selected_features_cv,
                    detected_outliers_cv,
                    l_cv,
                    u_cv,
                ) = self.selection_event(X_tr, a_tr, b_tr, z, candidate_id, mask_id)
                l_list.append(l_cv)
                u_list.append(u_cv)

                X_tr = np.delete(X_tr, detected_outliers_cv, 0)
                a_tr = np.delete(a_tr, detected_outliers_cv)
                b_tr = np.delete(b_tr, detected_outliers_cv)
                X_val = np.delete(feature_matrix, mask, 0)
                a_val = np.delete(a, mask)
                b_val = np.delete(b, mask)
                num = X_val.shape[0]

                if len(selected_features_cv) == 0:
                    quadratic = [
                        b_val @ b_val / num,
                        2 * b_val @ a_val / num,
                        a_val @ a_val / num,
                    ]
                else:
                    F = (
                        X_tr[:, selected_features_cv]
                        @ np.linalg.inv(
                            X_tr[:, selected_features_cv].T
                            @ X_tr[:, selected_features_cv]
                        )
                        @ X_val[:, selected_features_cv].T
                    )
                    G = F @ F.T
                    alpha = b_val @ b_val - 2 * b_tr @ F @ b_val + b_tr @ G @ b_tr
                    beta = (
                        2 * b_val @ a_val
                        - 2 * b_tr @ F @ a_val
                        - 2 * a_tr @ F @ b_val
                        + 2 * a_tr @ G @ b_tr
                    )
                    gamma = a_val @ a_val - 2 * a_tr @ F @ a_val + a_tr @ G @ a_tr
                    quadratic = [alpha / num, beta / num, gamma / num]
                self.cv_quadratic[candidate_id][mask_id][(l_cv, u_cv)] = quadratic
                quadratic_list.append(quadratic)

            quadratic_at_each_candidate[candidate_id] = np.mean(quadratic_list, axis=0)
            alpha, beta, gamma = quadratic_at_each_candidate[candidate_id]
            mse = alpha * z**2 + beta * z + gamma
            # print(mse, candidate)  # activate
            if mse < old_mse:
                old_mse = mse
                selected_candidate = candidate
                selected_candidate_id = candidate_id
                selected_quadratic = (alpha, beta, gamma)
            selected_quadratic = np.array(selected_quadratic)

        # print(selected_candidate, old_mse)
        self.set_parameters(self.best_candidate)
        return (
            selected_candidate,
            selected_candidate_id,
            np.max(l_list),
            np.min(u_list),
            quadratic_at_each_candidate,
            old_mse,
        )

    def model_selector(self, indexes):
        if not self.tuned:
            M, O = indexes
            return (set(M) == set(self.M)) and (set(O) == set(self.O))
        else:
            M, O, candidate = indexes
            return (
                set(M) == set(self.M)
                and set(O) == set(self.O)
                # and candidate == self.best_candidate
            )

    def __or__(self, other):
        if isinstance(other, PipelineStructure):
            pl = PipelineStructure()
            pl.nodes = self.nodes | other.nodes
            pl.edges = self.edges | other.edges
            pl.components = {**self.components, **other.components}
            if self.nodes >= other.nodes:
                pl.current_node = self.current_node
            else:
                pl.current_node = other.current_node
            return pl
        else:
            raise TypeError("Input must be PipelineStructure")

    def __str__(self):
        edge_list = []
        for node in self.static_order:
            for edge in self.edges:
                if edge[0] == node:
                    edge_list.append(f"{edge[0]} -> {edge[1]}")
        return "\n".join(edge_list)

    def tune(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        n_iter=10,
        cv=5,
        random_state=None,
    ):
        X, y = feature_matrix, response_vector
        node = self.static_order[1]
        if isinstance(self.components[node], MissingImputation):
            self.imputer = self.components[node].compute_imputer(X, y)
        else:
            self.imputer = np.eye(y.shape[0])
        y = self.imputer @ y[~np.isnan(y)]

        n = response_vector.shape[0]

        self.cv = cv
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)

        self.make_candidates()
        self.cv_masks = np.array_split(self.rng.permutation(n), self.cv)

        mse_at_each_candidate = []
        for candidate in self.candidates:
            mse_list = []
            for mask in self.cv_masks:
                self.set_parameters(candidate)

                X_tr, y_tr = X[mask], y[mask]
                X_val, y_val = np.delete(X, mask, 0), np.delete(y, mask)
                M, O = self(X_tr, y_tr)
                X_tr, y_tr = np.delete(X_tr, O, 0), np.delete(y_tr, O)
                if len(M) == 0:
                    mse_list.append(np.mean(y_val**2))
                else:
                    y_error = (
                        y_val
                        - X_val[:, M]
                        @ np.linalg.inv(X_tr[:, M].T @ X_tr[:, M])
                        @ X_tr[:, M].T
                        @ y_tr
                    )
                    mse_list.append(np.mean(y_error**2))
            mse_at_each_candidate.append(np.mean(mse_list))

        # print(mse_at_each_candidate)  # activate
        best_index = np.argmin(mse_at_each_candidate)
        self.best_mse = mse_at_each_candidate[best_index]
        self.best_candidate = self.candidates[best_index]
        self.set_parameters(self.best_candidate)
        self.tuned = True

    def make_candidates(self):
        self.candidates = []

        rolled_candidates = self.rollout_candidates()
        if not any(rolled_candidates):
            if self.n_iter == 1:
                self.candidates = [dict()]
                return None
            else:
                raise ValueError(
                    "There is no candidates. Please set candidates when define the pipiline."
                )

        finite_keys, finite_candidates = [], []
        dist_keys, dist_candidates = [], []
        for node in rolled_candidates.keys():
            candidates = rolled_candidates[node]
            if isinstance(candidates, (list, set, tuple)):
                finite_keys.append(node)
                finite_candidates.append(candidates)
            elif hasattr(candidates, "rvs"):
                dist_keys.append(node)
                dist_candidates.append(candidates)
            else:
                raise TypeError(
                    "Candidates must be list, set, tuple or has rvs method."
                )

        finite_candidates_grids = list(product(*finite_candidates))
        num_grids = len(finite_candidates_grids)

        # finite_dict = dict(zip(finite_keys, finite_candidates))
        dist_dict = dict(zip(dist_keys, dist_candidates))
        if num_grids < self.n_iter and len(dist_dict) == 0:
            raise ValueError("The number of candidates must be larger than n_iter.")
        elif num_grids >= self.n_iter and len(dist_dict) == 0:
            indexes = self.rng.choice(num_grids, size=self.n_iter, replace=False)
            for index in indexes:
                self.candidates.append(
                    dict(zip(finite_keys, finite_candidates_grids[index]))
                )
        else:
            for _ in range(self.n_iter):
                i = self.rng.choice(num_grids)
                temp_dict = dict(zip(finite_keys, finite_candidates_grids[i]))
                for key, dist in dist_dict.items():
                    temp_dict[key] = dist.rvs()
                self.candidates.append(temp_dict)

    def rollout_candidates(self):
        candidates_dict = dict()
        for node in self.static_order:
            if isinstance(self.components[node], (FeatureSelection, OutlierDetection)):
                candidates = self.components[node].candidates
                if candidates is not None:
                    candidates_dict[node] = candidates
        return candidates_dict

    def set_parameters(self, candidates):
        for node, parameters in candidates.items():
            self.components[node].parameters = parameters


class MultiPipelineStructure:
    def __init__(self, *pipelines):
        self.pipelines = pipelines
        self.tuned = False

    def __call__(self, feature_matrix: np.ndarray, response_vector: np.ndarray):
        if self.tuned:
            return self.pipelines[self.best_index](feature_matrix, response_vector)
        else:
            outputs = []
            for pipeline in self.pipelines:
                outputs.append(pipeline(feature_matrix, response_vector))
            return outputs

    def reset_intervals(self):
        for pipeline in self.pipelines:
            pipeline.reset_intervals()

    def inference(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma=None,
        test_index=None,  # int from 0 to |self.M|-1
        is_result=False,
        **kwargs,
    ):
        if not self.tuned:
            raise ValueError("Please tune the pipelines before inference.")

        if "step" not in kwargs:
            kwargs["step"] = 1e-6

        self.X, self.y = feature_matrix, response_vector
        for pipeline in self.pipelines:
            pipeline.X = feature_matrix

        self.M, self.O = self(feature_matrix, response_vector)

        pipeline = self.pipelines[self.best_index]
        # shape of imputer is (n, n - num_missing)
        node = pipeline.static_order[1]
        if isinstance(pipeline.components[node], MissingImputation):
            self.imputer = pipeline.components[node].compute_imputer(self.X, self.y)
        else:
            self.imputer = np.eye(self.y.shape[0])

        if sigma is None:
            residuals = (
                self.imputer @ self.y[~np.isnan(self.y)]
                - self.X
                @ np.linalg.inv(self.X.T @ self.X)
                @ self.X.T
                @ self.imputer
                @ self.y[~np.isnan(self.y)]
            )
            sigma = np.std(residuals, ddof=self.X.shape[1])

        n = self.y.shape[0]
        X = np.delete(self.X, self.O, 0)  # shape (n - |O|, p)
        X = X[:, self.M]  # shape (n - |O|, |M|)
        Im = np.delete(np.eye(n), self.O, 0)  # shape (n - |O|, n)

        etas = np.linalg.inv(X.T @ X) @ X.T @ Im  # shape (|M|, n)
        self.etas = etas @ self.imputer  # shape (|M|, n - num_missing)

        # to delete
        for i in range(len(self.M)):
            assert np.allclose(
                etas[i] @ self.imputer @ self.y[~np.isnan(self.y)],
                self.etas[i] @ self.y[~np.isnan(self.y)],
            ), "etas"

        if test_index is not None:
            self.etas = [self.etas[test_index]]

        self.calculators = []
        results = []
        # return None
        for eta in self.etas:
            self.reset_intervals()
            max_tail = 20 * np.sqrt((sigma**2.0) * eta @ eta)

            calculator = SelectiveInferenceNorm(
                self.y[~np.isnan(self.y)], sigma**2.0, eta
            )
            result = calculator.inference(
                self.algorithm,
                self.model_selector,
                max_tail=max_tail,
                **kwargs,
            )
            results.append(result)
            self.calculators.append(calculator)

        if test_index is None:
            if is_result:
                return self.M, results
            else:
                return self.M, [result.p_value for result in results]
        else:
            if is_result:
                return self.M[test_index], results[0]
            else:
                return self.M[test_index], results[0].p_value

    def tune(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        n_iters=10,
        cv=5,
        random_state=None,
    ):
        if isinstance(n_iters, int):
            n_iters = [n_iters] * len(self.pipelines)
        elif isinstance(n_iters, (list, tuple)):
            assert len(n_iters) == len(self.pipelines)

        for pipeline, n_iter in zip(self.pipelines, n_iters):
            pipeline.tune(feature_matrix, response_vector, n_iter, cv, random_state)
        self.best_index = np.argmin([pipeline.best_mse for pipeline in self.pipelines])
        self.tuned = True

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float):
        feature_matrix = self.X
        old_mse = np.inf
        quadratic_list = []

        selected_features, detected_outliers, l, u = self.pipelines[
            self.best_index
        ].selection_event(
            feature_matrix, self.imputer @ a, self.imputer @ b, z, None, None
        )

        l_list, u_list = [l], [u]
        for i in range(len(self.pipelines)):
            pipeline = self.pipelines[i]
            imputer = pipeline.imputer
            (
                selected_candidate,
                selected_candidate_id,
                l_cv,
                u_cv,
                quadratic_at_each_candidate,
                mse,
            ) = pipeline.cross_validate_error(imputer @ a, imputer @ b, z, l, u)
            l_list.append(l_cv)
            u_list.append(u_cv)

            quadratic_list.append(quadratic_at_each_candidate[selected_candidate_id])
            if mse < old_mse:
                old_mse = mse
                index = i
                candidate = selected_candidate
                selected_quadratic = quadratic_at_each_candidate[selected_candidate_id]

        l_list, u_list = [np.max(l_list)], [np.min(u_list)]

        assert l_list[0] < z < u_list[0], "1: l < z < u"

        for quadratic in quadratic_list:
            intervals = poly_lt_zero(selected_quadratic - quadratic)
            for interval in intervals:
                if interval[0] < z < interval[1]:
                    l_list.append(interval[0])
                    u_list.append(interval[1])
        l, u = np.max(l_list), np.min(u_list)
        assert l < z < u, "2: l < z < u"
        return (selected_features, detected_outliers, index, candidate), [l, u]

    def model_selector(self, indexes_pipelines):
        M, O, index, candidate = indexes_pipelines
        return (
            set(M) == set(self.M)
            and set(O) == set(self.O)
            # and self.best_index == index
            # and self.pipelines[self.best_index].best_candidate == candidate
        )

    def __str__(self):
        return "\n\n".join([str(pipeline) for pipeline in self.pipelines])


def make_dataset():
    feature_matrix = FeatureMatrix(PipelineStructure())
    response_vector = ResponseVector(PipelineStructure())
    return feature_matrix, response_vector


def make_pipeline(output: SelectedFeatures):
    pl_structure = output.pl_structure
    pl_structure.update("end", None)
    pl_structure.make_graph()
    return pl_structure


def make_pipelines(*pipelines: list[PipelineStructure]):
    return MultiPipelineStructure(*pipelines)
