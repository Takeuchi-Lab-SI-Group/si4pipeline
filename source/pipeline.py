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

    def inference(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma: float,
        test_index=None,  # int from 0 to |self.M|-1
        is_result=False,
        **kwargs,
    ):
        if "step" not in kwargs:
            kwargs["step"] = 1e-6

        self.X, self.y, self.cov = feature_matrix, response_vector, sigma**2
        self.M, self.O = self(feature_matrix, response_vector)

        node = self.static_order[1]
        if isinstance(self.components[node], MissingImputation):
            self.y, self.cov = self.components[node].compute_covariance(
                feature_matrix, response_vector, sigma
            )

        n = self.y.shape[0]
        X = np.delete(self.X, self.O, 0)
        X = X[:, self.M]
        Im = np.delete(np.eye(n), self.O, 0)

        self.etas = np.linalg.inv(X.T @ X) @ X.T @ Im
        if test_index is not None:
            self.etas = [self.etas[test_index]]

        self.calculators = []
        results = []
        for eta in self.etas:
            self.reset_intervals()

            if len(np.array(self.cov).shape) == 0:
                stat_sigma = np.sqrt(self.cov * eta @ eta)
            else:
                stat_sigma = np.sqrt(eta @ self.cov @ eta)
            max_tail = 20 * stat_sigma

            calculator = SelectiveInferenceNorm(self.y, self.cov, eta)
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
        return self.M[test_index], results[0]

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float):
        outputs = dict()
        feature_matrix = self.X

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
                    a, b, z, feature_matrix, selected_features, detected_outliers, l, u
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
        if not self.tuned:
            return (selected_features, detected_outliers), [l, u]
        else:
            pass

    def model_selector(self, indexes):
        if not self.tuned:
            M, O = indexes
            return (set(M) == set(self.M)) and (set(O) == set(self.O))
        else:
            pass

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
                self.reset_intervals()
                self.set_parameters(candidate)

                X_tr, y_tr = X[mask], y[mask]
                X_val, y_val = np.delete(X, mask, 0), np.delete(y, mask)
                M, _ = self(X_tr, y_tr)
                if len(M) == 0:
                    mse_list.append(np.inf)
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

        best_index = np.argmin(mse_at_each_candidate)
        self.best_mse = mse_at_each_candidate[best_index]
        self.best_candidate = self.candidates[best_index]
        self.set_parameters(self.best_candidate)
        self.tuned = True

        # print(self.candidates)
        # print(mse_at_each_candidate)

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

    def inference(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma: float,
        test_index=None,  # int from 0 to |self.M|-1
        is_result=False,
        **kwargs,
    ):
        if self.tuned:
            return self.pipelines[self.best_index].inference(
                feature_matrix, response_vector, sigma, test_index, is_result, **kwargs
            )
        else:
            raise ValueError("Please tune the pipelines before inference.")

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
        pass

    def model_selector(self, indexes_pipelines):
        M, O, index, candidate = indexes_pipelines
        return (
            set(M) == set(self.M)
            and set(O) == set(self.O)
            and self.best_index == index
            and self.pipelines[self.best_index].best_candidate == candidate
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
