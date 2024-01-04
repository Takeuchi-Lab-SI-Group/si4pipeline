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
from sicore import SelectiveInferenceNorm


class PipelineStructure:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.components = {"start": None}
        self.current_node = "start"

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
        **kwargs,
    ):
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

        if node == "end":
            return (selected_features, detected_outliers), [l, u]
        raise ValueError("There is no end node")

    def model_selector(self, indexes):
        M, O = indexes
        return (set(M) == set(self.M)) and (set(O) == set(self.O))

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


def make_dataset():
    feature_matrix = FeatureMatrix(PipelineStructure())
    response_vector = ResponseVector(PipelineStructure())
    return feature_matrix, response_vector


def make_pipeline(output: SelectedFeatures):
    pl_structure = output.pl_structure
    pl_structure.update("end", None)
    pl_structure.make_graph()
    return pl_structure
