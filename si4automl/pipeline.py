"""Module containing entity for the data analysis pipeline and manager of it."""

from itertools import product
from typing import cast

import numpy as np

from si4automl.abstract import Node, Structure
from si4automl.feature_selection import FeatureSelection
from si4automl.index_operation import IndexOperation
from si4automl.missing_imputation import MissingImputation
from si4automl.outlier_detection import OutlierDetection
from si4automl.utils import conver_entities


class Pipeline:
    """An entity class for the data analysis pipeline."""

    def __init__(
        self,
        static_order: list[Node],
        graph: dict[Node, set[Node]],
        layers: dict[
            Node,
            MissingImputation
            | FeatureSelection
            | OutlierDetection
            | IndexOperation
            | None,
        ],
    ) -> None:
        """Initialize the Pipeline object."""
        self.static_order = static_order
        self.graph = graph
        self.layers = layers

        self.cache_cv_error: dict[int, list[float]] = {}
        self._validate()

    def _validate(self) -> None:
        """Validate the Pipeline object."""
        assert self.static_order[0].type == "start"
        assert self.static_order[-1].type == "end"
        for node in self.static_order:
            parents = list(self.graph[node])
            match node.type:
                case "start":
                    assert not parents
                case "end":
                    assert len(parents) == 1
                case "feature_extraction" | "outlier_removal":
                    assert len(parents) == 1
                case "missing_imputation":
                    assert len(parents) == 1
                    assert isinstance(self.layers[node], MissingImputation)
                case "feature_selection":
                    assert len(parents) == 1
                    assert isinstance(self.layers[node], FeatureSelection)
                case "outlier_detection":
                    assert len(parents) == 1
                    assert isinstance(self.layers[node], OutlierDetection)
                case "index_operation":
                    assert len(parents) >= 1
                    assert isinstance(self.layers[node], IndexOperation)
                case _:
                    raise ValueError

    def __call__(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """Perform the data analysis pipeline on the given feature matrix and response vector."""
        outputs: dict[Node, tuple[list[int], list[int]]] = {}
        for node in self.static_order:
            layer = self.layers[node]
            parents = list(self.graph[node])
            match node.type:
                case "start":
                    outputs[node] = (list(range(feature_matrix.shape[1])), [])
                case "end":
                    return outputs[parents[0]]
                case "feature_extraction" | "outlier_removal":
                    outputs[node] = outputs[parents[0]]
                case "missing_imputation":
                    assert isinstance(layer, MissingImputation)
                    response_vector = layer.impute_missing(
                        feature_matrix,
                        response_vector,
                    )
                    outputs[node] = outputs[parents[0]]
                case "feature_selection":
                    assert isinstance(layer, FeatureSelection)
                    selected_features, detected_outliers = outputs[parents[0]]
                    selected_features = layer.select_features(
                        feature_matrix,
                        response_vector,
                        selected_features,
                        detected_outliers,
                    )
                    outputs[node] = (selected_features, detected_outliers)
                case "outlier_detection":
                    assert isinstance(layer, OutlierDetection)
                    selected_features, detected_outliers = outputs[parents[0]]
                    detected_outliers = layer.detect_outliers(
                        feature_matrix,
                        response_vector,
                        selected_features,
                        detected_outliers,
                    )
                    outputs[node] = (selected_features, detected_outliers)
                case "index_operation":
                    assert isinstance(layer, IndexOperation)
                    outputs[node] = layer.index_operation(
                        *[outputs[parent] for parent in parents],
                    )
                case _:
                    raise ValueError
        raise ValueError

    def selection_event(
        self,
        X: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        mask_id: int = -1,
    ) -> tuple[list[int], list[int], float, float]:
        """Compute the selection event."""
        outputs: dict[Node, tuple[list[int], list[int], float, float]] = {}
        for node in self.static_order:
            layer = self.layers[node]
            parents = list(self.graph[node])
            match node.type:
                case "start":
                    outputs[node] = (list(range(X.shape[1])), [], -np.inf, np.inf)
                case "end":
                    return outputs[parents[0]]
                case "feature_extraction" | "outlier_removal" | "missing_imputation":
                    outputs[node] = outputs[parents[0]]
                case "feature_selection" | "outlier_detection":
                    assert isinstance(layer, FeatureSelection | OutlierDetection)
                    selected_features, detected_outliers, l, u = outputs[parents[0]]
                    outputs[node] = layer.perform_si(
                        a,
                        b,
                        z,
                        X,
                        selected_features,
                        detected_outliers,
                        l,
                        u,
                        mask_id,
                    )
                case "index_operation":
                    assert isinstance(layer, IndexOperation)
                    selected_features, detected_outliers = layer.index_operation(
                        *[outputs[parent][:2] for parent in parents],
                    )
                    l_list, u_list = zip(
                        *[outputs[parent][2:] for parent in parents],
                        strict=True,
                    )
                    outputs[node] = (
                        selected_features,
                        detected_outliers,
                        np.max(l_list).item(),
                        np.min(u_list).item(),
                    )
                case _:
                    raise ValueError

        raise ValueError

    def __str__(self) -> str:
        """Return the string representation of the Pipeline object."""
        edge_list = []
        for sender in self.static_order:
            for reciever, value in self.graph.items():
                if sender in value:
                    edge_list.append(f"{sender.name} -> {reciever.name}")
        return "\n".join(edge_list)


class PipelineManager:
    """A class to manage the data analysis pipelines."""

    def __init__(self, structure: Structure) -> None:
        """Initialize the PipelineManager object."""
        self.static_order = structure.static_order
        self.graph = structure.graph

        self.pipelines = []
        layers_list = product(*[conver_entities(node) for node in self.static_order])
        for layers_ in layers_list:
            layers_ = cast(
                tuple[
                    MissingImputation
                    | FeatureSelection
                    | OutlierDetection
                    | IndexOperation
                    | None
                ],
                layers_,
            )
            pipeline = Pipeline(
                static_order=self.static_order,
                graph=self.graph,
                layers=dict(zip(self.static_order, layers_, strict=True)),
            )
            self.pipelines.append(pipeline)
        self.representeing_index = 0
        self.tuned = False

    def __call__(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """Perform the representing data analysis pipeline on the given feature matrix and response vector."""
        assert self.tuned or len(self.pipelines) == 1
        return self.pipelines[self.representeing_index](feature_matrix, response_vector)

    def __str__(self) -> str:
        """Return the string representation of the PipelineManager object."""
        return (
            f"PipelineManager with {len(self.pipelines)} pipelines:\n"
            "Representing pipelines:\n"
            f"{self.pipelines[self.representeing_index]}"
        )
