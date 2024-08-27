"""Module containing an abstract classes for the components of the data analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import ClassVar, Literal


@dataclass(frozen=True)
class Node:
    """A class for the node of the data analysis pipeline."""

    type: Literal[
        "start",
        "end",
        "feature_extraction",
        "outlier_removal",
        "missing_imputation",
        "feature_selection",
        "outlier_detection",
        "index_operation",
    ]
    method: str = ""
    parameters: frozenset[float] | frozenset[int] | None = None
    count: int | None = None

    @property
    def name(self) -> str:
        """Return the name of the node."""
        if self.method == "":
            if self.count is None:
                return f"{self.type}"
            return f"{self.type}_{self.count}"
        return f"{self.method}_{self.count}"


class Structure:
    """An abstract class for the structure of the data analysis pipeline."""

    def __init__(self) -> None:
        """Initialize the Structure object."""
        self.graph: dict[Node, set[Node]] = {Node("start"): set()}
        self.current_node = Node("start")

    def update(self, node: Node) -> None:
        """Update the structure of the data analysis pipeline."""
        self.graph.setdefault(node, set()).add(self.current_node)
        self.current_node = node

    def __or__(self, other: Structure) -> Structure:
        """Take the union of the structures of the data analysis pipelines."""
        structure = Structure()

        for key in self.graph.keys() | other.graph.keys():
            structure.graph[key] = self.graph.get(key, set()) | other.graph.get(
                key,
                set(),
            )
        structure.current_node = (
            self.current_node
            if self.graph.keys() >= other.graph.keys()
            else other.current_node
        )
        return structure

    def sort_graph(self) -> None:
        """Topologically sort the graph of the data analysis pipeline."""
        ts = TopologicalSorter(self.graph)
        self.graph = {node: self.graph[node] for node in ts.static_order()}


class FeatureMatrix:
    """An abstract class for the feature matrix."""

    def __init__(self, structure: Structure) -> None:
        """Initialize the FeatureMatrix object."""
        self.structure = structure


class ResponseVector:
    """An abstract class for the response vector."""

    def __init__(self, structure: Structure) -> None:
        """Initialize the ResponseVector object."""
        self.structure = structure


class SelectedFeatures:
    """An abstract class for the selected features."""

    def __init__(self, structure: Structure) -> None:
        """Initialize the SelectedFeatures object."""
        self.structure = structure


class DetectedOutliers:
    """An abstract class for the detected outliers."""

    def __init__(self, structure: Structure) -> None:
        """Initialize the DetectedOutliers object."""
        self.structure = structure


class MissingImputationConstructor:
    """A class for constructing the abstract missing imputation."""

    count: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the MissingImputationConstructor object."""

    def __call__(
        self,
        name: Literal[
            "mean_value_imputation",
            "euclidean_imputation",
            "manhattan_imputation",
            "chebyshev_imputation",
            "definite_regression_imputation",
        ],
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
    ) -> ResponseVector:
        """Perform the missing imputation on the feature matrix and response vector."""
        structure = feature_matrix.structure | response_vector.structure
        node = Node(
            "missing_imputation",
            name,
            count=MissingImputationConstructor.count,
        )
        structure.update(node)
        MissingImputationConstructor.count += 1
        return ResponseVector(structure)


class FeatureSelectionConstructor:
    """A class for constructing the abstract feature selection."""

    counter: ClassVar[dict[str, int]] = {}

    def __init__(self) -> None:
        """Initialize the FeatureSelectionConstructor object."""

    def __call__(
        self,
        name: Literal["marginal_screening", "stepwise_feature_selection", "lasso"],
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        parameters: float | list[int] | list[float],
    ) -> SelectedFeatures:
        """Perform the feature selection on the feature matrix and response vector."""
        structure = feature_matrix.structure | response_vector.structure
        parameters = parameters if isinstance(parameters, list) else [parameters]
        node = Node(
            "feature_selection",
            name,
            frozenset(parameters),
            FeatureSelectionConstructor.counter.get(name, 0),
        )
        structure.update(node)

        FeatureSelectionConstructor.counter.setdefault(name, 0)
        FeatureSelectionConstructor.counter[name] += 1
        return SelectedFeatures(structure)


class OutlierDetectionConstructor:
    """A class for constructing the abstract outlier detection."""

    counter: ClassVar[dict[str, int]] = {}

    def __init__(self) -> None:
        """Initialize the OutlierDetectionConstructor object."""

    def __call__(
        self,
        name: Literal["cook_distance", "dffits", "soft_ipod"],
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        parameters: float | list[int] | list[float],
    ) -> DetectedOutliers:
        """Perform the outlier detection on the feature matrix and response vector."""
        structure = feature_matrix.structure | response_vector.structure
        parameters = parameters if isinstance(parameters, list) else [parameters]
        node = Node(
            "outlier_detection",
            name,
            frozenset(parameters),
            OutlierDetectionConstructor.counter.get(name, 0),
        )
        structure.update(node)

        OutlierDetectionConstructor.counter.setdefault(name, 0)
        OutlierDetectionConstructor.counter[name] += 1
        return DetectedOutliers(structure)


class IndexOperationConstructor:
    """A class for constructing the abstract index operation."""

    counter: ClassVar[dict[str, int]] = {}

    def __init__(self) -> None:
        """Initialize the IndexOperationConstructor object."""

    def __call__(
        self,
        name: Literal[
            "union_features",
            "union_outliers",
            "intersection_features",
            "intersection_outliers",
        ],
        *inputs: SelectedFeatures | DetectedOutliers,
    ) -> SelectedFeatures | DetectedOutliers:
        """Perform the index operation on the selected features or detected outliers."""
        assert all(type(inputs[0]) is type(input_) for input_ in inputs)

        node = Node(
            "index_operation",
            name,
            count=IndexOperationConstructor.counter.get(name, 0),
        )

        structure = inputs[0].structure
        for input_ in inputs:
            input_.structure.update(node)
            structure = structure | input_.structure

        IndexOperationConstructor.counter.setdefault(name, 0)
        IndexOperationConstructor.counter[name] += 1

        if "features" in name:
            return SelectedFeatures(structure)
        if "outliers" in name:
            return DetectedOutliers(structure)
        raise ValueError


class FeatureExtractionConstructor:
    """A class for constructing the abstract feature extraction."""

    count: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the FeatureExtractionConstructor object."""

    def __call__(
        self,
        feature_matrix: FeatureMatrix,
        selected_features: SelectedFeatures,
    ) -> FeatureMatrix:
        """Perform the feature extraction on the feature matrix based on the selected features."""
        structure = feature_matrix.structure | selected_features.structure
        node = Node("feature_extraction", count=FeatureExtractionConstructor.count)
        structure.update(node)
        FeatureExtractionConstructor.count += 1
        return FeatureMatrix(structure)


class OutlierRemovalConstructor:
    """A class for constructing the abstract outlier removal."""

    count: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the OutlierRemovalConstructor object."""

    def __call__(
        self,
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        detected_outliers: DetectedOutliers,
    ) -> tuple[FeatureMatrix, ResponseVector]:
        """Perform the outlier removal on the feature matrix and the response vector based on the detected outliers."""
        structure = (
            feature_matrix.structure
            | response_vector.structure
            | detected_outliers.structure
        )
        node = Node("outlier_removal", count=OutlierRemovalConstructor.count)
        structure.update(node)
        OutlierRemovalConstructor.count += 1
        return FeatureMatrix(structure), ResponseVector(structure)
