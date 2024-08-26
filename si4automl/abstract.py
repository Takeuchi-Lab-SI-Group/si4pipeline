"""Module containing an abstract classes for the components of the data analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import ClassVar, Literal


@dataclass
class Config:
    """A class for the configuration of the node of the data analysis pipeline."""

    name: str
    parameters: float | list[int] | list[float] | None

    def __post_init__(self) -> None:
        """Post-initialize the Config object."""
        if self.parameters is None or isinstance(self.parameters, list):
            self.parameters_ = self.parameters
        else:
            self.parameters_ = [self.parameters]

    # def entities(self) -> list[float] | list[int] | None:
    #     """Return the entities of the node."""
    #     return self.parameters_


class Structure:
    """An abstract class for the structure of the data analysis pipeline."""

    def __init__(self) -> None:
        """Initialize the Structure object."""
        self.graph: dict[str, set[str]] = {}
        self.configs: dict[str, Config] = {"start": Config("start", None)}
        self.current_node = "start"

    def update(self, node: str, config: Config) -> None:
        """Update the structure of the data analysis pipeline."""
        self.graph.setdefault(node, set()).add(self.current_node)
        self.configs[node] = config
        self.current_node = node

    def __or__(self, other: Structure) -> Structure:
        """Take the union of the structures of the data analysis pipelines."""
        structure = Structure()

        for key in self.graph.keys() | other.graph.keys():
            structure.graph[key] = self.graph.get(key, set()) | other.graph.get(
                key,
                set(),
            )
        structure.configs = {**self.configs, **other.configs}
        structure.current_node = (
            self.current_node
            if self.graph.keys() >= other.graph.keys()
            else other.current_node
        )
        return structure

    def make_sorted_node_list(self) -> None:
        """Make the topologically sorted graph of the data analysis pipeline."""
        ts = TopologicalSorter(self.graph)
        self.static_order = list(ts.static_order())


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


class IndexOperationConstructor:
    """A class for constructing the abstract index operation."""

    counter: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the IndexOperationConstructor object."""

    def __call__(
        self,
        name: Literal[
            "union_of_features",
            "union_of_outliers",
            "intersection_of_features",
            "intersection_of_outliers",
        ],
        *inputs: SelectedFeatures | DetectedOutliers,
    ) -> SelectedFeatures | DetectedOutliers:
        """Perform the index operation on the selected features or detected outliers."""
        assert all(type(inputs[0]) is type(input_) for input_ in inputs)

        node = f"index_operation_{IndexOperationConstructor.counter}"
        config = Config(name, None)

        structure = inputs[0].structure
        for input_ in inputs:
            input_.structure.update(node, config)
            structure = structure | input_.structure

        IndexOperationConstructor.counter += 1

        if "features" in name:
            return SelectedFeatures(structure)
        if "outliers" in name:
            return DetectedOutliers(structure)
        raise ValueError


class FeatureExtractionConstructor:
    """A class for constructing the abstract feature extraction."""

    counter: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the FeatureExtractionConstructor object."""

    def __call__(
        self,
        name: str,
        feature_matrix: FeatureMatrix,
        selected_features: SelectedFeatures,
    ) -> FeatureMatrix:
        """Perform the feature extraction on the feature matrix based on the selected features."""
        structure = feature_matrix.structure | selected_features.structure
        node = f"feature_extraction_{FeatureExtractionConstructor.counter}"
        config = Config(name, None)
        structure.update(node, config)
        FeatureExtractionConstructor.counter += 1
        return FeatureMatrix(structure)


class OutlierRemovalConstructor:
    """A class for constructing the abstract outlier removal."""

    counter: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the OutlierRemovalConstructor object."""

    def __call__(
        self,
        name: str,
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
        node = f"outlier_removal_{OutlierRemovalConstructor.counter}"
        config = Config(name, None)
        structure.update(node, config)
        OutlierRemovalConstructor.counter += 1
        return FeatureMatrix(structure), ResponseVector(structure)


class FeatureSelectionConstructor:
    """A class for constructing the abstract feature selection."""

    counter: ClassVar[int] = 0

    def __init__(self) -> None:
        """Initialize the FeatureSelectionConstructor object."""

    def __call__(
        self,
        name: str,
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        parameters: float | list[int] | list[float],
    ) -> SelectedFeatures:
        """Perform the feature selection on the feature matrix and response vector."""
        structure = feature_matrix.structure | response_vector.structure
        node = f"feature_selection_{FeatureSelectionConstructor.counter}"
        config = Config(name, parameters)
        structure.update(node, config)
        FeatureSelectionConstructor.counter += 1
        return SelectedFeatures(structure)


def make_dataset() -> tuple[FeatureMatrix, ResponseVector]:
    """Make the dataset for the data analysis pipeline."""
    return FeatureMatrix(Structure()), ResponseVector(Structure())


def union(
    *inputs: SelectedFeatures | DetectedOutliers,
) -> SelectedFeatures | DetectedOutliers:
    """Perform the union operation on the selected features or detected outliers."""
    match inputs[0]:
        case SelectedFeatures():
            return IndexOperationConstructor()("union_of_features", *inputs)
        case DetectedOutliers():
            return IndexOperationConstructor()("union_of_outliers", *inputs)


def intersection(
    *inputs: SelectedFeatures | DetectedOutliers,
) -> SelectedFeatures | DetectedOutliers:
    """Perform the intersection operation on the selected features or detected outliers."""
    match inputs[0]:
        case SelectedFeatures():
            return IndexOperationConstructor()("intersection_of_features", *inputs)
        case DetectedOutliers():
            return IndexOperationConstructor()("intersection_of_outliers", *inputs)


def extract_features(
    feature_matrix: FeatureMatrix,
    selected_features: SelectedFeatures,
) -> FeatureMatrix:
    """Perform the feature extraction on the feature matrix based on the selected features."""
    return FeatureExtractionConstructor()(
        "extract_features",
        feature_matrix,
        selected_features,
    )


def remove_outliers(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    detected_outliers: DetectedOutliers,
) -> tuple[FeatureMatrix, ResponseVector]:
    """Perform the outlier removal on the feature matrix and the response vector based on the detected outliers."""
    return OutlierRemovalConstructor()(
        "remove_outliers",
        feature_matrix,
        response_vector,
        detected_outliers,
    )


def marginal_screening(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: int | list[int],
) -> SelectedFeatures:
    """Perform the marginal screening on the feature matrix and response vector."""
    return FeatureSelectionConstructor()(
        "marginal_screening",
        feature_matrix,
        response_vector,
        parameters,
    )


def stepwise_feature_selection(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: int | list[int],
) -> SelectedFeatures:
    """Perform the stepwise feature selection on the feature matrix and response vector."""
    return FeatureSelectionConstructor()(
        "stepwise_feature_selection",
        feature_matrix,
        response_vector,
        parameters,
    )


def make_structure(output: SelectedFeatures) -> Structure:
    """Make the Structure object of defined data analysis pipeline."""
    structure = output.structure
    structure.update("end", Config("end", None))
    structure.make_sorted_node_list()
    return structure
