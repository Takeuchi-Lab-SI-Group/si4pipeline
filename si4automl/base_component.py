"""Module containing the base classes for the components of the data analysis pipeline."""

from typing import ClassVar

from si4automl.pipeline import PipelineStructure


class FeatureMatrix:
    """An abstract class for the feature matrix."""

    def __init__(self, pl_structure: PipelineStructure, data: None = None) -> None:
        """Initialize the FeatureMatrix object."""
        self.pl_structure = pl_structure
        self.data = data


class ResponseVector:
    """An abstract class for the response vector."""

    def __init__(self, pl_structure: PipelineStructure, data: None = None) -> None:
        """Initialize the ResponseVector object."""
        self.pl_structure = pl_structure
        self.data = data


class SelectedFeatures:
    """An abstract class for the selected features."""

    def __init__(self, pl_structure: PipelineStructure, data: None = None) -> None:
        """Initialize the SelectedFeatures object."""
        self.pl_structure = pl_structure
        self.data = data


class DetectedOutliers:
    """An abstract class for the detected outliers."""

    def __init__(self, pl_structure: PipelineStructure, data: None = None) -> None:
        """Initialize the DetectedOutliers object."""
        self.pl_structure = pl_structure
        self.data = data


class IndexesOperator:
    """An abstract class for the indexes operator such as union and intersection."""

    instance_counter: ClassVar[dict[str, int]] = {}

    def __init__(self, name: str) -> None:
        """Initialize the IndexesOperator object."""
        IndexesOperator.instance_counter.setdefault(name, 0)
        self.name = f"{name}_{IndexesOperator.instance_counter[name]}"
        IndexesOperator.instance_counter[name] += 1

    def __call__(
        self,
        *inputs: SelectedFeatures | DetectedOutliers,
    ) -> SelectedFeatures | DetectedOutliers:
        """Perform the union or intersection operation on the inputs."""
        pl_structure = inputs[0].pl_structure
        pl_structure.update(self.name, self)
        input_type = type(inputs[0])
        if len(inputs) > 1:
            for input_ in inputs[1:]:
                if input_type is not type(input_):  # Validity check
                    raise TypeError  # Inputs must be same type"
                input_.pl_structure.update(self.name, self)
                pl_structure = pl_structure | input_.pl_structure
        if isinstance(inputs[0], SelectedFeatures):
            self.mode = "selected_features"
            return SelectedFeatures(pl_structure)
        if isinstance(inputs[0], DetectedOutliers):
            self.mode = "detected_outliers"
            return DetectedOutliers(pl_structure)
        raise TypeError  # Inputs must be SelectedFeatures or DetectedOutliers


class RemoveOutliers:
    """A class for removing the outliers from the data."""

    counter: int = 0

    def __init__(self, name: str = "remove") -> None:
        """Initialize the RemoveOutliers object."""
        self.name = f"{name}_{RemoveOutliers.counter}"
        RemoveOutliers.counter += 1

    def __call__(
        self,
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        detected_outliers: DetectedOutliers,
    ) -> tuple[FeatureMatrix, ResponseVector]:
        """Perform the outlier removal operation."""
        pl_structure = (
            feature_matrix.pl_structure
            | response_vector.pl_structure
            | detected_outliers.pl_structure
        )
        pl_structure.update(self.name, self)
        return FeatureMatrix(pl_structure), ResponseVector(pl_structure)


class ExtractFeatures:
    """A class for extracting the features from the data."""

    counter: int = 0

    def __init__(self, name: str = "extract") -> None:
        """Initialize the ExtractFeatures object."""
        self.name = f"{name}_{ExtractFeatures.counter}"
        ExtractFeatures.counter += 1

    def __call__(
        self,
        feature_matrix: FeatureMatrix,
        selected_features: SelectedFeatures,
    ) -> FeatureMatrix:
        """Perform the feature extraction operation."""
        pl_structure = feature_matrix.pl_structure | selected_features.pl_structure
        pl_structure.update(self.name, self)
        return FeatureMatrix(pl_structure)


class Union(IndexesOperator):
    """A class for the union operation on the indexes."""

    def __init__(self, name: str = "union") -> None:
        """Initialize the Union object."""
        super().__init__(name)

    def union(self, *inputs: list[int]) -> list[int]:
        """Perform the union operation on the indexes."""
        if len(inputs) == 1:
            return inputs[0]
        temp_set = set(inputs[0])
        for input_ in inputs[1:]:
            temp_set = temp_set | set(input_)
        return list(temp_set)


class Intersection(IndexesOperator):
    """A class for the intersection operation on the indexes."""

    def __init__(self, name: str = "intersection") -> None:
        """Initialize the Intersection object."""
        super().__init__(name)

    def intersection(self, *inputs: list[int]) -> list[int]:
        """Perform the intersection operation on the indexes."""
        if len(inputs) == 1:
            return inputs[0]
        temp_set = set(inputs[0])
        for input_ in inputs[1:]:
            temp_set = temp_set & set(input_)
        return list(temp_set)


def union(
    *inputs: SelectedFeatures | DetectedOutliers,
) -> SelectedFeatures | DetectedOutliers:
    """Perform the union operation on the indexes."""
    return Union()(*inputs)


def intersection(
    *inputs: SelectedFeatures | DetectedOutliers,
) -> SelectedFeatures | DetectedOutliers:
    """Perform the intersection operation on the indexes."""
    return Intersection()(*inputs)


def remove_outliers(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    detected_outliers: DetectedOutliers,
) -> tuple[FeatureMatrix, ResponseVector]:
    """Perform the outlier removal operation."""
    return RemoveOutliers()(feature_matrix, response_vector, detected_outliers)


def extract_features(
    feature_matrix: FeatureMatrix,
    selected_features: SelectedFeatures,
) -> FeatureMatrix:
    """Perform the feature extraction operation."""
    return ExtractFeatures()(feature_matrix, selected_features)
