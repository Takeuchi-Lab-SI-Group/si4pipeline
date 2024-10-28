"""Module containing the constructor functions for the components of the data analysis pipeline."""

from typing import TypeVar

from si4automl.abstract import (
    DetectedOutliers,
    FeatureExtractionConstructor,
    FeatureMatrix,
    FeatureSelectionConstructor,
    IndexOperationConstructor,
    MissingImputationConstructor,
    Node,
    OutlierDetectionConstructor,
    OutlierRemovalConstructor,
    ResponseVector,
    SelectedFeatures,
    Structure,
)
from si4automl.pipeline import PipelineManager


def initialize_dataset() -> tuple[FeatureMatrix, ResponseVector]:
    """Make the dataset for the data analysis pipeline."""
    return FeatureMatrix(Structure()), ResponseVector(Structure())


def mean_value_imputation(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
) -> ResponseVector:
    """Perform the mean value imputation on the feature matrix and response vector."""
    return MissingImputationConstructor()(
        "mean_value_imputation",
        feature_matrix,
        response_vector,
    )


def euclidean_imputation(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
) -> ResponseVector:
    """Perform the euclidean imputation on the feature matrix and response vector."""
    return MissingImputationConstructor()(
        "euclidean_imputation",
        feature_matrix,
        response_vector,
    )


def manhattan_imputation(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
) -> ResponseVector:
    """Perform the manhattan imputation on the feature matrix and response vector."""
    return MissingImputationConstructor()(
        "manhattan_imputation",
        feature_matrix,
        response_vector,
    )


def chebyshev_imputation(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
) -> ResponseVector:
    """Perform the chebyshev imputation on the feature matrix and response vector."""
    return MissingImputationConstructor()(
        "chebyshev_imputation",
        feature_matrix,
        response_vector,
    )


def definite_regression_imputation(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
) -> ResponseVector:
    """Perform the definite regression imputation on the feature matrix and response vector."""
    return MissingImputationConstructor()(
        "definite_regression_imputation",
        feature_matrix,
        response_vector,
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


def lasso(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: float | list[float],
) -> SelectedFeatures:
    """Perform the lass on the feature matrix and response vector."""
    return FeatureSelectionConstructor()(
        "lasso",
        feature_matrix,
        response_vector,
        parameters,
    )


def cook_distance(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: float | list[float],
) -> DetectedOutliers:
    """Perform the cook distance on the feature matrix and response vector."""
    return OutlierDetectionConstructor()(
        "cook_distance",
        feature_matrix,
        response_vector,
        parameters,
    )


def dffits(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: float | list[float],
) -> DetectedOutliers:
    """Perform the dffits on the feature matrix and response vector."""
    return OutlierDetectionConstructor()(
        "dffits",
        feature_matrix,
        response_vector,
        parameters,
    )


def soft_ipod(
    feature_matrix: FeatureMatrix,
    response_vector: ResponseVector,
    parameters: float | list[float],
) -> DetectedOutliers:
    """Perform the soft ipod on the feature matrix and response vector."""
    return OutlierDetectionConstructor()(
        "soft_ipod",
        feature_matrix,
        response_vector,
        parameters,
    )


T = TypeVar("T", SelectedFeatures, DetectedOutliers)


def union(*inputs: T) -> T:
    """Perform the union operation on the selected features or detected outliers."""
    match inputs[0]:
        case SelectedFeatures():
            return IndexOperationConstructor()("union_features", *inputs)
        case DetectedOutliers():
            return IndexOperationConstructor()("union_outliers", *inputs)


def intersection(*inputs: T) -> T:
    """Perform the intersection operation on the selected features or detected outliers."""
    match inputs[0]:
        case SelectedFeatures():
            return IndexOperationConstructor()("intersection_features", *inputs)
        case DetectedOutliers():
            return IndexOperationConstructor()("intersection_outliers", *inputs)


def extract_features(
    feature_matrix: FeatureMatrix,
    selected_features: SelectedFeatures,
) -> FeatureMatrix:
    """Perform the feature extraction on the feature matrix based on the selected features."""
    return FeatureExtractionConstructor()(
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
        feature_matrix,
        response_vector,
        detected_outliers,
    )


def construct_pipelines(output: SelectedFeatures) -> PipelineManager:
    """Make the Structure object of defined data analysis pipeline."""
    structure = output.structure
    structure.update(Node("end"))
    structure.sort_graph()
    return PipelineManager(structure)
