"""Module containing utility for converting the node to the entities."""

from typing import cast

from si4automl.abstract import Node
from si4automl.feature_selection import (
    FeatureSelection,
    Lasso,
    MarginalScreening,
    StepwiseFeatureSelection,
)
from si4automl.index_operation import (
    IndexOperation,
    IntersectionFeatures,
    IntersectionOutliers,
    UnionFeatures,
    UnionOutliers,
)
from si4automl.missing_imputation import (
    ChebyshevImputation,
    DefiniteRegressionImputation,
    EuclideanImputation,
    ManhattanImputation,
    MeanValueImputation,
    MissingImputation,
)
from si4automl.outlier_detection import (
    CookDistance,
    Dffits,
    OutlierDetection,
    SoftIpod,
)


def conver_entities(
    node: Node,
) -> (
    list[MissingImputation]
    | list[FeatureSelection]
    | list[OutlierDetection]
    | list[IndexOperation]
    | list[None]
):
    """Convert the node to the entities."""
    match node.type:
        case "start" | "end" | "feature_extraction" | "outlier_removal":
            return [None]
        case "missing_imputation":
            return _convert_missing_imputation_entities(node)
        case "feature_selection":
            return _convert_feature_selection_entities(node)
        case "outlier_detection":
            return _convert_outlier_detection_entities(node)
        case "index_operation":
            return _convert_index_operation_entities(node)
        case _:
            raise ValueError


def _convert_missing_imputation_entities(node: Node) -> list[MissingImputation]:
    """Create MissingImputation objects."""
    match node.method:
        case "mean_value_imputation":
            return [MeanValueImputation()]
        case "euclidean_imputation":
            return [EuclideanImputation()]
        case "manhattan_imputation":
            return [ManhattanImputation()]
        case "chebyshev_imputation":
            return [ChebyshevImputation()]
        case "definite_regression_imputation":
            return [DefiniteRegressionImputation()]
        case _:
            raise ValueError


def _convert_feature_selection_entities(node: Node) -> list[FeatureSelection]:
    """Create FeatureSelection objects."""
    if node.parameters is None:
        raise ValueError

    match node.method:
        case "stepwise_feature_selection":
            parameters = cast(frozenset[int], node.parameters)
            return [StepwiseFeatureSelection(parameter) for parameter in parameters]
        case "marginal_screening":
            parameters = cast(frozenset[int], node.parameters)
            return [MarginalScreening(parameter) for parameter in parameters]
        case "lasso":
            return [Lasso(parameter) for parameter in node.parameters]
        case _:
            raise ValueError


def _convert_outlier_detection_entities(node: Node) -> list[OutlierDetection]:
    """Create OutlierDetection objects."""
    if node.parameters is None:
        raise ValueError

    match node.method:
        case "cook_distance":
            return [CookDistance(parameter) for parameter in node.parameters]
        case "soft_ipod":
            return [SoftIpod(parameter) for parameter in node.parameters]
        case "dffits":
            return [Dffits(parameter) for parameter in node.parameters]
        case _:
            raise ValueError


def _convert_index_operation_entities(node: Node) -> list[IndexOperation]:
    """Create IndexOperation objects."""
    match node.method:
        case "union_features":
            return [UnionFeatures()]
        case "union_outliers":
            return [UnionOutliers()]
        case "intersection_features":
            return [IntersectionFeatures()]
        case "intersection_outliers":
            return [IntersectionOutliers()]
        case _:
            raise ValueError
