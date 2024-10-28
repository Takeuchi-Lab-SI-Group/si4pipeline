"""Module containing utility for converting the node to the entities."""

from dataclasses import dataclass
from typing import Literal

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


@dataclass
class Config:
    """A class for the configuration of the entitity of the node."""

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
    parameter: float | None = None

    @property
    def entity(
        self,
    ) -> (
        MissingImputation | FeatureSelection | OutlierDetection | IndexOperation | None
    ):
        """Entity of the node."""
        match self.type:
            case "start" | "end" | "feature_extraction" | "outlier_removal":
                return None
            case "missing_imputation":
                return self._entity_of_missing_imputation
            case "feature_selection":
                return self._entity_of_feature_selection
            case "outlier_detection":
                return self._entity_of_outlier_detection
            case "index_operation":
                return self._entity_of_index_operation

    @property
    def _entity_of_missing_imputation(self) -> MissingImputation:
        """Entity of the missing imputation node."""
        match self.method:
            case "mean_value_imputation":
                return MeanValueImputation()
            case "euclidean_imputation":
                return EuclideanImputation()
            case "manhattan_imputation":
                return ManhattanImputation()
            case "chebyshev_imputation":
                return ChebyshevImputation()
            case "definite_regression_imputation":
                return DefiniteRegressionImputation()
            case _:
                raise ValueError

    @property
    def _entity_of_feature_selection(self) -> FeatureSelection:
        """Entity of the feature selection node."""
        assert self.parameter is not None
        match self.method:
            case "stepwise_feature_selection":
                assert isinstance(self.parameter, int)
                return StepwiseFeatureSelection(self.parameter)
            case "marginal_screening":
                assert isinstance(self.parameter, int)
                return MarginalScreening(self.parameter)
            case "lasso":
                return Lasso(self.parameter)
            case _:
                raise ValueError

    @property
    def _entity_of_outlier_detection(self) -> OutlierDetection:
        """Entity of the outlier detection node."""
        assert self.parameter is not None
        match self.method:
            case "cook_distance":
                return CookDistance(self.parameter)
            case "soft_ipod":
                return SoftIpod(self.parameter)
            case "dffits":
                return Dffits(self.parameter)
            case _:
                raise ValueError

    @property
    def _entity_of_index_operation(self) -> IndexOperation:
        """Entity of the index operation node."""
        match self.method:
            case "union_features":
                return UnionFeatures()
            case "union_outliers":
                return UnionOutliers()
            case "intersection_features":
                return IntersectionFeatures()
            case "intersection_outliers":
                return IntersectionOutliers()
            case _:
                raise ValueError


def convert_node_to_config_list(node: Node) -> list[Config]:
    """Convert the node to the list of the configuration."""
    match node.type:
        case (
            "start"
            | "end"
            | "feature_extraction"
            | "outlier_removal"
            | "missing_imputation"
            | "index_operation"
        ):
            assert node.parameters is None
            return [
                Config(type=node.type, method=node.method, parameter=node.parameters),
            ]
        case "feature_selection":
            assert node.parameters is not None
            return [
                Config(type=node.type, method=node.method, parameter=parameter)
                for parameter in node.parameters
            ]
        case "outlier_detection":
            assert node.parameters is not None
            return [
                Config(type=node.type, method=node.method, parameter=parameter)
                for parameter in node.parameters
            ]
