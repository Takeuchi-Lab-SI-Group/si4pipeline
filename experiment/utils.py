"""Module containing pipeline managers for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from sicore import SelectiveInferenceResult  # type: ignore[import]

from si4automl import (
    PipelineManager,
    construct_pipelines,
    cook_distance,
    definite_regression_imputation,
    extract_features,
    initialize_dataset,
    intersection,
    lasso,
    marginal_screening,
    mean_value_imputation,
    remove_outliers,
    soft_ipod,
    stepwise_feature_selection,
    union,
)


@dataclass
class Results:
    """Results dataclass for the data analysis pipeline."""

    results: list[SelectiveInferenceResult] = field(default_factory=list)
    oc_p_values: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __add__(self, other: Results) -> Results:
        """Take union of two results."""
        return Results(
            results=self.results + other.results,
            oc_p_values=self.oc_p_values + other.oc_p_values,
            times=self.times + other.times,
        )

    def __iadd__(self, other: Self) -> Self:
        """Take union of two results in place."""
        self.results += other.results
        self.oc_p_values += other.oc_p_values
        self.times += other.times
        return self


def option1() -> PipelineManager:
    """Pipeline manager for only one option 1 pipeline."""
    X, y = initialize_dataset()
    y = mean_value_imputation(X, y)

    O = soft_ipod(X, y, 0.02)
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = union(M1, M2)
    return construct_pipelines(output=M)


def option2() -> PipelineManager:
    """Pipeline manager for only one option 2 pipeline."""
    X, y = initialize_dataset()
    y = definite_regression_imputation(X, y)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    O = cook_distance(X, y, 3.0)
    X, y = remove_outliers(X, y, O)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = intersection(M1, M2)
    return construct_pipelines(output=M)


def option1_multi() -> PipelineManager:
    """Pipeline manager for multiple option 1 pipelines."""
    X, y = initialize_dataset()
    y = mean_value_imputation(X, y)

    O = soft_ipod(X, y, [0.02, 0.018])
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, [3, 5])
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, [2, 3])
    M2 = lasso(X, y, [0.08, 0.12])
    M = union(M1, M2)
    return construct_pipelines(output=M)


def option2_multi() -> PipelineManager:
    """Pipeline manager for multiple option 2 pipelines."""
    X, y = initialize_dataset()
    y = definite_regression_imputation(X, y)

    M = marginal_screening(X, y, [3, 5])
    X = extract_features(X, M)

    O = cook_distance(X, y, [2.0, 3.0])
    X, y = remove_outliers(X, y, O)

    M1 = stepwise_feature_selection(X, y, [2, 3])
    M2 = lasso(X, y, [0.08, 0.12])
    M = intersection(M1, M2)
    return construct_pipelines(output=M)


def option1_parallel() -> PipelineManager:
    """Pipeline manager for parallel option1 pipelines."""
    X, y = initialize_dataset()
    y = mean_value_imputation(X, y)

    O = soft_ipod(X, y, 0.02)
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 20)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 5)
    M2 = lasso(X, y, 0.06)
    M = union(M1, M2)

    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = union(M1, M2)
    return construct_pipelines(output=M)


def option1_serial() -> PipelineManager:
    """Pipeline manager for serial option1 pipelines."""
    X, y = initialize_dataset()
    y = mean_value_imputation(X, y)

    O = soft_ipod(X, y, 0.02)
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 20)
    X = extract_features(X, M)

    M = stepwise_feature_selection(X, y, 10)
    X = extract_features(X, M)

    M = lasso(X, y, 0.04)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = union(M1, M2)
    return construct_pipelines(output=M)
