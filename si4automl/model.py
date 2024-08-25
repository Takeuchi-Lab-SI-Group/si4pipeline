from source.pipeline import make_dataset, make_pipeline
from source.base_component import union, intersection, extract_features, remove_outliers
from source.missing_imputation import (
    mean_value_imputation,
    definite_regression_imputation,
)
from source.outlier_detection import cook_distance, dffits, soft_ipod
from source.feature_selection import (
    stepwise_feature_selection,
    lasso,
    marginal_screening,
)


def option1():
    X, y = make_dataset()
    y = mean_value_imputation(X, y)

    O = cook_distance(X, y, 3.0)
    X, y = remove_outliers(X, y, O)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = union(M1, M2)
    return make_pipeline(output=M)


def option2():
    X, y = make_dataset()
    y = definite_regression_imputation(X, y)

    M = marginal_screening(X, y, 5)
    X = extract_features(X, M)

    O = dffits(X, y, 3.0)
    X, y = remove_outliers(X, y, O)

    M1 = stepwise_feature_selection(X, y, 3)
    M2 = lasso(X, y, 0.08)
    M = intersection(M1, M2)
    return make_pipeline(output=M)
