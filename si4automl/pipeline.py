"""Module containing entity for the data analysis pipeline and manager of it."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.polynomial import Polynomial
from sicore import (  # type: ignore[import]
    RealSubset,
    SelectiveInferenceNorm,
    SelectiveInferenceResult,
    polynomial_below_zero,
)

if TYPE_CHECKING:
    from si4automl.abstract import Node, Structure
from si4automl.entity import convert_node_to_config_list
from si4automl.feature_selection import FeatureSelection
from si4automl.index_operation import IndexOperation
from si4automl.missing_imputation import MissingImputation
from si4automl.outlier_detection import OutlierDetection


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

        self.cache_quadratic_cross_validation_error: dict[
            int,
            dict[tuple[float, float], list[float]],
        ] = {}
        self.imputer: np.ndarray | None = None
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
        assert X.shape[0] == a.shape[0] == b.shape[0]
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

    def cross_validation_error(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        cross_validation_masks: list[np.ndarray],
    ) -> float:
        """Compute the cross validation error."""
        X, y = feature_matrix, response_vector
        # layer = self.layers[self.static_order[1]]
        # if isinstance(layer, MissingImputation):
        #     imputer = layer.compute_imputer(X, y)
        # else:
        #     imputer = np.eye(len(y))
        imputer = self.compute_imputer(X, y)
        y = imputer @ y[~np.isnan(y)]

        error_list = []
        for mask in cross_validation_masks:
            X_val, y_val = X[mask], y[mask]
            X_tr, y_tr = np.delete(X, mask, 0), np.delete(y, mask)
            M, O = self(X_tr, y_tr)
            # print(M, O)
            X_tr, y_tr = np.delete(X_tr, O, 0), np.delete(y_tr, O)
            if len(M) == 0:
                error_list.append(np.mean(y_val**2))
            else:
                y_error = (
                    y_val
                    - X_val[:, M]
                    @ np.linalg.inv(X_tr[:, M].T @ X_tr[:, M])
                    @ X_tr[:, M].T
                    @ y_tr
                )
                error_list.append(np.mean(y_error**2))
        return np.mean(error_list).item()

    def quadratic_cross_validation_error(
        self,
        X: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
        cross_validation_masks: list[np.ndarray],
    ) -> tuple[Polynomial, float, float]:
        """Compute the cross validation error in the quadratic form."""
        assert X.shape[0] == a.shape[0] == b.shape[0]
        l_list, u_list = [], []
        quadratic_list = []
        for mask_id, mask in enumerate(cross_validation_masks):
            load = self._load_quadratic_cross_validation_error(mask_id, z)
            if load is not None:
                quadratic, l, u = load
                quadratic_list.append(quadratic)
                l_list.append(l)
                u_list.append(u)
                continue

            X_val, a_val, b_val = X[mask], a[mask], b[mask]
            X_tr = np.delete(X, mask, 0)
            a_tr = np.delete(a, mask)
            b_tr = np.delete(b, mask)
            (
                selected_features_cv,
                detected_outliers_cv,
                l,
                u,
            ) = self.selection_event(X_tr, a_tr, b_tr, z, mask_id)
            l_list.append(l)
            u_list.append(u)

            X_tr = np.delete(X_tr, detected_outliers_cv, 0)
            a_tr = np.delete(a_tr, detected_outliers_cv)
            b_tr = np.delete(b_tr, detected_outliers_cv)
            num = X_val.shape[0]
            if len(selected_features_cv) == 0:
                quadratic = [
                    a_val @ a_val / num,
                    2 * b_val @ a_val / num,
                    b_val @ b_val / num,
                ]
            else:
                F = (
                    X_tr[:, selected_features_cv]
                    @ np.linalg.inv(
                        X_tr[:, selected_features_cv].T @ X_tr[:, selected_features_cv],
                    )
                    @ X_val[:, selected_features_cv].T
                )
                G = F @ F.T
                alpha = b_val @ b_val - 2 * b_tr @ F @ b_val + b_tr @ G @ b_tr
                beta = (
                    2 * b_val @ a_val
                    - 2 * b_tr @ F @ a_val
                    - 2 * a_tr @ F @ b_val
                    + 2 * a_tr @ G @ b_tr
                )
                gamma = a_val @ a_val - 2 * a_tr @ F @ a_val + a_tr @ G @ a_tr
                quadratic = [gamma / num, beta / num, alpha / num]
            quadratic_list.append(quadratic)
            self.cache_quadratic_cross_validation_error.setdefault(mask_id, {})[
                (l, u)
            ] = quadratic

        return (
            Polynomial(np.mean(quadratic_list, axis=0)),
            np.max(l_list).item(),
            np.min(u_list).item(),
        )

    def _load_quadratic_cross_validation_error(
        self,
        mask_id: int,
        z: float,
    ) -> tuple[list[float], float, float] | None:
        self.cache_quadratic_cross_validation_error.setdefault(mask_id, {})
        for interval in self.cache_quadratic_cross_validation_error[mask_id]:
            if interval[0] <= z <= interval[1]:
                return (
                    self.cache_quadratic_cross_validation_error[mask_id][interval],
                    *interval,
                )
        return None

    def compute_imputer(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the imputer matrix based on the data analysis pipeline."""
        layer = self.layers[self.static_order[1]]
        if isinstance(layer, MissingImputation):
            self.imputer = layer.compute_imputer(feature_matrix, response_vector)
        else:
            self.imputer = np.eye(len(response_vector))
        return self.imputer

    def load_imputer(self) -> np.ndarray:
        """Load the imputer matrix from the cache."""
        assert self.imputer is not None
        return self.imputer

    def reset_cache(self) -> None:
        """Reset the cache of the Pipeline object."""
        self.cache_quadratic_cross_validation_error = {}
        for node in self.static_order:
            layer = self.layers[node]
            if isinstance(layer, FeatureSelection | OutlierDetection):
                layer.reset_cache()

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

    def __init__(self, structure: Structure | None = None) -> None:
        """Initialize the PipelineManager object."""
        self.pipelines: list[Pipeline] = []
        self.representeing_index = 0
        self.tuned = False

        if structure is None:
            return
        static_order = structure.static_order
        graph = structure.graph

        configs_iters = product(
            *[convert_node_to_config_list(node) for node in static_order],
        )
        for configs in configs_iters:
            entities = [config.entity for config in configs]
            pipeline = Pipeline(
                static_order=static_order,
                graph=graph,
                layers=dict(zip(static_order, entities, strict=True)),
            )
            self.pipelines.append(pipeline)

    def tune(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        *,
        num_folds: int = 5,
        max_candidates: int | None = None,
        random_state: int | None = 0,
    ) -> None:
        """Tune to select the best data analysis pipeline using the cross validation."""
        rng = np.random.default_rng(random_state)

        if max_candidates is not None:
            num_candidates = np.min([max_candidates, len(self.pipelines)])
        else:
            num_candidates = len(self.pipelines)
        self.candidates_indices: list[int] = rng.choice(
            len(self.pipelines),
            num_candidates,
            replace=False,
        ).tolist()
        self.candidates_indices.sort()

        self.cross_validation_masks: list[np.ndarray] = np.array_split(
            rng.permutation(len(response_vector)),
            num_folds,
        )

        cross_validation_error_list = [
            self.pipelines[index].cross_validation_error(
                feature_matrix,
                response_vector,
                self.cross_validation_masks,
            )
            for index in self.candidates_indices
        ]
        self.representeing_index = self.candidates_indices[
            np.argmin(cross_validation_error_list)
        ]
        self.tuned = True

    def __call__(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
    ) -> tuple[list[int], list[int]]:
        """Perform the representing data analysis pipeline on the given feature matrix and response vector."""
        assert self.tuned or len(self.pipelines) == 1
        return self.pipelines[self.representeing_index](feature_matrix, response_vector)

    def inference(
        self,
        feature_matrix: np.ndarray,
        response_vector: np.ndarray,
        sigma: float | None = None,
        *,
        test_index: int | None = None,
        retain_result: bool = False,
    ) -> (
        tuple[list[int], list[float] | list[SelectiveInferenceResult]]
        | tuple[int, float | SelectiveInferenceResult]
    ):
        """Inference the representing data analysis pipeline on the given feature matrix and response vector."""
        assert self.tuned or len(self.pipelines) == 1
        self.M, self.O = self(feature_matrix, response_vector)
        self.X = feature_matrix

        node = self.pipelines[self.representeing_index].static_order[1]
        self.exist_missing = np.any(np.isnan(response_vector))
        if node.type == "missing_imputation" and self.exist_missing:
            self.missing_imputation_method = node.method
        else:
            self.missing_imputation_method = "none"
        imputer = self.pipelines[self.representeing_index].compute_imputer(
            feature_matrix,
            response_vector,
        )
        X, y = feature_matrix, response_vector

        if sigma is None:
            residuals = (
                (np.eye(len(y)) - X @ np.linalg.inv(X.T @ X) @ X.T)
                @ imputer
                @ y[~np.isnan(y)]
            )
            sigma = np.std(residuals, ddof=X.shape[1])

        n = len(y)
        X_ = np.delete(X, self.O, 0)  # shape (n - |O|, p)
        X_ = X_[:, self.M]  # shape (n - |O|, |M|)
        Im = np.delete(np.eye(n), self.O, 0)  # shape (n - |O|, n)
        etas = np.linalg.inv(X_.T @ X_) @ X_.T @ Im  # shape (|M|, n)
        self.etas = etas @ imputer  # shape (|M|, n - num_missing)
        if test_index is not None:
            self.etas = self.etas[test_index].reshape(1, -1)
            test_index = int(test_index)

        results: list[SelectiveInferenceResult] = []
        for eta in self.etas:
            self.reset_cache_of_pipelines()
            si = SelectiveInferenceNorm(y[~np.isnan(y)], sigma**2.0, eta)
            results.append(si.inference(self._algorithm, self._model_selector))

        match test_index, retain_result:
            case None, False:
                return self.M, [result.p_value for result in results]
            case None, True:
                return self.M, results
            case int(), False:
                return test_index, results[0].p_value
            case int(), True:
                return test_index, results[0]
            case _, _:
                raise ValueError

        raise ValueError

    def _algorithm(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
    ) -> tuple[
        tuple[list[int], list[int]] | tuple[list[int], list[int], str],
        list[float],
    ]:
        """Algorithm to perform the selective inference."""
        if not self.tuned:
            imputer = self.pipelines[self.representeing_index].load_imputer()
            M, O, l, u = self.pipelines[self.representeing_index].selection_event(
                self.X,
                imputer @ a,
                imputer @ b,
                z,
            )
            return (M, O), [l, u]
        l_list, u_list = [-np.inf], [np.inf]
        polynomial_list: list[Polynomial] = []
        for index in self.candidates_indices:
            imputer = self.pipelines[index].load_imputer()
            quadratic, l, u = self.pipelines[index].quadratic_cross_validation_error(
                self.X,
                imputer @ a,
                imputer @ b,
                z,
                self.cross_validation_masks,
            )
            polynomial_list.append(quadratic)
            l_list.append(l)
            u_list.append(u)

        best_index = np.argmin([quadratic(z) for quadratic in polynomial_list])
        best_quadratic = polynomial_list[best_index]
        for quadratic in polynomial_list:
            l, u = RealSubset(
                polynomial_below_zero(best_quadratic - quadratic),
            ).find_interval_containing(z)
            l_list.append(l)
            u_list.append(u)

        imputer = self.pipelines[self.candidates_indices[best_index]].load_imputer()
        M, O, l, u = self.pipelines[
            self.candidates_indices[best_index]
        ].selection_event(
            self.X,
            imputer @ a,
            imputer @ b,
            z,
        )
        l_list.append(l)
        u_list.append(u)

        node = self.pipelines[self.candidates_indices[best_index]].static_order[1]
        if node.type == "missing_imputation" and self.exist_missing:
            missing_imputation_method = node.method
        else:
            missing_imputation_method = "none"

        l, u = np.max(l_list).item(), np.min(u_list).item()
        assert l <= z <= u
        return (
            (M, O, missing_imputation_method),
            [l, u],
        )

    def _model_selector(
        self,
        args: tuple[list[int], list[int]] | tuple[list[int], list[int], str],
    ) -> bool:
        """Model selector to perform the selective inference."""
        if not self.tuned:
            args = cast(tuple[list[int], list[int]], args)
            M, O = args
            return set(M) == set(self.M) and set(O) == set(self.O)
        args = cast(tuple[list[int], list[int], str], args)
        M, O, method = args
        return (
            set(M) == set(self.M)
            and set(O) == set(self.O)
            and method == self.missing_imputation_method
        )

    def reset_cache_of_pipelines(self) -> None:
        """Reset the cache of the all pipelines."""
        for pipeline in self.pipelines:
            pipeline.reset_cache()

    def __str__(self) -> str:
        """Return the string representation of the PipelineManager object."""
        return (
            f"PipelineManager with {len(self.pipelines)} pipelines:\n"
            "Representing pipelines:\n"
            f"{self.pipelines[self.representeing_index]}"
        )

    def __or__(self, other: PipelineManager) -> PipelineManager:
        """Merge the two PipelineManager objects."""
        manager = PipelineManager()
        manager.pipelines = self.pipelines + other.pipelines
        manager.representeing_index = 0
        manager.tuned = False
        return manager

    def show_parameter(self) -> str:
        """Return the string representation of the PipelineManager object."""
        list_ = []
        for node in self.pipelines[self.representeing_index].static_order:
            layer = self.pipelines[self.representeing_index].layers[node]
            if isinstance(layer, FeatureSelection | OutlierDetection):
                list_.append(f"{node.type}: {node.method} with {layer.parameter}")
        return "\n".join(list_)
