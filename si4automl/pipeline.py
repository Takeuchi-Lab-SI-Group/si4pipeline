"""Module containing entity for the data analysis pipeline and manager of it."""

from itertools import product
from typing import cast

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

        layers_list = list(
            product(*[conver_entities(node) for node in self.static_order]),
        )
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

    def __str__(self) -> str:
        """Return the string representation of the PipelineManager object."""
        return (
            f"PipelineManager with {len(self.pipelines)} pipelines:\n"
            "Representing pipelines:\n"
            f"{self.pipelines[self.representeing_index]}"
        )
