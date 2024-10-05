"""Module containing index operation methods."""


class IndexOperation:
    """A class for the index operation."""

    def __init__(self) -> None:
        """Initialize the IndexOperation object."""

    def index_operation(
        self,
        *inputs: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Perform the index operation."""
        raise NotImplementedError

    def union(self, *inputs: list[int]) -> list[int]:
        """Perform the union operation."""
        input_set = set(inputs[0])
        for input_ in inputs:
            input_set = input_set | set(input_)
        return list(input_set)

    def intersection(self, *inputs: list[int]) -> list[int]:
        """Perform the intersection operation."""
        input_set = set(inputs[0])
        for input_ in inputs:
            input_set = input_set & set(input_)
        return list(input_set)


class UnionFeatures(IndexOperation):
    """A class for the union features method."""

    def __init__(self) -> None:
        """Initialize the UnionFeatures object."""

    def index_operation(
        self,
        *inputs: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Perform the index operation."""
        return super().union(*[features for (features, _) in inputs]), inputs[0][1]


class UnionOutliers(IndexOperation):
    """A class for the union outliers method."""

    def __init__(self) -> None:
        """Initialize the UnionOutliers object."""

    def index_operation(
        self,
        *inputs: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Perform the index operation."""
        return inputs[0][0], super().union(*[outliers for (_, outliers) in inputs])


class IntersectionFeatures(IndexOperation):
    """A class for the intersection features method."""

    def __init__(self) -> None:
        """Initialize the IntersectionFeatures object."""

    def index_operation(
        self,
        *inputs: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Perform the index operation."""
        return (
            super().intersection(*[features for (features, _) in inputs]),
            inputs[0][1],
        )


class IntersectionOutliers(IndexOperation):
    """A class for the intersection outliers method."""

    def __init__(self) -> None:
        """Initialize the IntersectionOutliers object."""

    def index_operation(
        self,
        *inputs: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Perform the index operation."""
        return (
            inputs[0][0],
            super().intersection(*[outliers for (_, outliers) in inputs]),
        )
