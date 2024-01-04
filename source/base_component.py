class FeatureMatrix:
    def __init__(self, pl_structure, data=None):
        self.pl_structure = pl_structure
        self.data = data


class ResponseVector:
    def __init__(self, pl_structure, data=None):
        self.pl_structure = pl_structure
        self.data = data


class SelectedFeatures:
    def __init__(self, pl_structure, data=None):
        self.pl_structure = pl_structure
        self.data = data


class DetectedOutliers:
    def __init__(self, pl_structure, data=None):
        self.pl_structure = pl_structure
        self.data = data


class IndexesOperator:
    instance_counter = dict()

    def __init__(self, name: str):
        IndexesOperator.instance_counter.setdefault(name, 0)
        self.name = f"{name}_{IndexesOperator.instance_counter[name]}"
        IndexesOperator.instance_counter[name] += 1

    def __call__(
        self, *inputs: tuple[SelectedFeatures] | tuple[DetectedOutliers]
    ) -> SelectedFeatures | DetectedOutliers:
        pl_structure = inputs[0].pl_structure
        pl_structure.update(self.name, self)
        input_type = type(inputs[0])
        if len(inputs) > 1:
            for input in inputs[1:]:
                if input_type != type(input):
                    raise TypeError("Inputs must be same type")
                input.pl_structure.update(self.name, self)
                pl_structure = pl_structure | input.pl_structure
        if isinstance(inputs[0], SelectedFeatures):
            self.mode = "selected_features"
            return SelectedFeatures(pl_structure)
        elif isinstance(inputs[0], DetectedOutliers):
            self.mode = "detected_outliers"
            return DetectedOutliers(pl_structure)
        else:
            raise TypeError("Inputs must be SelectedFeatures or DetectedOutliers")


class RemoveOutliers:
    counter = 0

    def __init__(self, name="remove"):
        self.name = f"{name}_{RemoveOutliers.counter}"
        RemoveOutliers.counter += 1

    def __call__(
        self,
        feature_matrix: FeatureMatrix,
        response_vector: ResponseVector,
        detected_outliers: DetectedOutliers,
    ) -> (FeatureMatrix, ResponseVector):
        pl_structure = (
            feature_matrix.pl_structure
            | response_vector.pl_structure
            | detected_outliers.pl_structure
        )
        pl_structure.update(self.name, self)
        return FeatureMatrix(pl_structure), ResponseVector(pl_structure)


class ExtractFeatures:
    counter = 0

    def __init__(self, name="extract"):
        self.name = f"{name}_{ExtractFeatures.counter}"
        ExtractFeatures.counter += 1

    def __call__(
        self, feature_matrix: FeatureMatrix, selected_features: SelectedFeatures
    ) -> FeatureMatrix:
        pl_structure = feature_matrix.pl_structure | selected_features.pl_structure
        pl_structure.update(self.name, self)
        return FeatureMatrix(pl_structure)


class Union(IndexesOperator):
    def __init__(self, name="union"):
        super().__init__(name)

    def union(self, *inputs: tuple[list[int]]) -> list[int]:
        if len(inputs) == 1:
            return inputs[0]
        else:
            temp_set = set(inputs[0])
            for input in inputs[1:]:
                temp_set = temp_set | set(input)
            return list(temp_set)


class Intersection(IndexesOperator):
    def __init__(self, name="intersection"):
        super().__init__(name)

    def intersection(self, *inputs: tuple[list[int]]) -> list[int]:
        if len(inputs) == 1:
            return inputs[0]
        else:
            temp_set = set(inputs[0])
            for input in inputs[1:]:
                temp_set = temp_set & set(input)
            return list(temp_set)


def union(*inputs):
    return Union()(*inputs)


def intersection(*inputs):
    return Intersection()(*inputs)


def remove_outliers(feature_matrix, response_vector, detected_outliers):
    return RemoveOutliers()(feature_matrix, response_vector, detected_outliers)


def extract_features(feature_matrix, selected_features):
    return ExtractFeatures()(feature_matrix, selected_features)
