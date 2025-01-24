import enum
import typing
import numpy


VectorDataType = typing.Union[numpy.ndarray, typing.List[float]]


class DistanceMetric(enum.Enum):
    """
    An enumeration representing different types of distance metrics.

    - `DistanceMetric.L2`: L2 (Euclidean) distance metric.
    - `DistanceMetric.COSINE`: Cosine distance metric.
    """

    L2 = "L2"
    COSINE = "COSINE"

    def to_sql_func(self):
        """
        Converts the DistanceMetric to its corresponding SQL function name.

        Returns:
            str: The SQL function name.

        Raises:
            ValueError: If the DistanceMetric enum member is not supported.
        """
        if self == DistanceMetric.L2:
            return "l2_distance"
        elif self == DistanceMetric.COSINE:
            return "cosine_distance"
        else:
            raise ValueError("unsupported distance metric")


