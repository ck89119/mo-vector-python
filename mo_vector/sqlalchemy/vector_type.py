import sqlalchemy
import mo_vector
import mo_vector.utils


class VectorType(sqlalchemy.types.UserDefinedType):
    """
    Represents a vector column type in MO.
    """

    dim: int

    cache_ok = True

    def __init__(self, dim):
        if not isinstance(dim, int):
            raise ValueError("expected dimension to be an integer")

        super(sqlalchemy.types.UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        return f"vecf64({self.dim})"

    def bind_processor(self, dialect):
        """Convert the vector float array to a string representation suitable for binding to a database column."""

        def process(value):
            return mo_vector.utils.encode_vector(value, self.dim)

        return process

    def result_processor(self, dialect, coltype):
        """Convert the vector data from the database into vector array."""

        def process(value):
            return mo_vector.utils.decode_vector(value)

        return process

    class comparator_factory(sqlalchemy.types.UserDefinedType.Comparator):
        """Returns a comparator factory that provides the distance functions."""

        def l2_distance(self, other: mo_vector.VectorDataType):
            formatted_other = mo_vector.utils.encode_vector(other)
            return sqlalchemy.func.l2_distance(self, formatted_other).label(
                "l2_distance"
            )

        def cosine_distance(self, other: mo_vector.VectorDataType):
            formatted_other = mo_vector.utils.encode_vector(other)
            return sqlalchemy.func.cosine_distance(self, formatted_other).label(
                "cosine_distance"
            )

        def cosine_similarity(self, other: mo_vector.VectorDataType):
            formatted_other = mo_vector.utils.encode_vector(other)
            return sqlalchemy.func.cosine_similarity(self, formatted_other).label(
                "cosine_similarity"
            )
