import sqlalchemy
from .vector_type import VectorType


class VectorAdaptor:
    """
    A wrapper over existing SQLAlchemy engine to provide additional vector search capabilities.
    """

    engine: sqlalchemy.Engine

    def __init__(self, engine: sqlalchemy.Engine):
        self.engine = engine

    def _check_vector_column(self, column: sqlalchemy.Column):
        if not isinstance(column.type, VectorType):
            raise ValueError("Not a vector column")

    def has_vector_index(self, column: sqlalchemy.Column) -> bool:
        """
        Check if the index for the vector column exists.
        """

        self._check_vector_column(column)

        with self.engine.begin() as conn:
            table_name = conn.dialect.identifier_preparer.format_table(column.table)
            query = sqlalchemy.text(f"SHOW INDEX FROM {table_name}")
            result = conn.execute(query)
            result_dict = result.mappings().all()
            for row in result_dict:
                if row["Column_name"].lower() == column.name.lower():
                    return True
        return False

    def create_vector_index(
        self,
        column: sqlalchemy.Column,
        skip_existing: bool = False,
    ):
        self._check_vector_column(column)

        if column.type.dim is None:
            raise ValueError(
                "Vector index is only supported for fixed dimension vectors"
            )

        if skip_existing:
            if self.has_vector_index(column):
                # TODO: Currently there is no easy way to verify whether the distance
                # metric is correct. We should check it and throw error if distance metric is not matching
                return

        with self.engine.begin() as conn:
            table_name = conn.dialect.identifier_preparer.format_table(column.table)
            column_name = conn.dialect.identifier_preparer.format_column(column)
            index_name = conn.dialect.identifier_preparer.quote(
                f"vec_idx_{column.name}"
            )

            query = sqlalchemy.text("SET experimental_ivf_index = 1")
            conn.execute(query)
            query = sqlalchemy.text(
                f'create index {index_name} using ivfflat on {table_name}({column_name}) lists=1000 op_type "vector_l2_ops"'
            )
            conn.execute(query)

