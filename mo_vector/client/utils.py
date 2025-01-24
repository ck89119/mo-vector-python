import math
import sqlalchemy
import re
from typing import Any, Dict, Optional


class EmbeddingColumnMismatchError(ValueError):
    """
    Exception raised when the existing embedding column does not match the expected dimension.

    Attributes:
        existing_col (str): The definition of the existing embedding column.
        expected_col (str): The definition of the expected embedding column.
    """

    def __init__(self, existing_col, expected_col):
        self.existing_col = existing_col
        self.expected_col = expected_col
        super().__init__(
            f"The existing embedding column ({existing_col}) does not match the expected dimension ({expected_col})."
        )


def check_table_existence(
    connection_string: str,
    table_name: str,
    engine_args: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Check if the vector table exists in the database

    Args:
        connection_string (str): The connection string for the database.
        table_name (str): The name of the table to check.
        engine_args (Optional[Dict[str, Any]]): Additional arguments for the engine.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))
    try:
        inspector = sqlalchemy.inspect(engine)
        return table_name in inspector.get_table_names()
    finally:
        engine.dispose()


def get_embedding_column_definition(
    connection_string: str,
    table_name: str,
    column_name: str,
    engine_args: Optional[Dict[str, Any]] = None,
):
    """
    Retrieves the column definition of an embedding column from a database table.

    Args:
        connection_string (str): The connection string to the database.
        table_name (str): The name of the table.
        column_name (str): The name of the column.
        engine_args (Optional[Dict[str, Any]]): Additional arguments for the engine.

    Returns:
        tuple: A tuple containing the dimension (int or None) and distance metric (str or None).
    """
    engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))
    try:
        with engine.connect() as connection:
            query = f"""SELECT COLUMN_TYPE, COLUMN_COMMENT
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'"""
            result = connection.execute(sqlalchemy.text(query)).fetchone()
            if result:
                return extract_info_from_column_definition(result[0], result[1])
    finally:
        engine.dispose()

    return None, None


def extract_info_from_column_definition(column_type, column_comment):
    """
    Extracts the dimension and distance metric from a column definition,
    supporting both optional dimension and optional comment.

    Args:
        column_type (str): The column definition, possibly including dimension and a comment.

    Returns:
        tuple: A tuple containing the dimension (int or None) and the distance metric (str or None).
    """
    # Try to extract the dimension, which is optional.
    dimension_match = re.search(r"VECTOR(?:\((\d+)\))?", column_type, re.IGNORECASE)
    dimension = (
        int(dimension_match.group(1))
        if dimension_match and dimension_match.group(1)
        else None
    )

    # Extracting index type and distance metric from the comment, supporting both single and double quotes.
    distance_match = re.search(r"distance=([^,\)]+)", column_comment)
    distance = distance_match.group(1) if distance_match else None

    return dimension, distance


def rerank_data(
    vector_data: list[int, str],
    full_text_data: list[int, str],
    k: int,
    rerank_option: dict[str, any]
):
    rerank_type = rerank_option["rerank_type"]
    rank_value = rerank_option["rank_value"]
    weighted_score = rerank_option.get("weighted_score", [])
    rerank_score_threshold = rerank_option["rerank_score_threshold"]

    if rerank_type == 'RRF':
        return rrf_rerank(vector_data, full_text_data, k, rank_value)
    elif rerank_type == 'WeightedRank':
        return weighted_rank(vector_data, full_text_data, k, weighted_score)
    else:
        return list(set(vector_data + full_text_data))


def rrf_rerank(
    vector_query_result: list[int, str],
    full_text_query_result: list[int, str],
    k: int,
    rank_value: int,
):
    """
    使用 Reciprocal Rank Fusion (RRF) 对两个检索模型的结果进行融合，并返回前 top_n 个文档。

    :param vector_query_result: 向量检索模型的排名列表 [(score, text), ...]
    :param full_text_query_result: 全文检索的排名列表 [(score, text), ...]
    :param rank_value: RRF 中的常数 k
    :param k: 返回前 k 个文档
    :return: 按 RRF 得分排序后的文档列表 [(rrf_score, text), ...]
    """

    # 创建一个字典，用于存储每个文档的 RRF 得分
    rrf_scores = {}

    # 对第一个模型的排名结果计算 RRF 得分
    for score, text in enumerate(vector_query_result, start=1):
        rrf_score = 1 / (rank_value + score)
        if text in rrf_scores:
            rrf_scores[text] += rrf_score
        else:
            rrf_scores[text] = rrf_score

    # 对第二个模型的排名结果计算 RRF 得分
    for score, text in enumerate(full_text_query_result, start=1):
        rrf_score = 1 / (rank_value + score)
        if text in rrf_scores:
            rrf_scores[text] += rrf_score
        else:
            rrf_scores[text] = rrf_score

    # 根据 RRF 得分对文档进行排序，得分从高到低
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 返回前 top_k 个文档，并以 [score, text] 格式返回
    return [[score, text] for text, score in sorted_results[:k]]


def weighted_rank(
    vector_query_result: list[int, str],
    full_text_query_result: list[int, str],
    k: int,
    weights: list[float]
):
    """
    使用加权得分对向量检索结果和全文检索结果进行重排序。

    :param vector_query_result: 向量检索模型的排名结果 [(score, text), ...]
    :param full_text_query_result: 全文检索模型的排名结果 [(score, text), ...]
    :param weights: 对应每个检索模型的权重 [0,1]之间
    :param k: 返回前 k 个文档
    :return: 按加权得分排序后的文档列表 [[score, text], ...]
    """

    # 合并两个检索模型的结果
    all_results = []
    all_results.extend([(score, text, "vector") for score, text in vector_query_result])  # 向量检索结果
    all_results.extend([(score, text, "full_text") for score, text in full_text_query_result])  # 全文检索结果

    # 计算加权得分
    doc_score_map = {}
    for score, text, model_type in all_results:
        weight = weights[0] if model_type == "vector" else weights[1]  # 确定权重 weight0 表示向量检索的权重，weight1 表示全文检索的权重
        norm_score = convert_metric_score(score, model_type)  # 对得分进行归一化
        weighted_score = norm_score * weight  # 加权
        if text in doc_score_map:
            doc_score_map[text] += weighted_score  # 累加得分
        else:
            doc_score_map[text] = weighted_score

    # 根据加权得分进行排序，得分从高到低
    sorted_results = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)

    # 返回前 k 个文档
    return [[score, text] for text, score in sorted_results[:k]]


def convert_metric_score(
    original_score: float,
    metric_type: str
) -> float:
    """
    根据不同的距离/相似度类型，对 original_score 做预处理，转成“分数越大表示越相似”的形式。
    :param original_score: 原始得分
    :param metric_type: 度量类型 "l2", "ip", "cosine" 等
    :return: 归一化后的得分
    目前MO 的向量索引仅支持L2 距离, 所以当前版本的sdk 仅支持L2 距离的预处理。
    """

    # 这里直接设置距离为l2,待2.1.0 发布之后，支持ip 和cosine 距离，去掉这行代码。
    metric_type = "l2"

    if metric_type.lower() in ["l2", "euclidean"]:
        # 距离越小越相似，所以这里取 -distance
        score_for_arctan = -original_score
    elif metric_type.lower() in ["ip", "inner_product"]:
        # 内积值越大越相似，通常可以直接使用
        score_for_arctan = original_score
    elif metric_type.lower() in ["cosine"]:
        # cosine 相似度通常在[-1, 1]之间，直接用
        score_for_arctan = original_score
    else:
        # 默认不做特殊处理
        score_for_arctan = original_score

    return arctan_normalize(score_for_arctan)


def arctan_normalize(score: float) -> float:
    """
    将任意实数score映射到(0,1)之间。
    Formula: normalized_score = (1 / pi) * arctan(score) + 0.5
    当 score -> +∞ 时, 归一化结果 -> 1
    当 score -> -∞ 时, 归一化结果 -> 0
    """
    return (1.0 / math.pi) * math.atan(score) + 0.5

