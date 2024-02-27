from typing import ClassVar, List, Tuple
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from es_vector_store.utils.exclusive_args import (
    is_mutually_exclusive,
    MutuallyExclusiveStatusCode,
)

StringOrTuple = str | Tuple[str, str]


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN_DISTANCE = "l2_norm"
    DOT_PRODUCT = "dot_product"
    MAX_INNER_PRODUCT = "max_inner_product"


class VectorIndexType(Enum):
    HNSW = "hnsw"
    INT8_HSNW = "int8_hnsw"


class VectorSearchMode(Enum):
    KNN = "knn"
    TEXT = "query"
    HYBRID = "hybrid"

    def query_body(self) -> bool:
        match self:
            case VectorSearchMode.KNN:
                return {
                    "knn": {
                        "filter": filter,
                        "field": self.vector_field,
                        "query_vector": query_embedding,
                        "k": query.similarity_top_k,
                        "num_candidates": query.similarity_top_k * 10,
                    }
                }
            case VectorSearchMode.TEXT:
                return {
                    "query": {
                        "bool": {
                            "must": {
                                "match": {self.text_field: {"query": query.query_str}}
                            },
                            "filter": filter,
                        }
                    }
                }


class ESFilterKV(BaseModel):
    field: str
    value: str


class ESFilters(BaseModel):
    filters: List[ESFilterKV]

    def dict(self):
        if len(self.filters) == 1:
            f = self.filters[0]
            return {"term": {f.field: {"value": f.value}}}
        else:
            operands = [{"term": {f.field: {"value": f.value}}} for f in self.filters]
            return {"bool": {"must": operands}}


class KNNQuery(BaseModel):
    filter: ESFilters | None = Field(default=None)
    vector_field: str
    vector: List[float]
    top_k: int
    num_candidates: int | None = Field(default=None)
    DEFAULT_NUM_CANDIDATES_FACTOR_OF_TOP_K: ClassVar[int] = 10

    def dict(self):
        num_candidates = (
            KNNQuery.DEFAULT_NUM_CANDIDATES_FACTOR_OF_TOP_K * self.top_k
            if self.num_candidates is None
            else self.num_candidates
        )
        params = {
            "knn": {
                "field": self.vector_field,
                "query_vector": self.vector,
                "k": self.top_k,
                "num_candidates": num_candidates,
            }
        }
        if self.filter:
            params["knn"]["filter"] = self.filter.dict()
        return params


class TextQuery(BaseModel):
    field: str
    query: str
    filter: ESFilters

    def dict(self):
        return {
            "query": {
                "bool": {
                    "must": {"match": {self.field: {"query": self.query}}},
                    "filter": self.filter.dict(),
                }
            }
        }


class ConnectionParams(BaseModel):
    """
    Refer to the following for extra_config:
    - https://github.com/elastic/elasticsearch-py/blob/44c5c6993f67daf2a56d2310e67af6874c4bdc8f/elasticsearch/_sync/client/__init__.py#L99
    - https://github.com/elastic/elasticsearch-py/blob/44c5c6993f67daf2a56d2310e67af6874c4bdc8f/elasticsearch/_async/client/__init__.py#L99

    Args:
        url: The URL of the Elasticsearch cluster.
        cloud_id: The cloud ID of the Elasticsearch cluster.
        api_key: The API key to use for authentication.
        username: The username to use for basic authentication.
        password: The password to use for basic authentication.
        extra_config: Full set of parameters to pass to the Elasticsearch client.
    """

    url: str | None = Field(default=None)
    cloud_id: str | None = Field(default=None)
    api_key: StringOrTuple | None = Field(default=None)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None, repr=False)
    extra_config: dict | None = Field(default=None)

    def generate_config(self) -> dict:
        base_config = ConnectionConfig(
            hosts=[self.url],
            cloud_id=self.cloud_id,
            api_key=self.api_key,
            basic_auth=(self.username, self.password),
        ).dict()
        extra_config = self.extra_config or {}
        return {**base_config, **extra_config}


class ConnectionConfig(BaseModel):
    hosts: List[str]
    cloud_id: str | None
    api_key: StringOrTuple | None
    basic_auth: StringOrTuple | None = Field(repr=False)

    def dict(self):
        return self.model_dump()

    @model_validator(mode="after")
    def validate_url_or_cloud_id(self) -> "ConnectionConfig":
        url_cloud_id_status = is_mutually_exclusive(self.hosts, self.cloud_id)
        if url_cloud_id_status == MutuallyExclusiveStatusCode.ALL_NONE:
            raise ValueError("At least one of url or cloud_id must be set.")

        if url_cloud_id_status == MutuallyExclusiveStatusCode.MULTIPLE_SET:
            raise ValueError("Only one of url or cloud_id must be set.")
        return self


class VectorIndexOptions(BaseModel):
    type: VectorIndexType
    m: int | None = Field(default=None)
    ef_construction: int | None = Field(default=None)
    confidence_interval: float | None = Field(default=None)

    @model_validator(mode="after")
    def confidence_interval_only_when_int8_hsnw(self):
        using_hnsw_index = self.type == VectorIndexType.HNSW
        using_confidence_interval = isinstance(self.confidence_interval, float)
        if using_confidence_interval and using_hnsw_index:
            raise ValueError(
                "confidence_interval only works when "
                f"index type is {VectorIndexType.INT8_HSNW.value}"
            )
        return self

    def dict(self):
        raw_obj = self.model_dump()
        raw_obj["type"] = self.type.value
        obj = {k: v for k, v in raw_obj.items() if v is not None}
        return obj


class DenseVectorField(BaseModel):
    type: str = "dense_vector"
    dims: int
    similarity: DistanceMetric
    index: bool
    index_options: VectorIndexOptions | None = Field(default=None)

    @model_validator(mode="after")
    def vector_indexing_and_settings(self):
        if self.index and self.index_options is None:
            raise ValueError(f"`index_options` is required when index is True.")
        return self

    def dict(self):
        obj = self.model_dump()
        obj["similarity"] = obj["similarity"].value
        obj["index_options"] = self.index_options.dict()
        return obj
