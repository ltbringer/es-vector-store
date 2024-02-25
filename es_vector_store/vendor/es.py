"""
Code snippet: llama_index
permalink: https://github.com/run-llama/llama_index/blob/8118a45ea86f0dd391e4838e5423d67849c34955/llama-index-legacy/llama_index/legacy/vector_stores/elasticsearch.py
"""

from typing import Any, ClassVar, List, Mapping, Tuple
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    PrivateAttr,
    field_validator,
    ValidationInfo,
)
from elasticsearch import Elasticsearch, AsyncElasticsearch
from es_vector_store.utils.exclusive_args import (
    is_mutually_exclusive,
    MutuallyExclusiveStatusCode,
)
from es_vector_store import project

StringOrTuple = str | Tuple[str, str]


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


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN_DISTANCE = "l2_norm"
    DOT_PRODUCT = "dot_product"
    MAX_INNER_PRODUCT = "max_inner_product"


class VectorIndexType(Enum):
    HNSW = "hnsw"
    INT8_HSNW = "int8_hnsw"


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


class VectorStore(BaseModel):
    """
    ElasticsearchStore is a vector store that uses Elasticsearch as the backend.
    Use https://www.elastic.co/guide/en/elasticsearch/reference/current/explicit-mapping.html
    to define the mappings for other fields.

    Args:
        index: The name of the index to use.
        vector_field: A tuple containing the (name, dimension) of the vector.
        mappings: Other fields in the index.
                        To contain metadata about the vector or related vectors.
                        At least one field is required
        connection_params: The parameters to connect with an Elasticsearch cluster.
        async_client: If true, use the async Elasticsearch client.
    """

    index: str
    vector_field: Tuple[str, int]
    enable_vector_index: bool = True
    vector_index_options: VectorIndexOptions | None = Field(default=None)
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    mappings: Mapping[str, Any] = Field(default_factory=dict)
    connection_params: ConnectionParams
    async_client: bool = False
    MIN_FIELDS: ClassVar[int] = 1
    _client: Elasticsearch | AsyncElasticsearch = PrivateAttr()

    @model_validator(mode="after")
    def vector_indexing_and_settings(self):
        if self.enable_vector_index and self.vector_index_options is None:
            raise ValueError(
                f"vector_index_options is required when enable_vector_index is True."
            )
        return self

    @field_validator("mappings")
    def mappings_schema(
        cls, mappings: Mapping[str, Any], info: ValidationInfo
    ) -> "VectorStore":
        invalid_mappings_error = (
            f"Invalid schema because {info.field_name}={mappings}."
            " Explicit mappings are expected. "
            "Ref: "
        )
        missing_field_def_error = (
            f"{info.field_name} must contain at least "
            "one field to map the vector to text, url or other metadata."
        )
        if mappings is None:
            raise ValueError(invalid_mappings_error)

        if "properties" not in mappings:
            raise ValueError(invalid_mappings_error)

        if not mappings["properties"]:
            raise ValueError(missing_field_def_error)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = self._create_client()
        self._create_index()

    def _create_vector_mapping(self) -> DenseVectorField:
        _, dimension = self.vector_field
        return DenseVectorField(
            dims=dimension,
            similarity=self.distance_metric,
            index=self.enable_vector_index,
            index_options=(
                self.vector_index_options.dict() if self.enable_vector_index else None
            ),
        )

    def _create_client(self):
        connection_config = self.connection_params.generate_config()
        if self.async_client:
            return AsyncElasticsearch(**connection_config)
        return Elasticsearch(**connection_config)

    def _create_index(self):
        if self._client.indices.exists(index=self.index):
            print(f"Index {self.index} already exists.")
            return
        vector_name, _ = self.vector_field
        vector_mapping = {vector_name: self._create_vector_mapping().dict()}
        explicit_mapping = {"properties": {**self.mappings, **vector_mapping}}

        import pprint

        pprint.pprint(explicit_mapping)
        self._client.indices.create(index=self.index, mappings=explicit_mapping)
