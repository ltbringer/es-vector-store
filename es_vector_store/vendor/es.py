"""
Code snippet: llama_index
permalink: https://github.com/run-llama/llama_index/blob/8118a45ea86f0dd391e4838e5423d67849c34955/llama-index-legacy/llama_index/legacy/vector_stores/elasticsearch.py
"""

import uuid
from typing import Any, ClassVar, Generator, List, Mapping, Tuple

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    PrivateAttr,
)
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch import helpers
from es_vector_store.vendor.schemas import (
    ConnectionParams,
    DistanceMetric,
    VectorIndexOptions,
    DenseVectorField,
    KNNQuery,
    TextQuery,
)


def generate_records(
    data: Generator[Mapping[str, Any], None, None],
    index_name: str,
    expected_fields: set,
):
    for item in data:
        input_fields = set(item.keys())
        extra_fields = input_fields - expected_fields
        missing_fields = expected_fields - input_fields

        if extra_fields:
            raise ValueError(
                f"Found extra fields {extra_fields} in the"
                f" input data only provide {expected_fields}."
            )
        if missing_fields:
            raise ValueError(
                f"Missing fields {missing_fields} in "
                f"the input data provide {expected_fields}."
            )

        id_ = uuid.uuid4().hex
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": id_,
            **item,
        }


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
    vector_fields: List[Tuple[str, int]]
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

    @model_validator(mode="after")
    def mappings_schema(self) -> "VectorStore":
        invalid_mappings_error = (
            f"Invalid schema because mappings={self.mappings}."
            " Explicit mappings are expected. Ref:"
            " https://www.elastic.co/guide/en/elasticsearch/reference/current/explicit-mapping.html"
        )
        missing_field_def_error = (
            f"mappings must contain at least "
            "one field to map the vector to text, url or other metadata."
        )
        if not self.mappings:
            raise ValueError(invalid_mappings_error)

        if "properties" not in self.mappings:
            raise ValueError(invalid_mappings_error)

        if not self.mappings["properties"]:
            raise ValueError(missing_field_def_error)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = self._create_client()
        self._create_index()

    def _create_vector_mappings(self) -> List[Mapping[str, DenseVectorField]]:
        es_vector_fields = []
        for vector_field in self.vector_fields:
            vector_field_name, dimension = vector_field
            es_vector_fields.append(
                {
                    vector_field_name: DenseVectorField(
                        dims=dimension,
                        similarity=self.distance_metric,
                        index=self.enable_vector_index,
                        index_options=(
                            self.vector_index_options.dict()
                            if self.enable_vector_index
                            else None
                        ),
                    ).dict()
                }
            )
        return es_vector_fields

    def _create_client(self):
        connection_config = self.connection_params.generate_config()
        if self.async_client:
            return AsyncElasticsearch(**connection_config)
        return Elasticsearch(**connection_config)

    def _create_index(self):
        if self._client.indices.exists(index=self.index):
            print(f"Index {self.index} already exists.")
            return
        vector_mappings = self._create_vector_mappings()
        explicit_mapping = self.mappings
        for vector_mapping in vector_mappings:
            explicit_mapping["properties"].update(vector_mapping)
        self._client.indices.create(index=self.index, mappings=explicit_mapping)

    def bulk_insert(self, g: Generator[Mapping[str, Any], None, None]):
        vector_field_names = [vector_field[0] for vector_field in self.vector_fields]
        other_fields = list(self.mappings["properties"].keys())
        expected_fields = set(vector_field_names + other_fields)
        helpers.bulk(self._client, generate_records(g, self.index, expected_fields))

    def search(
        self, knn_query: KNNQuery | None = None, text_query: TextQuery | None = None
    ):
        knn_query_dict = knn_query.dict() if knn_query else {}
        text_query_dict = text_query.dict() if text_query else {}
        query = {**knn_query_dict, **text_query_dict}
        return self._client.search(index=self.index, body=query)

    def drop(self):
        return self._client.indices.delete(index=self.index)
