import pytest
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer
from qdrant_client.http.models import ScalarQuantizationConfig
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, ScalarQuantization, ScalarType
from albalat.scripts.embeddings_to_vector_db import QdrantCli, initialize_client, create_collection

@pytest.fixture
def in_memory_qdrant_client():
    collection_name = "test_collection"
    vectors_param = VectorParams(size=1024, distance=Distance.COSINE)
    scalar_params = ScalarQuantizationConfig(
        type=ScalarType.INT8,
        always_ram=True,
    )
    quantization_params = ScalarQuantization(scalar=scalar_params)

    hnsw_config = HnswConfigDiff(
        m=16,
        ef_construct=200,
    )


    client = QdrantClient(":memory:")


    qdrant_cli = QdrantCli(client=client,
                           collection_name=collection_name,
                           construction_parameters=hnsw_config,
                           vector_params=vectors_param,
                           collection_url="testURL",
                           quantization_config=quantization_params)


    yield qdrant_cli

    if client.collection_exists(qdrant_cli.collection_name):
        client.delete_collection(qdrant_cli.collection_name)

@pytest.fixture(scope="session")
def qdrant_client():
    with DockerContainer("qdrant/qdrant:latest") \
            .with_exposed_ports(6333) as container:
        collection_name = "test_collection"
        vectors_param = VectorParams(size=1024, distance=Distance.COSINE)
        scalar_params = ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=True,
        )
        quantization_params = ScalarQuantization(scalar=scalar_params)

        hnsw_config = HnswConfigDiff(
            m=16,
            ef_construct=200,
        )
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        qdrant_cli_quant = QdrantCli(
            collection_name=collection_name,
            construction_parameters=hnsw_config,
            vector_params=vectors_param,
            quantization_config=quantization_params,
            collection_url=f"http://{host}:{port}")


        initialize_client(qdrant_cli_quant)
        create_collection(qdrant_cli_quant)
        yield qdrant_cli_quant

        if qdrant_cli_quant.client.collection_exists(qdrant_cli_quant.collection_name):
            qdrant_cli_quant.client.delete_collection(qdrant_cli_quant.collection_name)
