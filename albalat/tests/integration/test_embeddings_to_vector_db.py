import yaml
import pytest
import numpy as np
from datasets import Dataset
from testcontainers.core.container import DockerContainer
from albalat.scripts.embeddings_to_vector_db import (
    create_vector_db,
    parse_yaml,
    initialize_client,
    add_embeddings,
    create_collection,
)


@pytest.mark.integration
def test_create_vector_db(tmp_path):
    with DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333) as container:
        embedding_db = {
            "index": [i for i in range(25000)],
            "embeddings": [
                np.random.normal(size=1024) for _ in range(25000)
            ],
        }

        dataset = Dataset.from_dict(embedding_db)
        dataset_path = tmp_path / "embeddings.parquet"
        dataset.to_parquet(dataset_path)

        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        config = {
            "poll_time": 30,
            "stable_time": 120,
            "embeddings_path": str(tmp_path),
            "collection_name": "test_collection3",
            "collection_url": f"http://{host}:{port}",
            "n_processes": 4,
            "upload_batch_size": 256,
            "hnsw": {
                "m": 16,
                "ef_construct": 100,
                "on_disk": True
            },
            "vectors": {
                "datatype": "float16",
                "on_disk": False
            },
            "quantization": {
                "use_quant": True,
            },
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

            create_vector_db(yaml_path)


@pytest.mark.integration
def test_add_embeddings(tmp_path):
    with DockerContainer("qdrant/qdrant:v1.18.3").with_exposed_ports(6333) as container:
        embedding_db = {
            "index": [0, 1, 2],
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        }
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        dataset = Dataset.from_dict(embedding_db)
        config = {
            "poll_time": 1,
            "stable_time": 1,
            "embeddings_path": str(tmp_path),
            "collection_name": "test_collection",
            "collection_url": f"http://{host}:{port}",
            "n_processes": 4,
            "upload_batch_size": 256,
            "hnsw": {
                "m": 16,
                "ef_construct": 100,
                "on_disk": False
            },
            "vectors": {
                "datatype": "float16",
                "on_disk": False
            },
            "quantization": {
                "use_quant": True,
            },
        }

        qdrant_config = parse_yaml(config, 3)
        initialize_client(qdrant_config)
        if qdrant_config.client.collection_exists(qdrant_config.collection_name):
            qdrant_config.client.delete_collection(qdrant_config.collection_name)

        create_collection(qdrant_config)
        add_embeddings(qdrant_config, dataset)
