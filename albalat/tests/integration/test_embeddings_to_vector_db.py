import yaml
import numpy as np
from datasets import Dataset
from testcontainers.core.container import DockerContainer
from albalat.scripts.embeddings_to_vector_db import create_vector_db, parse_yaml, initialize_client, add_embeddings, create_collection

def test_create_vector_db(tmp_path):
    with DockerContainer("qdrant/qdrant:latest") \
            .with_exposed_ports(6333) as container:

        embedding_db =  {
            "index": [0, 1, 2],
            "embeddings": [
                np.random.normal(size=1024),
                np.random.normal(size=1024),
                np.random.normal(size=1024),
            ],
        }

        dataset = Dataset.from_dict(embedding_db)
        dataset_path =  tmp_path / "embeddings.parquet"
        dataset.to_parquet(dataset_path)

        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        config = {
            "embeddings_path": str(tmp_path),
            "collection_name": "test_collection3",
            "collection_url": f"http://{host}:{port}",
            "n_processes": 4,
            "upload_batch_size": 256,
            "hnsw": {
                "m": 16,
                "ef_construct": 100,
            },
            "quantization": {
                "use_quant": True,
            },
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        try:
            create_vector_db(yaml_path)
        except:
            assert False, "Could not go through the create_vector_db function."

def test_add_embeddings(tmp_path):
    with DockerContainer("qdrant/qdrant:latest") \
            .with_exposed_ports(6333) as container:

        embedding_db =  {
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
            "embeddings_path": str(tmp_path),
            "collection_name": "test_collection",
            "collection_url": f"http://{host}:{port}",
            "n_processes": 4,
            "upload_batch_size": 256,
            "hnsw": {
                "m": 16,
                "ef_construct": 100,
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
        try:
            add_embeddings(qdrant_config, dataset)
        except:
            assert False, "Could not go through adding the embeddings to the db"
