import numpy as np
import pandas as pd
from datasets import Dataset
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScalarQuantizationConfig
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, QuantizationConfig, ScalarQuantization, ScalarType
from albalat.scripts.embeddings_to_vector_db import QdrantCli, initialize_client, create_collection, define_path, verify_parquet_folder, load_embeddings_dataset, validate_dataset_format, convert_embeddings_float16, disable_index_construction, enable_index_construction, parse_yaml


class TestEmbeddingsToVectorDB:
    def test_initialize_qdrant_client(self, in_memory_qdrant_client):
        initialize_client(in_memory_qdrant_client)
        assert isinstance(in_memory_qdrant_client.client,QdrantClient), """Created client is not a Qdrant client."""

    def test_define_path_wrong_path(self):
        test_folder = "test_folder"
        has_raised = False
        try:
            define_path(test_folder)
        except:
            has_raised = True

        assert has_raised, f"""Folder {test_folder} does not exist and should have raised an exception"""

    def verify_parquet_folder_no_parquet(self, tmp_path):
        has_raised = False
        try:
            verify_parquet_folder(tmp_path)
        except:
            has_raised = True

        assert not has_raised, f"""Folder {tmp_path} has parquet file(s) and should not raise an error."""

    def verify_parquet_folder_with_parquet(self, tmp_path):
        data = {
            "index": [0, 1, 2],
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        }

        dataset = Dataset.from_dict(data)

        dataset.to_parquet(tmp_path / "embeddings.parquet")

        has_raised = False
        try:
            verify_parquet_folder(tmp_path)
        except:
            has_raised = True

        assert not has_raised, f"""Folder {tmp_path} has parquet file(s) and should not raise an error."""

    def test_load_embeddings(self, tmp_path):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        })

        df.to_parquet(tmp_path / "part-000.parquet")
        df.to_parquet(tmp_path / "part-001.parquet")
        loaded_dataset = load_embeddings_dataset(tmp_path)
        assert len(loaded_dataset) == 6, f"""Dataset length should be 3, currently {len(loaded_dataset)}"""

    def test_validate_dataset_format_missing_embeddings(self):
        has_raised = False
        try:
            dataset = Dataset.from_dict({
                "index": [0, 1],
                "test": [
                    np.array([0.1, 0.2], dtype=np.float16),
                    np.array([0.3, 0.4], dtype=np.float16),
                ],
            })
            validate_dataset_format(dataset)
        except:
            has_raised = True

        assert has_raised, """Missing embeddings column should have raised an error."""

    def test_validate_dataset_format_missing_index(self):
        has_raised = False
        try:
            dataset = Dataset.from_dict({
                "test": [0, 1],
                "embeddings": [
                    np.array([0.1, 0.2], dtype=np.float16),
                    np.array([0.3, 0.4], dtype=np.float16),
                ],
            })
            validate_dataset_format(dataset)
        except:
            has_raised = True

        assert has_raised, """Missing index column should have raised an error."""

    def test_validate_dataset_format_valid(self):
        dataset = Dataset.from_dict({
            "index": [0, 1],
            "embeddings": [
                np.array([0.1, 0.2], dtype=np.float16),
                np.array([0.3, 0.4], dtype=np.float16),
            ],
        })
        dataset = convert_embeddings_float16(dataset)
        dataset.set_format("numpy")
        validate_dataset_format(dataset)

    def test_disable_indexing_integration(self, qdrant_client):
        disable_index_construction(qdrant_client)
        collection = qdrant_client.client.get_collection(qdrant_client.collection_name)
        assert collection.config.optimizer_config.indexing_threshold == 0, f"""The indexing threhsold should be at 0, currently
                                                                                {collection.config.optimizer_config.indexing_threshold} """
    def test_enable_indexing_integration(self, qdrant_client):
        disable_index_construction(qdrant_client)
        enable_index_construction(qdrant_client)
        collection = qdrant_client.client.get_collection(qdrant_client.collection_name)
        assert collection.config.optimizer_config.indexing_threshold == 20000, f"""The indexing threshold should be at 20000, currently
                                                                                {collection.config.optimizer_config.indexing_threshold} """


    def test_parse_yaml(self):
        config = {"hnsw":{"m":13, "ef_construct": 200},
                  "collection_name":"test_collec",
                  "collection_url":"http://localhost:6333",
                  "n_processes": 4,
                  "upload_batch_size": 200,
                  "quantization":{"use_quant":False}}
        qdrant_cli = parse_yaml(config, 1024)

        assert qdrant_cli.collection_name == "test_collec", f"""Expected collection name 'test_collec', recovered 
                                                                {qdrant_cli.collection_name}"""
        assert qdrant_cli.collection_url == "http://localhost:6333", f"""Expected collection url 'http://localhost:6333'
                                                                         {qdrant_cli.collection_url}"""
        assert qdrant_cli.n_processes == 4, f"""Expected n_processes 4, recovered {qdrant_cli.n_processes}"""
        assert qdrant_cli.upload_batch_size == 200, f"""Expected Upload batch size 200, recovered {qdrant_cli.upload_batch_size}"""

        assert qdrant_cli.construction_parameters.m == 13
        assert qdrant_cli.construction_parameters.ef_construct == 200

        assert qdrant_cli.vector_params.size == 1024
        assert qdrant_cli.vector_params.distance == Distance.COSINE
        assert qdrant_cli.quantization_config is None


    def test_parse_yaml_with_quantization(self):
        config = {
            "collection_name": "test_collection",
            "collection_url": "http://localhost:6333",
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

        qdrant = parse_yaml(config, embeddings_dim=768)

        assert qdrant.quantization_config is not None
        assert qdrant.quantization_config.scalar.type == ScalarType.INT8
        assert qdrant.quantization_config.scalar.always_ram is True
