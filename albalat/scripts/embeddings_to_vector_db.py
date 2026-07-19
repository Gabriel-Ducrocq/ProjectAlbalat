"""
This script creates a HNSW vector database using quadrant, based on a database of embeddings.

"""
import os
from pathlib import Path
from dataclasses import dataclass
from qdrant_client import QdrantClient
from datasets import Dataset, load_dataset
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.http.models import ScalarQuantizationConfig
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, QuantizationConfig, ScalarQuantization, ScalarType


@dataclass
class QdrantCli:
    client: QdrantClient
    collection_name: str
    construction_parameters: HnswConfigDiff
    vector_params: VectorParams
    quantization_config: QuantizationConfig | None = None

    def to_qdrant_kwargs(self):
        kwargs = {"collection_name":self.collection_name,
                  "vectors_config": self.vector_params,
                  "hnsw_config": self.construction_parameters}

        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        return kwargs

def initialize_client(url: str = "http://localhost:6333") -> QdrantClient:
    """
    Initalizes qdrant client
    :param url: url where to reach the client.
    :return: the qdrant client
    """
    client = QdrantClient(url=url)
    return client

def create_collection(qdrant_cli: QdrantCli):
    """
    Creates a qdrant collection with specified name and parameters.
    :param client: qdrant client to use to create a collection.
    :param collection_name: name of the collection we want to create.
    :param construction_parameters: parameters for the construction of the HNSW graph.
    :param url: url where to access the qdrant vector DB started from ocker/apptainer.
    :return: None
    """
    try:
        client.create_collection(
        **qdrant_cli.to_qdrant_kwargs()
        )
    except UnexpectedResponse as e:
        # Server rejected the request (bad params, collection already exists, etc.)
        print(f"Qdrant rejected the request: {e}")
        raise
    except ResponseHandlingException as e:
        # Network/connection-level failure (server down, timeout, etc.)
        print(f"Could not reach Qdrant: {e}")
        raise

def define_path(path: str) -> Path:
    """
    Check whether the path define by the string exists and if yes return it as a Path object.
    :param path: path to the folder.
    :return: path to the folder as Path object
    """
    path = Path(path)
    assert path.exists(), f"""The path {path} does not exists"""
    return path

def verify_parquet_folder(path: Path) -> None:
    """
    Verifies that the folder has some parquet files
    :param path: path to the folder containing parquet.
    :return: None
    """
    all_files = os.listdir(path)
    assert all_files != [], f"""Path {all_files} contains no file."""
    parquet_files = [file for file in all_files if file.endswith(".parquet")]
    assert parquet_files != [], f"""Path {all_files} contains no parquet file."""

def load_embeddings_dataset(path: Path) -> Dataset:
    """
    Reads the dataset of embeddings in parquet format
    :param path: path to the folder containing the embeddings in parquet format.
    :return: hugging face dataset
    """
    dataset = load_dataset("parquet", data_files=str(path / "*.parquet"), split="train")
    return dataset


client = initialize_client("testURL")
assert isinstance(client,QdrantClient), """Created client is not a Qdrant client."""
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
                       vector_params=vectors_param)
create_collection(qdrant_cli)

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

qdrant_cli_quant = QdrantCli(client=client,
                       collection_name=collection_name,
                       construction_parameters=hnsw_config,
                       vector_params=vectors_param,
                       quantization_config=quantization_params)

create_collection(qdrant_cli_quant)

test_folder = "test_folder"
has_raised = False
try:
    define_path(test_folder)
except:
    has_raised = True

assert has_raised, f"""Folder {test_folder} does not exist and should have raised an exception"""



test_folder = "../../albalat/data/processed/"
has_raised = False
try:
    test_folder = define_path(test_folder)
except:
    has_raised = True

assert not has_raised, f"""Folder {test_folder} does not exist and should have raised an exception"""

has_raised = False
try:
    verify_parquet_folder(test_folder)
except:
    has_raised = True

assert not has_raised, f"""Folder {test_folder} has parquet file(s) and should not raise an error."""

test_folder = define_path("../../albalat/data/raw/")
has_raised = False
try:
    verify_parquet_folder(test_folder)
except:
    has_raised = True

assert has_raised, f"""Folder {test_folder} has not parquet file and should raise an error."""


#['n_words', 'text_ids', 'chapters', 'spans', 'splitted_paragraphs', 'index', 'embeddings']