"""
This script creates a HNSW vector database using quadrant, based on a database of embeddings.

"""
import os
import time
import yaml
import typer
from pathlib import Path
from dataclasses import dataclass
from qdrant_client import QdrantClient
from datasets import Dataset, load_dataset, Sequence, Value
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.http.models import ScalarQuantizationConfig
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType,
    Datatype
)


@dataclass
class QdrantCli:
    collection_name: str
    collection_url: str
    construction_parameters: HnswConfigDiff
    vector_params: VectorParams
    quantization_config: QuantizationConfig | None = None
    n_processes: int = 4
    upload_batch_size: int = 256
    client: QdrantClient | None = None

    def to_qdrant_kwargs(self):
        kwargs = {
            "collection_name": self.collection_name,
            "vectors_config": self.vector_params,
            "hnsw_config": self.construction_parameters,
        }

        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        return kwargs


def initialize_client(qdrant_cli: QdrantCli) -> None:
    """
    Initializes qdrant client. This function does not return anything but has a side effect of updating the state of the
    qdrant_cli object.
    :param qdrant_cli: object of class QdrantCli containing all the necessary information about the collection.
    :return: None
    """
    qdrant_cli.client = QdrantClient(url=qdrant_cli.collection_url)


def create_collection(qdrant_cli: QdrantCli) -> None:
    """
    Creates a qdrant collection with specified name and parameters.
    :param qdrant_cli: object of type QdrantCli containing all the necessary information about the collection.
    :return: None
    """
    try:
        qdrant_cli.client.create_collection(**qdrant_cli.to_qdrant_kwargs())
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
    if not path.exists():
        raise FileNotFoundError(f"""The path {path} does not exists""")

    return path


def verify_parquet_folder(path: Path) -> None:
    """
    Verifies that the folder has some parquet files
    :param path: path to the folder containing parquet.
    :return: None
    """
    all_files = os.listdir(path)
    if all_files == []:
        raise ValueError(f"""Path {all_files} contains no file.""")
    parquet_files = [file for file in all_files if file.endswith(".parquet")]
    if parquet_files == []:
        raise ValueError(f"""Path {all_files} contains no parquet file.""")


def convert_embeddings_float16(dataset: Dataset) -> Dataset:
    """
    Convert the embeddings column of a dataset to float16.
    :param dataset: dataset with at least an "embeddings" column.
    :return: dataset.
    """
    dataset = dataset.cast_column("embeddings", Sequence(Value(dtype="float16")))
    return dataset


def load_embeddings_dataset(path: Path) -> Dataset:
    """
    Reads the dataset of embeddings in parquet format
    :param path: path to the folder containing the embeddings in parquet format.
    :return: hugging face dataset
    """
    dataset = load_dataset("parquet", data_files=str(path / "*.parquet"), split="train")
    dataset = convert_embeddings_float16(dataset)
    dataset.set_format("numpy")
    return dataset


def validate_dataset_format(dataset: Dataset) -> None:
    """
    Validates whether the dataset has an "embeddings" column and an "index" columns.
    :param dataset: Hugging Face dataset
    :return: None
    """
    colnames = set(dataset.column_names)
    expected_colnames = {"embeddings", "index"}
    if expected_colnames != expected_colnames.intersection(colnames):
        raise ValueError(f"""The columns {expected_colnames} are expected
                                                                    to be part of the Hugging Face dataset. Currently
                                                                    has columns {colnames}.""")

    if dataset.features["embeddings"].feature.dtype != "float16":
        raise TypeError(f"""The embeddings are supposed to be float16, currently
                                                            {dataset[0]["embeddings"].dtype}""")


def disable_index_construction(qdrant_client: QdrantCli) -> None:
    """
    Disables client construction for maximum efficiency in uploading the collection. This function returns nothing
    but has the side effect of changing the parameterization of the collection.
    :param qdrant_client: QdrantCli object with all the parameters of the qdrant collection/client.
    :return: None.
    """
    qdrant_client.client.update_collection(
        collection_name=qdrant_client.collection_name,
        optimizers_config={"indexing_threshold": 0},
    )


def enable_index_construction(qdrant_client: QdrantCli) -> None:
    """
    Enables client construction for maximum efficiency in uploading the collection. This function returns nothing
    but has the side effect of changing the parameterization of the collection.
    :param qdrant_client: QdrantCli object with all the parameters of the qdrant collection/client.
    :return: None.
    """
    qdrant_client.client.update_collection(
        collection_name=qdrant_client.collection_name,
        optimizers_config={"indexing_threshold": 20000},
    )


def add_embeddings(qdrant_cli: QdrantCli, embeddings_and_indexes: Dataset) -> None:
    """
    Adds the embeddings to the collection. As there are roughly 10 millions embeddings, we let Qdrant deal with the
    batching and insertion logic.
    :param qdrant_cli: QdrantCli object containing all the information related to the collection.
    :param embeddings_and_indexes: datasets of embeddings with the index of the graph.
    :return: None
    """
    qdrant_cli.client.upload_collection(
        qdrant_cli.collection_name,
        ids=[int(ind) for ind in embeddings_and_indexes["index"]],
        vectors=embeddings_and_indexes["embeddings"],
        parallel=qdrant_cli.n_processes,
        batch_size=qdrant_cli.upload_batch_size,
    )


def load_yaml(yaml_path: str) -> dict:
    """
    Load the yaml file into a dictionary.
    :param yaml_path: path to the yaml file.
    :return: a dictionary with the configuration.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    return config


def parse_yaml(config: dict, embeddings_dim: int) -> QdrantCli:
    """
    Loads the yaml file and translate it into a QdrantCli object.
    :param config: dict of the yaml file configuring the vector db.
    :param embeddings_dim: dimension of the embeddings.
    :return: the QdrantCli object containing all the parameters.
    """
    hnsw_config = HnswConfigDiff(
        m=config["hnsw"]["m"],
        ef_construct=config["hnsw"]["ef_construct"],
        on_disk=config["hnsw"]["on_disk"]
    )
    vectors_param = VectorParams(size=embeddings_dim,
                                 distance=Distance.COSINE,
                                 on_disk=config["vectors"]["on_disk"],
                                 datatype = Datatype.FLOAT16 if config["vectors"]["datatype"] == "float16" else Datatype.FLOAT32)

    scalar_params = ScalarQuantizationConfig(
        type=ScalarType.INT8,
        always_ram=True,
    )
    quantization_params = ScalarQuantization(scalar=scalar_params)
    return QdrantCli(
        collection_name=config["collection_name"],
        collection_url=config["collection_url"],
        construction_parameters=hnsw_config,
        n_processes=config["n_processes"],
        upload_batch_size=config["upload_batch_size"],
        vector_params=vectors_param,
        quantization_config=None
        if not config["quantization"]["use_quant"]
        else quantization_params,
    )


def prepare_embeddings(embedding_path: str) -> Dataset:
    """
    Given the path to the embeddings folder, creates the correct pattern for reading the parquet files, verifies its
    existence and the existence of at least one parquet file inside, load the embedding dataset and returns
    the "index" and "embeddings" columns.
    Convention: the data to be read is in parquet format and contains at least an "index" and "embeddings" columns.
                The path in the argument of the function is the path of the folder containing at least one parqet file.
                The script will load all parquet files in that folder.
    :param embedding_path: path to the embedding folder.
    :return: Hugging Face dataset with at least "index" and "embeddings" columns.
    """
    embedding_path = define_path(embedding_path)
    verify_parquet_folder(embedding_path)
    embedding_dataset = load_embeddings_dataset(embedding_path)
    validate_dataset_format(embedding_dataset)
    return embedding_dataset.select_columns(["index", "embeddings"])


def efficient_add_embeddings(qdrant_config: QdrantCli, embedding_data: Dataset) -> None:
    """
    Efficiently embeds the embeddings by disabling the indexing in the Qdrant collection, then adding all the embeddings
    and finally re-enabling the indexing.
    :param qdrant_config: all the parameters of the collection.
    :param embedding_data: Hugging Face Dataset with only columns "index" and "embeddings".
    :return: None
    """
    disable_index_construction(qdrant_config)
    add_embeddings(qdrant_config, embedding_data)
    enable_index_construction(qdrant_config)


def wait_indexing(client: QdrantClient, collection_name: str, stable_time: float, poll_time: float) -> None:
    """
    Wait for the index to be built: checks every minute if the status is green.
    :param: qdrant client
    :param collection_name: name of the collection.
    :param stable_time: time interval in seconds during which the collection is stable to consider that the index has been built.
    :param poll_time: time interval, in second, before we check the status of the collection again.
    :return: None
    """
    previous_info = None
    stable_start = None
    start_time = time.time()
    while True:
        info = client.get_collection(collection_name)

        current_info = (info.status,
            info.indexed_vectors_count,
            info.points_count
        )

        print(
            "status:", info.status,
            "indexed vectors count:", info.indexed_vectors_count,
            "points count:", info.points_count
        )

        if current_info == previous_info and info.status == "green":
            if stable_start is None:
                stable_start = time.time()
            elif time.time() - stable_start > stable_time:
                print("Collection stable")
                break

        previous_info = current_info
        time.sleep(poll_time)

    print(f"HNSW indexing completed in {time.time() - start_time}")

def create_vector_db(yaml_path: str) -> None:
    """
    Creates and initialize the client, disable the indexing, upload the embeddings and enable the indexing.
    :param yaml_path: path to the yaml file for getting the configuration of the experiment.
    :return: None
    """
    config = load_yaml(yaml_path)
    embedding_dataset = prepare_embeddings(config["embeddings_path"])
    qdrant_config = parse_yaml(config, embedding_dataset[0]["embeddings"].shape[-1])
    initialize_client(qdrant_config)
    if qdrant_config.client.collection_exists(qdrant_config.collection_name):
        qdrant_config.client.delete_collection(qdrant_config.collection_name)

    create_collection(qdrant_config)
    efficient_add_embeddings(qdrant_config, embedding_dataset)
    wait_indexing(qdrant_config.client, qdrant_config.collection_name, stable_time=config["stable_time"],
                  poll_time=config["poll_time"])


if __name__ == "__main__":
    typer.run(create_vector_db)
