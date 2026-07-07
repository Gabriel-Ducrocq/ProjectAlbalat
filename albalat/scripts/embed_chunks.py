"""
This scripts takes chunks and embed them with the given model.

The chunk dataset is supposed to be in parquet format and have at least a "paragraphs" column and "chunk_index" column.
It loads the model, places it on GPU.
"""

import os
import torch
import typer
from tqdm import tqdm
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


def get_device(local_rank: int = 0) -> str:
    """
    Gets the device, either cuda, mps or cpu.
    :param local_rank: GPU number.
    :return: device name.
    """
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"

    if torch.backends.mps.is_available():
        return f"mps:{local_rank}"

    return "cpu"


def compute_start_end_indexes(
    local_rank: int, world_size: int, n_groups: int
) -> tuple[int, int]:
    """
    Computes the start and end chunk indices corresponding to the portion assigned to this GPU.
    The end index is supposed to be exclusive.
    :param local_rank: rank of the GPU.
    :param world_size: total number of GPUs.
    :param n_groups: total number of chunks.
    :return: start and end where end is exclusive.
    """
    n_assigned_chunks = n_groups // world_size
    start_index = local_rank * n_assigned_chunks
    end_index = (local_rank + 1) * n_assigned_chunks
    if local_rank == world_size - 1:
        end_index = n_groups + 1

    return start_index, end_index


def load_chunks_datasets(dataset_path: str) -> pq.ParquetFile:
    """
    Reads the dataset only made of the chunks we assign to that GPU.
    :param dataset_path: path to the dataset in arrow format.
    :return: the part of the dataset containing the chunks assigned to the local_rank GPU.
    """
    assert dataset_path.endswith((".parquet")), (
        f"""The dataset file must be an arrow file. Currently {dataset_path}."""
    )
    chunk_dataset = pq.ParquetFile(dataset_path)
    return chunk_dataset


def load_model(
    model_name: str,
    device: str = "cpu",
    model_dtype: str = "torch.bfloat16",
    atten_implem: str = "sdpa",
) -> SentenceTransformer:
    """
    Load the model from the Hugging Face Hub with possibility to tune its type, device and attention implementation.
    :param model_name: name of the model.
    :param model_type: dtype of the model, must be in ["torch.bfloat16", "torch.float32", "torch.float16"]
    :param atten_implem: implementation of the attention mechanism, must be in ["sdpa", "flash_attention_2"]
    :return: the embedding model
    """
    assert model_dtype in [
        "torch.bfloat16",
        "torch.float32",
        "torch.float16",
    ], f"""The dtype of the model must be in 
                                                            ["torch.bfloat16","torch.float32", "torch.float16"], 
                                                            currently {model_dtype}."""

    assert atten_implem in [
        "sdpa",
        "flash_attention_2",
    ], f"""The attention implementation must be in
                                            ["sdpa", "flash_attention_2"], currently {atten_implem}."""

    return SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"dtype": eval(model_dtype), "attn_implementation": "sdpa"},
    )


def embed(
    dataset_path: str,
    model_name: str,
    batch_size_arrow: int = 4096,
    model_dtype: str = "torch.bfloat16",
    atten_implem: str = "sdpa",
    batch_size_encoder: int = 8,
) -> None:
    """
    Function responsible for embedding the chunks. It only embeds the chunks based on its GPU number ahd the total
    number of GPUs in the distributed training.
    :param dataset_path: path to the dataset containing the chunks, it must be an arrow file.
    :param model_name: name of the model that will loaded by Hugging Face.
    :param local_rank: GPU number.
    :param batch_size_arrow: size of the batch used to read each row group in the dataset.
    :param model_dtype: dtype of the embedding model, in ["torch.bfloat16", "torch.float32", "torch.float16"].
    :param atten_implem: implementation of the attention mechanism, must be in ["sdpa", "flash_attention_2"].
    :param batch_size_encoder: batch size to use in the encoding.
    :return: None.
    """

    assert dataset_path is not None and isinstance(
        dataset_path, str
    ), f"""The dataset_path must be a string, currently
                                                                        {type(dataset_path)}."""
    assert model_name is not None and isinstance(
        model_name, str
    ), f"""The model_name must be a string, currently
                                                                    {type(model_name)}."""

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"local_rank={local_rank}, world_size={world_size}")
    chunk_dataset = load_chunks_datasets(dataset_path)
    start_index, end_index = compute_start_end_indexes(
        local_rank, world_size, chunk_dataset.num_row_groups
    )
    device = get_device(local_rank)
    model = load_model(model_name, device, model_dtype, atten_implem)
    for row_group in tqdm(
        range(start_index, end_index), desc=f"GPU:{local_rank}", position=local_rank
    ):
        row_group_chunks = chunk_dataset.read_row_group(
            row_group, columns=["text_ids", "spans", "chapters", "splitted_paragraphs"]
        )
        for batch in row_group_chunks.to_batches(
            max_chunksize=batch_size_arrow
        ):
            texts = batch["splitted_paragraphs"].to_pylist()
            model.encode(
                texts,
                batch_size=batch_size_encoder,
                normalize_embeddings=True,
            )


if __name__ == "__main__":
    typer.run(embed)
