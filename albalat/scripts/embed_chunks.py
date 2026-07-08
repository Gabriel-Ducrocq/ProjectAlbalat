"""
This scripts takes chunks and embed them with the given model.

The chunk dataset is supposed to be in parquet format and have at least a "paragraphs" column and "chunk_index" column.
It loads the model, places it on GPU.
"""

import os

import datasets
import torch
import typer
from datasets import Dataset
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from datasets.distributed import split_dataset_by_node


@dataclass
class encodingParameters:
    """Class that contains all the parameters for the encoding."""

    num_proc: int = 4
    batch_size: int = 8
    convert_to_numpy: bool = True
    normalize_embeddings: bool = True
    batched: bool = True


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


def load_chunks_datasets(dataset_path: str) -> Dataset:
    """
    Reads the dataset only made of the chunks we assign to that GPU.
    :param dataset_path: path to the dataset in arrow format.
    :return: the part of the dataset containing the chunks assigned to the local_rank GPU.
    """
    assert dataset_path.endswith((".parquet")), (
        f"""The dataset file must be an arrow file. Currently {dataset_path}."""
    )
    chunk_dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
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


def encode_chunk(
    sample: str,
    model: SentenceTransformer,
    encoding_parameters: encodingParameters,
) -> dict:
    """
    Encode the chunks in that batch.
    :param model: model that we use for encoding.
    :param sample: samples of the dataset.
    :param batch_size: size of the batch for the encode method.
    :param convert_to_numpy: whether to convert to numpy or not.
    :param normalize_embeddings: whether to normalize embeddings.
    :param num_proc: number of workers.
    :return: a dictionnary corresponding to the embeddings output of the batch.
    """
    return {
        "embeddings": model.encode(
            sample,
            batch_size=encoding_parameters.batch_size,
            convert_to_numpy=encoding_parameters.convert_to_numpy,
            normalize_embeddings=encoding_parameters.normalize_embeddings,
            num_proc=encoding_parameters.num_proc,
        )
    }


def embed(
    output_dataset_path: str,
    dataset_path: str,
    model_name: str = "IEITYuan/Yuan-embedding-2.0-en",
    batch_size: int = 8,
    model_dtype: str = "torch.bfloat16",
    atten_implem: str = "sdpa",
    num_proc: int = 4,
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = True,
    batched: bool = True,
    writer_batch_size=128,
) -> None:
    """
    Function responsible for embedding the chunks. It only embeds the chunks based on its GPU number ahd the total
    number of GPUs in the distributed training.
    :param output_dataset_path: path where to save the dataset.
    :param dataset_path: path to the dataset containing the chunks, it must be an arrow file.
    :param model_name: name of the model that will loaded by Hugging Face.
    :param local_rank: GPU number.
    :param batch_size: size of the batch used to read each row group in the dataset.
    :param model_dtype: dtype of the embedding model, in ["torch.bfloat16", "torch.float32", "torch.float16"].
    :param atten_implem: implementation of the attention mechanism, must be in ["sdpa", "flash_attention_2"].
    :param batch_size_encoder: batch size to use in the encoding.
    :param num_proc: number of proc to use.
    :param convert_to_numpy: whether we should convert to numpy.
    :param normalize_embeddings: whether we should normalize the embeddings.
    :param batched: whether to use the batched version of the .map method.
    :param writer_batch_size: how samples to accumulate before writing on disk.
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

    assert output_dataset_path.endswith(".parquet"), f"""
                                        The output file be in parquet format, currently {output_dataset_path}"""

    encoding_parameters = encodingParameters(
        num_proc=num_proc,
        batch_size=batch_size,
        convert_to_numpy=convert_to_numpy,
        normalize_embeddings=normalize_embeddings,
        batched=batched,
    )
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = get_device(local_rank)
    print(f"local_rank={local_rank}, world_size={world_size}")
    model = load_model(model_name, device, model_dtype, atten_implem)
    chunk_dataset = load_chunks_datasets(dataset_path)
    chunk_dataset = split_dataset_by_node(
        chunk_dataset, rank=local_rank, world_size=world_size
    )
    chunk_dataset = chunk_dataset.map(
        lambda chunk, encoding_params, model_embed: encode_chunk(
            chunk["paragraphs"], model_embed, encoding_parameters
        ),
        fn_kwargs={"encoding_params": encoding_parameters, "model": model},
        batch_size=encoding_parameters.batch_size,
        batched=encoding_parameters.batched,
        writer_batch_size=writer_batch_size,
    )

    chunk_dataset.to_parquet(output_dataset_path)


if __name__ == "__main__":
    typer.run(embed)
