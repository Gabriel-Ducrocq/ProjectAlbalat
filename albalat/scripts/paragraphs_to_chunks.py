"""
This script takes the dataset of paragraphs obtained by the script books_to_paragraphs.py and cut the paragraphs into
chunks. Too small paragraphs are
The chunking strategy is the following:
- Too sma
"""
import typer
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from dataclasses import dataclass
from datasets import load_dataset
from nltk.tokenize import sent_tokenize


@dataclass
class State:
    """
    State of the paragraph/aggregate.
    length: length in words of the aggregate
    text: content
    paragraph_id: id of the paragraph obtained when breaking a book into paragraphs
    chapter: chapter in which the aggregate is situated
    span_start: character number within the book of the starting character of the aggregate.
    span_stop: character number within the book of the stopping character of the aggregate.
    """

    length: int = 0
    text: str = ""
    text_id: int | None = None
    paragraph_id: int | None = None
    chapter: str | None = None
    span_start: int | None | tuple[int] = None
    span_stop: int | None | tuple[int] = None
    index: int | None = None


def update_data(state: State, data: dict) -> None:
    """
    Updates the dictionnary data with the values provided in the arguments.
    This function does not return anything but has a side effect: modifies the data dictionnary.
    """
    data["paragraphs"].append(state.text)
    data["n_words"].append(state.length)
    data["text_ids"].append(state.text_id)
    data["chapters"].append(state.chapter)
    data["spans"].append([state.span_start, state.span_stop])


def reset(state: State) -> None:
    """
    This function reset the internal state of the State object
    :param state: current aggregate state
    :return: None
    """
    state.length = 0
    state.text = ""
    state.span_start = None
    state.span_stop = None


def flush_aggregate(state: State, data: dict) -> None:
    """
    Register the state of the aggregate in the data dictonnary and reset the State object.
    This function does not return anything but ha the side effects of modifying both the internal state of the State
    object and the data dictionnary.
    :param state: state of the aggregate
    :param data: data dictionnary containing the content of the previous aggregates
    :return: None
    """
    update_data(state, data)
    reset(state)


def set_state(paragraph: dict, state: State) -> None:
    """
    Set the new state with the values of the current paragraph.
    This function does not return anything but has a side effect of changing the state of the State object.
    """
    state.text_id = paragraph["text_ids"]
    state.paragraph_id = paragraph.get("paragraphs_index", None)
    state.chapter = paragraph["chapters"]
    state.text = paragraph["paragraphs"]
    state.length = paragraph["n_words"]
    if any(isinstance(x, list) for x in paragraph["spans"]):
        state.span_start = list(zip(*paragraph["spans"]))[0]
        state.span_stop = list(zip(*paragraph["spans"]))[1]
    else:
        state.span_start = paragraph["spans"][0]
        state.span_stop = paragraph["spans"][1]


def is_new_aggregate(state: State, state_new: State) -> bool:
    """
    Computes whether we need to start a new aggregate
    :param state: current state (i.e paragraph_id, book_id, chapter)
    :param state_new: state of the paragraph (same attribute as state)
    :return: whether to start a new aggregate
    """
    return (
        (state.text_id != state_new.text_id and state.text_id is not None)
        or (
            (state_new.paragraph_id - 1) != state.paragraph_id
            and state.paragraph_id is not None
        )
        or (state.chapter != state_new.chapter)
    )


def start_new_aggregate(
    current_state: State, new_state: State, data: dict, threshold_min: int, paragraph
) -> None:
    """
    Deals with the current and next aggregate in case we have to change aggregate because of new chapter, new book id,
    or paragraph_id has skipped one.
    This function has the side effect of changing the state of the current_state and data
    :param current_state: state of the current aggregate.
    :param new_state: state of the current paragraph.
    :param data: dictionnary of all the past aggregates.
    :param threshold_min: minimum number of words in the aggregate.
    :param paragraph: dictionnary of the content and metadata of the current paragraph.
    :return: None.
    """
    if current_state.text != "":
        flush_aggregate(current_state, data)
    if new_state.length >= threshold_min:
        update_data(new_state, data)
        reset(current_state)
    else:
        if current_state.text == "":
            current_state.text = paragraph["paragraphs"]
        else:
            current_state.text = "\n\n".join(
                [current_state.text, paragraph["paragraphs"]]
            )

        current_state.length += new_state.length
        current_state.text_id = new_state.text_id
        current_state.paragraph_id = new_state.paragraph_id
        current_state.chapter = new_state.chapter
        current_state.span_start = new_state.span_start
        current_state.span_stop = new_state.span_stop


def continue_aggregate(
    current_state: State,
    new_state: State,
    data: dict,
    threshold_min: int,
    paragraph: dict,
) -> None:
    """

    :param current_state:
    :param new_state:
    :param data:
    :param threshold_min:
    :param paragraph:
    :return:
    """
    if current_state.text == "":
        current_state.text = paragraph["paragraphs"]
    else:
        current_state.text = "\n\n".join([current_state.text, paragraph["paragraphs"]])

    current_state.length += new_state.length
    current_state.text_id = new_state.text_id
    current_state.paragraph_id = new_state.paragraph_id
    current_state.chapter = new_state.chapter
    current_state.span_stop = new_state.span_stop
    if current_state.length >= threshold_min:
        flush_aggregate(current_state, data)


def aggregate_paragraphs(hf_dataset: Dataset, threshold_min: int) -> Dataset:
    """
    Aggregate the paragraphs together so that they have at least threshold_min words, according to the following rule:
    - Paragraphs are aggregated in order of appearance in the text.
    - Two paragraphs not belonging to the same book cannot be aggregated together.
    - Two paragraphs not belonging to the same chapter cannot be aggregated together.
    - We aggregate paragraphs until this aggregate has more than threshold_min_words. As a result, if the aggregate has
        less than threshold_min_words but the next paragraphs has more than threshold_min_words, thi paragraph is still
        aggregated.
    - A paragraph that has more than threshold_min_words does not trigger aggregation of the subsequent paragraph.
        Therefore, it stays untouched, unless it is aggregated to the previous one.
    :param hf_dataset, the dataset to aggregate. Should have at least columns:
            ["text_ids", "index", "paragraphs", "n_words", "spans"]
    :param threshold_min: integer, minimum number of words in the paragraph.
    """
    current_state = State()
    data = {
        col_num: []
        for col_num in hf_dataset.features.keys()
        if col_num != "paragraphs_index"
    }
    for paragraph in tqdm(hf_dataset):
        new_state = State()
        set_state(paragraph, new_state)
        if current_state.span_start is None:
            current_state.span_start = paragraph["spans"][0]
            current_state.span_stop = paragraph["spans"][1]

        if is_new_aggregate(current_state, new_state):
            start_new_aggregate(
                current_state, new_state, data, threshold_min, paragraph
            )
        else:
            continue_aggregate(current_state, new_state, data, threshold_min, paragraph)

    if current_state.text != "":
        flush_aggregate(current_state, data)

    return Dataset.from_dict(data)


def split_paragraph(paragraph: str, max_p_length: int) -> list[str]:
    """
    Splits a paragraph in as many needed to be smaller than the maximum length.

    :param paragraphs: content of the paragraph.
    :param max_p_length: maximum size of the paragraph.
    return at least one paragraph (if unchanged) or more (if needed to break)
    """
    all_subparagraphs = []
    sentences = sent_tokenize(paragraph)
    agg_sentences = []
    length_sentence = 0
    for sentence_number, sent in enumerate(sentences):
        words = sent.split()
        length_sent = len(words)
        length_sentence += length_sent
        if length_sentence > max_p_length and sentence_number > 0:
            all_subparagraphs.append(" ".join(agg_sentences))
            agg_sentences = [sent]
            length_sentence = length_sent
            continue

        agg_sentences.append(sent)

    all_subparagraphs.append(" ".join(agg_sentences))

    return all_subparagraphs


def update_splitted_paragraph_data(
    state: State,
    splitted_paragraphs_data: dict,
    p: str | list[str],
    length_p: int | list[int],
    num_p: int,
) -> None:
    """
    Takes the current state of the paragraph and add it to the splitted_paragraph_data.
    There are two logics: if the paragraph content did not exceed the maximum threshold, we just add a row made up of
                          this paragraphs. If it did exceed it, we break it into paragraphs of roughly equal length and
                          add each one of them as a row.

    This function does not return anything but has the side effect of changing the state of splitted_paragraph_data.
    :param state: state of the paragraph.
    :param splitted_paragraphs_data: dictionnary containing all the paragraphs (broken or full)
    :return: None
    """
    if not isinstance(p, list):
        splitted_paragraphs_data["paragraphs"].append(p)
        splitted_paragraphs_data["n_words"].append(length_p)
        splitted_paragraphs_data["text_ids"].append(state.text_id[num_p])
        splitted_paragraphs_data["spans"].append(
            [state.span_start[num_p], state.span_stop[num_p]]
        )
        splitted_paragraphs_data["chapters"].append(state.chapter[num_p])
        splitted_paragraphs_data["splitted_paragraphs"].append(False)
    else:
        splitted_paragraphs_data["paragraphs"] += p
        splitted_paragraphs_data["n_words"] += length_p
        splitted_paragraphs_data["text_ids"] += [state.text_id[num_p]] * len(length_p)
        splitted_paragraphs_data["spans"] += [
            [state.span_start[num_p], state.span_stop[num_p]]
        ] * len(length_p)
        splitted_paragraphs_data["chapters"] += [state.chapter[num_p]] * len(length_p)
        splitted_paragraphs_data["splitted_paragraphs"] += [True] * len(length_p)


def compute_max_length(length_p: int, max_p_length: int) -> int:
    """
    Computes the maximum number of words subparagraphs can have.
    :param length_p: length of the current paragraphs.
    :param max_p_length: maximum length a paragraph can have
    :return: the rough size of each subparagraphs.
    """
    number_of_cuts = length_p // max_p_length
    rest_of_cuts = length_p % max_p_length
    if rest_of_cuts > 0:
        number_of_cuts += 1

    approx_parag_length = int(np.floor(max_p_length / number_of_cuts))
    return approx_parag_length


def break_paragraphs(paragraphs: dict, max_p_length: int) -> dict:
    """
    This function breaks paragraphs that are too big.
    :param paragraphs: content of the paragraph.
    :param max_p_length: maximum size of the paragraph.
    return at least one paragraph (if unchanged) or more (if needed to break)
    """
    current_state = State()
    set_state(paragraphs, current_state)
    splitted_paragraphs_data = {
        "paragraphs": [],
        "n_words": [],
        "text_ids": [],
        "spans": [],
        "chapters": [],
        "splitted_paragraphs": [],
    }
    for num_p, parag in enumerate(paragraphs["paragraphs"]):
        all_words = parag.split()
        length_p = len(all_words)
        if length_p < max_p_length:
            update_splitted_paragraph_data(
                current_state, splitted_paragraphs_data, parag, length_p, num_p
            )
        else:
            approx_parag_length = compute_max_length(length_p, max_p_length)
            parag = split_paragraph(parag, approx_parag_length)
            length_p = [len(p.split()) for p in parag]
            update_splitted_paragraph_data(
                current_state, splitted_paragraphs_data, parag, length_p, num_p
            )

    return splitted_paragraphs_data



def paragraphs_to_chunks(hf_dataset: str, hf_output: str, max_p_length: int, threshold_min: int, batched: bool = True,
                   batch_size: int = 256, num_proc: int = 16)->None:
    """
    First aggregates the paragraphs when possible/necessary. Second, breaks the paragraphs that are too long.
    This function does not return anything.
    :param hf_dataset: path to the Hugging Face dataset of paragraphs.
    :param hf_output: path to the output Hugging Face dataset.
    :param max_p_length: maximum length a chunk can have.
    :param threshold_min: minimum desired length of a chunk.
    :return: None.
    """
    dataset_paragraphs = load_dataset("parquet", data_files=hf_dataset, split="train")
    dataset_paragraphs = Dataset.from_dict(dataset_paragraphs[:100000])
    dataset_paragraphs = dataset_paragraphs.map(lambda batch, idx: {"paragraphs_index": idx,
                                                                    "n_words": [len(p.split()) for p in batch["paragraphs"]]},
                                                with_indices=True, batched=True, num_proc=16, batch_size=256)
    print(dataset_paragraphs)
    aggregated_paragraphs = aggregate_paragraphs(dataset_paragraphs, threshold_min)
    chunks = aggregated_paragraphs.map(break_paragraphs, batched=batched, batch_size = batch_size, num_proc= num_proc,
                                       fn_kwargs={"max_p_length": max_p_length})
    chunks.to_parquet(hf_output)

if __name__ == "__main__":
    typer.run(paragraphs_to_chunks)

