"""
This script takes the dataset of paragraphs obtained by the script books_to_paragraphs.py and cut the paragraphs into
chunks. Too small paragraphs are
The chunking strategy is the following:
- Too sma
"""
from tqdm import tqdm
from datasets import Dataset
from dataclasses import dataclass

@dataclass
class State:
    length = 0
    text = ""
    text_id = None
    paragraph_id = None
    chapter = None
    span_start = None
    span_stop = None

def update_data(current_text: str, current_length: int, text_id: int, chapter: str, span_start: int, span_stop: int, data: dict )-> None:
    """
    Updates the dictionnary data with the values provided in the arguments.
    This function does not return anything but has a side effect: modifies the data dictionnary.
    """
    data["paragraphs"].append(current_text)
    data["n_words"].append(current_length)
    data["text_ids"].append(text_id)
    data["chapters"].append(chapter)
    data["spans"].append((span_start, span_stop))


def update_data_tidy(state: State, data: dict )-> None:
    """
    Updates the dictionnary data with the values provided in the arguments.
    This function does not return anything but has a side effect: modifies the data dictionnary.
    """
    data["paragraphs"].append(state.text)
    data["n_words"].append(state.length)
    data["text_ids"].append(state.text_id)
    data["chapters"].append(state.chapter)
    data["spans"].append((state.span_start, state.span_stop))

def reset_tidy(state: State) -> None:
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
    update_data_tidy(state, data)
    reset_tidy(state)

def reset():
    """
    Reset values to None, None, "", 0 for span_start, span_stop, current_text, current_length
    """
    return None, None, "", 0


def set_state(paragraph: dict, state: State) -> None:
    """
    Set the new state with the values of the current paragraph.
    This function does not return anything but has a side effect of changing the state of the State object.
    """
    state.text_id = paragraph["text_ids"]
    state.paragraph_id = paragraph["paragraph_index"]
    state.chapter = paragraph["chapters"]
    state.text = paragraph["paragraphs"]
    state.length = paragraph["n_words"]
    state.span_start = paragraph["spans"][0]
    state.span_stop = paragraph["spans"][1]


def is_new_aggregate(state: State, state_new: State) -> bool:
    """
    Computes whether we need to start a new aggregate
    :param state: current state (i.e paragraph_id, book_id, chapter)
    :param state_new: state of the paragraph (same attribute as state)
    :return: whether to start a new aggregate
    """
    return ((state.text_id != state_new.text_id and state.text_id is not None) or
            ((state_new.paragraph_id - 1) != state.paragraph_id and state.paragraph_id is not None) or
            (state.chapter != state_new.chapter))


def start_new_aggregate(current_state: State, new_state: State, data: dict, threshold_min: int, paragraph) -> None:
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
        update_data_tidy(new_state, data)
        reset_tidy(current_state)
    else:
        if current_state.text == "":
            current_state.text = paragraph["paragraphs"]
        else:
            current_state.text = "\n\n".join([current_state.text, paragraph["paragraphs"]])

        current_state.length += new_state.length
        current_state.text_id = new_state.text_id
        current_state.paragraph_id = new_state.paragraph_id
        current_state.chapter = new_state.chapter
        current_state.span_start = new_state.span_start
        current_state.span_stop = new_state.span_stop


def continue_aggregate(current_state: State, new_state: State, data: dict, threshold_min: int, paragraph: dict)-> None:
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





def aggregate_paragraphs_tidy(hf_dataset: Dataset, threshold_min: int) -> Dataset:
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
    data = {col_num: [] for col_num in hf_dataset.features.keys() if col_num != "paragraph_index"}
    for paragraph in tqdm(hf_dataset):
        new_state = State()
        set_state(paragraph, new_state)
        if current_state.span_start is None:
            current_state.span_start = paragraph["spans"][0]
            current_state.span_stop = paragraph["spans"][1]

        if is_new_aggregate(current_state, new_state):
            start_new_aggregate(current_state, new_state, data, threshold_min, paragraph)
        else:
            continue_aggregate(current_state, new_state, data, threshold_min, paragraph)

    if current_state.text != "":
        flush_aggregate(current_state, data)

    return data






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
    data = {col_num :[] for col_num in hf_dataset.features.keys() if col_num != "paragraph_index"}
    current_length = 0
    current_text = ""
    text_id = None
    paragraph_id = None
    chapter = None
    current_span_start = None
    current_span_stop = None
    for paragraph in tqdm(hf_dataset):
        text_id_new = paragraph["text_ids"]
        paragraph_id_new = paragraph["paragraph_index"]
        chapter_new = paragraph["chapters"]
        text_new = paragraph["paragraphs"]
        length_new = paragraph["n_words"]
        if current_span_start is None:
            current_span_start = paragraph["spans"][0]
            current_span_stop = paragraph["spans"][1]

        new_span_start = paragraph["spans"][0]
        new_span_stop = paragraph["spans"][1]
        if ((text_id != text_id_new and text_id is not None) or
                ((paragraph_id_new - 1) != paragraph_id and paragraph_id is not None) or
                (chapter != chapter_new)):
            if current_text != "":
                update_data(current_text, current_length, text_id, chapter, current_span_start, current_span_stop, data)
                current_span_start, current_span_stop, current_text, current_length = reset()
            if length_new >= threshold_min:
                update_data(text_new, length_new, text_id_new, chapter_new, new_span_start, new_span_stop, data)
                current_span_start, current_span_stop, current_text, current_length = reset()
            else:
                if current_text == "":
                    current_text = paragraph["paragraphs"]
                else:
                    current_text = "\n\n".join([current_text, paragraph["paragraphs"]])

                current_length += length_new
                text_id = text_id_new
                paragraph_id = paragraph_id_new
                chapter = chapter_new
                current_span_start = new_span_start
                current_span_stop = new_span_stop


        else:
            if current_text == "":
                current_text = paragraph["paragraphs"]
            else:
                current_text = "\n\n".join([current_text, paragraph["paragraphs"]])

            current_length += length_new
            text_id = text_id_new
            paragraph_id = paragraph_id_new
            chapter = chapter_new
            current_span_stop = new_span_stop
            if current_length >= threshold_min:
                update_data(current_text, current_length, text_id, chapter, current_span_start, current_span_stop, data)
                current_span_start, current_span_stop, current_text, current_length = reset()

    if current_text != "":
        update_data(current_text, current_length, text_id, chapter, current_span_start, current_span_stop, data)


    return data



data = {"paragraphs":["hhb"], "n_words":[1, 2, 3], "text_ids":[], "chapters":[], "spans":[(0, 10)]}
data_expected = {"paragraphs":["hhb", "aaa"], "n_words":[1, 2, 3, 10], "text_ids":[1],
                 "chapters":["A"], "spans":[(0, 10), (35, 50)]}
update_data("aaa", 10, 1, "A", 35,50, data)
assert data_expected == data, f"""The updating data does not output the correct dictionnary. 
                                        Expected: {data_expected} \n
                                        Recovered: {data}"""


assert reset() == (None, None, "", 0), f"""Reset function should return {(None, None, "", 0)} but returns
                                            {reset()}."""

dict_data = {"paragraphs":["This is a test paragraphs.",
                           "and another one",
                           "yet another one",
                           "another small one",
                           "a very very very very very very very very very very long paragraph.",
                           "another small paragraph.",
                           "but with a skipped paragraph id.",
                           "another example",
                           "with different chapter"
                           ],
             "chapters":["I", "I", "II", "II", "II", "III", "III", "I", "II"],
             "n_words":[5, 3, 3, 3, 13, 3, 6, 2, 3],
             "text_ids":[0, 0, 1, 1, 1, 2, 2, 3, 3],
             "paragraph_index":[0, 1, 3,4,5, 14, 16, 1, 2],
             "spans":[(0, 10), (14, 35), (10, 14), (14, 28), (35, 67), (18, 24), (36, 45), (104, 110), (205, 309)]}

expected_data = {"paragraphs":["This is a test paragraphs.\n\nand another one""",
                           "yet another one\n\nanother small one",
                           "a very very very very very very very very very very long paragraph.",
                           "another small paragraph.",
                            "but with a skipped paragraph id.",
                            "another example",
                            "with different chapter"
                           ],
             "chapters":["I", "II", "II", "III", "III", "I", "II"],
             "n_words":[8, 6, 13, 3, 6, 2, 3],
             "text_ids":[0, 1, 1, 2, 2, 3, 3],
             "spans":[(0, 35), (10, 28), (35, 67), (18, 24), (36, 45), (104, 110), (205, 309)]}

hf_dataset = Dataset.from_dict(dict_data)
min_threshold = 6
#aggregated_dataset = aggregate_paragraphs(hf_dataset, min_threshold)
aggregated_dataset = aggregate_paragraphs_tidy(hf_dataset, min_threshold)
for column in expected_data.keys():
    assert expected_data[column] == aggregated_dataset[column], f"""Colums {column} wrong:
                                                                    expected: {expected_data[column]}
                                                                    Recovered: {aggregated_dataset[column]}"""
assert aggregated_dataset == expected_data, "Fail"

