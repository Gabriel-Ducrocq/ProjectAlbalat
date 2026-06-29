"""
This script takes the dataset of paragraphs obtained by the script books_to_paragraphs.py and cut the paragraphs into
chunks. Too small paragraphs are
The chunking strategy is the following:
- Too sma
"""
from tqdm import tqdm
from datasets import Dataset


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

def reset():
    """
    Reset values to None, None, "", 0 for span_start, span_stop, current_text, current_length
    """
    return None, None, "", 0


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
    span_start = None
    for paragraph in tqdm(hf_dataset):
        text_id_new = paragraph["text_ids"]
        paragraph_id_new = paragraph["paragraph_index"]
        chapter_new = paragraph["chapters"]
        text_new = paragraph["paragraphs"]
        length_new = paragraph["n_words"]
        if span_start is None:
            span_start = paragraph["spans"][0]

        span_stop = paragraph["spans"][1]
        if ((text_id != text_id_new and text_id is not None) or
                ((paragraph_id_new - 1) != paragraph_id and paragraph_id is not None) or
                (chapter != chapter_new)):
            if current_text != "":
                update_data(current_text, current_length, text_id, chapter, span_start, span_stop, data)
                span_start, span_stop, current_text, current_length = reset()
            if length_new >= threshold_min:
                update_data(text_new, length_new, text_id_new, chapter_new, span_start, span_stop, data)
                span_start, span_stop, current_text, current_length = reset()
            else:
                if current_text == "":
                    current_text = paragraph["paragraphs"]
                else:
                    current_text = "\n\n".join([current_text, paragraph["paragraphs"]])

                current_length += length_new
                text_id = text_id_new
                paragraph_id = paragraph_id_new
                chapter = chapter_new

        else:
            if current_text == "":
                current_text = paragraph["paragraphs"]
            else:
                current_text = "\n\n".join([current_text, paragraph["paragraphs"]])

            current_length += length_new
            text_id = text_id_new
            paragraph_id = paragraph_id_new
            chapter = chapter_new
            if current_length >= threshold_min:
                update_data(current_text, current_length, text_id, chapter, span_start, span_stop, data)
                span_start, span_stop, current_text, current_length = reset()

    if current_text != "":
        update_data(current_text, current_length, text_id, chapter, span_start, span_stop, data)


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

print("RECOVERED", expected_data)
#[(0, 35), (10, 28), (35, 67)]
hf_dataset = Dataset.from_dict(dict_data)
print("RECOVERED", expected_data)
min_threshold = 6
aggregated_dataset = aggregate_paragraphs(hf_dataset, min_threshold)
print(aggregated_dataset)
for column in expected_data.keys():
    assert expected_data[column] == aggregated_dataset[column], f"""Colums {column} wrong:
                                                                    expected: {expected_data[column]}
                                                                    Recovered: {aggregated_dataset[column]}"""
assert aggregated_dataset == expected_data, "Fail"

