"""
This script processes the "common-pile/project_gutenberg" Hugging Face dataset.
Several steps are involved:
1/ Download the Gutenberg project catalog.
2/ Keep only the books in English
3/ Keep only the books belonging to specific bookshelves (genres)
4/ Use this list to keep the corresponding texts in the "common-pile/project_gutenberg" dataset
5/ Chunk each book into paragraphs:
    1. Normalize text
    2. Break text into paragraphs.
    3. Remove dialogues, footnotes, transcriber's notes.
    4. Keeps track of the chapterization

The output of this script is Hugging Face dataset in Parquet format where each row is a
paragraph and colums are:
- text_id: identifier of the text
- paragraph: actual text
- span: starting and ending character numbers of the paragraph in the original text
- chapter: chapter number in which the paragraph occurs if relevent.
"""

import re
import typer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).parent.parent
PATH_TO_METADATA = BASE_DIR / "data" / "interim" / "metadata_ids.csv"

GENRES_KEPT = {
    "novels",
    "britishliterature",
    "adventure",
    "americanliterature",
    "children",
    "youngadultreading",
    "sciencefiction",
    "biographies",
    "historyamerican",
    "historymodern(1750+)",
    "fantasy",
    "shortstories",
    "historicalnovels",
    "historybritish",
    "travelwriting",
    "historywarfare",
    "historyother",
    "folklore",
    "mythology",
    "legends",
    "historyeuropean",
    "romance",
    "crime",
    "thrillersandmystery",
    "historyearlymodern(c.14501750)",
    "frenchliterature",
    "classicsofliterature",
}

BOOKSHELVES_REGEX = re.compile(r"[,;&]")
PATTERN_CHAPTER = re.compile(
    r"^(?i:CHAPTER) [IVXLCDM]+|^[IVXLCDM]+$|^[A-Z ]{2,}\.{0,1}$|^No. [0-9]+--[A-Z ]+|^Part [IVXLCDM]+.{0,1}$|^Chapter [A-Z][a-zA-Z]{2,13}|^\s*(\*\s*){3,}\*\s*$"
)

PATTERN_FOOTNOTE = re.compile(r"\[footnote.+")
PATTERN_TRANSCRIBER = re.compile(r"\[transcriber.+")
PATTERN_ILLUSTRATION = re.compile(r"\[illustration")
PATTERN_3_MORE_SPACE = re.compile(r"\s+")


def get_bookshelves(bookshelf):
    """
    Gets all the bookshelves for a book
    :param bookshelf: string of potentially multiple bookshelves.
    :return: list of bookshelves.
    """
    if bookshelf is None:
        return None

    splitted_result = re.split(
        BOOKSHELVES_REGEX,
        bookshelf.replace("Category: ", "").lower().replace("-", "").replace(" ", ""),
    )
    return splitted_result if isinstance(splitted_result, list) else [splitted_result]


def keep_on_genre_exclusive(bookshelves):
    """
    Whether the books that have all their bookshelves included in GENRES_KEPT.
    :param bookshelves: list of bookshelves
    :return: boolean, True if all bookshelves are to keep.
    """
    return set(bookshelves).issubset(GENRES_KEPT)


def extract_metadata_hf(hf_dataset):
    """
    Extracts the metadata and the text ids from the Hugging Face dataset
    :param hf_dataset: Hugging Face dataset woth columns ['license', 'language', 'url', 'title'].
    :return: a pandas dataframe of the mentionned columns + 1 column ['id']
    """
    tuples_metadata_id = []
    print("Extracting the metadata from the common_pile dataset:")
    for t in tqdm(hf_dataset):
        row = t["metadata"]
        row.update({"text_id": t["id"]})
        tuples_metadata_id.append(row)

    metadata_id = pd.DataFrame.from_dict(tuples_metadata_id)
    return metadata_id


def create_metadata_table(merged_catalog_meta_data, language_list=["en"]):
    """
    Takes the catalog of the Gutenberg project and the common_pile/gutenberg-project dataset
    and creates a metadata table, containing data on which we will filter the dataset:
    - Language
    - Bookshelves (genre)
    - Is there any missing value


    We also keep the id of the work so we know what to filter in and out in the common_pile dataset
    :param catalog: pandas dataframe
    :param metadata_common_pile: pandas dataframe with the metadata of the common pile dataset
    :return: merged tables, filtered.
    """
    merged_catalog_meta_data = merged_catalog_meta_data.loc[
        merged_catalog_meta_data.Language.isin(language_list)
    ]
    merged_catalog_meta_data = merged_catalog_meta_data[
        ~merged_catalog_meta_data.Bookshelves.isna()
    ]
    mask_genres = merged_catalog_meta_data.Bookshelves_bag_of_words.apply(
        keep_on_genre_exclusive
    )
    kept_books = merged_catalog_meta_data[mask_genres]
    return kept_books


def compute_inclusion_condition(words, p):
    """
    Computes whether to keep the paragraph or not. Discards if:
    - Less or equal than 3 words
    - Is entire upper characters
    - Is a footnote
    - Is a transcriber's note
    - Is an illustration

    Note that paragraph that are just chapter titles are discarded and will not be embedded, even though we keep track
    of them.
    arguments:
    :param words: list words in the paragraph
    :param p: full paragraph.
    :return: boolean, False if discarded, True otherwise.
    """
    p = p.strip()
    more_three_words = len(words) > 3
    not_dialog = len(words) > 10 or (
        """\"""" not in p and """“""" not in p and """\'""" not in p
    )
    not_all_upper = not p.isupper()
    p = p.lower()
    is_not_footnote = re.search(PATTERN_FOOTNOTE, p) is None
    is_not_transcribers = re.search(PATTERN_TRANSCRIBER, p) is None
    is_not_illustration = re.search(PATTERN_ILLUSTRATION, p) is None
    return (
        more_three_words
        and not_dialog
        and not_all_upper
        and is_not_footnote
        and is_not_illustration
        and is_not_transcribers
    )


def determine_chapterization(text):
    """
    Determines whether this book uses a chapterization scheme or not at all.
    :param text: content of the paragraph
    :return: boolean, True of there is chapterization, False otherwise.
    """
    no_chapterization = False
    if re.search(PATTERN_CHAPTER, text) is None:
        no_chapterization = True

    return no_chapterization


def normalize_book(book: str) -> str:
    """
    Normalizes a text by replacing special characters introduced by Project Gutenberg, replacing the paragraphs spacing
    (i.e \n\n) by a specific character: ||||, replacing the line skips introduced by Project Gutenberg with a mere
    spacing, and removing the escaped apostrophe.
    :param book: content of the book
    :return: normalized content of the book
    """
    paragraphs = (
        book.replace("\r\n", "\n")
        .replace("\xa0", "")
        .replace("\n\n", "||||")
        .replace("\n", " ")
        .replace("'", "'")
    )
    paragraphs = re.sub(PATTERN_3_MORE_SPACE, " ", paragraphs)
    return paragraphs


def pipeline(book, text_id):
    """
    This function chunks a book into paragraphs, keeps track of the chapterization, span in the origina text
    and the text id.
    parameters:
    book -- str, content of the book, unprocessed.
    text_id -- integer, id of the text.
    return:
    list of dict, paragraphs in the book.
    """
    paragraphs = normalize_book(book)
    no_chapterization = determine_chapterization(paragraphs)
    all_samples = {"paragraphs": [], "text_ids": [], "spans": [], "chapters": []}
    paragraphs_chunks = paragraphs.split("||||")
    chapter = None
    for paragraph_num, p in enumerate(paragraphs_chunks):
        p = p.strip()
        match = re.search(PATTERN_CHAPTER, p)
        if match is not None:
            chapter = match.group()
            continue

        if chapter is not None or no_chapterization:
            words = p.split()
            condition_inclusion = compute_inclusion_condition(words, p)
            if condition_inclusion:
                result_search = re.search(re.escape(p), paragraphs)
                assert result_search is not None, (
                    f"Cannot find paragraph number {paragraph_num} in original text id {text_id}: {p}"
                )
                all_samples["paragraphs"].append(p)
                all_samples["text_ids"].append(text_id)
                all_samples["spans"].append(result_search.span())
                all_samples["chapters"].append(chapter)

    assert len(set([len(v) for k, v in all_samples.items()])) == 1, (
        "All attributes must have the same length."
    )
    return all_samples


def map_pipeline(sample_batch):
    """
    Wrapper function apply pipeline to the rows of the Hugging Face dataset.
    """
    batch = {"paragraphs": [], "text_ids": [], "spans": [], "chapters": []}
    for num_sample in range(len(sample_batch["id"])):
        all_samples = pipeline(
            sample_batch["text"][num_sample], int(sample_batch["id"][num_sample])
        )
        batch["paragraphs"] += all_samples["paragraphs"]
        batch["text_ids"] += all_samples["text_ids"]
        batch["spans"] += all_samples["spans"]
        batch["chapters"] += all_samples["chapters"]

    return batch


def merge_and_select_columns(catalog, metadata_hf):
    """
    Merges the two dataframes and select only the colums ["Language","text_id", "Bookshelves"]
    :param catalog: dataframe made of the catalog of the Gutenberg project
    :param metadata_hf: dataframe made of the metadata present in the common_pile/gutenberg-project dataset.
    :return: a merged dataframe
    """
    merged_catalog_meta_data = pd.merge(
        catalog, metadata_hf, how="inner", left_on="Text#", right_on="text_id"
    )
    merged_catalog_meta_data = merged_catalog_meta_data[
        ["Language", "text_id", "Bookshelves"]
    ]
    return merged_catalog_meta_data


def save_dataframe(df, path=PATH_TO_METADATA):
    """
    Saves a dataframe at ../data/interim/metadata_ids.csv
    :param df: dataframe of metadata from common_pile/gutenberg-project
    :param path: path to which we save the dataframe
    :return: None
    """
    df.to_csv(path)


def filter_hf_dataset(hf_dataset, values):
    """
    Filters the HF dataset to keep only the given book ids.
    :param hf_dataset: Hugging Face dataset of common_pile/gutenberg-project
    :param values: values of books id to keep
    :return: a hf_dataset filtered to keep only the correct indexes.
    """
    return hf_dataset.filter(lambda x: eval(x["id"]) in values)


def map_hf_dataset(hf_dataset_filtered, batched=True, num_proc=8, batch_size=128):
    """
    Applies the map_pipeline function to the filtered HF dataset to split the books into paragraphs. Each paragraphs
    have 5 fields:
    - paragraphs: the content of the paragraph.
    - text_ids: the book id.
    - spans: a tuple (start character number, end character number) know where to find this paragraph in the book.
    - chapters: the chapter number it belongs to, as a string (e.g "CHAPTER IV").

    :param hf_dataset_filtered: common_pile/gutenberg-project filtered dataset.
    :param batched: whether to use the batch version of map.
    :param num_proc: number of processors to use.
    :param batch_size: the number of books in each batch.
    :return: Hugging face dataset with
    """
    return hf_dataset_filtered.map(
        map_pipeline,
        batched=batched,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=["id", "text", "source", "added", "metadata"],
    )


def books_to_paragraphs(output_file,
    languages=["en"], batched=True, num_proc=8, batch_size=128
):
    """
    Defines the entire pipeline to get from raw gutenberg-project books to paragraphs.
    :param output_file: string describing the location of the output file
    :param languages: languages to keep in the dataset.
    :param batched: whether to batch the application of the pipeline
    :param num_proc: number of processors.
    :param batch_size: size of the batch for the pipeline.
    :return: None.
    """
    assert output_file.endswith(".parquet"), "Output file must be in parquet format."
    common_pile_dataset = load_dataset(
        "common-pile/project_gutenberg", split="train", streaming=False
    )
    catalog = pd.read_csv(
        "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz",
        compression="gzip",
    )
    metadata_hf = extract_metadata_hf(common_pile_dataset)
    save_dataframe(metadata_hf)
    merged_catalog_meta_data = merge_and_select_columns(catalog, metadata_hf)
    merged_catalog_meta_data.Bookshelves_bag_of_words = (
        merged_catalog_meta_data.Bookshelves.apply(get_bookshelves)
    )
    kept_books = create_metadata_table(
        merged_catalog_meta_data, language_list=languages
    )
    common_pile_dataset_filtered = filter_hf_dataset(
        common_pile_dataset, kept_books.text_id.values
    )
    processed_pararaphs = map_hf_dataset(
        common_pile_dataset_filtered, batched, num_proc, batch_size
    )
    processed_pararaphs.to_parquet(output_file)


if __name__ == "__main__":
    typer.run(books_to_paragraphs)
