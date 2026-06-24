import os
import pandas as pd
from pathlib import Path
from datasets import Dataset
from albalat.scripts.books_to_paragraphs import (
    extract_metadata_hf,
    get_bookshelves,
    keep_on_genre_exclusive,
    compute_inclusion_condition,
    determine_chapterization,
    normalize_book,
    pipeline,
    map_pipeline,
    create_metadata_table,
    merge_and_select_columns,
    save_dataframe,
    filter_hf_dataset,
    map_hf_dataset,
)

BASE_DIR = Path(__file__).parent.parent


class TestBooksToParagraphs:
    def test_metadata_extractor(self):
        fake_data = {
            "id": [0, 1, 2, 3],
            "metadata": [
                {"language": "en", "url": "test"},
                {"language": "fr", "url": "test"},
            ]
            * 2,
            "extra": [0, 3, 5, 2],
        }
        fake_hf = Dataset.from_dict(fake_data)
        fake_pd = pd.DataFrame.from_dict(
            {"language": ["en", "fr"] * 2, "url": ["test"] * 4, "text_id": [0, 1, 2, 3]}
        )
        extracted_fake_pd = extract_metadata_hf(fake_hf)
        assert set(extracted_fake_pd.columns.values) == set(fake_pd.columns.values), (
            "Wrong column extraction."
        )
        assert extracted_fake_pd.equals(fake_pd), (
            "Extracted and expected dataframes differ."
        )

    def test_get_bookshelves(self):
        test_bookshelf = "Category: test1&TeSt2 -; test3, test4"
        bookshelves = get_bookshelves(test_bookshelf)
        assert bookshelves == [
            "test1",
            "test2",
            "test3",
            "test4",
        ], f"""Booksehlves not correctly extracted:
                                                                            {bookshelves} != ["test1", "test2", "test3", "test4"]"""

    def test_keep_on_genre(self):
        assert keep_on_genre_exclusive(
            ["americanliterature", "children", "youngadultreading"]
        ), (
            "Should keep the bookshelves:['americanliterature','children','youngadultreading']"
        )

        assert not keep_on_genre_exclusive(
            ["americanliterature", "test", "youngadultreading"]
        ), (
            "Should not keep the bookshelves:['americanliterature','test','youngadultreading']"
        )

    def test_compute_inclusion_condition(self):
        p = "[Footnote this is a footnote"
        words = p.split()
        assert not compute_inclusion_condition(words, p), (
            "Footnote should be discarded."
        )
        p = "[Transcriber: this is a transcriber note"
        words = p.split()
        assert not compute_inclusion_condition(words, p), (
            "Transcriber's note should be discarded."
        )
        p = "[IllUSTration: this is an illustration"
        words = p.split()
        assert not compute_inclusion_condition(words, p), (
            "Illustration should be discarded."
        )

        p = "'Hehehe' the crowd was laughing"
        words = p.split()
        assert not compute_inclusion_condition(words, p), "Dialog should be discarded."

        p = "Said Mary"
        words = p.split()
        assert not compute_inclusion_condition(words, p), (
            "Less than 3 words should be discarded."
        )

        p = "The bird is flying, the computer computing, and the music playing."
        words = p.split()
        assert compute_inclusion_condition(words, p), (
            f"This sentence '{p}' should be kept."
        )

    def test_determine_chapterization(self):
        text = [
            " CHAPTER VI. pjojm",
            " IX",
            "THE CAT.",
            "No. 10--THE CAT",
            " Part IX.",
            " Chapter Test",
            "  *   *   *   *   *",
        ]
        for t in text:
            assert determine_chapterization(t.strip()) is False, (
                f"This text {t} should trigger chapterization"
            )

    def test_normalize_book(self):
        assert (
            normalize_book("Test \r\n sentence \xa0 is \n\n different\n from ' another")
            == """Test sentence is |||| different from ' another"""
        )

    def test_pipeline(self):
        book = """
        Test book \n\n This is \n\n CHAPTER I \n\n this is a fake book \n that mimicks a gutenberg project \r\n by \xa0 adding
        many special \' characters \n\n and creating a fake paragraph. \n\n CHAPTER II \n\n This is a new \n\n chapter in two
        paragraphs
        """
        expected_paragraphs = {
            "paragraphs": [
                """this is a fake book that mimicks a gutenberg project by adding many special ' characters""",
                "and creating a fake paragraph.",
                "This is a new",
                "chapter in two paragraphs",
            ],
            "text_ids": [1, 1, 1, 1],
            "spans": [(44, 132), (138, 168), (190, 203), (209, 234)],
            "chapters": ["CHAPTER I", "CHAPTER I", "CHAPTER II", "CHAPTER II"],
        }
        paragraphs = pipeline(book, 1)
        assert expected_paragraphs["paragraphs"] == paragraphs["paragraphs"], (
            "Paragraphs differ"
        )
        assert expected_paragraphs["spans"] == paragraphs["spans"], "Spans differ"
        assert expected_paragraphs["text_ids"] == paragraphs["text_ids"], (
            "text_id differ"
        )
        assert expected_paragraphs["chapters"] == paragraphs["chapters"], (
            "chapter differ"
        )
        assert (
            expected_paragraphs == paragraphs
        ), f"""The text is not correctly cut into paragraphs:\n\n
        cut: {paragraphs}\n\n
        expected paragraphs: {expected_paragraphs}"""

    def test_map_pipeline(self):
        book = """
        Test book \n\n This is \n\n CHAPTER I \n\n this is a fake book \n that mimicks a gutenberg project \r\n by \xa0 adding
        many special \' characters \n\n and creating a fake paragraph. \n\n CHAPTER II \n\n This is a new \n\n chapter in two
        paragraphs
        """
        sample_batch = {"text": [book, book], "id": [1, 2]}

        expected_paragraphs = {
            "paragraphs": [
                """this is a fake book that mimicks a gutenberg project by adding many special ' characters""",
                "and creating a fake paragraph.",
                "This is a new",
                "chapter in two paragraphs",
            ]
            * 2,
            "text_ids": [1, 1, 1, 1, 2, 2, 2, 2],
            "spans": [(44, 132), (138, 168), (190, 203), (209, 234)] * 2,
            "chapters": ["CHAPTER I", "CHAPTER I", "CHAPTER II", "CHAPTER II"] * 2,
        }
        all_paragraphs = map_pipeline(sample_batch)
        assert expected_paragraphs["paragraphs"] == all_paragraphs["paragraphs"], (
            "Paragraphs differ"
        )
        assert expected_paragraphs["spans"] == all_paragraphs["spans"], "Spans differ"
        assert expected_paragraphs["text_ids"] == all_paragraphs["text_ids"], (
            "text_id differ"
        )
        assert expected_paragraphs["chapters"] == all_paragraphs["chapters"], (
            "chapter differ"
        )
        assert (
            expected_paragraphs == all_paragraphs
        ), f"""The text is not correctly cut into paragraphs:\n\n
        cut: {all_paragraphs}\n\n
        expected paragraphs: {expected_paragraphs}"""

    def test_merge_and_select(self):
        catalog = pd.DataFrame.from_dict(
            {
                "Text#": [1, 2, 3, 4],
                "Type": ["test"] * 4,
                "Issued": ["test"] * 4,
                "Title": ["title"] * 4,
                "Language": ["fr", "en"] * 2,
                "Authors": ["author"] * 4,
                "Subjects": ["subject"] * 4,
                "LoCC": ["test"] * 4,
                "Bookshelves": ["bookshelf"] * 4,
            }
        )
        fake_pd = pd.DataFrame.from_dict(
            {"language": ["fr", "en"] * 2, "url": ["test"] * 4, "text_id": [0, 1, 2, 3]}
        )
        expected_table = pd.DataFrame.from_dict(
            {
                "Language": ["fr", "en", "fr"],
                "text_id": [1, 2, 3],
                "Bookshelves": ["bookshelf"] * 3,
            }
        )
        merged_table = merge_and_select_columns(catalog, fake_pd)
        assert expected_table.equals(
            merged_table
        ), f"""Merging and selecting columns failed for catalog and metadata."
        expected: {expected_table}\n\n
        obtained: {merged_table}  
        """

    def test_create_metadata_table(self):
        merged_catalog_meta_data = pd.DataFrame.from_dict(
            {
                "Language": ["en", "fr", "it"] * 2,
                "Bookshelves": [
                    "novels & british literature",
                    "novels; folklore",
                    "biographies",
                    "science",
                    "poetry",
                    "economics",
                ],
            }
        )
        merged_catalog_meta_data["Bookshelves_bag_of_words"] = (
            merged_catalog_meta_data.Bookshelves.apply(get_bookshelves)
        )
        expected = merged_catalog_meta_data.loc[[0, 1]]
        kept_books = create_metadata_table(
            merged_catalog_meta_data, language_list=["en", "fr"]
        )
        assert kept_books.equals(expected), (
            f"The books kept table: {kept_books}\n\n does not match the expected table {expected}"
        )

    def test_save_dataframe(self):
        df = pd.DataFrame.from_dict({"test": [1] * 4})
        save_dataframe(df)
        has_raised = False
        print("__file__ =", __file__)
        print("BASE_DIR =", BASE_DIR)
        print("cwd =", os.getcwd())
        try:
            df_loaded = pd.read_csv(
                BASE_DIR / "data" / "interim" / "metadata_ids.csv", index_col=0
            )
        except FileNotFoundError:
            has_raised = True
        assert not has_raised, (
            "The file '../data/interim/metadata_ids.csv' does not exists"
        )
        assert df_loaded.equals(df), (
            "The saved and loaded files differ when they should be the same"
        )
        os.remove(BASE_DIR / "data" / "interim" / "metadata_ids.csv")

    def test_filter_hf(self):
        hf_dataset = Dataset.from_dict(
            {"test_col": ["t"] * 5, "id": ["1", "2", "3", "4", "5"]}
        )
        expected = Dataset.from_dict(
            {"test_col": ["t"] * 3, "id": ["1", "3", "5"]}
        ).to_pandas()
        vals_to_keep = [1, 3, 5]
        filtered = filter_hf_dataset(hf_dataset, vals_to_keep)
        filtered_pandas = filtered.to_pandas()
        assert filtered_pandas.equals(
            expected
        ), f"""Expected and filtered datasets do not match \n\n expected: {expected}
                                                  filtered: {filtered_pandas}"""

    def test_map_hf_dataset(self):
        ["id", "text", "source", "added", "metadata"]
        hf_dataset = Dataset.from_dict(
            {
                "id": [0, 0, 1, 1],
                "text": [
                    "This is \n\n CHAPTER I \n\n a text with many \n\n different \xa0 paragraphs \r\n for testing",
                    "Another \n\n I \n\n \n\n different text \xa0 paragraphs \r\n for testing \n\n small",
                ]
                * 2,
                "source": ["test_source"] * 4,
                "added": ["test_added"] * 4,
                "metadata": [
                    {"language": "en", "url": "test"},
                    {"language": "fr", "url": "test"},
                ]
                * 2,
            }
        )
        expected = Dataset.from_dict(
            {
                "paragraphs": [
                    "a text with many",
                    "different paragraphs for testing",
                    "different text paragraphs for testing",
                ]
                * 2,
                "text_ids": [0, 0, 0, 1, 1, 1],
                "spans": [(28, 44), (50, 82), (25, 62)] * 2,
                "chapters": ["CHAPTER I", "CHAPTER I", "I"] * 2,
            }
        ).to_pandas()
        mapped_dataset = map_hf_dataset(hf_dataset).to_pandas()
        assert mapped_dataset.equals(
            expected
        ), f"""Mapped and expected dataset do not match. Expected {expected}\n\n
                                                    recovered {mapped_dataset}"""
