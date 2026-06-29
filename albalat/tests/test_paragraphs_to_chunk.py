from pathlib import Path
from datasets import Dataset
from albalat.scripts.paragraphs_to_chunks import (
    State,
    update_data,
    reset,
    aggregate_paragraphs,
    split_paragraph,
    break_paragraphs,
)

BASE_DIR = Path(__file__).parent.parent


class TestParagraphAggregation:
    test_state = State()
    test_state.length = 10
    test_state.text = "aaa"
    test_state.paragraph_id = 10
    test_state.text_id = 1
    test_state.chapter = "A"
    test_state.span_start = 35
    test_state.span_stop = 50

    def test_update_data(self):
        data = {
            "paragraphs": ["hhb"],
            "n_words": [1, 2, 3],
            "text_ids": [],
            "chapters": [],
            "spans": [[0, 10]],
        }
        data_expected = {
            "paragraphs": ["hhb", "aaa"],
            "n_words": [1, 2, 3, 10],
            "text_ids": [1],
            "chapters": ["A"],
            "spans": [[0, 10], [35, 50]],
        }

        update_data(self.test_state, data)
        assert (
            data_expected == data
        ), f"""The updating data does not output the correct dictionnary. 
                                                Expected: {data_expected} \n
                                                Recovered: {data}"""

    def test_reset(self):
        reset(self.test_state)
        expected_state = State()
        expected_state.paragraph_id = 10
        expected_state.text_id = 1
        expected_state.chapter = "A"
        assert (
            self.test_state.__dict__ == expected_state.__dict__
        ), f"""Reset function does not reset properly.\n
                                                                         Reset state {self.test_state.__dict__} \n
                                                                         Expected state {expected_state.__dict__}"""

    def test_aggregate_paragraphs(self):
        dict_data = {
            "paragraphs": [
                "This is a test paragraphs.",
                "and another one",
                "yet another one",
                "another small one",
                "a very very very very very very very very very very long paragraph.",
                "another small paragraph.",
                "but with a skipped paragraph id.",
                "another example",
                "with different chapter",
            ],
            "chapters": ["I", "I", "II", "II", "II", "III", "III", "I", "II"],
            "n_words": [5, 3, 3, 3, 13, 3, 6, 2, 3],
            "text_ids": [0, 0, 1, 1, 1, 2, 2, 3, 3],
            "paragraphs_index": [0, 1, 3, 4, 5, 14, 16, 1, 2],
            "spans": [
                [0, 10],
                [14, 35],
                [10, 14],
                [14, 28],
                [35, 67],
                [18, 24],
                [36, 45],
                [104, 110],
                [205, 309],
            ],
        }
        expected_data = {
            "paragraphs": [
                "This is a test paragraphs.\n\nand another one",
                "yet another one\n\nanother small one",
                "a very very very very very very very very very very long paragraph.",
                "another small paragraph.",
                "but with a skipped paragraph id.",
                "another example",
                "with different chapter",
            ],
            "chapters": ["I", "II", "II", "III", "III", "I", "II"],
            "n_words": [8, 6, 13, 3, 6, 2, 3],
            "text_ids": [0, 1, 1, 2, 2, 3, 3],
            "spans": [
                [0, 35],
                [10, 28],
                [35, 67],
                [18, 24],
                [36, 45],
                [104, 110],
                [205, 309],
            ],
        }

        hf_dataset = Dataset.from_dict(dict_data)
        min_threshold = 6
        aggregated_dataset = aggregate_paragraphs(hf_dataset, min_threshold)

        for column in expected_data.keys():
            assert (
                expected_data[column] == aggregated_dataset[column]
            ), f"""Colums {column} wrong:
                                                                            expected: {expected_data[column]}
                                                                            Recovered: {aggregated_dataset[column]}"""

        assert (
            aggregated_dataset .to_dict()== expected_data
        ), f"""Aggregated and expected data do not match:\n
                                                        aggregated: {aggregated_dataset}\n
                                                        expected: {expected_data}"""


class TestBreakParagraphs:
    def test_split_paragraphs(self):
        paragraph = "This is a very long sentence, long long long long. It should be. It is small. Small sentence."
        all_subparagraphs = split_paragraph(paragraph, 9)
        expected_subparagraphs = [
            "This is a very long sentence, long long long long.",
            "It should be. It is small. Small sentence.",
        ]
        assert (
            all_subparagraphs == expected_subparagraphs
        ), f"""Expected and recovered subparagraphs do not match.\n
                                                                 Expected: {expected_subparagraphs}\n
                                                                 Recovered: {all_subparagraphs}."""

        all_subparagraphs = split_paragraph(paragraph, 3)
        expected_subparagraphs = [
            "This is a very long sentence, long long long long.",
            "It should be.",
            "It is small.",
            "Small sentence.",
        ]
        assert (
            all_subparagraphs == expected_subparagraphs
        ), f"""Expected and recovered subparagraphs do not match.\n
                                                                 Expected: {expected_subparagraphs}\n
                                                                 Recovered: {all_subparagraphs}."""

        all_subparagraphs = split_paragraph(paragraph, 4)
        expected_subparagraphs = [
            "This is a very long sentence, long long long long.",
            "It should be.",
            "It is small.",
            "Small sentence.",
        ]
        assert (
            all_subparagraphs == expected_subparagraphs
        ), f"""Expected and recovered subparagraphs do not match.\n
                                                                 Expected: {expected_subparagraphs}\n
                                                                 Recovered: {all_subparagraphs}."""

        all_subparagraphs = split_paragraph(paragraph, 6)
        expected_subparagraphs = [
            "This is a very long sentence, long long long long.",
            "It should be. It is small.",
            "Small sentence.",
        ]
        assert (
            all_subparagraphs == expected_subparagraphs
        ), f"""Expected and recovered subparagraphs do not match.\n
                                                                 Expected: {expected_subparagraphs}\n
                                                                 Recovered: {all_subparagraphs}."""

    def test_break_paragraphs(self):
        batch_paragraphs = {
            "paragraphs": [
                "This is a very long sentence, long long long long. It should be. It is small. Small sentence."
            ],
            "text_ids": [0],
            "paragraphs_index": [1],
            "chapters": ["IV"],
            "n_words": [18],
            "spans": [[14, 18]],
        }

        broken_paragraphs = break_paragraphs(batch_paragraphs, 17)
        expected_subparagraphs = {
            "paragraphs": [
                "This is a very long sentence, long long long long.",
                "It should be. It is small. Small sentence.",
            ],
            "text_ids": [0, 0],
            "chapters": ["IV", "IV"],
            "n_words": [10, 8],
            "spans": [[14, 18], [14, 18]],
            "splitted_paragraphs": [True, True],
        }

        for field in broken_paragraphs.keys():
            assert (
                expected_subparagraphs[field] == broken_paragraphs[field]
            ), f"""Expected and recovered column {field} do not 
                                                                                  match.\n
                                                                                  Expected {expected_subparagraphs[field]} \n
                                                                                  Recovered {broken_paragraphs[field]}"""

        assert (
            expected_subparagraphs == broken_paragraphs
        ), f"""Expected and recovered broken paragraphs do not match:\n
                                                                Expected: {expected_subparagraphs}\n
                                                                Recovered: {broken_paragraphs}."""
