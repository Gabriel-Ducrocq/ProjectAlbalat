from pathlib import Path
from datasets import Dataset
from albalat.scripts.paragraphs_to_chunks import (
    State,
    reset,
    aggregate_paragraphs,
    split_paragraph,
    break_paragraphs,
)

BASE_DIR = Path(__file__).parent.parent


class TestParagraphAggregation2:
    test_state = State()
    test_state.length = 10
    test_state.text = "aaa"
    test_state.paragraph_id = 10
    test_state.text_id = 1
    test_state.chapter = "A"
    test_state.span_start = 35
    test_state.span_stop = 50

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

    def test_aggregate(self):
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
                (0, 35),
                (10, 28),
                (35, 67),
                (18, 24),
                (36, 45),
                (104, 110),
                (205, 309),
            ],
        }

        total_length = len(expected_data["n_words"])
        expected_data = [
            {k: v[row_number] for k, v in expected_data.items()}
            for row_number in range(total_length)
        ]

        hf_dataset = Dataset.from_dict(dict_data)
        min_threshold = 6
        aggregated_dataset = list(aggregate_paragraphs(hf_dataset, min_threshold))
        for row_number, row in enumerate(expected_data):
            for k, v in row.items():
                assert (
                    row[k] == aggregated_dataset[row_number][k]
                ), f"""Row {row_number} column {k} wrong:
                                                                                expected: {expected_data[row_number]}
                                                                                Recovered: {row}"""


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
