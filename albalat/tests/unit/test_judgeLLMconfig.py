import pytest
from pathlib import Path
from pydantic import ValidationError
from albalat.utils.judgeLLMconfig import load_prompts


class TestLoadPrompts:
    def test_attributes(self, tmp_path):
        config_file_path = tmp_path / "config.yaml"
        config_file_path.write_text("""
        openai_llm: gpt-4o-mini
        judge_1: |
          Judge prompt one.
        judge_2: |
          Judge prompt two.
        """)

        config = load_prompts(config_file_path)

        assert config.openai_llm == "gpt-4o-mini"
        assert config.judge_1 == "Judge prompt one.\n"
        assert config.judge_2 == "Judge prompt two.\n"

    def test_file_not_found(self):
        config_file_path = Path("path/to/file/file.yaml")
        has_raised = False
        try:
            load_prompts(config_file_path)
        except FileNotFoundError:
            has_raised = True

        assert has_raised, f"Loading the judge config file {config_file_path} should have crashed as it does not exist."

    def test_file_not_yaml(self, tmp_path):
        config_file_path = tmp_path / "config.txt"
        config_file_path.write_text("""This is a plain text file.
        """)
        has_raised = False
        try:
            load_prompts(config_file_path)
        except ValueError:
            has_raised = True

        assert has_raised, f"Loading the judge file {config_file_path} should have crashed as it is not a yaml file."

    def test_empty_prompt(self, tmp_path):
        config_file = tmp_path / "config.yaml"

        config_file.write_text("""
        judge_1: ""
        judge_2: |
          Valid prompt
        openai_llm: gpt-4o-mini
        """)

        with pytest.raises(ValidationError):
            load_prompts(config_file)

    def test_missing_model(self, tmp_path):
        config_file = tmp_path / "config.yaml"

        config_file.write_text("""
    judge_1: |
      Valid prompt
    judge_2: |
      Valid prompt
    """)

        with pytest.raises(ValidationError) as exc_info:
            load_prompts(config_file)

        assert "openai_llm" in str(exc_info.value)

    def test_invalid_type(self, tmp_path):
        config_file = tmp_path / "config.yaml"

        config_file.write_text("""
        judge_1: |
          Valid prompt
        judge_2: |
          Valid prompt
        openai_llm: 123
        """)

        with pytest.raises(ValidationError):
            load_prompts(config_file)

    def test_extra_field(self, tmp_path):
        config_file = tmp_path / "config.yaml"

        config_file.write_text("""
        judge_1: |
          Valid prompt
        judge_2: |
          Valid prompt
        openai_llm: gpt-4o-mini
        something_else: invalid
        """)

        with pytest.raises(ValidationError):
            load_prompts(config_file)







