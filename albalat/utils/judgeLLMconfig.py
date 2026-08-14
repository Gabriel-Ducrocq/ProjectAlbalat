"""
This file defines how to read the yaml config file for the LLM as a judge evaluation.
"""

import yaml
from pathlib import Path
from albalat.evaluation.models import OpenAIScorerConfig

def load_prompts(path: Path) -> OpenAIScorerConfig:
    """
    Reads the prompt from the yaml file defining the LLM as a judge scoring.
    This yaml file contains:
    - the prompt for judge1.
    - the prompt for judge2.
    - the openAI LLM to use.
    :param path: Path to the file
    :return: an object containing the prompts and LLM model for the LLM judges.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"""LLM as a judge configuration file {path} not found.""")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML prompt file: {path}") from e

    return OpenAIScorerConfig.model_validate(data)