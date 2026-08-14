import yaml
from pathlib import Path
from albalat.evaluation.models import OpenAIScorerConfig

def load_prompts(path: Path) -> OpenAIScorerConfig:
    """
    Reads the prompt from the yaml file defining the LLM as a judge scoring.
    :param path: path t
    :return:
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"""LLM as a judge configuration file {path} not found.""")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML prompt file: {path}") from e

    return OpenAIScorerConfig.model_validate(data)