from pydantic import BaseModel, Field, ConfigDict

class Score(BaseModel):
    """
    Pydantic class to check that the data grading of the LLM follows the right format
    """
    model_config = ConfigDict(extra='forbid', strict=True)
    rating: int = Field(ge=0, le=2)


class OpenAIScorerConfig(BaseModel):
    """
    Pydantic class to check that the LLm evaluation yaml is correct.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    judge_1: str = Field(min_length=1)
    judge_2: str = Field(min_length=2)
    openai_llm: str = Field(min_length=1)










