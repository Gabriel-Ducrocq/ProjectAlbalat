"""
This file contains the class for computing the Context relevance of the RAG triad.
Since we have no annotated data set, we use a LLM-as-a-judge framework to evaluate how much the context is relevant.

Contrary to e.g RAGAS, the retrieved chunks are not aggregated together into a single context for grading, as we want
to evaluate each chunk separately against the query. Instead, the LLM score each chunk against the query.
We use the OpenAI API for grading.

For robustness, the same LLM is used to times to grade each chunk, with two different prompts. It is tasked to grade
the semantic and narrative similarity between the chunk and the query, giving:
- 0 if they are not semantically and narratively similar to each other.
- 1 if they are partially semantically and narratively similar to each other.
- 2 if they are semantically and narratively similar to each other.

For each chunk, the scores output by the two LLMs are averaged to give the final score.
"""

import asyncio
from collections.abc import Iterable
from albalat.evaluation.models import Score
from albalat.evaluation.models import OpenAIScorerConfig


def average_scores(scores: tuple[Score, Score]) -> dict[str, float]:
    """
    Averages the scores of all the judges for a single context and keeping them.
    """
    average = sum(score.rating for score in scores) / len(scores)
    grades = {f"judge{i}": float(score.rating) for i, score in enumerate(scores)}
    grades.update({"average": average})
    return grades


class OpenAIScorer:
    """
    Uses the OpenAI API to score the retrieved chunks against the query chunks.

    We use the same LLM with two different prompts, for robustness, following RAGAS.
    """

    def __init__(self, client, openai_api_config: OpenAIScorerConfig):
        super(OpenAIScorer, self).__init__()
        """
        llm is an evaluator llm from llm_factory in RAGAS. 
        """
        self.client = client
        self.prompt1 = openai_api_config.judge_1
        self.prompt2 = openai_api_config.judge_2

    def create_queries_unique_prompt(self, query, chunks, prompt_id):
        """
        Create the queries corresponding to a unique prompt.
        """
        if prompt_id not in (1, 2):
            raise ValueError(f"""prompt_id must be 1 or 2, currently {prompt_id}""")

        if prompt_id == 1:
            prompt = self.prompt1
        else:
            prompt = self.prompt2

        all_openAIQuery = [f"""Passage: {query}\n\nContext:{chunk}\n\n {prompt}""" for chunk in chunks]
        return all_openAIQuery

    def create_queries(self, query, chunks):
        """
        Create the queries by inserting the query and the chunks in both prompts.
        """
        queriesJudge1 = self.create_queries_unique_prompt(query, chunks, 1)
        queriesJudge2 = self.create_queries_unique_prompt(query, chunks, 2)
        return queriesJudge1, queriesJudge2

    async def get_scores(self, queriesJudge, temperature=0.1)-> Score:
        """
        Actually sends the full prompt to openAI.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": "You are a literary relevance evaluator. Output only JSON."
            },
                {"role": "user", "content": queriesJudge}],
            temperature=temperature,
        )
        json_grade = Score.model_validate_json(response.choices[0].message.content)
        return json_grade

    def process_responses(self, score_judges_1_2: Iterable[tuple[Score, Score]]) -> list[dict[str, float]]:
        """
        Average the scores for each context given by the two judges
        """
        averaged_scores = [average_scores(scores_1_2) for scores_1_2 in score_judges_1_2]
        return averaged_scores

    async def query_openai(self, query: str, chunks: list[str]) -> list[dict[str, float]]:
        """
        Queries the self.llm from OpenAI API, with all the pairs (query, chunks), with two prompts each.
        """
        queriesJudge1, queriesJudge2 = self.create_queries(query, chunks)
        scores1, scores2 = await asyncio.gather(
            asyncio.gather(*(self.get_scores(q) for q in queriesJudge1)),
            asyncio.gather(*(self.get_scores(q) for q in queriesJudge2)),
        )
        averaged_scores = self.process_responses(zip(scores1, scores2))
        return averaged_scores



