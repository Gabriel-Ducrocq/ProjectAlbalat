from albalat.evaluation.models import Score
from albalat.evaluation.scorer import average_scores, OpenAIScorerConfig, create_queries_unique_prompt, process_responses


def test_average_scores():
    judges = (Score(rating = 0), Score(rating = 0), Score(rating = 2))
    result_dict = average_scores(judges)
    expected_dict = {"judge1":judges[0].rating,"judge2":judges[1].rating, "judge3":judges[2].rating, "average": (judges[0].rating+judges[1].rating+judges[2].rating)/3}
    assert result_dict == expected_dict

def test_create_queries_unique_prompt():
    query = "This is a query"
    chunks = ["this is a chunk", "this is a second chunk"]
    prompt = "These are the instructions to follow"
    resulting_prompts = create_queries_unique_prompt(query, chunks, prompt)
    expected_prompts = ["""Passage: This is a query\n\nContext:this is a chunk\n\n These are the instructions to follow""",
                        """Passage: This is a query\n\nContext:this is a second chunk\n\n These are the instructions to follow""",]
    [f"""Passage: {query}\n\nContext:{chunk}\n\n {prompt}""" for chunk in chunks]
    assert resulting_prompts == expected_prompts, f"""The prompts {resulting_prompts} and the expected prompts {expected_prompts}
                                                        do not match."""

def test_process_responses():
    judges = [(Score(rating = 1), Score(rating = 1)),(Score(rating = 1), Score(rating = 2))]
    result_dict = process_responses(judges)
    expected_dict = [{"judge1":judges[i][0].rating,"judge2":judges[i][1].rating,
                     "average": (judges[i][0].rating+judges[i][1].rating)/2} for i in range(2)]
    assert result_dict == expected_dict, f"""Expected processed response {expected_dict} does not match the processed
                                                outputs {result_dict}"""