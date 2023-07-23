import pytest

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Thought, Hypothesis, TargettedThought
from ai.question_answering.generators.base import (
    TargettedThoughtGenerator,
    HypothesisAnswerGenerator,
)
from ai.question_answering.generators.thought.llm_discussion import (
    LLMHypothesisDataExplainer,
)
from ai.question_answering.generators.hypothesis_answer.llm_hypothesis_concluder import (
    LLMHypothesisConcluder,
)


thought0 = Thought(
    discussion="""This data source will provide the total number of customers in each country. Combined with the already selected 'Per Customer Quartiles UK' data source, we can calculate the average sales per customer in the UK. To confirm the hypothesis, we need to compare this value with the average sales per customer in other countries, which requires the number of customers in those countries.

Number of customers per country. Argentina: 12315, Australia: 4812, UK: 158, Romainia: 1465.""",
    score=0.5,
)
thought1 = Thought(
    discussion="""To calculate sales per customer, we need the total revenue sales in each country. This data source will provide us with the necessary information to determine the sales per customer for each country, including the UK. By comparing the sales per customer across different countries, we can determine if the UK has the highest sales per customer.

Total Sales: Argentina 1%, Australia 34%, UK 3%, Romania 19%, USA 25%
    """,
    score=0.8,
)


def test_LLMHypothesisConcluder():
    from langchain.chat_models import ChatOpenAI

    # model_name = "gpt-4"  # GPT4 does better
    model_name = "gpt-3.5-turbo"
    temperature = 0.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer",
        data_sources=[],
    )
    concluder = LLMHypothesisConcluder(model)
    conclusion = concluder(hypothesis, [thought0, thought1])

    print(conclusion)
    print()
    print(conclusion.discussion)
