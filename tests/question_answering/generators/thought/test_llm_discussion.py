import pytest

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Hypothesis
from ai.question_answering.generators.thought.llm_discussion import (
    LLMHypothesisDataExplainer,
)


def test_llm_hypothesis_explainer():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-4"  # GPT4 does better
    temperature = 0.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    class ToolInput(BaseModel):
        question: str

    data_sources_desc = [
        "SQL table of sales data",
        "view of sales per country",
        "view of sales per product",
        "view of sales cadence per customer",
        "view of change in sales over time per customer",
    ]
    data_sources = [
        Tool.from_function(
            func=lambda hypothesis: None,
            name=desc,
            description=desc,
            args_schema=ToolInput,
        )
        for desc in data_sources_desc
    ]

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer",
        data_sources=data_sources,
    )

    explainer = LLMHypothesisDataExplainer(llm=model)

    discussion = explainer(
        hypothesis,
        "Uk sales = 3%. Average customer spend is low, being in top quartile.",
    )

    print(discussion)
