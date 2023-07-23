import pytest
from typing import List

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Question, Hypothesis
from ai.question_answering.data.llm_single import LLMDataSourceSelector


def test_llm_multiple_datasource_selector():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-3.5-turbo"  # GPT4 does better
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

    selector = LLMDataSourceSelector(
        data_sources=data_sources,
        llm=model,
    )

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer",
        data_sources=data_sources,
    )

    selection = selector(hypothesis, [])
