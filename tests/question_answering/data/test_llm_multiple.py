import pytest
from typing import List

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Question, Hypothesis
from ai.question_answering.data.llm_multiple import LLMMultipleDataSourceSelector


def test_llm_multiple_datasource_selector():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-3.5-turbo"  # GPT4 does better
    temperature = 0.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    class ToolInput(BaseModel):
        question: str

    def total_country_sales(question: str) -> str:
        return "Total Sales: Argentina 1%, Australia 34%, UK 3%, Romania 19%, USA 25%"

    def per_customer_uk_distribution(question: str) -> str:
        return "Per-Customer Quartiles UK: 25, 50, 66, 81"

    data_sources = [
        Tool.from_function(
            func=total_country_sales,
            name="Total revenue percentage sales in each country.",
            description="Gets the total sales percentage of each country",
            # args_schema=ToolInput,
        ),
        Tool.from_function(
            func=lambda hypothesis: "Number of customers per country. Argentina: 12315, Australia: 4812, UK: 158, Romainia: 1465.",
            name="Customers in each Country",
            description="Customers Per Country",
        ),
        Tool.from_function(
            func=lambda hypothesis: "Per-Customer Quertiles Argentia: 13, 24, 34, 45",
            name="Per Customer Quartiles Argentina",
            description="Gets the sales sales distribution for per-customer spend in Argentina.",
            # args_schema=ToolInput,
        ),
        Tool.from_function(
            func=lambda hypothesis: "Per-Customer Quertiles Australia: 16, 34, 66, 72",
            name="Per Customer Quartiles Australia",
            description="Gets the sales sales distribution for per-customer spend in Australia.",
            # args_schema=ToolInput,
        ),
        Tool.from_function(
            func=per_customer_uk_distribution,
            name="Per Customer Quartiles UK",
            description="Gets the sales sales distribution for per-customer spend in UK.",
            # args_schema=ToolInput,
        ),
        Tool.from_function(
            func=lambda hypothesis: "Per-Customer Quertiles UK: 3, 6, 9, 12",
            name="Per Customer Quartiles Romania",
            description="Gets the sales sales distribution for per-customer spend in Romania.",
            # args_schema=ToolInput,
        ),
    ]


    selector = LLMMultipleDataSourceSelector(
        data_sources=data_sources,
        llm=model,
    )

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer. Sales per customer is revenue per customer.",
        data_sources=data_sources,
    )

    selection = selector(hypothesis, [])

    print(str(selection))

