import pytest

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Thought, Hypothesis
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
from ai.question_answering.generators.hypothesis_answer.llm_hypothesis_answer import (
    LLMHypothesisEvaluator,
)
from ai.question_answering.thought.llm_comparison import LLThoughtPairComparer
from ai.question_answering.thought.llm_summarizer import LLMThoughtSummarizer
from ai.question_answering.data import (
    LLMDataSourceSelector,
    LLMMultipleDataSourceSelector,
)
from ai.question_answering.generators.thought.llm_discussion import (
    LLMHypothesisDataExplainer,
)
from ai.question_answering.generators.thought.llm_scoring import LLMDataExplainerScorer

from tests.question_answering.generators.test_thought import (
    MockAlwaysFalseComparisonFilter,
)


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

thought = Thought(
    discussion="The UK accounts for 3% of customers spend "
    "but customers from that region are in the upper quartile "
    "of spend. 5 of the top 10 customres come from the UK. "
    "As a result the majority of customers from the UK account "
    "for the highest sales per customer. ",
    score=0.8,
)


def test_LLMHypothesisEvaluator():
    from langchain.chat_models import ChatOpenAI

    # model_name = "gpt-4"  # GPT4 does better
    model_name = "gpt-3.5-turbo"
    temperature = 0.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer",
        data_sources=data_sources,
    )
    evaluator = LLMHypothesisEvaluator(model)

    evaluation = evaluator(hypothesis, [], thought)

    print(evaluation)


def test_llm_hypothesis_answer():
    from langchain.chat_models import ChatOpenAI

    model_name_4 = "gpt-4"  # GPT4 does better
    model_name = "gpt-3.5-turbo"
    temperature = 0.0
    model_4 = ChatOpenAI(model_name=model_name_4, temperature=temperature)
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    def total_country_sales(question: str) -> str:
        return "Total Sales: Argentina 1%, Australia 34%, UK 3%, Romania 19%"

    def per_customer_uk_distribution(question: str) -> str:
        return "Per-Customer Quartiles UK: 25, 50, 66, 81"

    llm_data_sources = [
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
            args_schema=ToolInput,
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

    # Provide a wrapped agent as a tool? It can go get multiple data?

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales per customer. Sales per customer is revenue per customer.",
        data_sources=llm_data_sources,
    )

    # LLMDataSourceSelector(llm_data_sources, model),
    data_selector = LLMMultipleDataSourceSelector(llm_data_sources, model)

    generator = TargettedThoughtGenerator(
        MockAlwaysFalseComparisonFilter(),
        data_selector,
        LLMHypothesisDataExplainer(model),
        LLMDataExplainerScorer(model),
    )
    comparer = LLThoughtPairComparer(model)
    summarizer = LLMThoughtSummarizer(model, comparer)
    concluder = LLMHypothesisConcluder(model_4)
    evaluator = LLMHypothesisEvaluator(model_4)
    hypothesis_answerer = HypothesisAnswerGenerator(
        generator, summarizer, concluder, evaluator
    )

    hypothesis_answer = hypothesis_answerer.generate(hypothesis)

    print(hypothesis_answer.discussion)
    print(hypothesis_answer.score)

    # Maybe also an agent based tool to pick apart the hypothesis into its constituent elelment
    # Do some augmentation to pick apart what it means to be sales
    # or sales per customer. keywords ect
