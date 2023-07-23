from abc import ABC, abstractclassmethod

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Hypothesis, Thought


class DiscussionGenerator:
    def __call__(self, hypothesis: Hypothesis, data: str) -> str:
        return data


EXPLAINER_PROMPT_TEMPLATE = """
You are to act as a data insights analyst. 
You will be given a hypothesis and a piece of data that has been evaluated to relate to the hypothesis. 
You are to make inferences about the data which are useful for answering the hypothesis.

Hypothesis: 
{hypothesis}

Data:
{data}


You should be factual in your response. 
You must refer to the data specifically in your discussion.
Do not discuss how suitable or admissible your data is. Just make inferences about data relating to the hypothesis.
Consider if the response actually answers the hypothesis.
Answer only with the schema required.

{format_instructions}

"""


class ExplainerQuery(BaseModel):
    explanation: str


OUTPUT_PARSER = PydanticOutputParser(pydantic_object=ExplainerQuery)


EXPLAINER_PROMPT = PromptTemplate(
    template=EXPLAINER_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "data"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)


class LLMHypothesisDataExplainer(DiscussionGenerator):
    def __init__(self, llm):
        self.chain = LLMChain(
            llm=llm,
            prompt=EXPLAINER_PROMPT,
            output_parser=OUTPUT_PARSER,
        )

    def __call__(self, hypothesis: Hypothesis, data: str) -> str:
        return data
        # TODO:  reimplement
        return self.chain.predict(
            hypothesis=hypothesis,
            data=data,
        ).explanation
