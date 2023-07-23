from typing import List
from abc import ABC, abstractmethod

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import (
    Thought,
    Hypothesis,
    DataSourceSelection,
    MultipleDataSourceSelection,
)


class DataSourceSelector:
    def __init__(self, data_sources: List[Tool]):
        self.data_sources = data_sources

    @abstractmethod
    def __call__(
        self, hypothesis: Hypothesis, prior_thoughts: List[Thought]
    ) -> MultipleDataSourceSelection | DataSourceSelection | None:
        ...
