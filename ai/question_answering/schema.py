from __future__ import annotations
from typing import Tuple

from pydantic import BaseModel
from langchain.agents import Tool

from ai.question_answering.thought import Thought, ThoughtSummary


class HashableTool(Tool):
    kind: str = "Datasource"

    def __hash__(self) -> int:
        # Because tools themselves arent hashable
        return (self.name + self.description).__hash__()

    def __str__(self) -> str:
        return (
            self.kind
            + "(name: '"
            + self.name
            + "' description: '"
            + self.description
            + "')"
        )


class Question(BaseModel):
    question: str
    data_sources: Tuple[HashableTool, ...]


class Hypothesis(BaseModel):
    hypothesis: str
    data_sources: Tuple[HashableTool, ...]

    class Config:
        frozen = True


class TargettedThought(Thought):
    hypothesis: Hypothesis
    data: str

    def __str__(self):
        return (
            f"data: '{self.data}', discussion: '{self.discussion}', score: {self.score}"
        )


class AnsweredHypothesis(Hypothesis, Thought):
    thoughts: Tuple[Thought, ...]
    comparison: ThoughtSummary
    answer: Thought


class Answer(Question, Thought):
    hypothesis: Tuple[AnsweredHypothesis, ...]


class DataSourceSelection(BaseModel):
    data_source: HashableTool
    reason: str


class MultipleDataSourceSelection(BaseModel):
    objective: str
    selection: Tuple[DataSourceSelection, ...]
