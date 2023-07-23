from typing import Callable, List

from langchain.tools import Tool
from langchain.agents.agent_toolkits.base import BaseToolkit

from ai.question_answering.generators.base import (
    Thought,
    TargettedThought,
    TargettedThoughtGenerator,
    ThoughtSummarizer,
    Hypothesis,
    HypothesisAnswerGenerator,
    HypothesisConcluder,
)
from ai.question_answering.data import DataSourceSelection


def create_targetted_thought_generator_toolkit(
    hypothesis: Hypothesis,
    thoughts: List[Thought],  # The weakref to thoughts
    comparison_filter: Callable[[TargettedThought], List[TargettedThought]],
    data_source_selector: Callable[[Hypothesis], DataSourceSelection],
    # We need an agent to handle asking questions from the data source selector
    discussion_generator: Callable[[Hypothesis, str], str],
    discussion_scorer: Callable[[Hypothesis, str, str], float],
):
    data_source_selector_tool = Tool.from_function(
        func=data_source_selector.generate(hypothesis, thoughts),
        name="Evidence Thought Getter",
        description="Queries a data getter to provide evidence and discusses that data.",
    )

    thought_summarizer = Tool.from_function(
        func=
    )
