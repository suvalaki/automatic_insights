from typing import Callable, List, TypeVar, Generic, Optional
from abc import ABC, abstractmethod
from itertools import chain

from ai.question_answering.thought import Thought, ThoughtSummary, ThoughtSummarizer
from ai.question_answering.schema import (
    Question,
    Hypothesis,
    TargettedThought,
    AnsweredHypothesis,
    Answer,
    DataSourceSelection,
    MultipleDataSourceSelection,
)


T0 = TypeVar("T0")
T1 = TypeVar("T1")


class BaseGenerator(Generic[T0, T1], ABC):
    def __init__(self, comparison_filter: Callable[[T1], List[T1]]):
        self.comparison_filter = comparison_filter

    @abstractmethod
    def _generate_single(
        self, question: T0, other_hypothesis: Optional[List[T1]]
    ) -> T1:
        pass

    def generate(self, question: T0, other_hypothesis: List[T1]) -> T1:
        while True:
            hypothesis = self._generate_single(question, other_hypothesis)
            if self.comparison_filter(hypothesis, other_hypothesis):
                return hypothesis


class HypothesisGenerator(BaseGenerator[Question, Hypothesis]):
    pass


class TargettedThoughtGenerator(BaseGenerator[Hypothesis, TargettedThought]):
    def __init__(
        self,
        comparison_filter: Callable[[TargettedThought], List[TargettedThought]],
        data_source_selector: Callable[[Hypothesis], DataSourceSelection],
        # We need an agent to handle asking questions from the data source selector
        discussion_generator: Callable[[Hypothesis, str], str],
        discussion_scorer: Callable[[Hypothesis, str, str], float],
        data_getter=1,
    ):
        super().__init__(comparison_filter)
        self.data_source_selector = data_source_selector
        self.discussion_generator = discussion_generator
        self.discussion_scorer = discussion_scorer
        self.data_getter = data_getter

    def _generate_single(
        self, hypothesis: Hypothesis, thoughts: List[Thought], **kwargs
    ) -> TargettedThought:
        relevant_data_source: MultipleDataSourceSelection | DataSourceSelection | None = self.data_source_selector(
            hypothesis, thoughts
        )
        if isinstance(relevant_data_source, DataSourceSelection):
            if relevant_data_source is None or relevant_data_source.data_source == "":
                raise Exception("No relevant data source found")
            data: str = relevant_data_source.data_source(
                hypothesis.hypothesis
            )  # TODO: dynamic for question based
            reason = relevant_data_source.reason
        elif isinstance(relevant_data_source, MultipleDataSourceSelection):
            if len(relevant_data_source.selection) == 0:
                raise Exception("No relevant data source found")
            data_elements: str = [
                r.data_source(hypothesis.hypothesis)
                for r in relevant_data_source.selection
            ]
            data = "\n".join(data_elements)
            reason = "\n".join([r.reason for r in relevant_data_source.selection])

        discussion = self.discussion_generator(
            hypothesis,  # reason + "\n\n" + data
            data,
        )
        score = self.discussion_scorer(hypothesis, data, discussion)

        return TargettedThought(
            hypothesis=hypothesis, data=data, discussion=discussion, score=score
        )


class HypothesisConcluder(ABC):

    """Makes conclusions about the hypothesis based on the evidence.
    This is a standin for a summary. It might be an agent which combines the
    input thoughts together
    """

    @abstractmethod
    def __call__(self, h: Hypothesis, thoughts: List[Thought]) -> Thought:
        ...


class HypothesisAnswerGenerator(ABC):
    def __init__(
        self,
        thought_generator: TargettedThoughtGenerator,
        thought_summarizer: ThoughtSummarizer,
        concluder: HypothesisConcluder,
        evaluator: Callable[[Hypothesis, List[Thought], Thought], bool],
    ):
        self.thought_generator = thought_generator
        self.thought_summarizer = thought_summarizer
        self.concluder = concluder
        self.evaluator = evaluator

    def generate(self, hypothesis: Hypothesis) -> AnsweredHypothesis:
        thoughts = []
        answers: List[ThoughtSummary] = []
        answered = None

        while True:
            print("THOUGHT SIZE: ", len(thoughts))

            # Genenerate an additional piece of evidence
            # TODO: make other  the other answered hypothesis?
            if answered is not None:
                thoughts.append(
                    self.thought_generator.generate(
                        hypothesis, thoughts + [answered, summary]
                    )
                )
            else:
                thoughts.append(self.thought_generator.generate(hypothesis, thoughts))

            # Given the evidence available answer the hypothesis
            summary: ThoughtSummary = self.thought_summarizer.summarize(thoughts)

            # Filter out contradictions
            contradictory_thoughts = list(
                set(tuple(chain(*[c for c in summary.contradictions])))
            )
            thoughts = [t for t in thoughts if t not in contradictory_thoughts]
            thoughts = list(set(thoughts))

            if len(thoughts) == 0:
                continue

            format_thoughts = "\n-thought: ".join([str(t) for t in thoughts])
            print("THOUGHTS: ", format_thoughts)

            answered = self.concluder(hypothesis, thoughts)

            print("ASNWER: ", answered.discussion)

            discussion: Thought = self.evaluator(hypothesis, thoughts, answered)

            print("DISCUSSION: ", discussion)

            # Validate that the hypothesis has been answered adequately
            is_answered_adequately: bool = discussion.score > 0.5

            if is_answered_adequately:
                break

        return AnsweredHypothesis(
            **hypothesis.dict(),
            thoughts=thoughts,
            comparison=summary,
            answer=answered,
            discussion=discussion.discussion,
            score=discussion.score,
        )


class AnswerGenerator(ABC):
    def __init__(
        self,
        question: Question,
        hypothesis_generator: HypothesisGenerator,
        thought_generator: TargettedThoughtGenerator,
        thought_summarizer: ThoughtSummarizer,
        hypothesis_answer_generator: HypothesisAnswerGenerator,
        evaluator: Callable[[Question, List[AnsweredHypothesis], Thought], bool],
    ):
        self.question = question
        self.hypothesis_generator = hypothesis_generator
        self.thought_generator = thought_generator
        self.thought_summarizer = thought_summarizer
        self.hypothesis_answer_generator = hypothesis_answer_generator
        self.evaluator = evaluator

    def generate(self) -> Answer:
        hypothesis: List[Hypothesis] = []
        answered_hypothesis: List[AnsweredHypothesis] = []
        while True:
            additional_hypothesis = self.hypothesis_generator.generate(
                self.question, hypothesis
            )
            hypothesis.append(additional_hypothesis)
            answered_hypothesis.append(additional_hypothesis)
            conclusions: ThoughtSummary = self.thought_summarizer(answered_hypothesis)

            is_answered: bool = self.evaluator(
                self.question, answered_hypothesis, conclusions
            )
            if is_answered:
                break

        return Answer(
            question=self.question,
            hypothesis=hypothesis,
            answered_hypothesis=answered_hypothesis,
            discussion=conclusions.discussion,
            score=conclusions.score,
        )
