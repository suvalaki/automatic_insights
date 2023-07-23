from typing import List, Tuple
from abc import ABC, abstractmethod
from functools import reduce

from pydantic import BaseModel


class Thought(BaseModel):
    discussion: str
    score: float

    class Config:
        frozen = True


class ThoughtPairComparison(Thought):
    thoughts: Tuple[Thought, Thought]
    shared: Tuple[str, ...]
    unique: Tuple[Tuple[str, ...], Tuple[str, ...]]
    contradiction: Tuple[Tuple[str, str, str], ...]

    class Config:
        frozen = True

    @property
    def contradictory(self) -> bool:
        return len(self.contradiction) > 0


class ThoughtPairComparer(ABC):
    @abstractmethod
    def _get_shared(self, thought1: Thought, thought2: Thought) -> List[str]:
        pass

    @abstractmethod
    def _get_unique_from_1(self, thought1: Thought, thought2: Thought) -> List[str]:
        pass

    @abstractmethod
    def _get_unique_from_2(self, thought1: Thought, thought2: Thought) -> List[str]:
        pass

    def _get_unique(
        self, thought1: Thought, thought2: Thought
    ) -> Tuple[List[str], List[str]]:
        unique_in_1 = self._get_unique_from_1(thought1, thought2)
        unique_in_2 = self._get_unique_from_2(thought1, thought2)
        return (
            unique_in_1 if unique_in_1 is not None else [],
            unique_in_2 if unique_in_2 is not None else [],
        )

    @abstractmethod
    def _get_contradictions(
        self, thought1: Thought, thought2: Thought
    ) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def _get_discussion(
        self,
        thought1: Thought,
        thought2: Thought,
        shared: List[str],
        unique: Tuple[List[str], List[str]],
        contraditions: List[Tuple[str, str]],
    ) -> str:
        pass

    @abstractmethod
    def _get_score(
        self,
        thought1: Thought,
        thought2: Thought,
        shared: List[str],
        unique: Tuple[List[str], List[str]],
        contraditions: List[Tuple[str, str]],
        conclusion: str,
    ) -> float:
        pass

    def compare(self, thought1: Thought, thought2: Thought) -> ThoughtPairComparison:
        shared = self._get_shared(thought1, thought2)
        unique = self._get_unique(thought1, thought2)
        contradiction = self._get_contradictions(thought1, thought2)
        discussion = self._get_discussion(
            thought1, thought2, shared, unique, contradiction
        )
        score = self._get_score(
            thought1, thought2, shared, unique, contradiction, discussion
        )
        return ThoughtPairComparison(
            thoughts=(thought1, thought2),
            shared=shared if shared is not None else [],
            unique=unique,
            contradiction=contradiction if contradiction is not None else [],
            discussion=discussion,
            score=score,
        )


class ThoughtPairCombination(Thought):
    thoughts: Tuple[Thought, Thought]
    comparison: ThoughtPairComparison
    combined: Thought

    class Config:
        frozen = True


class ThoughtSummary(Thought):
    thoughts: Tuple[Thought, ...]
    contradictions: Tuple[Thought, ...]

    class Config:
        frozen = True


# Intermediate calculations based on throughts?
# For example call a calculator to build additional metrics...
# Or is this better fed into the data getter? Let one of the tools be a calculator?


class ThoughtSummarizer(ABC):
    @abstractmethod
    def _collect_contradictory_thoughts(
        self, thoughts: List[Thought]
    ) -> List[ThoughtPairCombination]:
        pass

    def _collect_non_contradictory_thoughts(
        self, thoughts: Tuple[List[Thought], List[Thought]]
    ) -> List[Thought]:
        contradictions = self._collect_contradictory_thoughts(thoughts)
        non_contradictory = [t for t in thoughts if t not in contradictions]
        return contradictions, non_contradictory

    @abstractmethod
    def _summarize_non_contradictory_thoughts(self, thoughts: List[Thought]) -> Thought:
        pass

    def summarize(self, thoughts: List[Thought]) -> ThoughtSummary:
        if len(thoughts) < 2:
            print("too few thoughts: ", len(thoughts))
            return ThoughtSummary(
                thoughts=thoughts,
                discussion=thoughts[0].discussion,
                score=thoughts[0].score,
                contradictions=[],
            )

        contradictions, non_contradictory = self._collect_non_contradictory_thoughts(
            thoughts
        )
        summarized = self._summarize_non_contradictory_thoughts(non_contradictory)

        return ThoughtSummary(
            thoughts=non_contradictory,
            discussion=summarized.discussion,
            score=summarized.score,
            contradictions=contradictions,
        )
