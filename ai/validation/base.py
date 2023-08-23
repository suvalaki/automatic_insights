from typing import Generic, List, TypeVar, Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ai.chains.parallel_chain import ParallelChain

# The type for the thing we are validating
InputT = TypeVar("InputT")

class ValidationSignal(Enum):
    VALID = "VALID" 
    INVALID = "INVALID"


class ValidationError(BaseModel):
    pass


class Validation(Generic[InputT], BaseModel):
    signal: ValidationSignal
    errors: List[ValidationError]


class ValidatorBase(Generic[InputT]):

    @abstractmethod
    def validate(self, _input: InputT, **kwargs) -> Validation[InputT]:
        ...

    @abstractmethod 
    async def avalidate(self, _input: InputT, **kwargs) -> Validation[InputT]:
        ...


class Validator(Generic[InputT], Chain, ABC):

    input_key:str = "input"
    output_key:str = "validation"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key] 

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]


    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        validation = self.validate(inputs[self.input_key], **inputs)
        return {self.output_key: validation}


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        validation = self.avalidate(inputs[self.input_key], **inputs)
        return {self.output_key: validation}


class ParalellValidator(Validator[InputT], Chain, ABC):

    validators: List[Validator]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        pass


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        ...


class SequentialValidator(Validator[InputT], Chain, ABC):

    validators: List[Validator]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        pass


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Validation]:
        ...

