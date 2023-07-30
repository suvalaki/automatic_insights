
from typing import List, Dict, Optional, Any, Tuple

from pydantic import validator, root_validator

from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.sql_database import SQLDatabase

# BFS filter does all the tables in paralel
# its probably better to periodically look at the data and just 
# write a plain text description of the data.

from ai.chains.sql.name_evaluation.base import TableSelectionDetailThought

ADDITIONAL_CONTEXT = (
    "This is the first part of a process where later you will "
    "query the table schemas for more information in building sql "
    "queries to respond to the objective"
)

SINGLE_EVALUATION_PROMPT_TEMPLATE = (
    "You will be provided with an objective, a table name, and a "
    "list of other tables. "
    "You are to evaluate the relevance of the table (for completing "
    " the objective) from just the name. "
    "Speculate about the table. "
    " Guess what columns might be in the table. "
    "If the table looks useful in combination with another table "
    "then you should say so. "
    " You should be critical when scoring the likelihood."
    "{additional_context}"
    "\n\nObjective: {objective}"
    "\n\nTable: {table}"
    "\n\nTable Schema and information: {table_info}"
    "\n\nOther Tables: {tables}"
    "\n\n{format_instructions}"
)

SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=TableSelectionDetailThought)

SINGLE_EVALUATION_PROMPT = PromptTemplate(
    template=SINGLE_EVALUATION_PROMPT_TEMPLATE,
    input_variables=["objective", "table", "table_info", "tables"],
    partial_variables={
        "additional_context" : ADDITIONAL_CONTEXT,
        "format_instructions": SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER.get_format_instructions()},
    output_parser=SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER,
)

class SingleTablenameRelevanceEvaluationChain(LLMChain):
    prompt: PromptTemplate = SINGLE_EVALUATION_PROMPT
    output_parser: PydanticOutputParser = SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER


class MultipleTablenameRelevanceEvaluationChain(Chain):

    llm : BaseLanguageModel
    db: SQLDatabase
    llm_chain : SingleTablenameRelevanceEvaluationChain 
    output_key: str = "tablename_evaluations"

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            values["llm_chain"] = SingleTablenameRelevanceEvaluationChain(
                llm=values["llm"]
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        return ["objective", "tables"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[TableSelectionDetailThought]]:

        replies = [
            self.llm_chain.predict(table=table, 
                table_info=self.db.get_table_info_no_throw([table]),
                                   **inputs, 
                                   run_manager=run_manager)
            for table in inputs["tables"]]

        return {self.output_key: replies}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        replies = [
            self.llm_chain.predict(
            table=table, 
            table_info=self.db.get_table_info_no_throw([table]),
            **inputs, 
            run_manager=run_manager)
            async for table in inputs["tables"]]

        return {self.output_key: replies}


    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> List[TableSelectionDetailThought]:
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> List[TableSelectionDetailThought]:
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    @property
    def _chain_type(self) -> str:
        return "multiple_tablename_relevance_evaluation_chain"