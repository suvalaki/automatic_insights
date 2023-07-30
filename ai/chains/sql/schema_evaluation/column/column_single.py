from pydantic import BaseModel, Field
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser


class ColumneEvaluatedDescription(BaseModel):
    column: str = Field(description="The column name being described")
    possible_description: str = Field(
        description="A detailed description and discussion of the column"
    )
    short_description: str = Field(
        description="A short summary description of the column and its contents"
    )


EVAL_COL_DESC_TEMPLATE = (
    "You will be presented with a tablename, column, the table schema and "
    "a table extract (table info), and some additional entries from the column specified. "
    "You are to evaulate what the column contains into detailed and short "
    "summaries."
    "The detailed summary should be exploratory and seeks to make inferences "
    "about the data present in the column. "
    "The short description should be a summary of only the important details "
    "previously provided. "
    "\n\nTablename: {table_name}"
    "\n\nColumn: {column_name}"
    "\n\nTable Info: {table_info}"
    "\n\nAdditional column data: {column_extract}"
    "\n\n{format_instructions}"
)

EVAL_COL_DESC_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=ColumneEvaluatedDescription
)

EVAL_COL_DESC_PROMPT = PromptTemplate(
    template=EVAL_COL_DESC_TEMPLATE,
    input_variables=["table_name", "column_name", "table_info", "column_extract"],
    partial_variables={
        "format_instructions": EVAL_COL_DESC_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=EVAL_COL_DESC_OUTPUT_PARSER,
)


def get_column_samples(table: str, column: str, samples: int = 10):
    query = select(table).limit(samples)


class SingleColumnEvaluateDescriptionChain(LLMChain):
    prompt: PromptTemplate = EVAL_COL_DESC_PROMPT
    output_parser: PydanticOutputParser = EVAL_COL_DESC_PROMPT.output_parser
