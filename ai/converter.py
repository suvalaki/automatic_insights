from typing import List, Optional, Protocol, Dict, Generic, TypeVar
import re

from pydantic_yaml import YamlModel

from langchain import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain, TransformChain
from langchain.chains import SequentialChain
from langchain.output_parsers import RetryWithErrorOutputParser

from ai.yaml_utils import create_schema_prompt


OutputFormat = TypeVar("OutputFormat")


class ChatMsgConverter(Protocol, Generic[OutputFormat]):
    def validate(self, msg: str) -> bool:
        ...

    def convert(self, msg: str) -> OutputFormat:
        ...

    def try_fix(self, msg: str) -> Optional[OutputFormat]:
        # eg. for yaml loaders ask gpt to reformulate it.
        ...

    def __call__(self, msg: str) -> OutputFormat:
        if self.validate(msg):
            return self.convert(msg)
        return self.try_fix(msg)


class YamlChatMsgConverter(ChatMsgConverter[OutputFormat]):
    def __init__(self, schema: OutputFormat):
        self.schema = schema

    def _remove_extraneous(self, msg: str) -> str:
        """Clean up the returned message from GPT-3."""

        if r := re.search(r"```yaml(.*)```", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"```(.*)```", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"---yaml(.*)---", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"---(.*)---", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"```yaml(.*)", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"```(.*)", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"---yaml(.*)", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        if r := re.search(r"---(.*)", msg, re.DOTALL):
            msg = r.group(1)
            return msg

        return msg

    def validate(self, msg: str) -> bool:
        return True

    def convert(self, msg: str) -> str:
        return self.schema.parse_raw(self._remove_extraneous(msg))

    def try_fix(self, msg: str) -> Optional[str]:
        return None


class LangchainYamlChatMsgConverter(BaseOutputParser[OutputFormat]):
    converter: YamlChatMsgConverter

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, schema: OutputFormat, **kwargs):
        super().__init__(converter=YamlChatMsgConverter(schema), **kwargs)

    def parse(self, completion: str) -> OutputFormat:
        return self.converter.convert(completion)


def create_pydantic_yaml_validated_chain(llm, schema, prompt, retry: int = 5, **kwargs):
    """
    Creates a Langchain for parsing yaml schemas - as KOR doesnt work...

    usage:
    ```
    validation_chain = create_pydantic_yaml_validated_chain(
        llm, MissionPromptOutput, prompt
    )

    r = validation_chain({"query": formatted_total_prompt})
    ```
    """

    complete_prompt = PromptTemplate.from_template(
        "Answer the user query. Reply with YAML in the schema specified. You must only reply with yaml.\n"
        + f"\n{prompt.template}\n"
        + "\n{format_instructions}\n",
        partial_variables={"format_instructions": create_schema_prompt(schema)},
    )

    retry_prompt = PromptTemplate.from_template(
        "Answer the user query. Reply with YAML in the schema specified. You must only reply with yaml.\n"
        + f"\n{prompt.template}\n"
        + "\n{format_instructions}\n"
        + "\nThe following input was invalid:\n"
        + "{raw}\n"
        + "\n{error}\n\nTry Again\n",
        partial_variables={"format_instructions": create_schema_prompt(schema)},
    )

    llm_chain = LLMChain(
        prompt=complete_prompt,
        llm=llm,
        output_key="raw",
    )

    parser = LangchainYamlChatMsgConverter(schema)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)

    def parse_output(inputs: dict, retry: int = retry) -> dict:
        text = inputs["raw"]
        print(text)

        error = ""
        try:
            converted = parser.parse(str(text))
        except Exception as e:
            print(str(e))
            converted = None
            error = str(e)

            if retry > 0:
                r = retry_parser.parse_with_prompt(
                    e,
                    complete_prompt.format(
                        **{
                            k: v
                            for k, v in inputs.items()
                            if k in prompt.input_variables
                        }
                    ),
                )
                inputs["raw"] = r
                return parse_output(inputs, retry - 1)

        return {"validated": converted, "error": error}

    transform_chain = TransformChain(
        input_variables=["raw"],
        output_variables=["validated"],
        transform=parse_output,
    )

    chain = SequentialChain(
        input_variables=prompt.input_variables,
        output_variables=["raw", "validated"],
        chains=[llm_chain, transform_chain],
    )

    return chain
