from string import Template
from polyfactory.factories.pydantic_factory import ModelFactory


# TODO: ASK GPT TO fix YAML

YAML_FIX_PROMPT = Template(
    "The following YAML is not valid:\n\n$yaml\n\nPlease fix it and try again."
)


yaml_schema_prompt_t = Template(
    """
You should only reply with a YAML code block following this json schema: $schema
The keys are case sensitive. Respond only with supplied keys. Dont change the format or add keys. 
Reply between ```yaml{Your yaml response goes here}```. Stop replying after the final '```'.
An example response (with random mock data) looks like:
'```yaml
$response
```'
You should not reply with the mock values.
You must reply with all the required fields in the YAML format specified.
"""
)


# Doesnt handle references properly
def simplify_schema(schema: dict):
    if isinstance(schema, dict):
        if all(
            [
                x in schema.keys()
                for x in ["title", "description", "type", "properties", "required"]
            ]
        ):
            schema = schema["properties"]
        for k, v in schema.items():
            schema[k] = simplify_schema(v)

    return schema


def create_schema_prompt(model) -> str:
    """schema: json schema as string"""

    class MockFactory(ModelFactory[model]):
        __model__ = model

    # This is important to remove the description from the schema
    # The description is provided by PYDANTIC_YAML
    # Its extra characters which are not needed
    # It also can cause Chat Continuation to get confused
    UNNEEDED_SCHEMA_STR = """ "description": "`pydantic.BaseModel` class with built-in YAML support.\\n\\nYou can alternatively inherit from this to implement your model:\\n`(pydantic_yaml.YamlModelMixin, pydantic.BaseModel)`\\n\\nSee Also\\n--------\\npydantic-yaml: https://github.com/NowanIlfideme/pydantic-yaml\\npydantic: https://pydantic-docs.helpmanual.io/\\npyyaml: https://pyyaml.org/\\nruamel.yaml: https://yaml.readthedocs.io/en/latest/index.html","""

    # return yaml_schema_prompt_t.substitute(schema=str(simplify_schema(schema)))
    schema = model.schema_json().replace(UNNEEDED_SCHEMA_STR, "")
    mock = MockFactory.build().yaml()
    return yaml_schema_prompt_t.substitute(schema=str(schema), response=str(mock))


def replace_parameter(yaml_dict, param_name, param_value):
    """
    Recursively replaces a parameter with a given name and value in a YAML dictionary.
    """
    if isinstance(yaml_dict, dict):
        for key, value in yaml_dict.items():
            yaml_dict[key] = replace_parameter(value, param_name, param_value)
    elif isinstance(yaml_dict, list):
        for i in range(len(yaml_dict)):
            yaml_dict[i] = replace_parameter(yaml_dict[i], param_name, param_value)
    elif isinstance(yaml_dict, str):
        yaml_dict = yaml_dict.replace("${{ " + param_name + " }}", param_value)
    return yaml_dict
