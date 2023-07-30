from typing import List, Dict, Optional, Any, Tuple


from pydantic import BaseModel, Field, root_validator, Field


class TableSelectionDetailThought(BaseModel):

    table: str = Field(
        description="Name of the usefull table.")
    reasons: List[str] = Field(description=
        "Explain how the likely_contents relates to the objective."
    )
    score: float = Field(
        description=
        "Likelihood that the table will assist in meeting the objective. "
        "A score between 0 and 1. "
        "0.0 means that the table wont be usefull at all. "
        "0.5 means that we dont know if the table will be usefull or not. "
        "1.0 means that the table is definitely usefull. "
    )


class TableSelectionsThought(BaseModel):
    table: str
    details: List[TableSelectionDetailThought]


class TableEvaluation(BaseModel):
    table: str = Field(description="Table being evaluated.")
    likely_contents: str = Field(description="Speculation on the likely description of the table.")
    relates_to_objective: float = Field(
        description="Likelihood that the table contents relates to the objective."
    )


class TableSelectionsDetailThought(BaseModel):
    evaluations: List[TableEvaluation]
    selection: List[str] | None 
    details: list[TableSelectionDetailThought] | None 