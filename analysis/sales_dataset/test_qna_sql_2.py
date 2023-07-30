from typing import List, Any

from pydantic import BaseModel

from ai.chains.sql.name_evaluation.database_filter import TableSelectionChain
from ai.chains.sql.name_evaluation.bfs_filter import (
    SingleTablenameRelevanceEvaluationChain,
)
from ai.chains.sql.name_evaluation.bfs_filter import (
    MultipleTablenameRelevanceEvaluationChain,
)

from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI


model_name = "gpt-3.5-turbo"
model = ChatOpenAI(model_name=model_name, temperature=0.0)


db = SQLDatabase.from_uri("sqlite:///./analysis/sales_dataset/data/data.db")

chain = TableSelectionChain(llm=model, db=db)

reply = chain.predict(objective="What are the number of customers in the UK.")

print(reply.json(indent=2))


from typing import Dict, Any
from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler


prompt = "Critique the output: "


class CritiqueLLMCallback(BaseCallbackHandler, LLMChain):
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass


single_chain = SingleTablenameRelevanceEvaluationChain(llm=model)
single_chain.predict(
    objective="What are the number of customers in the UK.",
    table="retail",
    table_info=db.get_table_info_no_throw(["retail"]),
    tables=["calendar", "retail", "population"],
)

single_chain.predict(
    objective="What are the number of customers in the UK.",
    table="calendar",
    table_info="",
    tables=["calendar", "retail", "population"],
)


multiple_chain = MultipleTablenameRelevanceEvaluationChain(llm=model, db=db)
evaluations = multiple_chain.predict(
    objective="What are the number of customers in the UK.",
    tables=["calendar", "retail", "population"],
)


# Setup a critique for the outputs?


# Now filter to tables with high enough score. Provide those
# tables to the query planner with their columns.
# Do we augment each column with a potential description?


def dummy_filter(obj):
    return [o for o in obj if o.score > 0.7]


filtered = dummy_filter(evaluations)
