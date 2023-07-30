from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from langchain.sql_database import SQLDatabase


class ExtendedSQLDatabase(SQLDatabase):

    """An extension of the SQLDatabase with some additional helper methods."""

    def get_table(self, tablename: str) -> Table:
        for tbl in self._metadata.sorted_tables:
            if tbl.name == tablename:
                return tbl

        raise "Table Not Found Error"
