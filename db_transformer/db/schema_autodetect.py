from functools import lru_cache
from typing import Iterable, List, Set, Tuple, Type, TypeVar, Union
from sqlalchemy import Numeric, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import distinct
import sqlalchemy.sql.functions as fn
from sqlalchemy.orm import Session
from sqlalchemy.types import TypeEngine

from db_transformer.db.db_wrapper import (
    CachedEngineWrapper,
    EngineWrapper,
    EngineWrapperInterface,
)
from db_transformer.helpers.collections.set_filter import SetFilter, SetFilterProtocol
from db_transformer.schema import (
    CategoricalColumnDef,
    ForeignKeyColumnDef,
    NumericColumnDef,
    OmitColumnDef,
    Schema,
    ColumnDefs,
)
from db_transformer.schema.schema import ForeignKeyDef, TableSchema


class SchemaAnalyzer:
    def __init__(self,
                 engine: Union[Engine, EngineWrapper, EngineWrapperInterface],
                 session: Session,
                 omit_filters: Union[SetFilterProtocol[Tuple[str, str]],
                                     Iterable[Tuple[str, str]], Tuple[str, str], None] = None,
                 ) -> None:
        if isinstance(engine, CachedEngineWrapper):
            wrapper = engine
        elif isinstance(engine, EngineWrapperInterface):
            # we want to cache it anyway for a single SchemaAnalyzer instance
            wrapper = CachedEngineWrapper(engine)
        elif isinstance(engine, Engine):
            wrapper = CachedEngineWrapper(EngineWrapper(engine))
        else:
            raise TypeError(
                f"database is neither {Engine}, nor an implementation of {EngineWrapperInterface}: {engine}")

        self._wrapper = wrapper
        self._session = session

        if isinstance(omit_filters, tuple):
            omit_filters = [omit_filters]
        if isinstance(omit_filters, Iterable):
            omit_filters = SetFilter(exclude=omit_filters)
        if callable(omit_filters):
            self._not_omitted = omit_filters(self._wrapper.get_table_column_pairs())
        else:
            self._not_omitted = self._wrapper.get_table_column_pairs()

    @property
    def engine(self) -> Engine:
        return self._wrapper.engine

    @property
    def engine_wrapper(self) -> EngineWrapperInterface:
        return self._wrapper

    @lru_cache(maxsize=None)
    def get_all_foreign_key_columns(self, table: str) -> Set[str]:
        fks = self.engine_wrapper.get_foreign_keys(table)
        out = set()
        for fk in fks.keys():
            out |= fk

        return out

    @lru_cache(maxsize=None)
    def guess_categorical_cardinality(self, table: str, column: str) -> int:
        tbl = self.engine_wrapper.get_orm_table(table)
        col = getattr(tbl, column)
        query = select(fn.count(distinct(col))).select_from(tbl)
        return self._session.scalar(query)

    def guess_likely_categorical(self, table: str, column: str, col_type: TypeEngine) -> bool:
        if isinstance(col_type, Numeric):
            return False

        if column in self.engine_wrapper.get_primary_key(table):
            return False

        return True

    def guess_column_type(self, table: str, column: str) -> object:
        if (table, column) not in self._not_omitted:
            return OmitColumnDef()

        col_type = self.engine_wrapper.get_columns(table)[column]
        pk = self.engine_wrapper.get_primary_key(table)
        is_in_pk = column in pk

        fks = self.get_all_foreign_key_columns(table)
        if column in fks:
            return ForeignKeyColumnDef(key=is_in_pk)

        if self.guess_likely_categorical(table, column, col_type):
            cardinality = self.guess_categorical_cardinality(table, column)
            return CategoricalColumnDef(key=is_in_pk, card=cardinality)

        return NumericColumnDef(key=is_in_pk)

    def guess_schema(self) -> Schema:
        schema = Schema()
        for table in self.engine_wrapper.get_tables():
            column_defs = ColumnDefs()
            fks: List[ForeignKeyDef] = list(self.engine_wrapper.get_foreign_keys(table).values())
            for column in self.engine_wrapper.get_columns(table):
                column_defs[column] = self.guess_column_type(table, column)

            schema[table] = TableSchema(columns=column_defs, foreign_keys=fks)

        return schema


if __name__ == "__main__":
    engine = create_engine("mysql+pymysql://guest:relational@relational.fit.cvut.cz:3306/mutagenesis")
    with Session(engine) as session:
        schema = SchemaAnalyzer(engine, session, omit_filters=('molecule', 'mutagenic')).guess_schema()
    print(schema)
