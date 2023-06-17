from functools import lru_cache
from sqlalchemy.exc import OperationalError
from tqdm.std import tqdm
from typing import Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import distinct
import sqlalchemy.sql.functions as fn
from sqlalchemy.orm import Session
from sqlalchemy.types import Integer, Boolean, Date, DateTime, LargeBinary, MatchType, PickleType, String, Numeric, Integer, Interval, Time, TypeEngine, Uuid

from db_transformer.db.db_inspector import (
    CachedDBInspector,
    DBInspector,
    DBInspectorInterface,
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
    ALWAYS_CATEGORICAL_TYPES = (
        Boolean, String, Uuid,
    )

    ALWAYS_NUMERICAL_TYPES = (
        Date, DateTime, Numeric, Interval, Time,
    )

    IGNORED_TYPES = (
        LargeBinary, MatchType, PickleType,
    )

    AMBIGUOUS_TYPES = (
        Integer,
    )

    AMBIGUOUS_CARDINALITY_THRESHOLD = 10000

    COMMON_NUMERICAL_COLUMN_NAMES = [
        'balance', 'amount', 'size', 'duration', 'frequency', 'count', 'votes', 'score',
        '_id', 'id_',
    ]

    COMMON_NUMERICAL_COLUMN_NAMES_CASE_SENSITIVE = [
        'Id',
    ]

    COMMON_CATEGORICAL_COLUMN_NAMES = [
        'name',
    ]

    COMMON_CATEGORICAL_COLUMN_NAMES_CASE_SENSITIVE = [
    ]

    def __init__(self,
                 engine: Union[Engine, DBInspector, DBInspectorInterface],
                 session: Session,
                 omit_filters: Union[SetFilterProtocol[Tuple[str, str]],
                                     Iterable[Tuple[str, str]], Tuple[str, str], None] = None,
                 verbose=False,
                 ) -> None:
        if isinstance(engine, CachedDBInspector):
            inspector = engine
        elif isinstance(engine, DBInspectorInterface):
            # we want to cache it anyway for a single SchemaAnalyzer instance
            inspector = CachedDBInspector(engine)
        elif isinstance(engine, Engine):
            inspector = CachedDBInspector(DBInspector(engine))
        else:
            raise TypeError(
                f"database is neither {Engine}, nor an implementation of {DBInspectorInterface}: {engine}")

        self._inspector = inspector
        self._session = session

        if isinstance(omit_filters, tuple):
            omit_filters = [omit_filters]
        if isinstance(omit_filters, Iterable):
            omit_filters = SetFilter(exclude=omit_filters)
        if callable(omit_filters):
            self._not_omitted = omit_filters(self._inspector.get_table_column_pairs())
        else:
            self._not_omitted = self._inspector.get_table_column_pairs()

        self._verbose = verbose

    @property
    def engine(self) -> Engine:
        return self._inspector.engine

    @property
    def db_inspector(self) -> DBInspectorInterface:
        return self._inspector

    @lru_cache(maxsize=None)
    def get_all_foreign_key_columns(self, table: str) -> Set[str]:
        fks = self.db_inspector.get_foreign_keys(table)
        out = set()
        for fk in fks.keys():
            out |= fk

        return out

    @lru_cache(maxsize=None)
    def guess_categorical_cardinality(self, table: str, column: str) -> Optional[int]:
        try:
            tbl = self.db_inspector.get_orm_table(table)
            col = getattr(tbl.c, column)
            query = select(fn.count(distinct(col))).select_from(tbl)
            if self._verbose:
                print(query)
            return self._session.scalar(query)
        except OperationalError as e:
            return None

    def _do_guess_column_type(self, table: str, column: str, col_type: TypeEngine) -> Union[
            Type[CategoricalColumnDef],
            Type[NumericColumnDef],
            Type[OmitColumnDef]]:
        if isinstance(col_type, self.IGNORED_TYPES):
            return OmitColumnDef

        if isinstance(col_type, self.ALWAYS_CATEGORICAL_TYPES):
            return CategoricalColumnDef

        if isinstance(col_type, self.ALWAYS_NUMERICAL_TYPES):
            return NumericColumnDef

        if isinstance(col_type, self.AMBIGUOUS_TYPES):
            for common_name in self.COMMON_CATEGORICAL_COLUMN_NAMES:
                if common_name in column.lower():
                    return CategoricalColumnDef

            for common_name in self.COMMON_CATEGORICAL_COLUMN_NAMES_CASE_SENSITIVE:
                if common_name in column:
                    return CategoricalColumnDef

            for common_name in self.COMMON_NUMERICAL_COLUMN_NAMES:
                if common_name in column.lower():
                    return NumericColumnDef

            for common_name in self.COMMON_NUMERICAL_COLUMN_NAMES_CASE_SENSITIVE:
                if common_name in column:
                    return NumericColumnDef

            cardinality = self.guess_categorical_cardinality(table, column)
            if cardinality is None or cardinality > self.AMBIGUOUS_CARDINALITY_THRESHOLD:
                return NumericColumnDef
            else:
                return CategoricalColumnDef

        return OmitColumnDef

    def guess_column_type(self, table: str, column: str) -> object:
        if (table, column) not in self._not_omitted:
            return OmitColumnDef()

        col_type = self.db_inspector.get_columns(table)[column]
        pk = self.db_inspector.get_primary_key(table)
        is_in_pk = column in pk

        fks = self.get_all_foreign_key_columns(table)
        if column in fks:
            return ForeignKeyColumnDef(key=is_in_pk)

        guessed_type = self._do_guess_column_type(table, column, col_type)

        if guessed_type == CategoricalColumnDef:
            cardinality = self.guess_categorical_cardinality(table, column)
            if cardinality is not None:
                return CategoricalColumnDef(key=is_in_pk, card=cardinality)
        elif guessed_type == NumericColumnDef:
            return NumericColumnDef(key=is_in_pk)

        return OmitColumnDef(key=is_in_pk)

    def guess_schema(self) -> Schema:
        schema = Schema()
        progress = self.db_inspector.get_tables()
        if self._verbose:
            progress = tqdm(self.db_inspector.get_tables(), desc="Table")

        for table in progress:
            column_defs = ColumnDefs()
            fks: List[ForeignKeyDef] = list(self.db_inspector.get_foreign_keys(table).values())
            for column in self.db_inspector.get_columns(table):
                column_defs[column] = self.guess_column_type(table, column)

            schema[table] = TableSchema(columns=column_defs, foreign_keys=fks)

        return schema


if __name__ == "__main__":
    engine = create_engine("mysql+pymysql://guest:relational@relational.fit.cvut.cz:3306/PTE")
    with Session(engine) as session:
        schema = SchemaAnalyzer(engine, session, verbose=True).guess_schema()
    print(schema)
