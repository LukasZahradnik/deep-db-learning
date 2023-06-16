from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
    Dict,
    FrozenSet,
    Optional,
    Set,
    Tuple,
)

import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeEngine
from sqlalchemy.ext.automap import automap_base

from db_transformer.helpers.collections.set_filter import SetFilterProtocol
from db_transformer.schema.schema import ForeignKeyDef


class EngineWrapperInterface(ABC):
    @property
    @abstractmethod
    def engine(self) -> Engine:
        ...

    @abstractmethod
    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        ...

    @abstractmethod
    def get_tables(self) -> Set[str]:
        ...

    @abstractmethod
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        ...

    @abstractmethod
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        ...

    @abstractmethod
    def get_primary_key(self, table: str) -> Set[str]:
        ...

    @abstractmethod
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        ...


class EngineWrapper(EngineWrapperInterface):
    def __init__(self,
                 engine: Engine,
                 table_filter: Optional[SetFilterProtocol[str]] = None,
                 column_filters: Optional[Dict[str, SetFilterProtocol[str]]] = None,
                 ):
        self._engine = engine
        self._inspect = sqlalchemy.inspect(engine)
        self._table_filter = table_filter
        self._column_filters = column_filters if column_filters is not None else {}

        self._automap_base = automap_base()
        self._automap_base.prepare(autoload_with=engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        return self._automap_base.classes[table]

    def get_tables(self) -> Set[str]:
        out = set(self._inspect.get_table_names())

        if self._table_filter is not None:
            out = self._table_filter(out)

        return out

    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        out = {col['name']: col['type'] for col in self._inspect.get_columns(table)}

        filt = self._column_filters.get(table, None)
        if filt is not None:
            out_keys = filt(set(out.keys()))
            out = {k: v for k, v in out.items() if k in out_keys}

        return out

    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        out = set()

        for tbl in self.get_tables():
            out |= {(tbl, col) for col in self.get_columns(tbl).keys()}

        return out

    def get_primary_key(self, table: str) -> Set[str]:
        return set(self._inspect.get_pk_constraint(table)['constrained_columns'])

    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return {
            frozenset(fk['constrained_columns']):
            ForeignKeyDef(
                columns=fk['constrained_columns'],
                ref_table=fk['referred_table'],
                ref_columns=fk['referred_columns']
            )
            for fk in self._inspect.get_foreign_keys(table)}


class CachedEngineWrapper(EngineWrapperInterface):
    def __init__(self, delegate: EngineWrapperInterface):
        if isinstance(delegate, CachedEngineWrapper):
            raise TypeError("DatabaseWrapper is already cached.")

        self._delegate = delegate

    @property
    def engine(self) -> Engine:
        return self._delegate.engine

    @lru_cache(maxsize=None)
    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        return self._delegate.get_orm_table(table)

    @lru_cache(maxsize=None)
    def get_tables(self) -> Set[str]:
        return self._delegate.get_tables()

    @lru_cache(maxsize=None)
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        return self._delegate.get_columns(table)

    @lru_cache(maxsize=None)
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        return self._delegate.get_table_column_pairs()

    @lru_cache(maxsize=None)
    def get_primary_key(self, table: str) -> Set[str]:
        return self._delegate.get_primary_key(table)

    @lru_cache(maxsize=None)
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        return self._delegate.get_foreign_keys(table)
