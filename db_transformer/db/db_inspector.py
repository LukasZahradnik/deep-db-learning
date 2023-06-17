from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
    Any,
    Dict,
    FrozenSet,
    Optional,
    Set,
    Tuple,
    Type,
)

import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.schema import ForeignKeyConstraint, Table
from sqlalchemy.types import TypeEngine
from sqlalchemy.ext.automap import automap_base

from db_transformer.helpers.collections.set_filter import SetFilterProtocol
from db_transformer.schema.schema import ForeignKeyDef


class DBInspectorInterface(ABC):
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


class DBInspector(DBInspectorInterface):
    def __init__(self,
                 engine: Engine,
                 table_filter: Optional[SetFilterProtocol[str]] = None,
                 column_filters: Optional[Dict[str, SetFilterProtocol[str]]] = None,
                 ):
        self._engine = engine
        self._inspect = sqlalchemy.inspect(engine)
        self._table_filter = table_filter
        self._column_filters = column_filters if column_filters is not None else {}

        # self._automap_base = automap_base()

        # def name_for_scalar_relationship(base: Type[Any], local_cls: Type[Any],
        #                                  referred_cls: Type[Any], constraint: ForeignKeyConstraint):
        #     out = '_'.join([c.name for c in constraint.columns]) + "_" + referred_cls.__name__.lower()
        #     return out


        # def name_for_collection_relationship(base: Type[Any], local_cls: Type[Any],
        #                                  referred_cls: Type[Any], constraint: ForeignKeyConstraint):
        #     out = '_'.join([c.name for c in constraint.columns]) + "_" + referred_cls.__name__.lower()
        #     return out

        # self._automap_base.prepare(autoload_with=engine,
        #                            name_for_scalar_relationship=name_for_scalar_relationship,
        #                            name_for_collection_relationship=name_for_collection_relationship)

    @property
    def engine(self) -> Engine:
        return self._engine

    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        pk = self.get_primary_key(table)
        pk_empty = len(pk) == 0

        # Note: SQLAlchemy's automap can be used here instead of creating a simple table model like below, but that is
        # unnecessarily complicated for our purposes here - we do not need foreign keys mapped, and stuff.
        # Also, automap doesn't support tables without primary keys, which is a dealbreaker.

        columns = [
            sqlalchemy.Column(col['name'], col['type'], primary_key=pk_empty or col['name'] in pk)
            for col in self._inspect.get_columns(table)
        ]

        tbl = Table(table,
                    sqlalchemy.MetaData(schema=self._inspect.default_schema_name),
                    *columns
                    )
        return tbl

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


class CachedDBInspector(DBInspectorInterface):
    def __init__(self, delegate: DBInspectorInterface):
        if isinstance(delegate, CachedDBInspector):
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
