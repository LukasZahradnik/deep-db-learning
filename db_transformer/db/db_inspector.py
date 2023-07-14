from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Set, Tuple

import sqlalchemy
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.schema import Table
from sqlalchemy.types import TypeEngine

from db_transformer.helpers.collections.set_filter import SetFilterProtocol
from db_transformer.schema.schema import ForeignKeyDef


class DBInspectorInterface(ABC):
    """
    The interface for :py:class:`DBInspector`.
    """

    @property
    @abstractmethod
    def connection(self) -> Connection:
        """
        The underlying SQLAlchemy `Connection`.
        """
        ...

    @property
    @abstractmethod
    def engine(self) -> Engine:
        """
        The underlying SQLAlchemy `Engine`.
        """
        ...

    @abstractmethod
    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        """
        Returns a `sqlalchemy.Table` instance, allowing for running queries on the database table.
        """
        ...

    @abstractmethod
    def get_tables(self) -> Set[str]:
        """
        Get all tables in the databases as a set.
        """
        ...

    @abstractmethod
    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        """
        Get all columns in a table as a dictionary of string -> SQLAlchemy type
        """
        ...

    @abstractmethod
    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        """
        Get all (table, column) pairs in the database as a set.
        """
        ...

    @abstractmethod
    def get_primary_key(self, table: str) -> Set[str]:
        """
        Get the primary key of the given table as a set of column names.
        """
        ...

    @abstractmethod
    def get_foreign_keys(self, table: str) -> Dict[FrozenSet[str], ForeignKeyDef]:
        """
        Get all foreign key constraints of the given table as a dictionary, where the key is a subset of table columns,
        and the value is SQLAlchemy's `ForeignKeyDef` instance.
        """
        ...


class DBInspector(DBInspectorInterface):
    """
    A simplified helper class that allows to retrieve select basic information about a database,
    its tables, columns, and values.
    """

    def __init__(self,
                 connection: Connection,
                 table_filter: Optional[SetFilterProtocol[str]] = None,
                 column_filters: Optional[Dict[str, SetFilterProtocol[str]]] = None,
                 ):
        """
        :field connection: The database connection - instance of SQLAlchemy's `Connection` class.
        :field table_filter: A :py:class:`db_transformer.helpers.collections.set_filter.SetFilter` instance or a callable that filters a set of values. \
All values that remain are the tables that the inspector will be aware of; excluded ones will be ignored.
        """

        self._connection = connection
        self._inspect = sqlalchemy.inspect(self._connection.engine)
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
    def connection(self) -> Connection:
        return self._connection

    @property
    def engine(self) -> Engine:
        return self._connection.engine

    def get_orm_table(self, table: str) -> sqlalchemy.Table:
        pk = self.get_primary_key(table)
        pk_empty = len(pk) == 0

        # Note: SQLAlchemy's automap can be used here instead of creating a simple table model like below, but that is
        # unnecessarily complicated for our purposes here - we do not need foreign keys mapped, for example.
        # Also, automap doesn't support tables without primary keys, which is a dealbreaker why we cannot use it.
        # We need very simple SQL queries here anyway.

        # Note: SQLAlchemy needs all tables to have a priamry key because it differentiates rows based on primary key values.
        # If a table doesn't have a primary key, SQLAlchemy doesn't default to anything and simply refuses to work with such table.
        # As a workaround, if primary key is empty, then we "trick" SQLAlchemy as follows: we mark all columns as primary key.
        # That way SQLAlchemy will differentiate rows based on the values in all columns.
        #
        # We do not propagate this "trick" to the machine learning pipeline - it is only for SQLAlchemy to work.
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
    """
    :py:class:`DBInspector`, but that caches its results instead of repeated SQL calls/retrievals.
    """

    def __init__(self, delegate: DBInspectorInterface):
        """
        :field delegate: The :py:class:`DBInspectorInterface` instance to cache the results of. :py:class:`CachedDBInspector` will delegate the function calls to this instance.
        """

        if isinstance(delegate, CachedDBInspector):
            raise TypeError("DatabaseWrapper is already cached.")

        self._delegate = delegate

    @property
    def connection(self) -> Connection:
        return self._delegate.connection

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
