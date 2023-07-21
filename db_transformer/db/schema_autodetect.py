from functools import lru_cache
import re
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import inflect
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import distinct, select
from sqlalchemy.sql.expression import null
import sqlalchemy.sql.functions as fn
from sqlalchemy.sql.operators import isnot
from sqlalchemy.types import (
    Boolean,
    Date,
    DateTime,
    Integer,
    Interval,
    Numeric,
    String,
    TEXT,
    Text,
    Time,
    TypeEngine,
    Unicode,
)

from db_transformer.db.db_inspector import (
    CachedDBInspector,
    DBInspector,
    DBInspectorInterface,
)
from db_transformer.helpers.collections.set_filter import SetFilter, SetFilterProtocol
from db_transformer.helpers.progress import wrap_progress
from db_transformer.schema import (
    CategoricalColumnDef,
    ColumnDef,
    ColumnDefs,
    DateColumnDef,
    DateTimeColumnDef,
    DurationColumnDef,
    ForeignKeyColumnDef,
    ForeignKeyDef,
    KeyColumnDef,
    NumericColumnDef,
    OmitColumnDef,
    Schema,
    TableSchema,
    TextColumnDef,
    TimeColumnDef,
)


__all__ = [
    "SchemaAnalyzer"
]


class SchemaAnalyzer:
    """
    Helper class for retrieving a database :py:class:`Schema`, where column types (as interpreted by the machine learning pipeline)
    are determined based on select heuristics. Can be subclassed to support other :py:class:`ColumnDef` instances as well.

    Caches the results of all database queries - in order to obtain fresh data after updating a database,
    please create a new instance of the class.
    """

    DETERMINED_TYPES: Dict[Type[ColumnDef], Tuple[Type[TypeEngine], ...]] = {
        TextColumnDef: (Text, TEXT, LONGTEXT, MEDIUMTEXT, Unicode, ),
        CategoricalColumnDef: (Boolean, ),
        NumericColumnDef: (Numeric, ),
        DateColumnDef: (Date, ),
        DateTimeColumnDef: (DateTime, ),
        DurationColumnDef: (Interval, ),
        TimeColumnDef: (Time, ),
    }
    """
    Per each ColumnDef, set of SQL types that are automatically matched to given ColumnDef.
    """

    ID_NAME_REGEX = re.compile(
        r"_id$|^id_|_id_|Id$|Id[^a-z]|[Ii]dentifier|IDENTIFIER|ID[^a-zA-Z]|ID$|[guGU]uid[^a-z]|[guGU]uid$|[GU]UID[^a-zA-Z]|[GU]UID$")

    COMMON_NUMERIC_COLUMN_NAME_REGEX = re.compile(
        r"balance|amount|size|duration|frequency|count|votes|score|number|age", re.IGNORECASE)  # TODO: add more?

    FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD = 0.95
    """
    The fraction of distinct values to total count of non-null values, which decides (in some situations) that type cannot be categorical.
    If the fraction exceeds this threshold, marks the column as something other than categorical.
    """

    STRING_CARDINALITY_THRESHOLD = 800
    """
    Cardinality threshold below which string types are assumed categorical, and above which are assumed numeric.
    """

    INTEGER_CARDINALITY_THRESHOLD = 10000
    """
    Cardinality threshold below which integer types are assumed categorical, and above which are assumed numeric.
    """

    def __init__(self,
                 connection: Union[Connection, DBInspector, DBInspectorInterface],
                 omit_filters: Union[SetFilterProtocol[Tuple[str, str]],
                                     Iterable[Tuple[str, str]], Tuple[str, str], None] = None,
                 target: Optional[Tuple[str, str]] = None,
                 verbose=False,
                 ) -> None:
        """
        :field connection: The database connection - instance of SQLAlchemny's `Connection` class, \
or a custom :py:class:`DBInspector` or :py:class:`DBInspectorInterface` instance, which allows to \
e.g. specify database tables or table columns to be completely ignored.
        :field omit_filters: A filter for (table_name, column_name) tuples. Can be one of the following:
            a) a list of such tuples, in which case they will all receive the :py:class:`OmitColumnDef` type
            b) a :py:class:`db_transformer.helpers.collections.set_filter.SetFilter` instance, allowing to specify either a whitelist or a blacklist (or both), in which case \
all that is *excluded* will receive the :py:class:`OmitColumnDef` type
            c) a callable which, given a set of values, returns their subset. In this case all that is *excluded* will receive \
the :py:class:`OmitColumnDef` type
        :field target: Tuple of (table_name, column_name) for the target table and column
        :field verbose: If true, will show executed `SELECT` statements, as well as a per-table progress bar.
        """

        if isinstance(connection, CachedDBInspector):
            inspector = connection
        elif isinstance(connection, DBInspectorInterface):
            # we want to cache it anyway for a single SchemaAnalyzer instance
            inspector = CachedDBInspector(connection)
        elif isinstance(connection, Connection):
            inspector = CachedDBInspector(DBInspector(connection))
        else:
            raise TypeError(
                f"database is neither {Connection.__name__}, nor an implementation of {DBInspectorInterface.__name__}: {connection}")

        self._inspector = inspector

        self._target = target

        if isinstance(omit_filters, tuple):
            omit_filters = [omit_filters]
        if isinstance(omit_filters, Iterable):
            omit_filters = SetFilter(exclude=omit_filters)
        if callable(omit_filters):
            self._not_omitted = omit_filters(self._inspector.get_table_column_pairs())
        else:
            self._not_omitted = self._inspector.get_table_column_pairs()

        self._verbose = verbose

        self._inflect = inflect.engine()

    @property
    def connection(self) -> Connection:
        """
        The underlying SQLAlchemy `Connection`.
        """
        return self._inspector.connection

    @property
    def db_inspector(self) -> CachedDBInspector:
        """
        The underlying :py:class:`CachedDBInspector` instance.
        """
        return self._inspector

    @lru_cache(maxsize=None)
    def _get_all_foreign_key_columns(self, table: str) -> Set[str]:
        """
        A helper function for obtaining a set of all columns of a table that are part of (any) foreign key 
        (doesn't say which foreign key - they can be mixed). Caches its outputs using `@lru_cache`.
        """

        fks = self.db_inspector.get_foreign_keys(table)
        out = set()
        for fk in fks.keys():
            out |= fk

        return out

    @lru_cache(maxsize=None)
    def query_no_distinct(self, table: str, column: str) -> Optional[int]:
        """
        Queries the DB for the total number of distinct values present in a column.

        Equivalent to something like `SELECT count(*) FROM (SELECT DISTINCT [column] FROM [table])`.

        Caches its outputs using `@lru_cache`.
        """

        try:
            tbl = self.db_inspector.get_orm_table(table)
            col = getattr(tbl.c, column)
            query = select(fn.count(distinct(col))).select_from(tbl)
            return self.connection.scalar(query)
        except OperationalError as e:
            return None

    @lru_cache(maxsize=None)
    def query_no_nonnull(self, table: str, column: str) -> Optional[int]:
        """
        Queries the DB for the total number of non-null values present in a column.

        Equivalent to `SELECT count([column]) FROM [table] WHERE [column] IS NOT NULL`

        Caches its outputs using `@lru_cache`
        """

        try:
            tbl = self.db_inspector.get_orm_table(table)
            col = getattr(tbl.c, column)
            query = select(fn.count(col)).select_from(tbl).where(isnot(col, null()))
            return self.connection.scalar(query)
        except OperationalError:
            return None

    def do_guess_column_type(self, table: str, column: str, in_primary_key: bool, must_have_type: bool, col_type: TypeEngine) -> Type[ColumnDef]:
        """
        Determine the :py:class:`ColumnDef` subclass to use with the given table column, based on the 
        SQL column type, the values of the data, and other heuristics.

        Returns the class itself, not an instance of :py:class:`ColumnDef`.

        You may override this method in order to provide custom logic 
        for returning custom :py:class:`ColumnDef` subclasses.
        """
        # check whether this column must be a specific column type
        for output_col_type, sql_col_types in self.DETERMINED_TYPES.items():
            if isinstance(col_type, sql_col_types):
                return output_col_type

        if isinstance(col_type, (Integer, String)):
            cardinality = self.query_no_distinct(table, column)

            # first check if it is an ID-like column name
            if not must_have_type and cardinality is not None and self.ID_NAME_REGEX.search(column):
                n_nonnull = self.query_no_nonnull(table, column)

                # check if there are too many distinct values compared to total
                if n_nonnull is not None and (
                        n_nonnull == 0 or
                        cardinality / n_nonnull > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD):
                    return OmitColumnDef

            if isinstance(col_type, Integer):
                # try matching based on common regex names
                if self.COMMON_NUMERIC_COLUMN_NAME_REGEX.search(column):
                    return NumericColumnDef

                # check if the column name is plural - then it is probably a count
                if self._inflect.singular_noun(column) is not False:
                    return NumericColumnDef

                if cardinality is None or cardinality > self.INTEGER_CARDINALITY_THRESHOLD:
                    return NumericColumnDef

                return CategoricalColumnDef
            elif isinstance(col_type, String):
                if cardinality is None or cardinality > self.STRING_CARDINALITY_THRESHOLD:
                    return TextColumnDef

                n_nonnull = self.query_no_nonnull(table, column)
                if (n_nonnull is not None and
                        cardinality / n_nonnull > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD):
                    return TextColumnDef

                return CategoricalColumnDef

        # no decision - omit
        return OmitColumnDef

    def instantiate_column_type(self, table: str, column: str, in_primary_key: bool, cls: Type[ColumnDef]) -> ColumnDef:
        """
        Instantiate the :py:class:`ColumnDef` subclass instance based on the subclass returned by :py:method:`do_guess_column_type`.

        You may override this method in order to instantiate custom subclasses of :py:class:`ColumnDef` if you've overridden :py:method:`do_guess_column_type`.
        """
        if cls == CategoricalColumnDef:
            cardinality = self.query_no_distinct(table, column)
            assert cardinality is not None, f"Column {table}.{column} was determined to be categorical but cardinality cannot be retrieved."
            return CategoricalColumnDef(key=in_primary_key, card=cardinality)

        if cls in {KeyColumnDef, ForeignKeyColumnDef, NumericColumnDef, DateColumnDef, DateTimeColumnDef, DurationColumnDef, TimeColumnDef, OmitColumnDef, TextColumnDef}:
            return cls(key=in_primary_key)

        raise TypeError(f"No logic for instantiating {cls.__name__} has been provided to {SchemaAnalyzer.__name__}.")

    def guess_column_type(self, table: str, column: str) -> ColumnDef:
        """
        Runs :py:method:`do_guess_column_type` as well as :py:method:`instantiate_column_type` together, and returns the instantiated :py:class:`ColumnDef`.

        Contains additional logic for foreign keys and filtering based on constructor input.
        """

        # omit based on column filters provided in the class constructor
        if (table, column) not in self._not_omitted:
            return OmitColumnDef()

        # retrieve info about the column
        col_type = self.db_inspector.get_columns(table)[column]
        pk = self.db_inspector.get_primary_key(table)
        is_in_pk = column in pk
        must_have_type = (table, column) == self._target

        if not must_have_type:
            if is_in_pk and len(pk) == 1:
                # This is the only primary key column.
                # The column is thus most likely purely an identifier of the row, without holding any extra information,
                # whereas if there are more columns part of the primary key, then we can more likely assume that it conveys more information.

                # Thus, we will mark this with the "purely a primary key" ColumnDef.
                return KeyColumnDef(key=True)

            # if the column is part of a foreign key constraint, return "foreign_key" ColumnDef
            # instead of the actual column type.
            fks = self._get_all_foreign_key_columns(table)
            if column in fks:
                return ForeignKeyColumnDef(key=is_in_pk)

        # delegate to other methods
        guessed_type = self.do_guess_column_type(
            table, column, in_primary_key=is_in_pk, must_have_type=must_have_type, col_type=col_type)

        if must_have_type and isinstance(guessed_type, (OmitColumnDef, KeyColumnDef, ForeignKeyColumnDef)):
            raise TypeError(f"Column '{column}' in table '{table}' cannot be omitted.")

        return self.instantiate_column_type(table, column, in_primary_key=is_in_pk, cls=guessed_type)

    def guess_schema(self) -> Schema:
        """
        Locates all database tables and all columns and runs :py:method:`guess_column_type` for all of them.
        Returns the result as a :py:class:`Schema`.
        """

        schema = Schema()

        for table in wrap_progress(self.db_inspector.get_tables(), verbose=self._verbose, desc="Tables"):
            column_defs = ColumnDefs()
            fks: List[ForeignKeyDef] = list(self.db_inspector.get_foreign_keys(table).values())
            for column in self.db_inspector.get_columns(table):
                column_defs[column] = self.guess_column_type(table, column)

            schema[table] = TableSchema(columns=column_defs, foreign_keys=fks)

        return schema


if __name__ == "__main__":
    import json
    from db_transformer.helpers.objectpickle import serialize
    from sqlalchemy import create_engine

    dataset = "mutagenesis"

    engine = create_engine(f"mariadb+mariadbconnector://guest:relational@relational.fit.cvut.cz:3306/{dataset}")
    with engine.connect() as connection:
        schema = SchemaAnalyzer(connection, verbose=True).guess_schema()

    print(schema)

    with open(f'dataset/{dataset}.json', 'w') as fp:
        json.dump(serialize(schema), fp, indent=3)
