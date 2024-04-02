import re
import warnings
from functools import lru_cache
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from typing import get_args as t_get_args

import inflect
import sqlalchemy.sql.functions as fn
from sqlalchemy import column, table
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import distinct, select
from sqlalchemy.sql.expression import null
from sqlalchemy.sql.operators import isnot
from sqlalchemy.types import (
    TEXT,
    Boolean,
    Date,
    DateTime,
    Integer,
    Interval,
    Numeric,
    String,
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
from db_transformer.db.distinct_cnt_retrieval import (
    DBDistinctCounter,
    DBFullDataFetchLocalDistinctCounter,
    SimpleDBDistinctCounter,
    SimpleStringSeriesMapper,
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
    ForeignKeyDef,
    NumericColumnDef,
    OmitColumnDef,
    Schema,
    TableSchema,
    TextColumnDef,
    TimeColumnDef,
)

__all__ = ["SchemaAnalyzer"]


TargetType = Literal["categorical", "numeric"]

BuiltinDBDistinctCounter = Literal[
    "db_distinct",
    "fetchall_noop",
    "fetchall_rstrip",
    "fetchall_strip",
    "fetchall_unidecode",
    "fetchall_ci",
    "fetchall_rstrip_ci",
    "fetchall_strip_ci",
    "fetchall_unidecode_ci",
    "fetchall_unidecode_rstrip",
    "fetchall_unidecode_strip",
    "fetchall_unidecode_rstrip_ci",
    "fetchall_unidecode_strip_ci",
]


def _get_db_distinct_counter(
    cnt: Union[DBDistinctCounter, BuiltinDBDistinctCounter],
    force_collation: Optional[str] = None,
) -> DBDistinctCounter:
    if isinstance(cnt, str):
        if cnt not in t_get_args(BuiltinDBDistinctCounter):
            raise ValueError(
                f"Unknown DBDistinctCounter '{cnt}'. "
                f"Must be a lambda or one of: {t_get_args(BuiltinDBDistinctCounter)}."
            )

        if cnt == "db_distinct":
            return SimpleDBDistinctCounter(force_collation=force_collation)

        assert cnt.startswith("fetchall_")
        mapper: SimpleStringSeriesMapper = cnt[len("fetchall_") :]

        if force_collation is not None:
            raise ValueError(
                "You can only use the 'force_collation' parameter with 'db_distinct' DBDistinctCounter."
            )

        return DBFullDataFetchLocalDistinctCounter(mapper)
    else:
        if force_collation is not None:
            raise ValueError(
                "You can only use the 'force_collation' parameter with 'db_distinct' DBDistinctCounter."
            )

        return cnt


class SchemaAnalyzer:
    """Auto-detect a database :py:class:`Schema`.

    Column types (as interpreted by the machine learning pipeline)
    are determined based on select heuristics.
    Can be subclassed to support other :py:class:`ColumnDef` instances as well.

    Caches the results of all database queries.
    In order to obtain fresh data after updating a database, please create a new instance of the class.
    """

    DETERMINED_TYPES: Dict[Type[ColumnDef], Tuple[Type[TypeEngine], ...]] = {
        TextColumnDef: (
            LONGTEXT,
            MEDIUMTEXT,
            Unicode,
        ),
        CategoricalColumnDef: (Boolean,),
        NumericColumnDef: (Numeric,),
        DateColumnDef: (Date,),
        DateTimeColumnDef: (DateTime,),
        DurationColumnDef: (Interval,),
        TimeColumnDef: (Time,),
    }
    """
    Per each ColumnDef, set of SQL types that are automatically matched to given ColumnDef.
    """

    ID_NAME_REGEX = re.compile(
        r"_id$|^id_|_id_|Id$|Id[^a-z]|[Ii]dentifier|IDENTIFIER|ID[^a-zA-Z]|ID$|[guGU]uid[^a-z]|[guGU]uid$|[GU]UID[^a-zA-Z]|[GU]UID$"
    )

    COMMON_NUMERIC_COLUMN_NAME_REGEX = re.compile(
        r"balance|amount|size|duration|frequency|count|cnt|votes|score|number|age|year|month|day",
        re.IGNORECASE,
    )  # TODO: add more?

    FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_GUARANTEED_THRESHOLD = 0.05
    """
    The fraction of distinct values to total count of non-null values,
    which decides (in some situations) that type must be categorical.
    If the fraction is below this threshold, marks the column as categorical.
    """

    FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD = 0.2
    """
    The fraction of distinct values to total count of non-null values,
    which decides (in some situations) that type cannot be categorical.
    If the fraction exceeds this threshold, marks the column as something other than categorical.
    """

    def __init__(
        self,
        connection: Union[Connection, DBInspector, DBInspectorInterface],
        omit_filters: Union[
            SetFilterProtocol[Tuple[str, str]],
            Iterable[Tuple[str, str]],
            Tuple[str, str],
            None,
        ] = None,
        target: Optional[Tuple[str, str]] = None,
        target_type: Optional[TargetType] = None,
        db_distinct_counter: Union[
            DBDistinctCounter, BuiltinDBDistinctCounter
        ] = "db_distinct",
        force_collation: Optional[str] = None,
        post_guess_schema_hook: Optional[Callable[[Schema], None]] = None,
        verbose=False,
    ) -> None:
        """Construct the SchemaAnalyzer.

        :field connection: The database connection - instance of SQLAlchemny's `Connection` class, \
or a custom :py:class:`DBInspector` or :py:class:`DBInspectorInterface` instance, which allows to \
e.g. specify database tables or table columns to be completely ignored.
        :field omit_filters: A filter for (table_name, column_name) tuples. Can be one of the following:
            a) a list of such tuples, in which case they will all receive the :py:class:`OmitColumnDef` type
            b) a :py:class:`db_transformer.helpers.collections.set_filter.SetFilter` instance, allowing to specify \
either a whitelist or a blacklist (or both), in which case \
all that is *excluded* will receive the :py:class:`OmitColumnDef` type
            c) a callable which, given a set of values, returns their subset. In this case all that is *excluded* \
will receive the :py:class:`OmitColumnDef` type
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
                f"database is neither {Connection.__name__}, nor "
                f"an implementation of {DBInspectorInterface.__name__}: {connection}"
            )

        self._inspector = inspector

        self._target = target
        self._target_type: Optional[TargetType] = target_type
        self._db_distinct_counter = _get_db_distinct_counter(
            db_distinct_counter, force_collation
        )
        self._force_collation = force_collation
        self._post_guess_schema_hook = post_guess_schema_hook

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
        """The underlying SQLAlchemy `Connection`."""
        return self._inspector.connection

    @property
    def db_inspector(self) -> CachedDBInspector:
        """The underlying :py:class:`CachedDBInspector` instance."""
        return self._inspector

    @lru_cache(maxsize=None)
    def _get_all_non_composite_foreign_key_columns(self, table: str) -> Set[str]:
        """Obtain a set of all columns of a table that are part of any non-composite foreign key.

        (doesn't say which foreign key - they can be mixed). Caches its outputs using `@lru_cache`.
        """
        fks = self.db_inspector.get_foreign_keys(table)
        out = set()
        for fk in fks.keys():
            if len(fk) <= 1:
                out |= fk

        return out

    @lru_cache(maxsize=None)
    def guess_categorical_cardinality(
        self, table_name: str, column_name: str, col_type: TypeEngine
    ) -> Optional[int]:
        """Query the DB for the total number of distinct values present in a column.

        Equivalent to something like `SELECT count(*) FROM (SELECT DISTINCT [column] FROM [table])`.

        Caches its outputs using `@lru_cache`.
        """
        try:
            return self._db_distinct_counter(
                self.connection, table_name, column_name, col_type
            )
        except OperationalError as e:
            if self._verbose:
                warnings.warn(str(e))
            return None

    @lru_cache(maxsize=None)
    def query_no_nonnull(self, table_name: str, column_name: str) -> Optional[int]:
        """Query the DB for the total number of non-null values present in a column.

        Equivalent to `SELECT count([column]) FROM [table] WHERE [column] IS NOT NULL`

        Caches its outputs using `@lru_cache`
        """
        try:
            tbl = table(table_name)
            col = column(column_name)
            query = select(fn.count(col)).select_from(tbl).where(isnot(col, null()))
            return self.connection.scalar(query)
        except OperationalError as e:
            if self._verbose:
                warnings.warn(str(e))
            return None

    def do_guess_column_type(
        self,
        table: str,
        column: str,
        in_primary_key: bool,
        must_have_type: bool,
        col_type: TypeEngine,
    ) -> Type[ColumnDef]:
        """
        Determine the :py:class:`ColumnDef` subclass to use with the given table column.

        Done based on the SQL column type, the values of the data, and other heuristics.

        Returns the class itself, not an instance of :py:class:`ColumnDef`.

        You may override this method in order to provide custom logic
        for returning custom :py:class:`ColumnDef` subclasses.
        """
        # check whether this column must be a specific column type
        for output_col_type, sql_col_types in self.DETERMINED_TYPES.items():
            if isinstance(col_type, sql_col_types):
                return output_col_type

        n_nonnull = self.query_no_nonnull(table, column)
        if n_nonnull == 0:
            if must_have_type:
                raise ValueError(
                    f"Column {column} in table {table} contains only NULL values, "
                    "but it cannot be omitted as it is the target."
                )
            return OmitColumnDef

        if isinstance(col_type, (Integer, String, Text, TEXT)):
            cardinality = self.guess_categorical_cardinality(table, column, col_type)

            if isinstance(col_type, Integer):
                # check if there are too many distinct values compared to total
                if cardinality is None or (
                    n_nonnull is not None
                    and cardinality / n_nonnull
                    > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD
                ):
                    if not must_have_type and self.ID_NAME_REGEX.search(column):
                        return OmitColumnDef

                    return NumericColumnDef

                # try matching based on common regex names
                if self.COMMON_NUMERIC_COLUMN_NAME_REGEX.search(column):
                    return NumericColumnDef

                # check if the column name is plural - then it is probably a count
                if self._inflect.singular_noun(column) is not False:
                    return NumericColumnDef

                return CategoricalColumnDef
            else:
                # check if there are too many distinct values compared to total
                if cardinality is None or (
                    n_nonnull is not None
                    and cardinality / n_nonnull
                    > self.FRACTION_COUNT_DISTINCT_TO_COUNT_NONNULL_IGNORE_THRESHOLD
                ):
                    if not must_have_type and self.ID_NAME_REGEX.search(column):
                        return OmitColumnDef

                    return TextColumnDef

                return CategoricalColumnDef

        # no decision - omit
        return OmitColumnDef

    def instantiate_column_type(
        self,
        table: str,
        column: str,
        in_primary_key: bool,
        col_type: TypeEngine,
        cls: Type[ColumnDef],
    ) -> ColumnDef:
        """
        Instantiate the :py:class:`ColumnDef` subclass instance.

        You may override this method in order to instantiate custom subclasses of :py:class:`ColumnDef`
        if you've overridden :py:method:`do_guess_column_type`.
        """
        if cls == CategoricalColumnDef:
            cardinality = self.guess_categorical_cardinality(table, column, col_type)
            assert cardinality is not None, (
                f"Column {table}.{column} was determined to be categorical "
                "but cardinality cannot be retrieved."
            )
            return CategoricalColumnDef(key=in_primary_key, card=cardinality)

        if cls in {
            NumericColumnDef,
            DateColumnDef,
            DateTimeColumnDef,
            DurationColumnDef,
            TimeColumnDef,
            OmitColumnDef,
            TextColumnDef,
        }:
            return cls(key=in_primary_key)

        raise TypeError(
            f"No logic for instantiating {cls.__name__} has been provided to {SchemaAnalyzer.__name__}."
        )

    def guess_column_type(self, table: str, column: str) -> ColumnDef:
        """Run :py:method:`do_guess_column_type` as well as :py:method:`instantiate_column_type` together.

        Returns the instantiated :py:class:`ColumnDef`.

        Contains additional logic for foreign keys and filtering based on constructor input.
        """
        # omit based on column filters provided in the class constructor
        if (table, column) not in self._not_omitted:
            return OmitColumnDef()

        # retrieve info about the column
        col_type = self.db_inspector.get_columns(table)[column]
        pk = self.db_inspector.get_primary_key(table)
        is_in_pk = column in pk
        is_target = (table, column) == self._target

        guessed_type: Optional[Type[ColumnDef]] = None
        if is_target and self._target_type is not None:
            if self._target_type == "categorical":
                guessed_type = CategoricalColumnDef
            elif self._target_type == "numeric":
                guessed_type = NumericColumnDef
            else:
                raise ValueError()
        else:
            if is_in_pk and len(pk) == 1:
                # This is the only primary key column.
                # The column is thus most likely purely an identifier of the row, without holding any extra information,
                # whereas if there are more columns part of the primary key, then we
                # can more likely assume that it conveys more information.

                # Thus, we will mark this as "omit", to signify that this column should not be a feature
                return OmitColumnDef(key=True)

            # if the column is part of a non-composite foreign key constraint, return "omit" ColumnDef
            # instead of the actual column type, as it most likely should not be a feature
            non_comp_fks = self._get_all_non_composite_foreign_key_columns(table)
            if column in non_comp_fks:
                return OmitColumnDef(key=is_in_pk)

        # delegate to other methods
        if guessed_type is None:
            guessed_type = self.do_guess_column_type(
                table,
                column,
                in_primary_key=is_in_pk,
                must_have_type=is_target,
                col_type=col_type,
            )

        if is_target and isinstance(guessed_type, OmitColumnDef):
            raise TypeError(f"Column '{column}' in table '{table}' cannot be omitted.")

        return self.instantiate_column_type(
            table, column, in_primary_key=is_in_pk, col_type=col_type, cls=guessed_type
        )

    def guess_schema(self) -> Schema:
        """Locate all database tables and all columns and run :py:method:`guess_column_type` for all of them.

        Returns the result as a :py:class:`Schema`.
        """
        schema = Schema()

        for table_name in wrap_progress(
            self.db_inspector.get_tables(), verbose=self._verbose, desc="Analyzing schema"
        ):
            column_defs = ColumnDefs()
            fks: List[ForeignKeyDef] = list(
                self.db_inspector.get_foreign_keys(table_name).values()
            )
            for column_name in self.db_inspector.get_columns(table_name):
                column_defs[column_name] = self.guess_column_type(table_name, column_name)

            schema[table_name] = TableSchema(columns=column_defs, foreign_keys=fks)

        if self._post_guess_schema_hook is not None:
            self._post_guess_schema_hook(schema)

        return schema
