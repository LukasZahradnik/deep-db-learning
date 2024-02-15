from collections import OrderedDict
from typing import Callable, Literal, Optional, Protocol, Union
from typing import get_args as t_get_args

import pandas as pd
import sqlalchemy.sql.functions as fn
from sqlalchemy import column, table
from sqlalchemy.engine import Connection
from sqlalchemy.sql import distinct, select
from sqlalchemy.types import (
    TEXT,
    String,
    Text,
    TypeEngine,
    Unicode,
)
from unidecode import unidecode

__ALL__ = [
    "DBDistinctCounter",
    "SimpleDBDistinctCounter",
    "DBFullDataFetchLocalDistinctCounter",
]


class DBDistinctCounter(Protocol):
    def __call__(
        self, conn: Connection, table_name: str, column_name: str, col_type: TypeEngine
    ) -> Optional[int]: ...


class SimpleDBDistinctCounter(DBDistinctCounter):
    def __init__(self, force_collation: Optional[str]) -> None:
        super().__init__()
        self.force_collation = force_collation

    def __call__(
        self, conn: Connection, table_name: str, column_name: str, col_type: TypeEngine
    ) -> Optional[int]:
        tbl = table(table_name)
        col = column(column_name)

        if self.force_collation is not None and isinstance(
            col_type, (String, Text, TEXT, Unicode)
        ):
            col = col.collate(self.force_collation)

        # subquery instead of COUNT(DISTINCT [col]) in order to include null values in the count
        query = select(fn.count()).select_from(
            select(distinct(col)).select_from(tbl).subquery()
        )
        return conn.scalar(query)


SimpleStringSeriesMapper = Literal[
    "noop",
    "rstrip",
    "strip",
    "unidecode",
    "ci",
    "rstrip_ci",
    "strip_ci",
    "unidecode_ci",
    "unidecode_rstrip",
    "unidecode_strip",
    "unidecode_rstrip_ci",
    "unidecode_strip_ci",
]

SeriesMapper = Callable[[pd.Series], pd.Series]


def _map_safe(c: Callable[[str], str]) -> Callable[[Optional[str]], Optional[str]]:
    def _func(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None

        return c(s)

    return _func


SIMPLE_STRING_SERIES_MAPPERS: OrderedDict[SimpleStringSeriesMapper, SeriesMapper] = (
    OrderedDict(
        [
            ("noop", lambda s: s),
            ("ci", lambda s: s.str.lower()),
            ("rstrip", lambda s: s.str.rstrip()),
            ("strip", lambda s: s.str.strip()),
            ("unidecode", lambda s: s.map(_map_safe(lambda s: unidecode(s)))),
            ("rstrip_ci", lambda s: s.str.lower().str.rstrip()),
            ("strip_ci", lambda s: s.str.lower().str.strip()),
            (
                "unidecode_ci",
                lambda s: s.str.lower().map(_map_safe(lambda s: unidecode(s))),
            ),
            (
                "unidecode_rstrip",
                lambda s: s.map(_map_safe(lambda s: unidecode(s))).str.rstrip(),
            ),
            (
                "unidecode_strip",
                lambda s: s.map(_map_safe(lambda s: unidecode(s))).str.strip(),
            ),
            (
                "unidecode_rstrip_ci",
                lambda s: s.str.lower().map(_map_safe(lambda s: unidecode(s))).str.rstrip(),
            ),
            (
                "unidecode_strip_ci",
                lambda s: s.str.lower().map(_map_safe(lambda s: unidecode(s))).str.strip(),
            ),
        ]
    )
)


def get_string_mapper(
    mapper: Union[SeriesMapper, SimpleStringSeriesMapper]
) -> SeriesMapper:
    if isinstance(mapper, str):
        if mapper not in SIMPLE_STRING_SERIES_MAPPERS:
            raise ValueError(
                f"Unknown mapper '{mapper}'. "
                f"Must be a lambda or one of: {t_get_args(SimpleStringSeriesMapper)}."
            )
        return SIMPLE_STRING_SERIES_MAPPERS[mapper]
    else:
        return mapper


class DBFullDataFetchLocalDistinctCounter(DBDistinctCounter):
    def __init__(
        self, string_mapper: Union[SeriesMapper, SimpleStringSeriesMapper] = "noop"
    ) -> None:
        super().__init__()
        self.string_mapper = get_string_mapper(string_mapper)

    def __call__(
        self, conn: Connection, table_name: str, column_name: str, col_type: TypeEngine
    ) -> Optional[int]:
        tbl = table(table_name)
        col = column(column_name)

        # subquery instead of COUNT(DISTINCT [col]) in order to include null values in the count
        the_data_df = pd.read_sql_query(select(col).select_from(tbl), conn)
        the_data_series = the_data_df[the_data_df.columns[0]]

        try:
            the_data_series = self.string_mapper(the_data_series)
        except AttributeError:
            # not a string series - leave as is
            pass

        return the_data_series.nunique(dropna=False)
