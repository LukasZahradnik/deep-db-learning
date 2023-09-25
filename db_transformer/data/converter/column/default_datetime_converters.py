import datetime
from typing import Optional

import pandas as pd

from db_transformer.schema.columns import NumericColumnDef

from .pandas_converter import PandasConverter

__ALL__ = ['DateConverter', 'DateTimeConverter', 'TimestampConverter', 'TimeConverter']


class DateConverter(PandasConverter):
    """Converts column to year and day of year."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('_year', lambda s: (s.dt.year, NumericColumnDef())),
            ('_dayofyear', lambda s: (s.dt.dayofyear, NumericColumnDef())),
            skip_if_allsame=skip_if_allsame)


def _get_seconds_since_midnight(s: pd.Series) -> pd.Series:
    return ((s - s.dt.normalize()) / pd.Timedelta('1 second')).fillna(0).astype(int)


def _get_seconds_since_midnight_time(t: Optional[datetime.time]) -> Optional[int]:
    if t is None:
        return None

    return t.second + (t.minute + t.hour * 60) * 60


class DateTimeConverter(PandasConverter):
    """Converts column to year, day of year, and seconds since midnight."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('_year', lambda s: (s.dt.year, NumericColumnDef())),
            ('_dayofyear', lambda s: (s.dt.dayofyear, NumericColumnDef())),
            ('_seconds_since_midnight', lambda s: (_get_seconds_since_midnight(s), NumericColumnDef())),
            skip_if_allsame=skip_if_allsame)


class TimeConverter(PandasConverter):
    """Converts column to seconds since midnight."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('', lambda s: (s.map(lambda v: _get_seconds_since_midnight_time(v)), NumericColumnDef())),
            skip_if_allsame=skip_if_allsame
        )


class TimestampConverter(PandasConverter):
    """Converts column to a timestamp since epoch in seconds."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('', lambda s: (s.astype('int64') // 10**9, NumericColumnDef())),
            skip_if_allsame=skip_if_allsame)
