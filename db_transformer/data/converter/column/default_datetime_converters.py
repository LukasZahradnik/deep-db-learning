import pandas as pd

from .pandas_converter import PandasConverter

__ALL__ = ['DateConverter', 'DateTimeConverter', 'TimestampConverter', 'TimeConverter']


class DateConverter(PandasConverter):
    """Converts column to year and day of year."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('_year', lambda s: s.dt.year),
            ('_dayofyear', lambda s: s.dt.dayofyear),
            skip_if_allsame=skip_if_allsame)


def _get_seconds_since_midnight(s: pd.Series) -> pd.Series:
    return ((s - s.dt.normalize()) / pd.Timedelta('1 second')).astype(int)


class DateTimeConverter(PandasConverter):
    """Converts column to year, day of year, and seconds since midnight."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('_year', lambda s: s.dt.year),
            ('_dayofyear', lambda s: s.dt.dayofyear),
            ('_seconds_since_midnight', _get_seconds_since_midnight),
            skip_if_allsame=skip_if_allsame)


class TimeConverter(PandasConverter):
    """Converts column to seconds since midnight."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('', _get_seconds_since_midnight),
            skip_if_allsame=skip_if_allsame
        )


class TimestampConverter(PandasConverter):
    """Converts column to a timestamp since epoch in seconds."""

    def __init__(self, skip_if_allsame=True) -> None:
        super().__init__(
            ('', lambda s: s.dt.astype('int64') // 10**9),
            skip_if_allsame=skip_if_allsame)
