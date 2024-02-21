from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, TypeVar
from typing import get_args as t_get_args
from datetime import datetime, time
import torch
from torch_geometric.data.dataset import Union

from db_transformer.data.convertor.columns.column_convertor import ColumnConvertor
from db_transformer.data.convertor.columns.num_convertor import NumConvertor
from db_transformer.schema.columns import (
    ColumnDef,
    DateColumnDef,
    DateTimeColumnDef,
    DurationColumnDef,
    TimeColumnDef,
)


__all__ = [
    "DateSegment",
    "DateConvertor",
    "TimeSegment",
    "TimeConvertor",
    "DateTimeSegment",
    "DateTimeConvertor",
    "DurationConvertor",
]

_TColumnDef = TypeVar("_TColumnDef", bound=ColumnDef)
_TSegment = TypeVar("_TSegment", bound=str)


class _SegmentedConvertor(
    ColumnConvertor[_TColumnDef], Generic[_TColumnDef, _TSegment], ABC
):
    def __init__(self, segments: List[_TSegment]) -> None:
        super().__init__()
        self.segment_convertors = torch.nn.ModuleDict(
            {segment: NumConvertor() for segment in segments}
        )

        if len(self.segment_convertors) == 0:
            raise ValueError(
                f"Must specify dimensionalities for at least one of the following segments (those you want represented): "
                f"{self.get_all_segments()}"
            )

    @classmethod
    @abstractmethod
    def get_all_segments(cls) -> List[str]:
        pass

    @property
    def segments(self) -> List[str]:
        return list(self.segment_convertors.keys())

    def create(self, column_def: DateColumnDef):
        for convertor in self.segment_convertors.values():
            convertor: NumConvertor
            convertor.create(None)

    @abstractmethod
    def _retrieve_segment_value(self, segment: _TSegment, value: Any) -> float:
        pass

    def forward(self, value) -> Optional[torch.Tensor]:
        tensors = []
        for segment in self.segments:
            v = self._retrieve_segment_value(segment, value)
            v = self.segment_convertors[segment](v)
            tensors.append(v)
        return torch.concat(tensors)


DateSegment = Literal["year", "month", "day", "ordinal", "timestamp"]


class DateConvertor(_SegmentedConvertor[DateColumnDef, DateSegment]):
    _DATE_SEGMENT_TO_NUMERIC: Dict[DateSegment, Callable[[datetime], float]] = {
        "year": lambda datetime: datetime.year,
        "month": lambda datetime: datetime.month,
        "day": lambda datetime: datetime.day,
        "ordinal": lambda datetime: datetime.toordinal(),
        "timestamp": lambda datetime: datetime.timestamp(),
    }

    @classmethod
    def get_all_segments(cls) -> List[str]:
        return list(t_get_args(DateSegment))

    def _retrieve_segment_value(self, segment: DateSegment, value: Any) -> float:
        if value is not None:
            value = datetime.strptime(value, "%Y-%m-%d")
            return self._DATE_SEGMENT_TO_NUMERIC[segment](value)

        return 0  # TODO how to handle None?


TimeSegment = Literal["hours", "minutes", "seconds", "total_seconds"]


class TimeConvertor(_SegmentedConvertor[TimeColumnDef, TimeSegment]):
    _TIME_SEGMENT_TO_NUMERIC: Dict[TimeSegment, Callable[[time], float]] = {
        "hours": lambda time: time.hour,
        "minutes": lambda time: time.minute,
        "seconds": lambda time: time.second,
        "total_seconds": lambda time: time.hour * 3600 + time.minute * 60 + time.second,
    }

    @classmethod
    def get_all_segments(cls) -> List[str]:
        return list(t_get_args(TimeSegment))

    def _retrieve_segment_value(self, segment: TimeSegment, value: Any) -> float:
        print(value, type(value))
        raise NotImplementedError()  # TODO


DateTimeSegment = Union[TimeSegment, DateSegment]


def _merge_segments() -> Dict[DateTimeSegment, Callable[[datetime], float]]:
    out: Dict[DateTimeSegment, Callable[[datetime], float]] = {}
    for segment, c_datetime in DateConvertor._DATE_SEGMENT_TO_NUMERIC.items():
        out[segment] = c_datetime
    for segment, c_time in TimeConvertor._TIME_SEGMENT_TO_NUMERIC.items():
        out[segment] = lambda datetime: c_time(datetime.time())
    return out


class DateTimeConvertor(_SegmentedConvertor[DateTimeColumnDef, DateTimeSegment]):
    _DATETIME_SEGMENT_TO_NUMERIC = _merge_segments()

    @classmethod
    def get_all_segments(cls) -> List[str]:
        return list(t_get_args(DateSegment)) + list(t_get_args(TimeSegment))

    def _retrieve_segment_value(self, segment: DateTimeSegment, value: Any) -> float:
        if value is not None:
            value = datetime.fromisoformat(value)
            return self._DATETIME_SEGMENT_TO_NUMERIC[segment](value)

        return 0  # TODO how to handle None?


class DurationConvertor(ColumnConvertor[DurationColumnDef]):
    def __init__(self) -> None:
        super().__init__()
        self.num = NumConvertor()

    def create(self, column_def: DurationColumnDef):
        self.num.create(None)

    def forward(self, value) -> torch.Tensor:
        print(value, type(value))
        raise NotImplementedError()  # TODO
