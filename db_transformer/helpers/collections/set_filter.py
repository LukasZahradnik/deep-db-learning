from typing import Generic, Hashable, Iterable, Optional, Protocol, Set, TypeVar, Union


_THashable = TypeVar("_THashable", bound=Hashable)

__all__ = [
    "SetFilter",
    "SetFilterProtocol",
]


class SetFilter(Generic[_THashable]):
    def __init__(
        self,
        include: Optional[Union[Iterable[_THashable], _THashable]] = None,
        exclude: Optional[Union[Iterable[_THashable], _THashable]] = None,
    ) -> None:
        self._include = set(include) if include is not None else None
        self._exclude = set(exclude) if exclude is not None else None

    def __call__(self, v: Set[_THashable]) -> Set[_THashable]:
        if self._include is not None:
            v = v & self._include

        if self._exclude is not None:
            v = v - self._exclude

        return v


class SetFilterProtocol(Protocol[_THashable]):
    def __call__(self, v: Set[_THashable]) -> Set[_THashable]: ...
