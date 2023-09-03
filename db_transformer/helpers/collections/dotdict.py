from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, Generic, Optional, TypeVar, Union


_Value = TypeVar('_Value')

__all__ = [
    'DotDict',
    'OrderedDotDict',
]

class DotDict(Generic[_Value], Mapping[str, _Value]):
    """
    A dict-like class that allows dot-notation access to individual elements.

    In addition, all methods that set values (such as `update()`) are routed through `__setitem__`.
    Thus, when re-defining `__setitem__`, you are guaranteed to intercept any value being set.
    This is not the case for traditional python `dict`.
    """

    def __new__(cls, *kargs, **kwargs) -> 'DotDict[_Value]':
        out = object.__new__(cls)
        object.__setattr__(out, '_DotDict__data', {})
        return out

    def __init__(self, __items: Optional[Union['DotDict[_Value]', Dict[str, _Value]]] = None, /, **kwargs: _Value):
        self.__data: Dict[str, _Value]
        self.update(__items, **kwargs)

    def __getstate__(self) -> dict:
        return dict(self)

    def __setstate__(self, state: dict):
        for k, v in state.items():
            self[k] = v

    def update(self, __items: Optional[Union['DotDict[_Value]', Dict[str, _Value]]] = None, /, **kwargs: _Value):
        ndata = {}
        if isinstance(__items, DotDict):
            ndata.update(__items.__data)
        elif __items is not None:
            ndata.update(__items)
        ndata.update(kwargs)

        # perform the update such that
        for k, v in ndata.items():
            self[k] = v

    def __setattr__(self, key: str, value: _Value):
        self[key] = value

    def __getattr__(self, key: str) -> _Value:
        try:
            return self[key]
        except KeyError:
            raise AttributeError()

    def __setitem__(self, key: str, value: _Value):
        self.__data[key] = value

    def __getitem__(self, key: str) -> _Value:
        return self.__data[key]

    def __delitem__(self, key: str):
        del self.__data[key]

    def __len__(self) -> int:
        return len(self.__data)

    def keys(self):
        return self.__data.keys()

    def values(self):
        return self.__data.values()

    def items(self):
        return self.__data.items()

    def __iter__(self):
        return self.__data.__iter__()

    def __repr__(self) -> str:
        out_inner = ',\n'.join((f"{k} = {v}" for k, v in self.items()))
        out_inner = '\n'.join(['    ' + line for line in out_inner.splitlines()])

        out = self.__class__.__name__ + '('
        if out_inner:
            out += '\n' + out_inner + '\n'
        out += ')'

        return out

    def __str__(self) -> str:
        return self.__repr__()


class OrderedDotDict(Generic[_Value], DotDict[_Value]):
    """
    A dict-like class that allows dot-notation access to individual elements. Ordered.

    In addition, all methods that set values (such as `update()`) are routed through `__setitem__`.
    Thus, when re-defining `__setitem__`, you are guaranteed to intercept any value being set.
    This is not the case for traditional python `dict`.
    """

    def __new__(cls, *kargs, **kwargs) -> 'OrderedDotDict[_Value]':
        out = object.__new__(cls)
        object.__setattr__(out, '_DotDict__data', OrderedDict())
        return out
