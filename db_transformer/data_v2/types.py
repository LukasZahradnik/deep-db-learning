from collections.abc import Callable, Mapping
from typing import Any, Dict, Generic, Optional, TypeVar, Union


_Value = TypeVar('_Value')


class DotDict(Generic[_Value], Mapping[str, _Value]):
    """
    A dict-like class that allows dot-notation access to individual elements.

    In addition, all methods that set values (such as `update()`) are routed through `__setitem__`.
    Thus, when re-defining `__setitem__`, you are guaranteed to intercept any value being set.
    This is not the case for traditional python `dict`.
    """

    def __init__(self, __items: Optional[Union['DotDict[_Value]', Dict[str, _Value]]] = None, /, **kwargs: _Value):
        self.__data: Dict[str, _Value]
        object.__setattr__(self, '_DotDict__data', {})
        self.update(__items, **kwargs)

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


class TypeCheckedDotDict(DotDict[_Value], Generic[_Value]):
    """
    A `DotDict` that is also type-checked or type-casted. 
    Type-checking / type-casting is performed by the `__mapping` argument passed in the constructor as the first positional argument.
    """

    def __init__(self, __mapping: Callable[[Any], _Value], __items: Optional[Union['DotDict[Any]', Dict[str, Any]]] = None, /, **kwargs: _Value):
        self.__mapping: Callable[[Any], _Value]
        object.__setattr__(self, '_TypeCheckedDotDict__mapping', __mapping)
        super().__init__(__items, **kwargs)

    def __setitem__(self, key: str, value: _Value):
        return super().__setitem__(key, self.__mapping(value))
