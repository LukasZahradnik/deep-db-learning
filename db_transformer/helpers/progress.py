from typing import Iterable, TypeVar


_T = TypeVar("_T")


def wrap_progress(vals: Iterable[_T], verbose: bool, **kwargs) -> Iterable[_T]:
    if verbose:
        from tqdm.std import tqdm

        return tqdm(vals, leave=None, **kwargs)
    return vals
