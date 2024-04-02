from typing import Iterable, TypeVar


_T = TypeVar("_T")


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def wrap_progress(vals: Iterable[_T], verbose: bool, **kwargs) -> Iterable[_T]:
    if verbose:
        if is_notebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm.std import tqdm

        return tqdm(vals, leave=True, **kwargs)
    return vals
